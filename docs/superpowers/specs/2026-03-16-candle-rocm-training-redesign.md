# Picochat Training Redesign: candle-rocm + Foundation-First Training

**Date:** 2026-03-16
**Status:** Approved

## Problem Statement

The current picochat training pipeline produces a 6M parameter model that generates incoherent text. Root causes:

1. **SFT-as-knowledge-database**: The SFT approach force-feeds question-answer pairs, causing catastrophic interference at small model sizes. Similar patterns (e.g., 7 capital city questions) collapse into a single dominant answer ("Paris").
2. **CPU-only training**: The Rust/candle training code has no ROCm support, limiting us to 95 tok/s on CPU. An AMD RX 5700 XT (8GB VRAM, ROCm 6.2) sits unused.
3. **Tiny model capacity**: 6M params with 4096 vocab cannot distinguish similar patterns or sustain coherent multi-token generation.
4. **Wrong tokenizer**: 4096-vocab BPE trained on children's stories fragments code badly.

## Goals

- **Minimal viable LLM**: Natural conversation, expresses uncertainty when unsure, general knowledge.
- **Deep ANSI C (C89/C90) ability**: Understand and generate C code, explain undefined behavior, strict aliasing, memory management. Fully Turing-complete code generation.
- **Open-source ROCm backend for candle**: Standalone library others can use for AMD GPU training with candle.

## Non-Goals

- Multi-GPU training
- C99/C11/C23 (focus on C89/C90 only)
- Convolution / vision model support in initial ROCm backend

---

## Phase 0: Hardware Proof-of-Concept

The RX 5700 XT is gfx1010 (RDNA 1), which was dropped from official ROCm support after 5.x. However, PyTorch 2.5.1+rocm6.2 detects the GPU and works (confirmed). Before building the full backend, we validate HIP/rocBLAS work at the FFI level:

1. Compile a trivial HIP kernel targeting gfx1010 with `hipcc`
2. Run `hipMalloc` / `hipMemcpy` roundtrip
3. Execute `rocblas_sgemm` on a small matrix
4. If gfx1010 is rejected, test with `HSA_OVERRIDE_GFX_VERSION=10.3.0` (masquerade as gfx1030)

This validates or kills the hardware path in an afternoon. If gfx1010 fails entirely, the fallback is Python/PyTorch training (which already works on this GPU).

---

## Phase A: candle-rocm (GPU Backend)

### Architecture

Three-layer crate workspace, added as a git submodule at `picochat/candle-rocm/`:

```
candle-rocm/
├── Cargo.toml                   (workspace)
│
├── hip-sys/                     Layer 1: Raw FFI bindings
│   ├── build.rs                 (find ROCm, link libamdhip64 + librocblas)
│   └── src/
│       ├── lib.rs
│       ├── hip_runtime.rs       (hipMalloc, hipFree, hipMemcpy, hipLaunchKernel, etc.)
│       └── rocblas.rs           (rocblas_sgemm_strided_batched, etc.)
│
├── hip-runtime/                 Layer 2: Safe Rust abstractions
│   └── src/
│       ├── lib.rs
│       ├── device.rs            (HipDevice — safe device handle, Drop cleans up)
│       ├── memory.rs            (DeviceBuffer<T> — typed GPU memory with RAII)
│       ├── module.rs            (HipModule — load .hsaco kernels, get functions)
│       └── blas.rs              (RocBlas — safe GEMM wrapper)
│
├── kernels/                     HIP kernel sources (.hip files)
│   ├── affine.hip
│   ├── unary.hip               (exp, log, relu, tanh, sqrt, neg, recip, etc.)
│   ├── binary.hip              (add, mul, sub, div, max, min + comparisons)
│   ├── ternary.hip             (where_cond)
│   ├── reduce.hip              (sum, min, max, argmin, argmax)
│   ├── cast.hip                (f32↔f16, f32↔bf16, etc.)
│   ├── indexing.hip            (gather, scatter_add, index_select, index_add)
│   ├── fill.hip                (fill, copy2d)
│   ├── softmax.hip             (fused numerically-stable softmax + log_softmax)
│   └── rope.hip                (fused rotary position embeddings)
│
└── candle-backend/              Layer 3: candle integration
    └── src/
        ├── lib.rs
        ├── device.rs            (BackendDevice impl for RocmDevice)
        ├── storage.rs           (BackendStorage impl — 21 methods)
        ├── error.rs             (RocmError enum mirroring CudaError)
        └── utils.rs             (Map1/Map2/Map3 type dispatch traits)
```

### Candle Integration — Drop-in Replacement

**Fork strategy:** We vendor candle-core 0.8 in the submodule and add a `rocm` feature flag. The diff is minimal — adding `Device::Rocm(RocmDevice)` to the `Device` enum, `Storage::Rocm(RocmStorage)` to the `Storage` enum, and match arms that dispatch to our backend. We pin to candle 0.8 and track upstream selectively. If the upstream PR is rejected, the fork remains viable since the diff is small (~200 lines of match arm additions).

candle-core's `Device` enum gets a new variant behind a `rocm` feature flag:

```rust
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),
    #[cfg(feature = "metal")]
    Metal(MetalDevice),
    #[cfg(feature = "rocm")]
    Rocm(RocmDevice),
}
```

Existing picochat code changes one line:
```rust
// Before
let device = Device::Cpu;
// After
let device = Device::Rocm(RocmDevice::new(0)?);
```

All model code, training loops, optimizers, and checkpointing remain unchanged. They operate on `&Device` and `Tensor` generically.

Picochat's Cargo.toml switches the candle dependency to the submodule:
```toml
candle-core = { path = "candle-rocm/candle-core", features = ["rocm"] }
```

### Autograd / Backward Pass

Candle's autograd system does NOT require custom backward kernels per operation. The backward pass decomposes into the same forward primitives (matmul with transposed args, element-wise mul for gradient scaling, sum for gradient accumulation). Our `BackendStorage` implementation of forward ops is sufficient — candle handles the backward graph automatically.

### Kernel Compilation

- `.hip` source files compiled to `.hsaco` (GPU binary) at build time via `hipcc`
- Loaded at runtime via `hipModuleLoad` / `hipModuleGetFunction`
- Kernels mirror candle's existing CUDA kernel naming convention
- All kernels handle strided/non-contiguous tensors via dims/strides parameters

### BackendDevice Method Coverage

All 12 methods on candle's `BackendDevice` trait:

| Method | Implementation | Notes |
|--------|---------------|-------|
| `new(ordinal)` | `hipSetDevice(ordinal)` | Create device handle, init rocBLAS handle |
| `location()` | Return `DeviceLocation::Rocm { gpu_id }` | Trivial |
| `same_device(other)` | Compare gpu_id | Trivial |
| `zeros_impl(shape, dtype)` | `hipMemset` or fill kernel | Zero-initialized GPU buffer |
| `ones_impl(shape, dtype)` | Fill kernel with 1.0 | Uses fill.hip |
| `alloc_uninit(shape, dtype)` | `hipMalloc` | Unsafe, no initialization |
| `storage_from_slice(data)` | `hipMemcpy` H2D | Copy host data to GPU |
| `storage_from_cpu_storage(s)` | `hipMemcpy` H2D | Copy CpuStorage to GPU |
| `storage_from_cpu_storage_owned(s)` | `hipMemcpy` H2D | Same as above, takes ownership |
| `rand_uniform(shape, dtype, lo, hi)` | hiprand or custom kernel | Needed for weight init |
| `rand_normal(shape, dtype, mean, std)` | hiprand or custom kernel | Needed for weight init |
| `set_seed(seed)` | Configure hiprand generator | Reproducibility |
| `synchronize()` | `hipDeviceSynchronize()` | Block until all ops complete |

**RNG:** hipRAND provides `hiprandGenerateUniform` / `hiprandGenerateNormal`. We wrap these in a `HipRng` struct (mirroring candle's `CudaRng`). Only F32/F64 supported; F16/BF16 generate in F32 then cast.

### BackendStorage Method Coverage

Every method on candle's `BackendStorage` trait, categorized:

**Implement with HIP kernel:**
| Method | Kernel | Notes |
|--------|--------|-------|
| `affine` | affine.hip | y = mul*x + add |
| `unary_impl` | unary.hip | exp, log, relu, tanh, sqrt, neg, recip, sqr, silu, gelu, erf, sign, abs, floor, ceil, round |
| `binary_impl` | binary.hip | add, mul, sub, div, maximum, minimum |
| `cmp` | binary.hip | eq, ne, lt, le, gt, ge (returns u8) |
| `where_cond` | ternary.hip | conditional select |
| `reduce_op` | reduce.hip | sum, min, max, argmin, argmax |
| `powf` | unary.hip | element-wise power |
| `elu` | unary.hip | exponential linear unit |
| `gather` | indexing.hip | gather along dimension |
| `scatter_add` | indexing.hip | scatter with add |
| `index_select` | indexing.hip | embedding lookup |
| `index_add` | indexing.hip | accumulate at indices |
| `to_dtype` | cast.hip | dtype conversion |
| `copy_strided_src` | fill.hip | strided GPU→GPU copy |
| `copy2d` | fill.hip | 2D block copy |

**Implement via rocBLAS:**
| Method | API | Notes |
|--------|-----|-------|
| `matmul` | `rocblas_sgemm_strided_batched` | Batched GEMM, mirrors cuBLAS exactly |

**Implement via HIP runtime (no kernel needed):**
| Method | API | Notes |
|--------|-----|-------|
| `try_clone` | hipMemcpy D2D | Deep copy |
| `to_cpu_storage` | hipMemcpy D2H | GPU→CPU transfer |
| `dtype` / `device` | — | Return stored metadata |

**Zero-copy (stride manipulation only):**
- `transpose`, `reshape`, `narrow`, `chunk`, `squeeze`, `unsqueeze`, `expand` — these are handled by candle's `Layout` and never call `BackendStorage` methods directly. They modify strides/offsets on the `Tensor`, and the kernels read dims/strides params to handle non-contiguous data.

**Not needed for MVP (return `Err` / unimplemented):**
| Method | Reason |
|--------|--------|
| `conv1d` | No CNN layers in transformer |
| `conv2d` | No CNN layers |
| `conv_transpose1d/2d` | No CNN layers |
| `avg_pool2d` / `max_pool2d` | No pooling |
| `upsample_nearest1d/2d` | No upsampling |

**Additional fused kernels for performance (not in BackendStorage but needed):**
| Kernel | Why |
|--------|-----|
| `softmax.hip` | Numerically stable fused softmax (max-subtract trick). Without this, decomposed exp→sum→div overflows on large logits. |
| `rope.hip` | Fused rotary position embeddings. Without this, interleaved sin/cos multiply-add decomposes into many small ops. |

**Not implemented initially:** convolutions, pooling, upsampling, cuDNN equivalents.

### TDD Plan

#### Phase 1: hip-sys (FFI bindings)
```
test_hip_get_device_count        → hipGetDeviceCount returns ≥ 1
test_hip_set_device              → hipSetDevice(0) succeeds
test_hip_malloc_free             → allocate 1MB, free it, no error
test_hip_memcpy_roundtrip        → write f32 array to GPU, read back, values match
test_hip_device_properties       → query name, memory size, compute capability
test_rocblas_create_destroy      → create/destroy rocblas handle
test_rocblas_sgemm               → 2x2 matmul, verify against known result
```

#### Phase 2: hip-runtime (safe wrappers)
```
test_device_new                  → HipDevice::new(0) succeeds
test_device_buffer_alloc         → DeviceBuffer::<f32>::alloc(1024)
test_device_buffer_roundtrip     → host→device→host preserves values
test_device_buffer_zeros         → alloc_zeros reads back all zeros
test_device_buffer_drop          → freed on drop, no leak
test_module_load_kernel          → compile trivial kernel, load .hsaco
test_kernel_launch_add_scalar    → kernel adds 1.0 to every element
test_rocblas_gemm_identity       → multiply by identity matrix
test_rocblas_gemm_known          → 4x4 matmul against reference
test_rocblas_gemm_batched        → batched GEMM, verify each batch
test_hiprand_uniform             → uniform random in [0,1], verify range
test_hiprand_normal              → normal random, verify mean/std within tolerance
test_hiprand_seeded              → same seed produces same values
```

#### Phase 3: kernels (numerical correctness)
```
test_affine_f32                  → y = 2.0*x + 3.0, compare to CPU
test_unary_relu                  → relu([-2,-1,0,1,2]) = [0,0,0,1,2]
test_unary_exp_log_roundtrip     → exp(log(x)) ≈ x within epsilon
test_binary_add                  → element-wise add matches CPU
test_binary_mul                  → element-wise mul matches CPU
test_reduce_sum                  → sum along axis matches CPU
test_reduce_max                  → max along axis matches CPU
test_cast_f32_to_f16             → cast preserves range
test_index_select                → embedding lookup matches CPU
test_fill                        → fill buffer with value, verify
test_strided_operations          → non-contiguous tensor ops correct
test_softmax                     → fused softmax matches CPU (test large logits for stability)
test_log_softmax                 → log_softmax matches CPU within epsilon
test_rope                        → RoPE matches CPU for known positions
test_where_cond                  → ternary select matches CPU
```

#### Phase 4: candle integration
```
test_backend_device_new          → Device::Rocm(0) creates valid device
test_backend_zeros               → Tensor::zeros on ROCm
test_backend_from_slice          → Tensor::from_slice transfers to GPU
test_backend_to_cpu              → .to_device(Device::Cpu) roundtrip
test_backend_matmul              → Tensor::matmul matches CPU
test_backend_add                 → (a + b) on GPU matches CPU
test_backend_softmax             → softmax matches CPU within epsilon
test_forward_pass                → picochat forward pass GPU = CPU
test_backward_pass               → gradient computation valid
```

---

## Phase B: Training Redesign

### New Tokenizer

- **Vocab size:** 16,384 (4x current, uses picochat's existing BPE trainer)
- **Training corpus:** Mixed English text + ANSI C code
- **Split pattern:** GPT-4 regex (already implemented, handles code well)
- **Special tokens:** Same 16 slots. Rename `<|python_start|>` / `<|python_end|>` to `<|code_start|>` / `<|code_end|>` (language-neutral, since we're targeting C not Python)

### Model Architecture

| Parameter | Old | New |
|-----------|-----|-----|
| Depth (n_layer) | 4 | 6 |
| n_embd | 256 | 384 |
| n_head | 4 | 6 |
| n_kv_head | 2 | 3 |
| head_dim | 64 | 64 |
| vocab_size | 4,096 | 16,384 |
| seq_len | 256 | 512 |
| Parameters | 6M | ~50M |
| Weights size | 24MB | ~200MB |

**Note:** `GPTConfig::from_depth(6)` produces `n_embd=384, seq_len=2048, vocab_size=32768`. We override `seq_len` to 512 and `vocab_size` to 16384 via manual config construction rather than `from_depth`.

**VRAM estimate (F32, batch_size=4, seq_len=512):**
- Weights: 50M × 4B = 200MB
- Gradients: 200MB
- Optimizer state (MuonAdamW — 2 momentum + 1 Muon buf): ~500MB
- Activations (6 layers × attention weights 4×6×512×512×4 ≈ 150MB + intermediates): ~400MB
- Total: ~1.3-1.5GB. Fits in 8GB with room for batch_size=8-16.

**Future optimization:** Mixed-precision (F16 forward/backward, F32 master weights) would halve activation memory and double GEMM throughput on the 5700 XT.

### Pretraining Data Strategy

**Core insight:** Knowledge comes from pretraining, not SFT. The model learns facts and C from reading massive amounts of text.

**Corpus mix (~80% English, ~20% C code initially, adjustable based on eval):**

Starting conservative on the C ratio. Code-focused models like Code Llama use 10-20% code in initial pretraining. We can increase the ratio in a continued-pretraining phase if C performance is lacking.

English sources:
- TinyStories GPT-4 Clean (already have — good for basic language fluency)
- Simple Wikipedia (factual knowledge in accessible language)
- Conversational/dialogue text (natural chat patterns)
- Educational Q&A text (models seeing question-answer patterns naturally)

C code sources:
- K&R "The C Programming Language" examples and exercises
- musl libc source (clean, well-commented ANSI C)
- Small canonical C projects: lua, tinycc, kilo (text editor), sbase
- C89 tutorials with explanations and comments
- Man pages for standard library functions
- C89 standard excerpts (type rules, undefined behavior, etc.)

**Data format:** All text concatenated into parquet files. No special formatting — just raw text with document boundaries. The model learns from immersion, not explicit teaching.

### SFT Strategy

**~100 examples maximum.** SFT teaches FORMAT, not KNOWLEDGE.

Categories:
- **Conversation format** (~30): How to greet, respond to casual chat, end conversations
- **Uncertainty** (~20): Natural "I'm not sure" / "I don't know" for questions outside knowledge
- **Code presentation** (~20): How to format C code in chat, explain code structure
- **Self-awareness** (~15): "I'm a small model", capabilities and limitations
- **Refusal** (~15): Graceful handling of requests it can't fulfill

### GRPO Strategy

Reinforce quality, penalize gibberish:
- **Coherence reward**: Penalize repetitive or degenerate output
- **Factual reward**: Correct answers to simple QA
- **Code correctness**: Generated C code compiles (can test with tcc)
- **Uncertainty reward**: Correctly expresses uncertainty on unknowable questions

### Evaluation

New eval suite testing 4 dimensions:

1. **Conversation coherence** — greetings, casual chat, multi-turn
2. **Factual knowledge** — basic facts, world knowledge, common sense
3. **C programming** — write functions, explain code, identify bugs, understand UB
4. **Uncertainty calibration** — knows what it knows vs doesn't know

Each dimension scored independently. C code generation tested by actually compiling output with `tcc -std=c89`.

---

## Phase C: Deploy

1. Export trained model as safetensors (same format, compatible with Rust serving)
2. Serve via Rust/candle (CPU or GPU)
3. Update Docker image to pull from HuggingFace
4. Cap `--max-tokens 128` to prevent degeneration during ramp-up

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| **gfx1010 not officially supported by ROCm 6.2** | Phase 0 proof-of-concept validates before any implementation. Fallback: `HSA_OVERRIDE_GFX_VERSION=10.3.0`. Last resort: Python/PyTorch training (already works). |
| ROCm/HIP FFI complexity | TDD approach, start minimal, build up |
| 50M params still too small for deep C | Can scale to depth 8 (~120M) if needed, fits in 8GB VRAM |
| C pretraining data quality | Curate carefully, prefer well-commented canonical code |
| candle-core fork maintenance | Pin to 0.8, minimal diff (~200 lines of match arms), aim for upstream PR |
| Training instability on GPU | Start with small batch sizes, validate loss curves match CPU |
| GRPO C compilation reward is blocking I/O | Acceptable at small scale — GPU is idle during generation anyway. Async if needed later. |

## Success Criteria

1. candle-rocm passes all TDD tests on RX 5700 XT
2. Picochat forward pass produces identical results on CPU and GPU
3. GPU training achieves ≥10x speedup over CPU
4. Trained model holds coherent multi-turn conversations
5. Model generates compilable C89 code for simple functions
6. Model naturally expresses uncertainty for unknowable questions
