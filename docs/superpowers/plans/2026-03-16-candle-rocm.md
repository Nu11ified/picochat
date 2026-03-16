# candle-rocm Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ROCm/HIP backend for candle as a standalone library, enabling GPU training of picochat on AMD RX 5700 XT.

**Architecture:** Three-layer crate workspace (hip-sys FFI → hip-runtime safe wrappers → candle-backend integration), added as a git submodule. Vendored candle-core fork adds `Device::Rocm` variant. TDD throughout.

**Tech Stack:** Rust, HIP/ROCm 6.2, rocBLAS, hipRAND, candle 0.8

**Spec:** `docs/superpowers/specs/2026-03-16-candle-rocm-training-redesign.md`

---

## Chunk 1: Environment Setup + Hardware Proof-of-Concept

**Execution order:** Task 1 → Task 4 (scaffold) → Task 2 → Task 3. Task 4 must run before Tasks 2-3 because the PoC files go inside the candle-rocm submodule directory.

### Task 1: Install ROCm SDK

ROCm is not installed on the host. The PyTorch pip wheel bundles HIP libs but we need the full SDK (hipcc compiler, headers, dev libs).

**Files:**
- No code files — system setup only

- [ ] **Step 1: Add AMD ROCm repo and install**

```bash
# Add ROCm apt repo for Ubuntu 24.04
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2 noble main" | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt-get update
sudo apt-get install -y rocm-hip-runtime-dev rocm-hip-sdk rocblas-dev hiprand-dev
```

- [ ] **Step 2: Verify installation**

Run:
```bash
/opt/rocm/bin/hipcc --version
ls /opt/rocm/lib/libamdhip64.so
ls /opt/rocm/lib/librocblas.so
ls /opt/rocm/lib/libhiprand.so
ls /opt/rocm/include/hip/hip_runtime.h
ls /opt/rocm/include/rocblas/rocblas.h
ls /opt/rocm/include/hiprand/hiprand.h
```
Expected: All files exist, hipcc reports ROCm 6.2.

- [ ] **Step 3: Verify GPU detection**

Run:
```bash
/opt/rocm/bin/rocm-smi --showid
/opt/rocm/bin/hipinfo
```
Expected: Shows AMD Radeon RX 5700 XT (gfx1010).

If gfx1010 is not detected or errors occur, set:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```
and retry. If this works, document it as a required env var.

### Task 2: Hardware Proof-of-Concept — HIP Kernel

Validate that we can compile and run HIP code targeting this GPU, before investing in the full backend.

**Prerequisite:** Task 4 (git submodule scaffold) must be completed first so that `/data/github/picochat/candle-rocm/` exists.

**Files:**
- Create: `/data/github/picochat/candle-rocm/poc/vector_add.hip`

- [ ] **Step 1: Create poc directory**

```bash
mkdir -p /data/github/picochat/candle-rocm/poc
```

- [ ] **Step 2: Write trivial HIP kernel**

Create `candle-rocm/poc/vector_add.hip`:
```cpp
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    hipLaunchKernelGGL(vector_add, dim3(grid_size), dim3(block_size), 0, 0, d_a, d_b, d_c, N);

    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(i * 2);
        if (h_c[i] != expected) {
            printf("FAIL at %d: got %f expected %f\n", i, h_c[i], expected);
            errors++;
            if (errors > 5) break;
        }
    }

    if (errors == 0) {
        printf("PASS: vector_add %d elements correct\n", N);
    }

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return errors > 0 ? 1 : 0;
}
```

- [ ] **Step 3: Compile and run**

```bash
cd /data/github/picochat/candle-rocm/poc
/opt/rocm/bin/hipcc --offload-arch=gfx1010 -o vector_add vector_add.hip
./vector_add
```

Expected: `PASS: vector_add 1024 elements correct`

If `--offload-arch=gfx1010` fails, try with `HSA_OVERRIDE_GFX_VERSION=10.3.0` and `--offload-arch=gfx1030`.

### Task 3: Hardware Proof-of-Concept — rocBLAS SGEMM

**Files:**
- Create: `candle-rocm/poc/rocblas_test.hip`

- [ ] **Step 1: Write rocBLAS SGEMM test**

Create `candle-rocm/poc/rocblas_test.hip`:
```cpp
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <math.h>

int main() {
    // 2x2 matmul: C = A * B
    // A = [[1, 2], [3, 4]]  B = [[5, 6], [7, 8]]
    // C = [[19, 22], [43, 50]]
    float h_A[] = {1, 3, 2, 4};  // column-major
    float h_B[] = {5, 7, 6, 8};  // column-major
    float h_C[] = {0, 0, 0, 0};

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, 4 * sizeof(float));
    hipMalloc(&d_B, 4 * sizeof(float));
    hipMalloc(&d_C, 4 * sizeof(float));

    hipMemcpy(d_A, h_A, 4 * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, 4 * sizeof(float), hipMemcpyHostToDevice);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    float alpha = 1.0f, beta = 0.0f;
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  2, 2, 2, &alpha, d_A, 2, d_B, 2, &beta, d_C, 2);

    hipMemcpy(h_C, d_C, 4 * sizeof(float), hipMemcpyDeviceToHost);

    // h_C is column-major: [19, 43, 22, 50]
    float expected[] = {19, 43, 22, 50};
    int errors = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(h_C[i] - expected[i]) > 1e-4) {
            printf("FAIL at %d: got %f expected %f\n", i, h_C[i], expected[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("PASS: rocblas_sgemm 2x2 correct\n");
    }

    rocblas_destroy_handle(handle);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return errors > 0 ? 1 : 0;
}
```

- [ ] **Step 2: Compile and run**

```bash
cd /data/github/picochat/candle-rocm/poc
/opt/rocm/bin/hipcc --offload-arch=gfx1010 -lrocblas -o rocblas_test rocblas_test.hip
./rocblas_test
```

Expected: `PASS: rocblas_sgemm 2x2 correct`

### Task 4: Create Git Submodule + Workspace Scaffold

**Files:**
- Create: `candle-rocm/` as git submodule with workspace structure

- [ ] **Step 1: Initialize the candle-rocm repo**

```bash
cd /data/github
mkdir candle-rocm && cd candle-rocm
git init
```

- [ ] **Step 2: Create workspace Cargo.toml**

Create `Cargo.toml`:
```toml
[workspace]
members = ["hip-sys", "hip-runtime", "candle-backend"]
resolver = "2"
```

- [ ] **Step 3: Scaffold hip-sys crate**

```bash
mkdir -p hip-sys/src
```

Create `hip-sys/Cargo.toml`:
```toml
[package]
name = "hip-sys"
version = "0.1.0"
edition = "2021"
description = "Raw FFI bindings to AMD HIP runtime, rocBLAS, and hipRAND"
license = "MIT OR Apache-2.0"

[build-dependencies]
```

Create `hip-sys/src/lib.rs`:
```rust
#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]
pub mod hip_runtime;
pub mod rocblas;
pub mod hiprand;
```

Create empty module files:
- `hip-sys/src/hip_runtime.rs` → `//! HIP runtime FFI bindings.`
- `hip-sys/src/rocblas.rs` → `//! rocBLAS FFI bindings.`
- `hip-sys/src/hiprand.rs` → `//! hipRAND FFI bindings.`

Create `hip-sys/build.rs`:
```rust
fn main() {
    let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());

    println!("cargo:rustc-link-search=native={rocm_path}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    println!("cargo:rustc-link-lib=dylib=hiprand");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}
```

- [ ] **Step 4: Scaffold hip-runtime crate**

```bash
mkdir -p hip-runtime/src
```

Create `hip-runtime/Cargo.toml`:
```toml
[package]
name = "hip-runtime"
version = "0.1.0"
edition = "2021"
description = "Safe Rust wrappers for AMD HIP runtime"
license = "MIT OR Apache-2.0"

[dependencies]
hip-sys = { path = "../hip-sys" }
```

Create `hip-runtime/src/lib.rs`:
```rust
pub mod device;
pub mod memory;
pub mod module;
pub mod blas;
pub mod rng;
pub mod error;
```

Create empty module files with a one-line `//!` doc comment describing each module's purpose.

- [ ] **Step 5: Scaffold candle-backend crate**

```bash
mkdir -p candle-backend/src
```

Create `candle-backend/Cargo.toml`:
```toml
[package]
name = "candle-rocm"
version = "0.1.0"
edition = "2021"
description = "ROCm/HIP backend for candle ML framework"
license = "MIT OR Apache-2.0"

[dependencies]
hip-runtime = { path = "../hip-runtime" }
candle-core = { version = "0.8" }
```

Create `candle-backend/src/lib.rs`:
```rust
pub mod device;
pub mod storage;
pub mod error;
pub mod utils;
```

Create empty module files with a one-line `//!` doc comment describing each module's purpose.

- [ ] **Step 6: Create kernels directory**

```bash
mkdir -p kernels
```

Create `kernels/README.md`:
```
# HIP Kernels
.hip source files compiled to .hsaco at build time via hipcc.
```

- [ ] **Step 7: Verify workspace compiles**

```bash
cd /data/github/picochat/candle-rocm
PATH="/home/nullify/.cargo/bin:$PATH" cargo check --workspace
```

Expected: Compiles (with warnings about unused modules). hip-sys linking may fail if ROCm SDK isn't installed yet — that's OK, we verify it after Task 1.

- [ ] **Step 8: Add as git submodule to picochat**

```bash
cd /data/github/picochat
git submodule add ../candle-rocm candle-rocm
```

- [ ] **Step 9: Initial commit**

```bash
cd /data/github/picochat/candle-rocm
git add -A
git commit -m "scaffold candle-rocm workspace: hip-sys, hip-runtime, candle-backend"
```

```bash
cd /data/github/picochat
git add candle-rocm .gitmodules
git commit -m "add candle-rocm as git submodule"
```

---

## Chunk 2: hip-sys — Raw FFI Bindings (TDD Phase 1)

### Task 5: HIP Runtime FFI Types and Functions

Write the raw `extern "C"` bindings for the HIP runtime functions we need. These are hand-written (not bindgen) to keep the dependency minimal and the types clear.

**Reference:** `/opt/rocm/include/hip/hip_runtime_api.h`

**Files:**
- Modify: `hip-sys/src/hip_runtime.rs`
- Create: `hip-sys/tests/hip_runtime_test.rs`

- [ ] **Step 1: Write the FFI types and function declarations**

Write `hip-sys/src/hip_runtime.rs`:
```rust
use std::ffi::c_void;
use std::os::raw::c_int;

pub type hipError_t = c_int;
pub type hipDevice_t = c_int;
pub type hipStream_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipDeviceptr_t = *mut c_void;

pub const HIP_SUCCESS: hipError_t = 0;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
}

#[repr(C)]
#[derive(Debug)]
pub struct hipDeviceProp_t {
    pub name: [std::os::raw::c_char; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    // hipDeviceProp_t has 100+ fields in ROCm 6.2 (~792+ bytes total).
    // We over-allocate to avoid writing past the struct boundary.
    // Validate with: assert!(size_of::<hipDeviceProp_t>() >= 792)
    pub _padding: [u8; 4096],
}

extern "C" {
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    pub fn hipGetDevice(device_id: *mut c_int) -> hipError_t;
    pub fn hipGetDeviceProperties(prop: *mut hipDeviceProp_t, device_id: c_int) -> hipError_t;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    pub fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipModuleLoad(module: *mut hipModule_t, fname: *const std::os::raw::c_char) -> hipError_t;
    pub fn hipModuleGetFunction(
        func: *mut hipFunction_t,
        module: hipModule_t,
        name: *const std::os::raw::c_char,
    ) -> hipError_t;
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipGetErrorString(error: hipError_t) -> *const std::os::raw::c_char;
}
```

- [ ] **Step 2: Write failing tests**

Create `hip-sys/tests/hip_runtime_test.rs`:
```rust
use hip_sys::hip_runtime::*;
use std::ptr;

#[test]
fn test_hip_get_device_count() {
    let mut count: i32 = 0;
    let err = unsafe { hipGetDeviceCount(&mut count) };
    assert_eq!(err, HIP_SUCCESS, "hipGetDeviceCount failed");
    assert!(count >= 1, "Expected at least 1 GPU, got {count}");
}

#[test]
fn test_hip_set_device() {
    let err = unsafe { hipSetDevice(0) };
    assert_eq!(err, HIP_SUCCESS, "hipSetDevice(0) failed");
}

#[test]
fn test_hip_malloc_free() {
    unsafe {
        hipSetDevice(0);
    }
    let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
    let size = 1024 * 1024; // 1MB
    let err = unsafe { hipMalloc(&mut ptr, size) };
    assert_eq!(err, HIP_SUCCESS, "hipMalloc failed");
    assert!(!ptr.is_null(), "hipMalloc returned null");

    let err = unsafe { hipFree(ptr) };
    assert_eq!(err, HIP_SUCCESS, "hipFree failed");
}

#[test]
fn test_hip_memcpy_roundtrip() {
    unsafe { hipSetDevice(0); }

    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let bytes = data.len() * std::mem::size_of::<f32>();

    let mut d_ptr: *mut std::ffi::c_void = ptr::null_mut();
    unsafe {
        assert_eq!(hipMalloc(&mut d_ptr, bytes), HIP_SUCCESS);
        assert_eq!(
            hipMemcpy(d_ptr, data.as_ptr() as *const _, bytes, hipMemcpyKind::hipMemcpyHostToDevice),
            HIP_SUCCESS
        );
    }

    let mut result = vec![0.0f32; 256];
    unsafe {
        assert_eq!(
            hipMemcpy(result.as_mut_ptr() as *mut _, d_ptr, bytes, hipMemcpyKind::hipMemcpyDeviceToHost),
            HIP_SUCCESS
        );
        hipFree(d_ptr);
    }

    assert_eq!(data, result, "H2D→D2H roundtrip mismatch");
}

#[test]
fn test_hip_device_properties() {
    unsafe { hipSetDevice(0); }
    let mut prop = std::mem::MaybeUninit::<hipDeviceProp_t>::zeroed();
    let err = unsafe { hipGetDeviceProperties(prop.as_mut_ptr(), 0) };
    assert_eq!(err, HIP_SUCCESS, "hipGetDeviceProperties failed");

    let prop = unsafe { prop.assume_init() };
    let name = unsafe { std::ffi::CStr::from_ptr(prop.name.as_ptr()) };
    let name_str = name.to_str().unwrap();
    println!("GPU: {name_str}");
    println!("VRAM: {} MB", prop.total_global_mem / (1024 * 1024));
    assert!(prop.total_global_mem > 0, "Expected non-zero VRAM");
}
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
cd /data/github/picochat/candle-rocm
PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p hip-sys -- --nocapture
```

Expected: All 4 tests pass. If linking fails, check that `/opt/rocm/lib` is in `LD_LIBRARY_PATH` or that `build.rs` paths are correct.

- [ ] **Step 4: Commit**

```bash
cd /data/github/picochat/candle-rocm
git add hip-sys/
git commit -m "hip-sys: add HIP runtime FFI bindings with passing tests"
```

### Task 6: rocBLAS FFI Bindings

**Files:**
- Modify: `hip-sys/src/rocblas.rs`
- Create: `hip-sys/tests/rocblas_test.rs`

- [ ] **Step 1: Write rocBLAS FFI declarations**

Write `hip-sys/src/rocblas.rs`:
```rust
use std::ffi::c_void;
use std::os::raw::c_int;

pub type rocblas_handle = *mut c_void;
pub type rocblas_status = c_int;

pub const ROCBLAS_STATUS_SUCCESS: rocblas_status = 0;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum rocblas_operation {
    rocblas_operation_none = 111,
    rocblas_operation_transpose = 112,
    rocblas_operation_conjugate_transpose = 113,
}

extern "C" {
    pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
    pub fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;

    pub fn rocblas_sgemm(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const c_void,
        lda: c_int,
        b: *const c_void,
        ldb: c_int,
        beta: *const f32,
        c: *mut c_void,
        ldc: c_int,
    ) -> rocblas_status;

    pub fn rocblas_sgemm_strided_batched(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const c_void,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        ldb: c_int,
        stride_b: i64,
        beta: *const f32,
        c: *mut c_void,
        ldc: c_int,
        stride_c: i64,
        batch_count: c_int,
    ) -> rocblas_status;
}
```

- [ ] **Step 2: Write failing test**

Create `hip-sys/tests/rocblas_test.rs`:
```rust
use hip_sys::hip_runtime::*;
use hip_sys::rocblas::*;
use std::ptr;

#[test]
fn test_rocblas_create_destroy() {
    unsafe { hipSetDevice(0); }
    let mut handle: rocblas_handle = ptr::null_mut();
    let status = unsafe { rocblas_create_handle(&mut handle) };
    assert_eq!(status, ROCBLAS_STATUS_SUCCESS);
    assert!(!handle.is_null());

    let status = unsafe { rocblas_destroy_handle(handle) };
    assert_eq!(status, ROCBLAS_STATUS_SUCCESS);
}

#[test]
fn test_rocblas_sgemm() {
    unsafe { hipSetDevice(0); }

    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = A * B = [[19, 22], [43, 50]]
    // Column-major layout
    let h_a: [f32; 4] = [1.0, 3.0, 2.0, 4.0];
    let h_b: [f32; 4] = [5.0, 7.0, 6.0, 8.0];
    let expected: [f32; 4] = [19.0, 43.0, 22.0, 50.0];

    let bytes = 4 * std::mem::size_of::<f32>();
    let mut d_a: *mut std::ffi::c_void = ptr::null_mut();
    let mut d_b: *mut std::ffi::c_void = ptr::null_mut();
    let mut d_c: *mut std::ffi::c_void = ptr::null_mut();

    unsafe {
        hipMalloc(&mut d_a, bytes);
        hipMalloc(&mut d_b, bytes);
        hipMalloc(&mut d_c, bytes);
        hipMemcpy(d_a, h_a.as_ptr() as *const _, bytes, hipMemcpyKind::hipMemcpyHostToDevice);
        hipMemcpy(d_b, h_b.as_ptr() as *const _, bytes, hipMemcpyKind::hipMemcpyHostToDevice);

        let mut handle: rocblas_handle = ptr::null_mut();
        rocblas_create_handle(&mut handle);

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = rocblas_sgemm(
            handle,
            rocblas_operation::rocblas_operation_none,
            rocblas_operation::rocblas_operation_none,
            2, 2, 2,
            &alpha,
            d_a, 2,
            d_b, 2,
            &beta,
            d_c, 2,
        );
        assert_eq!(status, ROCBLAS_STATUS_SUCCESS);

        let mut result = [0.0f32; 4];
        hipMemcpy(
            result.as_mut_ptr() as *mut _,
            d_c,
            bytes,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        );

        for i in 0..4 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-4,
                "Mismatch at {i}: got {} expected {}",
                result[i],
                expected[i]
            );
        }

        rocblas_destroy_handle(handle);
        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_c);
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cd /data/github/picochat/candle-rocm
PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p hip-sys --test rocblas_test -- --nocapture
```

Expected: Both tests pass.

- [ ] **Step 4: Commit**

```bash
cd /data/github/picochat/candle-rocm
git add hip-sys/
git commit -m "hip-sys: add rocBLAS FFI bindings with SGEMM test"
```

### Task 7: hipRAND FFI Bindings

**Files:**
- Modify: `hip-sys/src/hiprand.rs`
- Create: `hip-sys/tests/hiprand_test.rs`

- [ ] **Step 1: Write hipRAND FFI declarations**

Write `hip-sys/src/hiprand.rs`:
```rust
use std::ffi::c_void;
use std::os::raw::c_int;

pub type hiprandGenerator_t = *mut c_void;
pub type hiprandStatus_t = c_int;

pub const HIPRAND_STATUS_SUCCESS: hiprandStatus_t = 0;

#[repr(C)]
pub enum hiprandRngType_t {
    HIPRAND_RNG_PSEUDO_DEFAULT = 400,
    HIPRAND_RNG_PSEUDO_XORWOW = 401,
    HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 408,
}

extern "C" {
    pub fn hiprandCreateGenerator(
        generator: *mut hiprandGenerator_t,
        rng_type: hiprandRngType_t,
    ) -> hiprandStatus_t;

    pub fn hiprandDestroyGenerator(generator: hiprandGenerator_t) -> hiprandStatus_t;

    pub fn hiprandSetPseudoRandomGeneratorSeed(
        generator: hiprandGenerator_t,
        seed: u64,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateUniform(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateNormal(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateUniformDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: usize,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateNormalDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> hiprandStatus_t;
}
```

- [ ] **Step 2: Write tests**

Create `hip-sys/tests/hiprand_test.rs`:
```rust
use hip_sys::hip_runtime::*;
use hip_sys::hiprand::*;
use std::ptr;

#[test]
fn test_hiprand_uniform() {
    unsafe { hipSetDevice(0); }

    let n = 1024usize;
    let bytes = n * std::mem::size_of::<f32>();
    let mut d_out: *mut std::ffi::c_void = ptr::null_mut();
    unsafe {
        hipMalloc(&mut d_out, bytes);

        let mut gen: hiprandGenerator_t = ptr::null_mut();
        assert_eq!(
            hiprandCreateGenerator(&mut gen, hiprandRngType_t::HIPRAND_RNG_PSEUDO_DEFAULT),
            HIPRAND_STATUS_SUCCESS
        );
        assert_eq!(hiprandSetPseudoRandomGeneratorSeed(gen, 42), HIPRAND_STATUS_SUCCESS);
        assert_eq!(hiprandGenerateUniform(gen, d_out as *mut f32, n), HIPRAND_STATUS_SUCCESS);

        let mut result = vec![0.0f32; n];
        hipMemcpy(result.as_mut_ptr() as *mut _, d_out, bytes, hipMemcpyKind::hipMemcpyDeviceToHost);

        for &v in &result {
            assert!(v >= 0.0 && v <= 1.0, "Uniform value out of range: {v}");
        }

        hiprandDestroyGenerator(gen);
        hipFree(d_out);
    }
}

#[test]
fn test_hiprand_normal() {
    unsafe { hipSetDevice(0); }

    let n = 4096usize;
    let bytes = n * std::mem::size_of::<f32>();
    let mut d_out: *mut std::ffi::c_void = ptr::null_mut();
    unsafe {
        hipMalloc(&mut d_out, bytes);

        let mut gen: hiprandGenerator_t = ptr::null_mut();
        hiprandCreateGenerator(&mut gen, hiprandRngType_t::HIPRAND_RNG_PSEUDO_DEFAULT);
        hiprandSetPseudoRandomGeneratorSeed(gen, 42);
        hiprandGenerateNormal(gen, d_out as *mut f32, n, 0.0, 1.0);

        let mut result = vec![0.0f32; n];
        hipMemcpy(result.as_mut_ptr() as *mut _, d_out, bytes, hipMemcpyKind::hipMemcpyDeviceToHost);

        let mean: f32 = result.iter().sum::<f32>() / n as f32;
        let variance: f32 = result.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

        assert!((mean).abs() < 0.1, "Mean too far from 0: {mean}");
        assert!((variance - 1.0).abs() < 0.2, "Variance too far from 1: {variance}");

        hiprandDestroyGenerator(gen);
        hipFree(d_out);
    }
}

#[test]
fn test_hiprand_seeded_reproducibility() {
    unsafe { hipSetDevice(0); }

    let n = 256usize;
    let bytes = n * std::mem::size_of::<f32>();

    let generate = |seed: u64| -> Vec<f32> {
        let mut d_out: *mut std::ffi::c_void = ptr::null_mut();
        unsafe {
            hipMalloc(&mut d_out, bytes);
            let mut gen: hiprandGenerator_t = ptr::null_mut();
            hiprandCreateGenerator(&mut gen, hiprandRngType_t::HIPRAND_RNG_PSEUDO_DEFAULT);
            hiprandSetPseudoRandomGeneratorSeed(gen, seed);
            hiprandGenerateUniform(gen, d_out as *mut f32, n);

            let mut result = vec![0.0f32; n];
            hipMemcpy(result.as_mut_ptr() as *mut _, d_out, bytes, hipMemcpyKind::hipMemcpyDeviceToHost);
            hiprandDestroyGenerator(gen);
            hipFree(d_out);
            result
        }
    };

    let run1 = generate(12345);
    let run2 = generate(12345);
    assert_eq!(run1, run2, "Same seed should produce identical results");

    let run3 = generate(99999);
    assert_ne!(run1, run3, "Different seeds should produce different results");
}
```

- [ ] **Step 3: Run tests**

```bash
cd /data/github/picochat/candle-rocm
PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p hip-sys --test hiprand_test -- --nocapture
```

Expected: All 3 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /data/github/picochat/candle-rocm
git add hip-sys/
git commit -m "hip-sys: add hipRAND FFI bindings with seeded RNG tests"
```

---

## Chunk 3: hip-runtime — Safe Rust Wrappers (TDD Phase 2)

### Task 8: HipDevice — Safe Device Management

**Files:**
- Modify: `hip-runtime/src/error.rs`
- Modify: `hip-runtime/src/device.rs`
- Create: `hip-runtime/tests/device_test.rs`

- [ ] **Step 1: Write error type**

Write `hip-runtime/src/error.rs`:
```rust
use std::fmt;

#[derive(Debug)]
pub enum HipError {
    HipRuntimeError { code: i32, msg: String },
    RocblasError { code: i32 },
    HiprandError { code: i32 },
    KernelNotFound { name: String },
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HipRuntimeError { code, msg } => write!(f, "HIP error {code}: {msg}"),
            Self::RocblasError { code } => write!(f, "rocBLAS error {code}"),
            Self::HiprandError { code } => write!(f, "hipRAND error {code}"),
            Self::KernelNotFound { name } => write!(f, "kernel not found: {name}"),
        }
    }
}

impl std::error::Error for HipError {}

pub type Result<T> = std::result::Result<T, HipError>;

/// Check a HIP status code and convert to Result.
pub fn check_hip(code: i32) -> Result<()> {
    if code == hip_sys::hip_runtime::HIP_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = hip_sys::hip_runtime::hipGetErrorString(code);
            if ptr.is_null() {
                "unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(HipError::HipRuntimeError { code, msg })
    }
}

pub fn check_rocblas(code: i32) -> Result<()> {
    if code == hip_sys::rocblas::ROCBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(HipError::RocblasError { code })
    }
}

pub fn check_hiprand(code: i32) -> Result<()> {
    if code == hip_sys::hiprand::HIPRAND_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(HipError::HiprandError { code })
    }
}
```

- [ ] **Step 2: Write HipDevice**

Write `hip-runtime/src/device.rs`:
```rust
use crate::error::{check_hip, Result};
use hip_sys::hip_runtime;

#[derive(Debug, Clone)]
pub struct HipDevice {
    ordinal: usize,
}

impl HipDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        check_hip(unsafe { hip_runtime::hipSetDevice(ordinal as i32) })?;
        Ok(Self { ordinal })
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn set_current(&self) -> Result<()> {
        check_hip(unsafe { hip_runtime::hipSetDevice(self.ordinal as i32) })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.set_current()?;
        check_hip(unsafe { hip_runtime::hipDeviceSynchronize() })
    }

    pub fn device_count() -> Result<usize> {
        let mut count: i32 = 0;
        check_hip(unsafe { hip_runtime::hipGetDeviceCount(&mut count) })?;
        Ok(count as usize)
    }

    pub fn name(&self) -> Result<String> {
        let mut prop = std::mem::MaybeUninit::<hip_runtime::hipDeviceProp_t>::zeroed();
        check_hip(unsafe {
            hip_runtime::hipGetDeviceProperties(prop.as_mut_ptr(), self.ordinal as i32)
        })?;
        let prop = unsafe { prop.assume_init() };
        let name = unsafe { std::ffi::CStr::from_ptr(prop.name.as_ptr()) };
        Ok(name.to_string_lossy().into_owned())
    }

    pub fn total_memory(&self) -> Result<usize> {
        let mut prop = std::mem::MaybeUninit::<hip_runtime::hipDeviceProp_t>::zeroed();
        check_hip(unsafe {
            hip_runtime::hipGetDeviceProperties(prop.as_mut_ptr(), self.ordinal as i32)
        })?;
        let prop = unsafe { prop.assume_init() };
        Ok(prop.total_global_mem)
    }
}
```

- [ ] **Step 3: Write tests**

Create `hip-runtime/tests/device_test.rs`:
```rust
use hip_runtime::device::HipDevice;

#[test]
fn test_device_new() {
    let dev = HipDevice::new(0).expect("Failed to create device");
    assert_eq!(dev.ordinal(), 0);
}

#[test]
fn test_device_count() {
    let count = HipDevice::device_count().expect("Failed to get device count");
    assert!(count >= 1);
}

#[test]
fn test_device_name() {
    let dev = HipDevice::new(0).unwrap();
    let name = dev.name().unwrap();
    println!("Device name: {name}");
    assert!(!name.is_empty());
}

#[test]
fn test_device_memory() {
    let dev = HipDevice::new(0).unwrap();
    let mem = dev.total_memory().unwrap();
    println!("Device memory: {} MB", mem / (1024 * 1024));
    assert!(mem > 1024 * 1024 * 1024, "Expected at least 1GB VRAM");
}

#[test]
fn test_device_synchronize() {
    let dev = HipDevice::new(0).unwrap();
    dev.synchronize().expect("Synchronize failed");
}
```

- [ ] **Step 4: Run tests**

```bash
cd /data/github/picochat/candle-rocm
PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p hip-runtime --test device_test -- --nocapture
```

Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /data/github/picochat/candle-rocm
git add hip-runtime/
git commit -m "hip-runtime: add HipDevice safe wrapper with tests"
```

### Task 9: DeviceBuffer — Safe GPU Memory

**Files:**
- Modify: `hip-runtime/src/memory.rs`
- Create: `hip-runtime/tests/memory_test.rs`

- [ ] **Step 1: Write DeviceBuffer**

Write `hip-runtime/src/memory.rs`:
```rust
use crate::error::{check_hip, Result};
use hip_sys::hip_runtime::{self, hipMemcpyKind};
use std::marker::PhantomData;

/// Typed GPU memory buffer with RAII.
pub struct DeviceBuffer<T> {
    ptr: *mut std::ffi::c_void,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocate uninitialized GPU memory for `len` elements.
    pub fn alloc(len: usize) -> Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        check_hip(unsafe { hip_runtime::hipMalloc(&mut ptr, bytes) })?;
        Ok(Self { ptr, len, _phantom: PhantomData })
    }

    /// Allocate zero-initialized GPU memory.
    pub fn alloc_zeros(len: usize) -> Result<Self> {
        let buf = Self::alloc(len)?;
        let bytes = len * std::mem::size_of::<T>();
        check_hip(unsafe { hip_runtime::hipMemset(buf.ptr, 0, bytes) })?;
        Ok(buf)
    }

    /// Copy from host slice to device.
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let buf = Self::alloc(data.len())?;
        let bytes = data.len() * std::mem::size_of::<T>();
        check_hip(unsafe {
            hip_runtime::hipMemcpy(
                buf.ptr,
                data.as_ptr() as *const _,
                bytes,
                hipMemcpyKind::hipMemcpyHostToDevice,
            )
        })?;
        Ok(buf)
    }

    /// Copy device buffer back to host.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut result = vec![unsafe { std::mem::zeroed() }; self.len];
        let bytes = self.len * std::mem::size_of::<T>();
        check_hip(unsafe {
            hip_runtime::hipMemcpy(
                result.as_mut_ptr() as *mut _,
                self.ptr,
                bytes,
                hipMemcpyKind::hipMemcpyDeviceToHost,
            )
        })?;
        Ok(result)
    }

    /// Raw device pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Raw mutable device pointer.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Raw void pointer (for kernel params).
    pub fn as_void_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hip_runtime::hipFree(self.ptr) };
        }
    }
}

// DeviceBuffer cannot be sent across threads (GPU context is thread-local)
// but can be shared within a thread.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
```

- [ ] **Step 2: Write tests**

Create `hip-runtime/tests/memory_test.rs`:
```rust
use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;

#[test]
fn test_device_buffer_alloc() {
    let _dev = HipDevice::new(0).unwrap();
    let buf = DeviceBuffer::<f32>::alloc(1024).expect("alloc failed");
    assert_eq!(buf.len(), 1024);
    assert_eq!(buf.byte_size(), 1024 * 4);
}

#[test]
fn test_device_buffer_roundtrip() {
    let _dev = HipDevice::new(0).unwrap();
    let data: Vec<f32> = (0..256).map(|i| i as f32 * 1.5).collect();
    let buf = DeviceBuffer::from_slice(&data).unwrap();
    let result = buf.to_vec().unwrap();
    assert_eq!(data, result);
}

#[test]
fn test_device_buffer_zeros() {
    let _dev = HipDevice::new(0).unwrap();
    let buf = DeviceBuffer::<f32>::alloc_zeros(128).unwrap();
    let result = buf.to_vec().unwrap();
    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn test_device_buffer_drop() {
    let _dev = HipDevice::new(0).unwrap();
    // Allocate and drop many buffers — if drop doesn't free, we'll OOM
    for _ in 0..100 {
        let _buf = DeviceBuffer::<f32>::alloc(1024 * 1024).unwrap(); // 4MB each
    }
}

#[test]
fn test_device_buffer_u8() {
    let _dev = HipDevice::new(0).unwrap();
    let data: Vec<u8> = (0..255).collect();
    let buf = DeviceBuffer::from_slice(&data).unwrap();
    let result = buf.to_vec().unwrap();
    assert_eq!(data, result);
}
```

- [ ] **Step 3: Run tests**

```bash
cd /data/github/picochat/candle-rocm
PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p hip-runtime --test memory_test -- --nocapture
```

Expected: All 5 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /data/github/picochat/candle-rocm
git add hip-runtime/
git commit -m "hip-runtime: add DeviceBuffer safe GPU memory with RAII"
```

### Task 10: RocBlas — Safe GEMM Wrapper

**Files:**
- Modify: `hip-runtime/src/blas.rs`
- Create: `hip-runtime/tests/blas_test.rs`

- [ ] **Step 1: Write safe RocBlas wrapper**

Write `hip-runtime/src/blas.rs`:
```rust
use crate::error::{check_rocblas, Result};
use crate::memory::DeviceBuffer;
use hip_sys::rocblas;

pub struct RocBlas {
    handle: rocblas::rocblas_handle,
}

impl RocBlas {
    pub fn new() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        check_rocblas(unsafe { rocblas::rocblas_create_handle(&mut handle) })?;
        Ok(Self { handle })
    }

    /// C = alpha * A * B + beta * C
    /// All matrices in column-major. Dimensions: A is m×k, B is k×n, C is m×n.
    pub fn sgemm(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        lda: usize,
        b: &DeviceBuffer<f32>,
        ldb: usize,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        ldc: usize,
    ) -> Result<()> {
        let op_a = if trans_a {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };
        let op_b = if trans_b {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(unsafe {
            rocblas::rocblas_sgemm(
                self.handle,
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a.as_void_ptr(),
                lda as i32,
                b.as_void_ptr(),
                ldb as i32,
                &beta,
                c.as_void_ptr() as *mut _,
                ldc as i32,
            )
        })
    }

    /// Batched strided SGEMM.
    pub fn sgemm_strided_batched(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        lda: usize,
        stride_a: i64,
        b: &DeviceBuffer<f32>,
        ldb: usize,
        stride_b: i64,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        let op_a = if trans_a {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };
        let op_b = if trans_b {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(unsafe {
            rocblas::rocblas_sgemm_strided_batched(
                self.handle,
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a.as_void_ptr(),
                lda as i32,
                stride_a,
                b.as_void_ptr(),
                ldb as i32,
                stride_b,
                &beta,
                c.as_void_ptr() as *mut _,
                ldc as i32,
                stride_c,
                batch_count as i32,
            )
        })
    }
}

impl Drop for RocBlas {
    fn drop(&mut self) {
        unsafe { rocblas::rocblas_destroy_handle(self.handle) };
    }
}
```

- [ ] **Step 2: Write tests**

Create `hip-runtime/tests/blas_test.rs`:
```rust
use hip_runtime::blas::RocBlas;
use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;

#[test]
fn test_rocblas_gemm_identity() {
    let _dev = HipDevice::new(0).unwrap();
    let blas = RocBlas::new().unwrap();

    // A = [[1,0],[0,1]] (identity), B = [[3,4],[5,6]]
    // C = I * B = B
    // Column-major: A=[1,0,0,1], B=[3,5,4,6]
    let a = DeviceBuffer::from_slice(&[1.0f32, 0.0, 0.0, 1.0]).unwrap();
    let b = DeviceBuffer::from_slice(&[3.0f32, 5.0, 4.0, 6.0]).unwrap();
    let mut c = DeviceBuffer::<f32>::alloc_zeros(4).unwrap();

    blas.sgemm(false, false, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2)
        .unwrap();

    let result = c.to_vec().unwrap();
    assert_eq!(result, vec![3.0, 5.0, 4.0, 6.0]);
}

#[test]
fn test_rocblas_gemm_known() {
    let _dev = HipDevice::new(0).unwrap();
    let blas = RocBlas::new().unwrap();

    // A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2)
    // C = A*B = [[58,64],[139,154]] (2x2)
    // Col-major: A=[1,4,2,5,3,6], B=[7,9,11,8,10,12]
    let a = DeviceBuffer::from_slice(&[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let b = DeviceBuffer::from_slice(&[7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0]).unwrap();
    let mut c = DeviceBuffer::<f32>::alloc_zeros(4).unwrap();

    // m=2, n=2, k=3
    blas.sgemm(false, false, 2, 2, 3, 1.0, &a, 2, &b, 3, 0.0, &mut c, 2)
        .unwrap();

    let result = c.to_vec().unwrap();
    let expected = vec![58.0f32, 139.0, 64.0, 154.0]; // col-major
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "Mismatch at {i}: got {got} expected {exp}"
        );
    }
}

#[test]
fn test_rocblas_gemm_batched() {
    let _dev = HipDevice::new(0).unwrap();
    let blas = RocBlas::new().unwrap();

    // 2 batches of 2x2 matmul
    // Batch 0: A=[[1,0],[0,1]] * B=[[2,3],[4,5]] = [[2,3],[4,5]]
    // Batch 1: A=[[2,0],[0,2]] * B=[[1,1],[1,1]] = [[2,2],[2,2]]
    // Col-major, contiguous batches
    let a = DeviceBuffer::from_slice(&[
        1.0f32, 0.0, 0.0, 1.0, // batch 0 identity
        2.0, 0.0, 0.0, 2.0,    // batch 1 scale-by-2
    ]).unwrap();
    let b = DeviceBuffer::from_slice(&[
        2.0f32, 4.0, 3.0, 5.0, // batch 0
        1.0, 1.0, 1.0, 1.0,    // batch 1
    ]).unwrap();
    let mut c = DeviceBuffer::<f32>::alloc_zeros(8).unwrap();

    blas.sgemm_strided_batched(
        false, false, 2, 2, 2,
        1.0, &a, 2, 4, &b, 2, 4,
        0.0, &mut c, 2, 4, 2,
    ).unwrap();

    let result = c.to_vec().unwrap();
    let expected = vec![
        2.0f32, 4.0, 3.0, 5.0, // batch 0: identity * B = B
        2.0, 2.0, 2.0, 2.0,    // batch 1: 2*I * ones = 2*ones
    ];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "Mismatch at {i}: got {got} expected {exp}"
        );
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cd /data/github/picochat/candle-rocm
PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p hip-runtime --test blas_test -- --nocapture
```

Expected: All 3 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /data/github/picochat/candle-rocm
git add hip-runtime/
git commit -m "hip-runtime: add safe RocBlas GEMM wrapper with tests"
```

### Task 11: HipRng — Safe RNG Wrapper

**Files:**
- Modify: `hip-runtime/src/rng.rs`
- Create: `hip-runtime/tests/rng_test.rs`

- [ ] **Step 1: Write safe HipRng wrapper**

Write `hip-runtime/src/rng.rs`:
```rust
use crate::error::{check_hiprand, Result};
use crate::memory::DeviceBuffer;
use hip_sys::hiprand;

pub struct HipRng {
    gen: hiprand::hiprandGenerator_t,
}

impl HipRng {
    pub fn new(seed: u64) -> Result<Self> {
        let mut gen = std::ptr::null_mut();
        check_hiprand(unsafe {
            hiprand::hiprandCreateGenerator(
                &mut gen,
                hiprand::hiprandRngType_t::HIPRAND_RNG_PSEUDO_DEFAULT,
            )
        })?;
        check_hiprand(unsafe { hiprand::hiprandSetPseudoRandomGeneratorSeed(gen, seed) })?;
        Ok(Self { gen })
    }

    pub fn set_seed(&self, seed: u64) -> Result<()> {
        check_hiprand(unsafe { hiprand::hiprandSetPseudoRandomGeneratorSeed(self.gen, seed) })
    }

    pub fn uniform_f32(&self, buf: &mut DeviceBuffer<f32>) -> Result<()> {
        check_hiprand(unsafe {
            hiprand::hiprandGenerateUniform(self.gen, buf.as_mut_ptr(), buf.len())
        })
    }

    pub fn normal_f32(&self, buf: &mut DeviceBuffer<f32>, mean: f32, std: f32) -> Result<()> {
        check_hiprand(unsafe {
            hiprand::hiprandGenerateNormal(self.gen, buf.as_mut_ptr(), buf.len(), mean, std)
        })
    }

    pub fn uniform_f64(&self, buf: &mut DeviceBuffer<f64>) -> Result<()> {
        check_hiprand(unsafe {
            hiprand::hiprandGenerateUniformDouble(self.gen, buf.as_mut_ptr(), buf.len())
        })
    }

    pub fn normal_f64(&self, buf: &mut DeviceBuffer<f64>, mean: f64, std: f64) -> Result<()> {
        check_hiprand(unsafe {
            hiprand::hiprandGenerateNormalDouble(self.gen, buf.as_mut_ptr(), buf.len(), mean, std)
        })
    }
}

impl Drop for HipRng {
    fn drop(&mut self) {
        unsafe { hiprand::hiprandDestroyGenerator(self.gen) };
    }
}
```

- [ ] **Step 2: Write tests**

Create `hip-runtime/tests/rng_test.rs`:
```rust
use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;
use hip_runtime::rng::HipRng;

#[test]
fn test_rng_uniform() {
    let _dev = HipDevice::new(0).unwrap();
    let rng = HipRng::new(42).unwrap();
    let mut buf = DeviceBuffer::<f32>::alloc(1024).unwrap();
    rng.uniform_f32(&mut buf).unwrap();
    let vals = buf.to_vec().unwrap();
    assert!(vals.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_rng_normal() {
    let _dev = HipDevice::new(0).unwrap();
    let rng = HipRng::new(42).unwrap();
    let mut buf = DeviceBuffer::<f32>::alloc(4096).unwrap();
    rng.normal_f32(&mut buf, 0.0, 1.0).unwrap();
    let vals = buf.to_vec().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(mean.abs() < 0.1, "Mean {mean} too far from 0");
}

#[test]
fn test_rng_reproducibility() {
    let _dev = HipDevice::new(0).unwrap();

    let gen = |seed| {
        let rng = HipRng::new(seed).unwrap();
        let mut buf = DeviceBuffer::<f32>::alloc(256).unwrap();
        rng.uniform_f32(&mut buf).unwrap();
        buf.to_vec().unwrap()
    };

    assert_eq!(gen(123), gen(123));
    assert_ne!(gen(123), gen(456));
}
```

- [ ] **Step 3: Run tests**

```bash
cd /data/github/picochat/candle-rocm
PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p hip-runtime --test rng_test -- --nocapture
```

Expected: All 3 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /data/github/picochat/candle-rocm
git add hip-runtime/
git commit -m "hip-runtime: add safe HipRng wrapper with reproducibility tests"
```

### Task 12: HipModule — Kernel Loading

**Files:**
- Modify: `hip-runtime/src/module.rs`
- Create: `kernels/test_add.hip`
- Create: `hip-runtime/tests/module_test.rs`

- [ ] **Step 1: Write a trivial test kernel**

Create `kernels/test_add.hip`:
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void add_scalar(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += scalar;
    }
}
```

- [ ] **Step 2: Write HipModule wrapper**

Write `hip-runtime/src/module.rs`:
```rust
use crate::error::{check_hip, HipError, Result};
use hip_sys::hip_runtime;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

pub struct HipModule {
    module: hip_runtime::hipModule_t,
    functions: HashMap<String, hip_runtime::hipFunction_t>,
}

impl HipModule {
    /// Load a compiled .hsaco GPU binary.
    pub fn load(path: &Path) -> Result<Self> {
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let mut module = std::ptr::null_mut();
        check_hip(unsafe { hip_runtime::hipModuleLoad(&mut module, c_path.as_ptr()) })?;
        Ok(Self {
            module,
            functions: HashMap::new(),
        })
    }

    /// Get a kernel function by name. Caches the lookup.
    pub fn get_function(&mut self, name: &str) -> Result<hip_runtime::hipFunction_t> {
        if let Some(&func) = self.functions.get(name) {
            return Ok(func);
        }
        let c_name = CString::new(name).unwrap();
        let mut func = std::ptr::null_mut();
        check_hip(unsafe {
            hip_runtime::hipModuleGetFunction(&mut func, self.module, c_name.as_ptr())
        })?;
        self.functions.insert(name.to_string(), func);
        Ok(func)
    }

    /// Launch a kernel.
    pub unsafe fn launch(
        func: hip_runtime::hipFunction_t,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        check_hip(hip_runtime::hipModuleLaunchKernel(
            func,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_mem,
            std::ptr::null_mut(), // default stream
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        ))
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        unsafe { hip_runtime::hipModuleUnload(self.module) };
    }
}

/// Compile a .hip source file to .hsaco using hipcc.
pub fn compile_kernel(src: &Path, out: &Path, arch: &str) -> std::io::Result<()> {
    let status = std::process::Command::new("/opt/rocm/bin/hipcc")
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-o",
            out.to_str().unwrap(),
            src.to_str().unwrap(),
        ])
        .status()?;
    if !status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("hipcc failed with status {status}"),
        ));
    }
    Ok(())
}
```

- [ ] **Step 3: Write tests**

Create `hip-runtime/tests/module_test.rs`:
```rust
use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;
use hip_runtime::module::{compile_kernel, HipModule};
use std::path::Path;

#[test]
fn test_compile_and_load_kernel() {
    let _dev = HipDevice::new(0).unwrap();

    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/test_add.hip");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/test_add.hsaco");

    // Determine arch — try gfx1010 first, fall back to gfx1030
    let arch = std::env::var("HIP_ARCH").unwrap_or_else(|_| "gfx1010".to_string());
    compile_kernel(&src, &out, &arch).expect("Failed to compile kernel");

    let mut module = HipModule::load(&out).expect("Failed to load module");
    let func = module.get_function("add_scalar").expect("Failed to get function");
    assert!(!func.is_null());
}

#[test]
fn test_launch_add_scalar() {
    let _dev = HipDevice::new(0).unwrap();

    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/test_add.hip");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/test_add.hsaco");
    let arch = std::env::var("HIP_ARCH").unwrap_or_else(|_| "gfx1010".to_string());
    compile_kernel(&src, &out, &arch).unwrap();

    let mut module = HipModule::load(&out).unwrap();
    let func = module.get_function("add_scalar").unwrap();

    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let mut buf = DeviceBuffer::from_slice(&data).unwrap();

    let scalar: f32 = 10.0;
    let n: i32 = 256;
    let mut data_ptr = buf.as_void_ptr();
    let mut scalar_ptr = &scalar as *const f32 as *mut std::ffi::c_void;
    let mut n_ptr = &n as *const i32 as *mut std::ffi::c_void;

    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;

    unsafe {
        HipModule::launch(
            func,
            (grid_size, 1, 1),
            (block_size, 1, 1),
            0,
            &mut [
                &mut data_ptr as *mut _ as *mut std::ffi::c_void,
                &mut scalar_ptr as *mut _ as *mut std::ffi::c_void,
                &mut n_ptr as *mut _ as *mut std::ffi::c_void,
            ],
        )
        .unwrap();
    }

    let result = buf.to_vec().unwrap();
    for (i, &v) in result.iter().enumerate() {
        let expected = i as f32 + 10.0;
        assert!(
            (v - expected).abs() < 1e-5,
            "Mismatch at {i}: got {v} expected {expected}"
        );
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd /data/github/picochat/candle-rocm
PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p hip-runtime --test module_test -- --nocapture
```

Expected: Both tests pass. The add_scalar kernel compiles, loads, runs, and produces correct results.

- [ ] **Step 5: Commit**

```bash
cd /data/github/picochat/candle-rocm
git add hip-runtime/ kernels/
git commit -m "hip-runtime: add HipModule kernel loading with compile+launch test"
```

---

## Chunk 4: HIP Kernels (TDD Phase 3)

> **Note:** This chunk covers writing and testing the GPU compute kernels. Each kernel is a `.hip` file compiled to `.hsaco` at build time. Tests verify numerical correctness against CPU reference implementations.
>
> This chunk and Chunk 5 (candle integration) will be written in a separate plan document once Chunks 1-3 are validated on hardware. The kernel implementations depend on confirming the exact GPU architecture constraints from Phase 0, and the candle integration depends on the kernels.

### Placeholder Tasks (to be expanded after hardware validation)

- Task 13: affine.hip kernel + tests
- Task 14: unary.hip kernel (relu, exp, log, sqrt, tanh, neg, recip, sqr, silu, gelu, erf, sign, abs, floor, ceil, round, powf, elu) + tests
- Task 15: binary.hip kernel (add, mul, sub, div, max, min, cmp ops) + tests
- Task 16: ternary.hip kernel (where_cond) + tests
- Task 17: reduce.hip kernel (sum, max, argmax, argmin, min) + tests
- Task 18: cast.hip kernel (dtype conversions) + tests
- Task 19: indexing.hip kernel (gather, scatter_add, index_select, index_add) + tests
- Task 20: fill.hip kernel (fill, copy2d, copy_strided) + tests
- Task 21: softmax.hip kernel (fused numerically stable) + tests
- Task 22: rope.hip kernel (fused rotary position embeddings) + tests

---

## Chunk 5: candle Integration (TDD Phase 4)

> **Note:** To be expanded after Chunks 1-4 are working. Covers:
> - Forking candle-core 0.8, adding Device::Rocm variant
> - Implementing BackendDevice (12 methods)
> - Implementing BackendStorage (21 methods)
> - dummy_rocm_backend.rs
> - Integration tests: forward pass, backward pass, picochat model on GPU

### Placeholder Tasks (to be expanded after kernels work)

- Task 23: Fork candle-core, add Device::Rocm + Storage::Rocm + DeviceLocation::Rocm
- Task 24: Implement BackendDevice for RocmDevice
- Task 25: Implement BackendStorage for RocmStorage (dispatch to kernels)
- Task 26: Write dummy_rocm_backend.rs
- Task 27: Integration test — Tensor ops on GPU match CPU
- Task 28: Integration test — picochat forward pass GPU = CPU
- Task 29: Add --device rocm CLI flag to picochat
- Task 30: Verify training loop runs on GPU
