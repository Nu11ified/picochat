# Phase 2: Training Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the optimizer, weight initialization, data loading, and training loop so we can train a GPT model end-to-end on CPU.

**Architecture:** Three crates get filled in: picochat-optim (LR schedule, AdamW, Muon, combined router), picochat-data (token buffer + batch iterator), picochat-train (training step, checkpointing). Weight initialization is added to picochat-core. The CLI gets a `--train` flag. Distributed training (multi-GPU) is deferred to Phase 2.5.

**Tech Stack:** Rust, candle-core 0.8, candle-nn 0.8, safetensors (via candle-core)

---

### Task 1: LR Schedule (Warmup + Cosine Warmdown)

**Files:**
- Create: `crates/picochat-optim/src/schedule.rs`
- Modify: `crates/picochat-optim/src/lib.rs`
- Create: `crates/picochat-optim/tests/schedule_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-optim/tests/schedule_test.rs
use picochat_optim::schedule::LrSchedule;

#[test]
fn test_warmup_starts_at_zero() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    let lr = sched.get_lr(0);
    assert!((lr - 0.0).abs() < 1e-8, "lr at step 0 should be ~0, got {lr}");
}

#[test]
fn test_warmup_reaches_base_lr() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    let lr = sched.get_lr(100);
    assert!((lr - 0.01).abs() < 1e-8, "lr at warmup end should be base_lr, got {lr}");
}

#[test]
fn test_constant_phase() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    // Warmdown starts at step 800 (1000 - 0.2 * 1000)
    let lr_mid = sched.get_lr(500);
    assert!((lr_mid - 0.01).abs() < 1e-8, "lr in constant phase should be base_lr, got {lr_mid}");
}

#[test]
fn test_warmdown_ends_at_zero() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    let lr = sched.get_lr(1000);
    assert!(lr < 1e-6, "lr at final step should be ~0, got {lr}");
}

#[test]
fn test_warmdown_is_cosine() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    // Warmdown starts at 800, midpoint at 900
    let lr_mid = sched.get_lr(900);
    // Cosine at midpoint should be base_lr * 0.5 * (1 + cos(pi * 0.5)) = 0.005
    assert!((lr_mid - 0.005).abs() < 1e-4, "cosine midpoint should be ~0.005, got {lr_mid}");
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-optim`
Expected: FAIL — `schedule` module not found.

**Step 3: Implement LrSchedule**

```rust
// crates/picochat-optim/src/schedule.rs

/// Learning rate schedule: linear warmup → constant → cosine warmdown.
pub struct LrSchedule {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    warmdown_start: usize,
}

impl LrSchedule {
    /// Create a new LR schedule.
    /// - `base_lr`: Peak learning rate
    /// - `warmup_steps`: Steps to linearly ramp from 0 → base_lr
    /// - `total_steps`: Total training steps
    /// - `warmdown_frac`: Fraction of total steps for cosine warmdown at the end
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize, warmdown_frac: f64) -> Self {
        let warmdown_steps = (total_steps as f64 * warmdown_frac) as usize;
        let warmdown_start = total_steps.saturating_sub(warmdown_steps);
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            warmdown_start,
        }
    }

    /// Get the learning rate at a given step.
    pub fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup: 0 → base_lr
            self.base_lr * (step as f64 / self.warmup_steps as f64)
        } else if step < self.warmdown_start {
            // Constant phase
            self.base_lr
        } else {
            // Cosine warmdown: base_lr → 0
            let progress = (step - self.warmdown_start) as f64
                / (self.total_steps - self.warmdown_start) as f64;
            self.base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }
}
```

Update lib.rs:
```rust
// crates/picochat-optim/src/lib.rs
pub mod schedule;
```

**Step 4: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-optim`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add crates/picochat-optim/src/schedule.rs crates/picochat-optim/src/lib.rs crates/picochat-optim/tests/
git commit -m "feat: LR schedule with warmup and cosine warmdown"
```

---

### Task 2: AdamW Optimizer

**Files:**
- Create: `crates/picochat-optim/src/adamw.rs`
- Modify: `crates/picochat-optim/src/lib.rs`
- Create: `crates/picochat-optim/tests/adamw_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-optim/tests/adamw_test.rs
use candle_core::{Device, DType, Tensor};
use candle_nn::VarMap;
use picochat_optim::adamw::AdamW;

#[test]
fn test_adamw_single_step_changes_params() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let var = varmap.get((4, 4), "weight", candle_nn::Init::Const(1.0), DType::F32, &device).unwrap();

    // Create a fake gradient (all ones)
    let grad = Tensor::ones((4, 4), DType::F32, &device).unwrap();

    let mut opt = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.0);
    opt.step_var(&var, &grad, 0.01).unwrap();

    let updated: Vec<f32> = var.flatten_all().unwrap().to_vec1().unwrap();
    // After one step, params should have moved away from 1.0
    assert!(updated[0] < 1.0, "param should decrease, got {}", updated[0]);
}

#[test]
fn test_adamw_weight_decay() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let var = varmap.get((4,), "bias", candle_nn::Init::Const(2.0), DType::F32, &device).unwrap();

    // Zero gradient — only weight decay should change params
    let grad = Tensor::zeros((4,), DType::F32, &device).unwrap();

    let mut opt = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.1);
    opt.step_var(&var, &grad, 0.01).unwrap();

    let updated: Vec<f32> = var.flatten_all().unwrap().to_vec1().unwrap();
    // Weight decay should shrink params toward 0
    assert!(updated[0] < 2.0, "weight decay should shrink, got {}", updated[0]);
}

#[test]
fn test_adamw_multiple_steps_converge() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let var = varmap.get((2,), "w", candle_nn::Init::Const(5.0), DType::F32, &device).unwrap();

    let mut opt = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0);

    // Gradient always points toward reducing param (positive grad = reduce positive param)
    for _ in 0..100 {
        let grad = Tensor::ones((2,), DType::F32, &device).unwrap();
        opt.step_var(&var, &grad, 0.1).unwrap();
    }

    let updated: Vec<f32> = var.flatten_all().unwrap().to_vec1().unwrap();
    assert!(updated[0] < 2.0, "should converge significantly, got {}", updated[0]);
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-optim`
Expected: FAIL — `adamw` module not found.

**Step 3: Implement AdamW**

```rust
// crates/picochat-optim/src/adamw.rs
use candle_core::{Result, Tensor};
use candle_nn::Var;
use std::collections::HashMap;

struct AdamWState {
    m: Tensor,  // first moment
    v: Tensor,  // second moment
    step: usize,
}

/// Fused AdamW optimizer with decoupled weight decay.
pub struct AdamW {
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    default_lr: f64,
    states: HashMap<usize, AdamWState>,
}

impl AdamW {
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            beta1,
            beta2,
            eps,
            weight_decay,
            default_lr: lr,
            states: HashMap::new(),
        }
    }

    /// Apply one AdamW update step to a single variable.
    pub fn step_var(&mut self, var: &Var, grad: &Tensor, lr: f64) -> Result<()> {
        let id = var.as_tensor().id().0;
        let theta = var.as_tensor();

        let state = self.states.entry(id).or_insert_with(|| AdamWState {
            m: Tensor::zeros_like(theta).unwrap(),
            v: Tensor::zeros_like(theta).unwrap(),
            step: 0,
        });
        state.step += 1;

        // Decoupled weight decay: theta = theta * (1 - lr * wd)
        let theta_decayed = if self.weight_decay > 0.0 {
            (theta * (1.0 - lr * self.weight_decay))?
        } else {
            theta.clone()
        };

        // m = beta1 * m + (1 - beta1) * grad
        state.m = ((&state.m * self.beta1)? + (grad * (1.0 - self.beta1))?)?;
        // v = beta2 * v + (1 - beta2) * grad^2
        state.v = ((&state.v * self.beta2)? + (grad.sqr()? * (1.0 - self.beta2))?)?;

        // Bias correction
        let bc1 = 1.0 - self.beta1.powi(state.step as i32);
        let bc2 = 1.0 - self.beta2.powi(state.step as i32);
        let m_hat = (&state.m / bc1)?;
        let v_hat = (&state.v / bc2)?;

        // Update: theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
        let update = (m_hat / (v_hat.sqrt()? + self.eps)?)?;
        let new_theta = (theta_decayed - (update * lr)?)?;

        var.set(&new_theta)?;
        Ok(())
    }
}
```

Update lib.rs to add `pub mod adamw;`

**Step 4: Run tests**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-optim`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/picochat-optim/src/adamw.rs crates/picochat-optim/src/lib.rs crates/picochat-optim/tests/adamw_test.rs
git commit -m "feat: AdamW optimizer with decoupled weight decay"
```

---

### Task 3: Muon Optimizer with Polar Express

**Files:**
- Create: `crates/picochat-optim/src/muon.rs`
- Modify: `crates/picochat-optim/src/lib.rs`
- Create: `crates/picochat-optim/tests/muon_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-optim/tests/muon_test.rs
use candle_core::{Device, DType, Tensor};
use candle_nn::VarMap;
use picochat_optim::muon::{polar_express, Muon};

#[test]
fn test_polar_express_near_orthogonal() {
    let device = Device::Cpu;
    // Random 8x8 matrix
    let g = Tensor::randn(0f32, 1.0, (8, 8), &device).unwrap();
    let u = polar_express(&g).unwrap();

    // U @ U^T should be close to identity
    let uut = u.matmul(&u.t().unwrap()).unwrap();
    let identity = Tensor::eye(8, DType::F32, &device).unwrap();
    let diff = (uut - identity).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff < 0.1, "U@U^T should be near identity, frobenius_sq={diff}");
}

#[test]
fn test_polar_express_tall_matrix() {
    let device = Device::Cpu;
    // 16x4 matrix (tall: rows > cols, should transpose internally)
    let g = Tensor::randn(0f32, 1.0, (16, 4), &device).unwrap();
    let u = polar_express(&g).unwrap();
    assert_eq!(u.dims(), &[16, 4]);
}

#[test]
fn test_muon_single_step() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let var = varmap.get((8, 8), "w", candle_nn::Init::Const(1.0), DType::F32, &device).unwrap();

    let grad = Tensor::randn(0f32, 1.0, (8, 8), &device).unwrap();

    let mut muon = Muon::new(0.02, 0.95);
    muon.step_var(&var, &grad, 0.02).unwrap();

    // Params should have changed
    let updated: Vec<f32> = var.flatten_all().unwrap().to_vec1().unwrap();
    let all_ones = updated.iter().all(|&v| (v - 1.0).abs() < 1e-6);
    assert!(!all_ones, "params should have been updated");
}

#[test]
fn test_muon_momentum_accumulates() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let var = varmap.get((8, 8), "w", candle_nn::Init::Const(0.0), DType::F32, &device).unwrap();

    let mut muon = Muon::new(0.02, 0.95);

    // Apply same gradient multiple times — momentum should amplify the update
    let grad = Tensor::ones((8, 8), DType::F32, &device).unwrap();
    muon.step_var(&var, &grad, 0.02).unwrap();
    let after_1: f32 = var.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];

    muon.step_var(&var, &grad, 0.02).unwrap();
    let after_2: f32 = var.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];

    // Second step should move further in the same direction due to momentum
    let step1_size = after_1.abs();
    let step2_size = (after_2 - after_1).abs();
    assert!(step2_size > step1_size * 0.8, "momentum should amplify: step1={step1_size}, step2={step2_size}");
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-optim`
Expected: FAIL — `muon` module not found.

**Step 3: Implement Muon with Polar Express**

```rust
// crates/picochat-optim/src/muon.rs
use candle_core::{Result, Tensor};
use candle_nn::Var;
use std::collections::HashMap;

/// Polar Express orthogonalization coefficients (5 iterations).
/// From Amsel et al. 2025: https://arxiv.org/pdf/2505.16932
const POLAR_EXPRESS_COEFFS: [(f64, f64, f64); 5] = [
    (8.1566, -22.4833, 15.8788),
    (4.0429,  -2.8089,  0.5000),
    (3.8917,  -2.7725,  0.5061),
    (3.2858,  -2.3681,  0.4645),
    (2.3465,  -1.7098,  0.4232),
];

/// Compute the polar factor of a matrix G via Polar Express iteration.
/// Returns U such that G = U @ S where U is orthogonal, S is positive semidefinite.
/// For tall matrices (rows > cols), transposes internally for faster convergence.
pub fn polar_express(g: &Tensor) -> Result<Tensor> {
    let (rows, cols) = g.dims2()?;
    let transposed = rows > cols;

    let mut x = if transposed { g.t()? } else { g.clone() };

    // Normalize by Frobenius norm
    let norm = x.sqr()?.sum_all()?.sqrt()?;
    x = (x / (norm + 1e-7)?)?;

    // 5 iterations with per-iteration coefficients
    for &(a, b, c) in &POLAR_EXPRESS_COEFFS {
        // A = X @ X^T
        let a_mat = x.matmul(&x.t()?)?;
        // B = b * A + c * A @ A
        let b_mat = ((&a_mat * b)? + (a_mat.matmul(&a_mat)? * c)?)?;
        // X = a * X + B @ X
        x = ((&x * a)? + b_mat.matmul(&x)?)?;
    }

    if transposed {
        x.t()
    } else {
        Ok(x)
    }
}

struct MuonState {
    momentum: Tensor,
}

/// Muon optimizer: Nesterov momentum + Polar Express orthogonalization.
/// For 2D matrix parameters only.
pub struct Muon {
    default_lr: f64,
    beta: f64,  // momentum coefficient (typically 0.95)
    states: HashMap<usize, MuonState>,
}

impl Muon {
    pub fn new(lr: f64, beta: f64) -> Self {
        Self {
            default_lr: lr,
            beta,
            states: HashMap::new(),
        }
    }

    /// Apply one Muon step to a 2D variable.
    pub fn step_var(&mut self, var: &Var, grad: &Tensor, lr: f64) -> Result<()> {
        let id = var.as_tensor().id().0;

        let state = self.states.entry(id).or_insert_with(|| MuonState {
            momentum: Tensor::zeros_like(grad).unwrap(),
        });

        // Nesterov momentum:
        // buf = beta * buf + grad
        // grad_nesterov = grad + beta * buf
        state.momentum = ((&state.momentum * self.beta)? + grad)?;
        let nesterov_grad = (grad + (&state.momentum * self.beta)?)?;

        // Polar Express orthogonalization
        let update = polar_express(&nesterov_grad)?;

        // Apply update: theta = theta - lr * update
        let new_theta = (var.as_tensor() - (update * lr)?)?;
        var.set(&new_theta)?;
        Ok(())
    }
}
```

Update lib.rs to add `pub mod muon;`

**Step 4: Run tests**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-optim`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/picochat-optim/src/muon.rs crates/picochat-optim/src/lib.rs crates/picochat-optim/tests/muon_test.rs
git commit -m "feat: Muon optimizer with Polar Express orthogonalization"
```

---

### Task 4: Combined MuonAdamW Optimizer

**Files:**
- Create: `crates/picochat-optim/src/combined.rs`
- Modify: `crates/picochat-optim/src/lib.rs`
- Modify: `crates/picochat-optim/Cargo.toml` (add candle-nn dependency)
- Create: `crates/picochat-optim/tests/combined_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-optim/tests/combined_test.rs
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_optim::combined::{MuonAdamW, ParamGroup};

#[test]
fn test_classify_params_by_name() {
    // 2D weight → Muon, 1D/scalar → AdamW
    let groups = vec![
        ParamGroup::muon("h.0.attn.c_q.weight", 0.02),
        ParamGroup::adamw("wte.weight", 0.2, 0.0),
        ParamGroup::adamw("resid_lambdas", 0.005, 0.0),
    ];
    assert!(groups[0].is_muon());
    assert!(!groups[1].is_muon());
    assert!(!groups[2].is_muon());
}

#[test]
fn test_combined_step_updates_all_params() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Create a small "model" with 2D and 1D params
    let _w2d = candle_nn::linear_no_bias(8, 8, vb.pp("block.linear"))?;
    let _emb = candle_nn::embedding(16, 8, vb.pp("wte"))?;

    let mut optimizer = MuonAdamW::from_varmap(&varmap, 8);

    // Create fake loss: sum of all params squared
    let vars = varmap.all_vars();
    let mut loss = Tensor::new(0.0f32, &device)?;
    for var in &vars {
        loss = (loss + var.as_tensor().sqr()?.sum_all()?)?;
    }

    optimizer.backward_step(&loss).unwrap();

    // All params should have been updated (moved away from init)
    // We can't check exact values, but the step should not error
}

#[test]
fn test_combined_with_schedule() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _w = candle_nn::linear_no_bias(4, 4, vb.pp("block.linear"))?;

    let mut optimizer = MuonAdamW::from_varmap(&varmap, 4);

    // Step 0 with schedule
    let vars = varmap.all_vars();
    let loss = vars[0].as_tensor().sqr()?.sum_all()?;
    optimizer.backward_step_with_lr(&loss, 0.001).unwrap();
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-optim`
Expected: FAIL — `combined` module not found.

**Step 3: Implement MuonAdamW**

First, update `crates/picochat-optim/Cargo.toml`:
```toml
[package]
name = "picochat-optim"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = { workspace = true }
candle-nn = { workspace = true }
anyhow = { workspace = true }
```

```rust
// crates/picochat-optim/src/combined.rs
use candle_core::{Result, Tensor};
use candle_nn::{Var, VarMap};
use std::collections::HashMap;

use crate::adamw::AdamW;
use crate::muon::Muon;

/// Describes how a parameter should be optimized.
pub struct ParamGroup {
    pub name_pattern: String,
    pub lr: f64,
    pub use_muon: bool,
    pub weight_decay: f64,
}

impl ParamGroup {
    pub fn muon(name: &str, lr: f64) -> Self {
        Self { name_pattern: name.to_string(), lr, use_muon: true, weight_decay: 0.0 }
    }

    pub fn adamw(name: &str, lr: f64, weight_decay: f64) -> Self {
        Self { name_pattern: name.to_string(), lr, use_muon: false, weight_decay }
    }

    pub fn is_muon(&self) -> bool {
        self.use_muon
    }
}

/// Routing rule for a specific variable.
struct VarRoute {
    var: Var,
    name: String,
    use_muon: bool,
    lr: f64,
}

/// Combined Muon + AdamW optimizer.
/// Routes 2D matrix params to Muon, everything else to AdamW.
/// Per-spec LR assignments by parameter name.
pub struct MuonAdamW {
    adamw: AdamW,
    muon: Muon,
    routes: Vec<VarRoute>,
}

impl MuonAdamW {
    /// Create from VarMap with default routing based on tensor dimensionality.
    /// n_embd is used for dimension-aware LR scaling.
    pub fn from_varmap(varmap: &VarMap, n_embd: usize) -> Self {
        let adamw_scale = (n_embd as f64 / 768.0).powf(-0.5);
        let data = varmap.data().lock().unwrap();

        let mut routes = Vec::new();
        for (name, var) in data.iter() {
            let dims = var.as_tensor().dims().len();
            let (use_muon, lr) = classify_param(name, dims, adamw_scale);
            routes.push(VarRoute {
                var: var.clone(),
                name: name.clone(),
                use_muon,
                lr,
            });
        }

        Self {
            adamw: AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.0),
            muon: Muon::new(0.02, 0.95),
            routes,
        }
    }

    /// Run backward pass and optimizer step.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;

        for route in &self.routes {
            if let Some(grad) = grads.get(&route.var) {
                if route.use_muon {
                    self.muon.step_var(&route.var, &grad, route.lr)?;
                } else {
                    self.adamw.step_var(&route.var, &grad, route.lr)?;
                }
            }
        }
        Ok(())
    }

    /// Run backward pass with a specific LR multiplier (for schedule).
    pub fn backward_step_with_lr(&mut self, loss: &Tensor, lr_mult: f64) -> Result<()> {
        let grads = loss.backward()?;

        for route in &self.routes {
            if let Some(grad) = grads.get(&route.var) {
                let lr = route.lr * lr_mult;
                if route.use_muon {
                    self.muon.step_var(&route.var, &grad, lr)?;
                } else {
                    self.adamw.step_var(&route.var, &grad, lr)?;
                }
            }
        }
        Ok(())
    }
}

/// Classify a parameter into Muon or AdamW based on name and dimensionality.
/// Returns (use_muon, base_lr).
fn classify_param(name: &str, ndim: usize, adamw_scale: f64) -> (bool, f64) {
    // Per-spec routing: see spec.md "Parameter Groups" table
    if name.contains("lm_head") {
        (false, 0.004 * adamw_scale)
    } else if name.contains("wte") {
        (false, 0.2 * adamw_scale)
    } else if name.contains("ve.") {
        // Value embeddings
        (false, 0.2 * adamw_scale)
    } else if name.contains("resid_lambdas") {
        (false, 0.005)
    } else if name.contains("x0_lambdas") {
        (false, 0.5)
    } else if ndim == 2 {
        // All 2D weights in transformer blocks → Muon
        (true, 0.02)
    } else {
        // Anything else → AdamW with default LR
        (false, 0.001 * adamw_scale)
    }
}
```

Update lib.rs to add `pub mod combined;`

**Step 4: Run tests**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-optim`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/picochat-optim/
git commit -m "feat: combined MuonAdamW optimizer with parameter routing"
```

---

### Task 5: Weight Initialization

**Files:**
- Create: `crates/picochat-core/src/init.rs`
- Modify: `crates/picochat-core/src/lib.rs`
- Create: `crates/picochat-core/tests/init_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-core/tests/init_test.rs
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_core::init::initialize_weights;

#[test]
fn test_resid_lambdas_init_to_one() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();

    initialize_weights(&varmap, &config).unwrap();

    let data = varmap.data().lock().unwrap();
    let lambdas = data.get("resid_lambdas").unwrap();
    let vals: Vec<f32> = lambdas.flatten_all().unwrap().to_vec1().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-6, "resid_lambdas[{i}] should be 1.0, got {v}");
    }
}

#[test]
fn test_x0_lambdas_init_to_point_one() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();

    initialize_weights(&varmap, &config).unwrap();

    let data = varmap.data().lock().unwrap();
    let lambdas = data.get("x0_lambdas").unwrap();
    let vals: Vec<f32> = lambdas.flatten_all().unwrap().to_vec1().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!((v - 0.1).abs() < 1e-6, "x0_lambdas[{i}] should be 0.1, got {v}");
    }
}

#[test]
fn test_c_proj_init_to_zero() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();

    initialize_weights(&varmap, &config).unwrap();

    let data = varmap.data().lock().unwrap();
    // c_proj in attention and MLP should be zeros
    for (name, var) in data.iter() {
        if name.contains("c_proj") {
            let sum: f32 = var.as_tensor().abs()?.sum_all()?.to_scalar().unwrap();
            assert!(sum < 1e-6, "{name} should be zeros, got sum={sum}");
        }
    }
}

#[test]
fn test_wte_init_normal() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();

    initialize_weights(&varmap, &config).unwrap();

    let data = varmap.data().lock().unwrap();
    let wte = data.get("wte.weight").unwrap();
    let vals: Vec<f32> = wte.flatten_all().unwrap().to_vec1().unwrap();
    // Check it's not all zeros (was initialized with normal distribution)
    let sum_sq: f32 = vals.iter().map(|v| v * v).sum();
    assert!(sum_sq > 1.0, "wte should have non-trivial values after normal init");
    // Check approximate std dev ~1.0
    let mean_sq = sum_sq / vals.len() as f32;
    assert!(mean_sq > 0.5 && mean_sq < 2.0, "wte init std should be ~1.0, got rms={}", mean_sq.sqrt());
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-core`
Expected: FAIL — `init` module not found.

**Step 3: Implement weight initialization**

```rust
// crates/picochat-core/src/init.rs
use candle_core::{DType, Result, Tensor};
use candle_nn::VarMap;
use crate::config::GPTConfig;

/// Apply proper weight initialization per the spec.
///
/// | Parameter | Init |
/// |-----------|------|
/// | wte | Normal(0, 1.0) |
/// | lm_head | Normal(0, 0.001) |
/// | c_q, c_k, c_v, c_fc | Uniform(-s, s) where s = sqrt(3/n_embd) |
/// | c_proj (attn & MLP) | Zeros |
/// | resid_lambdas | Fill(1.0) |
/// | x0_lambdas | Fill(0.1) |
/// | value embeddings | Uniform(-s, s) |
/// | ve_gate | Zeros |
pub fn initialize_weights(varmap: &VarMap, config: &GPTConfig) -> Result<()> {
    let s = (3.0 / config.n_embd as f64).sqrt();
    let data = varmap.data().lock().unwrap();

    for (name, var) in data.iter() {
        let shape = var.as_tensor().shape().clone();
        let device = var.as_tensor().device().clone();
        let dtype = var.as_tensor().dtype();

        let new_val = if name == "resid_lambdas" {
            Tensor::ones(shape.dims(), dtype, &device)?
        } else if name == "x0_lambdas" {
            (Tensor::ones(shape.dims(), dtype, &device)? * 0.1)?
        } else if name.contains("c_proj") || name.contains("ve_gate") {
            Tensor::zeros(shape.dims(), dtype, &device)?
        } else if name.contains("lm_head") {
            Tensor::randn(0f32, 0.001, shape.dims(), &device)?.to_dtype(dtype)?
        } else if name.contains("wte") && !name.contains("ve.") {
            Tensor::randn(0f32, 1.0, shape.dims(), &device)?.to_dtype(dtype)?
        } else if name.contains("c_q") || name.contains("c_k") || name.contains("c_v")
            || name.contains("c_fc") || name.contains("ve.")
        {
            // Uniform(-s, s) via (rand * 2s - s)
            let uniform = Tensor::rand(0f32, 1.0, shape.dims(), &device)?.to_dtype(dtype)?;
            ((uniform * (2.0 * s))? - s)?
        } else {
            // Default: leave as-is (VarMap zero init)
            continue;
        };

        var.set(&new_val)?;
    }

    Ok(())
}
```

Update lib.rs to add `pub mod init;`

**Step 4: Run tests**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-core`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/picochat-core/src/init.rs crates/picochat-core/src/lib.rs crates/picochat-core/tests/init_test.rs
git commit -m "feat: weight initialization matching nanochat spec"
```

---

### Task 6: Token Data Buffer and Batch Iterator

**Files:**
- Modify: `crates/picochat-data/Cargo.toml` (add dependencies)
- Create: `crates/picochat-data/src/dataloader.rs`
- Modify: `crates/picochat-data/src/lib.rs`
- Create: `crates/picochat-data/tests/dataloader_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-data/tests/dataloader_test.rs
use picochat_data::dataloader::{TokenDataset, DataLoader};

#[test]
fn test_dataset_from_tokens() {
    let tokens: Vec<u32> = (0..1000).collect();
    let ds = TokenDataset::new(tokens);
    assert_eq!(ds.len(), 1000);
}

#[test]
fn test_dataloader_batch_shape() {
    let tokens: Vec<u32> = (0..10000).collect();
    let ds = TokenDataset::new(tokens);
    let mut dl = DataLoader::new(ds, 4, 64); // batch=4, seq_len=64

    let (input, target) = dl.next_batch().unwrap();
    assert_eq!(input.len(), 4);
    assert_eq!(input[0].len(), 64);
    assert_eq!(target.len(), 4);
    assert_eq!(target[0].len(), 64);
}

#[test]
fn test_dataloader_target_is_shifted() {
    // target[i] = input[i+1] (next-token prediction)
    let tokens: Vec<u32> = (0..10000).collect();
    let ds = TokenDataset::new(tokens);
    let mut dl = DataLoader::new(ds, 1, 8);

    let (input, target) = dl.next_batch().unwrap();
    // For contiguous data: target should be input shifted by 1
    for i in 0..7 {
        assert_eq!(target[0][i], input[0][i + 1],
            "target[{i}] should equal input[{}]", i + 1);
    }
}

#[test]
fn test_dataloader_wraps_around() {
    let tokens: Vec<u32> = (0..100).collect();
    let ds = TokenDataset::new(tokens);
    let mut dl = DataLoader::new(ds, 2, 32);

    // Should be able to get many batches without error
    for _ in 0..20 {
        let (input, _target) = dl.next_batch().unwrap();
        assert_eq!(input.len(), 2);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-data`
Expected: FAIL — `dataloader` module not found.

**Step 3: Implement DataLoader**

Update `crates/picochat-data/Cargo.toml`:
```toml
[package]
name = "picochat-data"
version = "0.1.0"
edition = "2021"

[dependencies]
picochat-tokenizer = { path = "../picochat-tokenizer" }
anyhow = { workspace = true }
serde = { workspace = true }
rand = { workspace = true }
```

```rust
// crates/picochat-data/src/dataloader.rs
use anyhow::Result;
use rand::Rng;

/// A buffer of pre-tokenized data (token IDs).
pub struct TokenDataset {
    tokens: Vec<u32>,
}

impl TokenDataset {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self { tokens }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Simple random-sampling data loader.
/// Yields (input, target) batches where target is input shifted by 1 position.
pub struct DataLoader {
    dataset: TokenDataset,
    batch_size: usize,
    seq_len: usize,
    rng: rand::rngs::ThreadRng,
}

impl DataLoader {
    pub fn new(dataset: TokenDataset, batch_size: usize, seq_len: usize) -> Self {
        Self {
            dataset,
            batch_size,
            seq_len,
            rng: rand::thread_rng(),
        }
    }

    /// Get the next batch: (input, target) where each is Vec<Vec<u32>> of shape (B, T).
    /// target[b][t] = input[b][t+1] (next-token prediction).
    pub fn next_batch(&mut self) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        let max_start = self.dataset.len().saturating_sub(self.seq_len + 1);
        let mut inputs = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            let start = if max_start > 0 {
                self.rng.gen_range(0..max_start)
            } else {
                0
            };
            let chunk = &self.dataset.tokens[start..start + self.seq_len + 1];
            inputs.push(chunk[..self.seq_len].to_vec());
            targets.push(chunk[1..self.seq_len + 1].to_vec());
        }

        Ok((inputs, targets))
    }
}
```

Update lib.rs:
```rust
// crates/picochat-data/src/lib.rs
pub mod dataloader;
```

**Step 4: Run tests**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-data`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add crates/picochat-data/
git commit -m "feat: token dataset and batch data loader"
```

---

### Task 7: Training Step Function

**Files:**
- Modify: `crates/picochat-train/Cargo.toml` (add all needed deps)
- Create: `crates/picochat-train/src/trainer.rs`
- Modify: `crates/picochat-train/src/lib.rs`
- Create: `crates/picochat-train/tests/trainer_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-train/tests/trainer_test.rs
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_core::init::initialize_weights;
use picochat_train::trainer::Trainer;
use picochat_data::dataloader::{TokenDataset, DataLoader};

#[test]
fn test_single_train_step() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();

    let mut trainer = Trainer::new(&varmap, &config);

    // Create a small batch
    let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], &device).unwrap();
    let target = Tensor::new(&[[2u32, 3, 4, 5, 6, 7, 8, 9]], &device).unwrap();

    let loss = trainer.train_step(&model, &input, &target).unwrap();
    let loss_val: f32 = loss.to_scalar().unwrap();
    // Initial loss should be ~log(vocab_size) = ~10.4
    assert!(loss_val > 5.0 && loss_val < 15.0, "initial loss={loss_val}");
}

#[test]
fn test_loss_decreases_over_steps() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();

    let mut trainer = Trainer::new(&varmap, &config);

    // Use the same batch repeatedly to test overfitting
    let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], &device).unwrap();
    let target = Tensor::new(&[[2u32, 3, 4, 5, 6, 7, 8, 9]], &device).unwrap();

    let mut losses = Vec::new();
    for _ in 0..20 {
        let loss = trainer.train_step(&model, &input, &target).unwrap();
        losses.push(loss.to_scalar::<f32>().unwrap());
    }

    // Loss should generally decrease when overfitting on a single batch
    let first_loss = losses[0];
    let last_loss = *losses.last().unwrap();
    assert!(last_loss < first_loss,
        "loss should decrease: first={first_loss}, last={last_loss}");
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train`
Expected: FAIL — `trainer` module not found.

**Step 3: Implement Trainer**

Update `crates/picochat-train/Cargo.toml`:
```toml
[package]
name = "picochat-train"
version = "0.1.0"
edition = "2021"

[dependencies]
picochat-core = { path = "../picochat-core" }
picochat-optim = { path = "../picochat-optim" }
picochat-data = { path = "../picochat-data" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
anyhow = { workspace = true }

[dev-dependencies]
picochat-data = { path = "../picochat-data" }
```

```rust
// crates/picochat-train/src/trainer.rs
use candle_core::{Result, Tensor};
use candle_nn::VarMap;
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_optim::combined::MuonAdamW;
use picochat_optim::schedule::LrSchedule;

/// Training driver: holds optimizer and schedule, executes train steps.
pub struct Trainer {
    optimizer: MuonAdamW,
    schedule: Option<LrSchedule>,
    step_count: usize,
}

impl Trainer {
    /// Create a new trainer with default optimizer settings.
    pub fn new(varmap: &VarMap, config: &GPTConfig) -> Self {
        let optimizer = MuonAdamW::from_varmap(varmap, config.n_embd);
        Self {
            optimizer,
            schedule: None,
            step_count: 0,
        }
    }

    /// Create with an LR schedule.
    pub fn with_schedule(varmap: &VarMap, config: &GPTConfig, schedule: LrSchedule) -> Self {
        let optimizer = MuonAdamW::from_varmap(varmap, config.n_embd);
        Self {
            optimizer,
            schedule: Some(schedule),
            step_count: 0,
        }
    }

    /// Execute one training step: forward → backward → optimizer step.
    /// Returns the loss tensor (scalar).
    pub fn train_step(&mut self, model: &GPT, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        // Forward pass with targets → cross-entropy loss
        let loss = model.forward(input, Some(target))?;

        // Backward + optimizer step
        if let Some(ref schedule) = self.schedule {
            let lr_mult = schedule.get_lr(self.step_count) / schedule.get_lr(1.max(self.step_count));
            self.optimizer.backward_step_with_lr(&loss, lr_mult)?;
        } else {
            self.optimizer.backward_step(&loss)?;
        }

        self.step_count += 1;
        Ok(loss)
    }

    /// Current step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}
```

Update lib.rs:
```rust
// crates/picochat-train/src/lib.rs
pub mod trainer;
```

**Step 4: Run tests**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train`
Expected: All tests PASS. The "loss decreases" test may be slow on CPU (~30s for depth=4, 20 steps).

**Step 5: Commit**

```bash
git add crates/picochat-train/
git commit -m "feat: training step with MuonAdamW optimizer"
```

---

### Task 8: Checkpoint Save/Load

**Files:**
- Create: `crates/picochat-train/src/checkpoint.rs`
- Modify: `crates/picochat-train/src/lib.rs`
- Create: `crates/picochat-train/tests/checkpoint_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-train/tests/checkpoint_test.rs
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_core::init::initialize_weights;
use picochat_train::checkpoint;

#[test]
fn test_save_and_load_roundtrip() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);

    // Create and initialize model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();

    // Save checkpoint
    let path = "/tmp/picochat_test_checkpoint.safetensors";
    checkpoint::save_varmap(&varmap, path).unwrap();

    // Load into a new VarMap + model
    let varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
    let _model2 = GPT::new(&config, vb2).unwrap();
    checkpoint::load_varmap(&varmap2, path, &device).unwrap();

    // Compare values
    let data1 = varmap.data().lock().unwrap();
    let data2 = varmap2.data().lock().unwrap();
    for (name, var1) in data1.iter() {
        let var2 = data2.get(name).unwrap_or_else(|| panic!("missing {name}"));
        let diff = (var1.as_tensor() - var2.as_tensor()).unwrap()
            .abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-6, "mismatch in {name}: diff={diff}");
    }

    // Cleanup
    std::fs::remove_file(path).ok();
}

#[test]
fn test_save_config_alongside() {
    let config = GPTConfig::from_depth(4);
    let path = "/tmp/picochat_test_config.json";
    checkpoint::save_config(&config, path).unwrap();

    let loaded = checkpoint::load_config(path).unwrap();
    assert_eq!(loaded.n_layer, config.n_layer);
    assert_eq!(loaded.n_embd, config.n_embd);
    assert_eq!(loaded.n_head, config.n_head);

    std::fs::remove_file(path).ok();
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train`
Expected: FAIL — `checkpoint` module not found.

**Step 3: Implement checkpoint save/load**

Add `serde_json` to `crates/picochat-train/Cargo.toml`:
```toml
[dependencies]
# ... existing deps ...
serde = { workspace = true }
serde_json = { workspace = true }
```

```rust
// crates/picochat-train/src/checkpoint.rs
use anyhow::Result;
use candle_core::{safetensors, Device, Tensor};
use candle_nn::VarMap;
use picochat_core::config::GPTConfig;
use std::collections::HashMap;
use std::path::Path;

/// Save all VarMap tensors to a safetensors file.
pub fn save_varmap<P: AsRef<Path>>(varmap: &VarMap, path: P) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let tensors: HashMap<String, Tensor> = data
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect();
    safetensors::save(&tensors, path)?;
    Ok(())
}

/// Load tensors from a safetensors file into an existing VarMap.
/// The VarMap must already have been populated (e.g., by building the model).
pub fn load_varmap<P: AsRef<Path>>(varmap: &VarMap, path: P, device: &Device) -> Result<()> {
    let saved = safetensors::load(path, device)?;
    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        if let Some(saved_tensor) = saved.get(name) {
            var.set(saved_tensor)?;
        }
    }
    Ok(())
}

/// Save GPTConfig as JSON.
pub fn save_config<P: AsRef<Path>>(config: &GPTConfig, path: P) -> Result<()> {
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load GPTConfig from JSON.
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<GPTConfig> {
    let json = std::fs::read_to_string(path)?;
    let config: GPTConfig = serde_json::from_str(&json)?;
    Ok(config)
}
```

Update lib.rs to add `pub mod checkpoint;`

**Step 4: Run tests**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/picochat-train/
git commit -m "feat: checkpoint save/load with safetensors"
```

---

### Task 9: CLI Train Command

**Files:**
- Modify: `crates/picochat-cli/Cargo.toml` (add picochat-train, picochat-data, picochat-optim deps)
- Modify: `crates/picochat-cli/src/main.rs`

**Step 1: Update Cargo.toml**

```toml
[package]
name = "picochat-cli"
version = "0.1.0"
edition = "2021"

[dependencies]
picochat-core = { path = "../picochat-core" }
picochat-train = { path = "../picochat-train" }
picochat-data = { path = "../picochat-data" }
picochat-optim = { path = "../picochat-optim" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
clap = { workspace = true }
anyhow = { workspace = true }
```

**Step 2: Implement train CLI**

```rust
// crates/picochat-cli/src/main.rs
use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_core::init::initialize_weights;
use picochat_data::dataloader::{TokenDataset, DataLoader};
use picochat_optim::schedule::LrSchedule;
use picochat_train::trainer::Trainer;
use picochat_train::checkpoint;

#[derive(Parser)]
#[command(name = "picochat", version, about = "Train and chat with small reasoning LLMs")]
struct Cli {
    /// Model depth (number of transformer layers)
    #[arg(long, default_value_t = 4)]
    depth: usize,

    /// Run a quick smoke test forward pass
    #[arg(long)]
    smoke_test: bool,

    /// Train the model on synthetic data
    #[arg(long)]
    train: bool,

    /// Number of training steps
    #[arg(long, default_value_t = 50)]
    steps: usize,

    /// Batch size
    #[arg(long, default_value_t = 2)]
    batch_size: usize,

    /// Sequence length
    #[arg(long, default_value_t = 64)]
    seq_len: usize,

    /// Save checkpoint path
    #[arg(long)]
    save: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = Device::Cpu;

    if cli.smoke_test {
        run_smoke_test(&cli, &device)?;
    } else if cli.train {
        run_train(&cli, &device)?;
    } else {
        println!("picochat v0.1.0");
        println!("  --smoke-test  Run forward pass verification");
        println!("  --train       Train on synthetic data");
    }

    Ok(())
}

fn run_smoke_test(cli: &Cli, device: &Device) -> Result<()> {
    println!("picochat smoke test (depth={})", cli.depth);
    let config = GPTConfig::from_depth(cli.depth);
    println!("Config: n_layer={}, n_embd={}, n_head={}, n_kv_head={}",
        config.n_layer, config.n_embd, config.n_head, config.n_kv_head);
    println!("Vocab: {} (padded: {})", config.vocab_size, config.padded_vocab_size());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(&config, vb)?;

    let num_params = model.num_parameters();
    println!("Total parameters: {num_params} ({:.2}M)", num_params as f64 / 1e6);

    let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], device)?;
    println!("Running forward pass with 8 tokens...");
    let logits = model.forward(&input, None)?;
    println!("Output logits shape: {:?}", logits.shape());
    println!("Smoke test PASSED!");
    Ok(())
}

fn run_train(cli: &Cli, device: &Device) -> Result<()> {
    let config = GPTConfig::from_depth(cli.depth);
    println!("picochat training (depth={})", cli.depth);
    println!("Config: n_layer={}, n_embd={}, n_head={}, n_kv_head={}",
        config.n_layer, config.n_embd, config.n_head, config.n_kv_head);

    // Create model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(&config, vb)?;
    initialize_weights(&varmap, &config)?;

    let num_params = model.num_parameters();
    println!("Parameters: {num_params} ({:.2}M)", num_params as f64 / 1e6);

    // Generate synthetic training data (random tokens)
    let num_tokens = cli.batch_size * cli.seq_len * cli.steps * 2;
    let tokens: Vec<u32> = (0..num_tokens)
        .map(|i| (i as u32) % config.vocab_size as u32)
        .collect();
    let dataset = TokenDataset::new(tokens);
    let mut dataloader = DataLoader::new(dataset, cli.batch_size, cli.seq_len);

    // Create trainer with schedule
    let schedule = LrSchedule::new(0.001, cli.steps / 10, cli.steps, 0.2);
    let mut trainer = Trainer::with_schedule(&varmap, &config, schedule);

    println!("Training for {} steps (batch_size={}, seq_len={})...", cli.steps, cli.batch_size, cli.seq_len);

    let start = std::time::Instant::now();
    for step in 0..cli.steps {
        let (input_vecs, target_vecs) = dataloader.next_batch()?;

        // Convert to tensors
        let input = Tensor::new(input_vecs.as_slice(), device)?;
        let target = Tensor::new(target_vecs.as_slice(), device)?;

        let loss = trainer.train_step(&model, &input, &target)?;

        if step % 10 == 0 || step == cli.steps - 1 {
            let loss_val: f32 = loss.to_scalar()?;
            let elapsed = start.elapsed().as_secs_f64();
            let tokens_per_sec = ((step + 1) * cli.batch_size * cli.seq_len) as f64 / elapsed;
            println!("step {step:>4}/{} | loss: {loss_val:.4} | tok/s: {tokens_per_sec:.0}",
                cli.steps);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("Training complete in {elapsed:.1}s");

    // Save checkpoint if requested
    if let Some(ref path) = cli.save {
        checkpoint::save_varmap(&varmap, format!("{path}/model.safetensors"))?;
        checkpoint::save_config(&config, format!("{path}/config.json"))?;
        println!("Checkpoint saved to {path}/");
    }

    Ok(())
}
```

**Step 3: Build and run**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo build -p picochat-cli`
Expected: Compiles.

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo run -p picochat-cli -- --train --depth 4 --steps 20 --batch-size 1 --seq-len 32`
Expected: Prints training progress with decreasing loss, then "Training complete".

**Step 4: Verify all tests still pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test --workspace`
Expected: All tests across all crates PASS.

**Step 5: Commit**

```bash
git add crates/picochat-cli/
git commit -m "feat: CLI training command with synthetic data"
```

---

## Summary

After completing all 9 tasks, you will have:

1. **LR Schedule** — warmup → constant → cosine warmdown
2. **AdamW optimizer** — per-variable step with decoupled weight decay
3. **Muon optimizer** — Nesterov momentum + Polar Express orthogonalization (5 iterations)
4. **MuonAdamW** — combined router dispatching 2D params → Muon, rest → AdamW, with per-spec LR assignments
5. **Weight initialization** — matching nanochat exactly (Normal, Uniform, Zeros, constant fills)
6. **Token data loader** — yields (input, target) batches from a token buffer
7. **Trainer** — forward/backward/step cycle with MuonAdamW and LR schedule
8. **Checkpointing** — save/load model weights (safetensors) and config (JSON)
9. **CLI** — `picochat --train --depth 4 --steps 50` runs end-to-end training

**Next plan**: Phase 2.5 will add the BPE tokenizer (picochat-tokenizer) and Parquet/FineWeb data loading for training on real text.
