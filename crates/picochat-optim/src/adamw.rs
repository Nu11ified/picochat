use candle_core::{Result, Tensor, TensorId, Var};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// Per-variable optimizer state for AdamW.
struct AdamWState {
    m: Tensor,  // first moment estimate
    v: Tensor,  // second moment estimate
    step: usize,
}

/// AdamW optimizer with decoupled weight decay.
///
/// Implements the algorithm from Loshchilov & Hutter (2019):
/// weight decay is applied directly to parameters rather than
/// being folded into the gradient, giving better regularization.
pub struct AdamW {
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    default_lr: f64,
    states: HashMap<TensorId, AdamWState>,
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

    /// Return the default learning rate.
    pub fn default_lr(&self) -> f64 {
        self.default_lr
    }

    /// Perform one AdamW update for a single variable.
    ///
    /// 1. Decoupled weight decay: `theta = theta * (1 - lr * wd)`
    /// 2. Update moments: `m = beta1*m + (1-beta1)*grad`,
    ///    `v = beta2*v + (1-beta2)*grad^2`
    /// 3. Bias correction: `m_hat = m / (1 - beta1^t)`,
    ///    `v_hat = v / (1 - beta2^t)`
    /// 4. Parameter update: `theta = theta - lr * m_hat / (sqrt(v_hat) + eps)`
    pub fn step_var(&mut self, var: &Var, grad: &Tensor, lr: f64) -> Result<()> {
        let var_id = var.as_tensor().id();

        // Initialize state on first call for this variable.
        if let Entry::Vacant(e) = self.states.entry(var_id) {
            e.insert(AdamWState {
                m: Tensor::zeros_like(var.as_tensor())?,
                v: Tensor::zeros_like(var.as_tensor())?,
                step: 0,
            });
        }

        let state = self.states.get_mut(&var_id).unwrap();
        state.step += 1;
        let t = state.step as f64;

        // 1. Decoupled weight decay
        let theta = (var.as_tensor() * (1.0 - lr * self.weight_decay))?;

        // 2. Update biased first and second moment estimates
        state.m = ((&state.m * self.beta1)? + (grad * (1.0 - self.beta1))?)?;
        state.v = ((&state.v * self.beta2)? + (grad.sqr()? * (1.0 - self.beta2))?)?;

        // 3. Bias correction
        let m_hat = (&state.m / (1.0 - self.beta1.powf(t)))?;
        let v_hat = (&state.v / (1.0 - self.beta2.powf(t)))?;

        // 4. Parameter update
        let update = (m_hat / (v_hat.sqrt()? + self.eps)?)?;
        let new_theta = (theta - (update * lr)?)?;

        var.set(&new_theta)?;
        Ok(())
    }
}
