use candle_core::{Result, Tensor, Var};
use candle_nn::VarMap;
use std::collections::HashMap;

use crate::adamw::AdamW;
use crate::muon::Muon;

/// Describes how a group of parameters should be optimized.
pub struct ParamGroup {
    /// Glob/substring pattern matched against parameter names.
    pub name_pattern: String,
    /// Base learning rate for this group.
    pub lr: f64,
    /// If true, use Muon; otherwise use AdamW.
    pub use_muon: bool,
    /// Weight decay (only relevant for AdamW params).
    pub weight_decay: f64,
}

impl ParamGroup {
    /// Convenience constructor for a Muon-optimized parameter group.
    pub fn muon(name: &str, lr: f64) -> Self {
        Self {
            name_pattern: name.to_string(),
            lr,
            use_muon: true,
            weight_decay: 0.0,
        }
    }

    /// Convenience constructor for an AdamW-optimized parameter group.
    pub fn adamw(name: &str, lr: f64, weight_decay: f64) -> Self {
        Self {
            name_pattern: name.to_string(),
            lr,
            use_muon: false,
            weight_decay,
        }
    }

    /// Returns true if this group uses the Muon optimizer.
    pub fn is_muon(&self) -> bool {
        self.use_muon
    }
}

/// Internal routing info: which optimizer and LR to use for each variable.
struct VarRoute {
    var: Var,
    name: String,
    use_muon: bool,
    lr: f64,
}

/// Combined optimizer that dispatches each parameter to either Muon or AdamW
/// based on its name and shape.
///
/// Muon is used for 2D weight matrices (where orthogonal updates help),
/// while AdamW handles embeddings, biases, layer-norm parameters, and
/// other special-cased groups.
pub struct MuonAdamW {
    adamw: AdamW,
    muon: Muon,
    routes: Vec<VarRoute>,
}

impl MuonAdamW {
    /// Build a `MuonAdamW` optimizer from a `VarMap`, classifying every
    /// parameter by name and shape.
    ///
    /// Classification rules (checked in order):
    /// - `lm_head`       -> AdamW, lr = 0.004 * scale
    /// - `wte` (not `ve.`) -> AdamW, lr = 0.2 * scale
    /// - `ve.`           -> AdamW, lr = 0.2 * scale
    /// - `resid_lambdas` -> AdamW, lr = 0.005
    /// - `x0_lambdas`    -> AdamW, lr = 0.5
    /// - 2D params       -> Muon,  lr = 0.02
    /// - everything else -> AdamW, lr = 0.001 * scale
    ///
    /// Where `scale = (n_embd / 768)^{-0.5}`.
    pub fn from_varmap(varmap: &VarMap, n_embd: usize) -> Self {
        let scale = (n_embd as f64 / 768.0).powf(-0.5);

        // Lock the VarMap, extract (name, var) pairs, then drop the lock.
        let pairs: Vec<(String, Var)> = {
            let data = varmap.data().lock().unwrap();
            data.iter()
                .map(|(name, var)| (name.clone(), var.clone()))
                .collect()
        };

        let mut routes = Vec::with_capacity(pairs.len());

        for (name, var) in pairs {
            let dims = var.as_tensor().dims();
            let (use_muon, lr) = if name.contains("lm_head") {
                (false, 0.004 * scale)
            } else if name.contains("wte") && !name.contains("ve.") {
                (false, 0.2 * scale)
            } else if name.contains("ve.") {
                (false, 0.2 * scale)
            } else if name.contains("resid_lambdas") {
                (false, 0.005)
            } else if name.contains("x0_lambdas") {
                (false, 0.5)
            } else if dims.len() == 2 {
                (true, 0.02)
            } else {
                (false, 0.001 * scale)
            };

            routes.push(VarRoute {
                var,
                name,
                use_muon,
                lr,
            });
        }

        // Default hyperparams for the sub-optimizers.
        let adamw = AdamW::new(0.001, 0.9, 0.95, 1e-8, 0.0);
        let muon = Muon::new(0.02, 0.95);

        Self {
            adamw,
            muon,
            routes,
        }
    }

    /// Run backward pass on `loss` and take one optimizer step on every
    /// routed parameter.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step_with_lr(loss, 1.0)
    }

    /// Run backward pass on `loss` and take one optimizer step, scaling
    /// every parameter group's LR by `lr_mult`.
    pub fn backward_step_with_lr(&mut self, loss: &Tensor, lr_mult: f64) -> Result<()> {
        let grads = loss.backward()?;

        for route in &self.routes {
            let grad = match grads.get(route.var.as_tensor()) {
                Some(g) => g.clone(),
                None => continue, // param not in the computation graph
            };

            let lr = route.lr * lr_mult;

            if route.use_muon {
                self.muon.step_var(&route.var, &grad, lr)?;
            } else {
                self.adamw.step_var(&route.var, &grad, lr)?;
            }
        }

        Ok(())
    }

    /// Return a reference to the internal routes (for diagnostics / logging).
    pub fn route_summary(&self) -> Vec<(&str, bool, f64)> {
        self.routes
            .iter()
            .map(|r| (r.name.as_str(), r.use_muon, r.lr))
            .collect()
    }

    /// Export all optimizer state tensors keyed by parameter name.
    ///
    /// Keys use the format `adamw.{name}.m`, `adamw.{name}.v`,
    /// `adamw.{name}.step` (as 1-element f32 tensor), or `muon.{name}.buf`.
    pub fn save_state(&self) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        for route in &self.routes {
            let id = route.var.as_tensor().id();
            if route.use_muon {
                if let Some(buf) = self.muon.get_state(id) {
                    tensors.insert(format!("muon.{}.buf", route.name), buf.clone());
                }
            } else if let Some((m, v, step)) = self.adamw.get_state(id) {
                tensors.insert(format!("adamw.{}.m", route.name), m.clone());
                tensors.insert(format!("adamw.{}.v", route.name), v.clone());
                let step_t = Tensor::new(&[step as f32], m.device())?;
                tensors.insert(format!("adamw.{}.step", route.name), step_t);
            }
        }
        Ok(tensors)
    }

    /// Restore optimizer state from named tensors (inverse of `save_state`).
    pub fn load_state(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        for route in &self.routes {
            let id = route.var.as_tensor().id();
            if route.use_muon {
                let key = format!("muon.{}.buf", route.name);
                if let Some(buf) = tensors.get(&key) {
                    self.muon.set_state(id, buf.clone());
                }
            } else {
                let m_key = format!("adamw.{}.m", route.name);
                let v_key = format!("adamw.{}.v", route.name);
                let step_key = format!("adamw.{}.step", route.name);
                if let (Some(m), Some(v)) = (tensors.get(&m_key), tensors.get(&v_key)) {
                    let step = tensors.get(&step_key)
                        .map(|t| t.to_vec1::<f32>().unwrap_or_default())
                        .and_then(|v| v.first().copied())
                        .unwrap_or(0.0) as usize;
                    self.adamw.set_state(id, m.clone(), v.clone(), step);
                }
            }
        }
        Ok(())
    }
}
