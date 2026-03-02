use candle_core::{Result, Tensor};
use candle_nn::VarMap;
use crate::config::GPTConfig;

/// Apply per-spec weight initialization to a VarMap that already has variables
/// created (by building the GPT model).
///
/// Initialization rules (from nanochat spec):
///   - `wte` (token embedding):   Normal(0, 1.0)
///   - `lm_head`:                 Normal(0, 0.001)
///   - `c_q`, `c_k`, `c_v`, `c_fc`: Uniform(-s, s) where s = sqrt(3/n_embd)
///   - `c_proj` (attn & MLP):     Zeros
///   - `resid_lambdas`:           Fill(1.0)
///   - `x0_lambdas`:              Fill(0.1)
///   - Value embeddings (`ve.`):  Uniform(-s, s)
///   - VE gate weights (`ve_gate`): Zeros
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
            // Uniform(-s, s) = rand(0,1) * 2s - s
            let uniform = Tensor::rand(0f32, 1.0, shape.dims(), &device)?.to_dtype(dtype)?;
            ((uniform * (2.0 * s))? - s)?
        } else {
            continue; // Leave as-is
        };

        var.set(&new_val)?;
    }
    Ok(())
}
