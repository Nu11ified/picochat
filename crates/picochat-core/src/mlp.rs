use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

pub struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    pub fn new(n_embd: usize, vb: VarBuilder) -> Result<Self> {
        let c_fc = linear_no_bias(n_embd, 4 * n_embd, vb.pp("c_fc"))?;
        let c_proj = linear_no_bias(4 * n_embd, n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = x.relu()?.sqr()?; // ReLU^2
        self.c_proj.forward(&x)
    }
}
