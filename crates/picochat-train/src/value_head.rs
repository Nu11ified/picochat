use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

/// Lightweight outcome reward model: single linear projection from hidden state to scalar.
pub struct ValueHead {
    proj: Linear,
}

impl ValueHead {
    pub fn new(n_embd: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear_no_bias(n_embd, 1, vb.pp("value_head"))?;
        Ok(Self { proj })
    }

    /// Forward pass: (batch, n_embd) -> (batch, 1)
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.proj.forward(hidden_states)
    }

    /// Compute MSE loss between predicted values and target rewards.
    /// hidden_states: (batch, n_embd), targets: (batch,)
    pub fn mse_loss(&self, hidden_states: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let predicted = self.forward(hidden_states)?.squeeze(1)?;
        let diff = (&predicted - targets)?;
        let sq = (&diff * &diff)?;
        sq.mean_all()
    }
}
