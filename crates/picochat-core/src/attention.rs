use candle_core::{Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::config::GPTConfig;
use crate::kv_cache::LayerCache;
use crate::norm::rms_norm;
use crate::rotary::RotaryEmbedding;

pub struct CausalSelfAttention {
    c_q: Linear,
    c_k: Linear,
    c_v: Linear,
    c_proj: Linear,
    ve_gate: Option<Linear>,
    n_head: usize,
    n_kv_head: usize,
    n_embd: usize,
    head_dim: usize,
    #[allow(dead_code)]
    layer_idx: usize,
    #[allow(dead_code)]
    n_layers: usize,
}

impl CausalSelfAttention {
    pub fn new(config: &GPTConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let c_q = linear_no_bias(config.n_embd, config.n_head * head_dim, vb.pp("c_q"))?;
        let c_k = linear_no_bias(config.n_embd, config.n_kv_head * head_dim, vb.pp("c_k"))?;
        let c_v = linear_no_bias(config.n_embd, config.n_kv_head * head_dim, vb.pp("c_v"))?;
        let c_proj = linear_no_bias(config.n_embd, config.n_embd, vb.pp("c_proj"))?;

        let ve_gate = if config.has_value_embedding(layer_idx) {
            let ve_gate_channels = 32;
            Some(linear_no_bias(
                ve_gate_channels,
                config.n_kv_head,
                vb.pp("ve_gate"),
            )?)
        } else {
            None
        };

        Ok(Self {
            c_q,
            c_k,
            c_v,
            c_proj,
            ve_gate,
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            n_embd: config.n_embd,
            head_dim,
            layer_idx,
            n_layers: config.n_layer,
        })
    }

    /// x: (B, T, C), ve: optional (B, T, n_kv_head*head_dim), returns (B, T, C)
    pub fn forward(
        &self,
        x: &Tensor,
        ve: Option<&Tensor>,
        rope: &RotaryEmbedding,
        rope_offset: usize,
        window_size: (usize, usize),
        kv_cache: Option<&mut LayerCache>,
    ) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;

        let q = self
            .c_q
            .forward(x)?
            .reshape((b, t, self.n_head, self.head_dim))?;
        let k = self
            .c_k
            .forward(x)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?;
        let mut v = self
            .c_v
            .forward(x)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?;

        // Value residual with gated blending
        if let (Some(ve_tensor), Some(ve_gate)) = (ve, &self.ve_gate) {
            let ve_reshaped = ve_tensor.reshape((b, t, self.n_kv_head, self.head_dim))?;
            let x_prefix = x.narrow(2, 0, 32)?; // first 32 channels
            let gate = ve_gate.forward(&x_prefix)?;
            // sigmoid: 1 / (1 + exp(-x)), scaled by 2
            let gate = ((gate.neg()?.exp()? + 1.0)?.recip()? * 2.0)?;
            let gate = gate.unsqueeze(3)?;
            v = (v + gate.broadcast_mul(&ve_reshaped)?)?;
        }

        // Apply RoPE
        let q = rope.apply(&q, rope_offset)?;
        let k = rope.apply(&k, rope_offset)?;

        // QK norm
        let q = rms_norm(&q)?;
        let k = rms_norm(&k)?;

        // Transpose to (B, H, T, D)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // KV cache handling: append new K, V and get full cached tensors
        let (k, v) = match kv_cache {
            Some(cache) => cache.update(&k, &v)?,
            None => (k, v),
        };

        // GQA: repeat KV heads
        let t_k_actual = k.dim(2)?;
        let repeat_factor = self.n_head / self.n_kv_head;
        let k = if repeat_factor > 1 {
            k.unsqueeze(2)?
                .expand((b, self.n_kv_head, repeat_factor, t_k_actual, self.head_dim))?
                .reshape((b, self.n_head, t_k_actual, self.head_dim))?
        } else {
            k
        };
        let v = if repeat_factor > 1 {
            v.unsqueeze(2)?
                .expand((b, self.n_kv_head, repeat_factor, t_k_actual, self.head_dim))?
                .reshape((b, self.n_head, t_k_actual, self.head_dim))?
        } else {
            v
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn_weights = apply_causal_mask(&attn_weights, window_size, rope_offset)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let y = attn_weights.matmul(&v)?;

        // Back to (B, T, C)
        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.n_embd))?;
        self.c_proj.forward(&y)
    }
}

fn apply_causal_mask(
    attn_weights: &Tensor,
    window_size: (usize, usize),
    offset: usize,
) -> Result<Tensor> {
    let (_, _, t_q, t_k) = attn_weights.dims4()?;
    let device = attn_weights.device();
    let neg_inf = f32::NEG_INFINITY;
    let mut mask_data = vec![0.0f32; t_q * t_k];
    for i in 0..t_q {
        let abs_i = offset + i;
        for j in 0..t_k {
            if j > abs_i {
                // Future positions: always masked
                mask_data[i * t_k + j] = neg_inf;
            } else if window_size.0 < t_k && (abs_i - j) > window_size.0 {
                // Outside sliding window: masked
                mask_data[i * t_k + j] = neg_inf;
            }
        }
    }
    let mask = Tensor::new(mask_data.as_slice(), device)?.reshape((1, 1, t_q, t_k))?;
    attn_weights.broadcast_add(&mask)
}
