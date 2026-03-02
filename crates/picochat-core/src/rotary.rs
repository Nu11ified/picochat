use candle_core::{Device, Result, Tensor};

pub struct RotaryEmbedding {
    cos: Tensor, // (1, max_seq_len, 1, head_dim/2)
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, base: f64, device: &Device) -> Result<Self> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::new(t.as_slice(), device)?;
        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?.unsqueeze(0)?.unsqueeze(2)?;
        let sin = freqs.sin()?.unsqueeze(0)?.unsqueeze(2)?;
        Ok(Self { cos, sin })
    }

    pub fn cos(&self) -> &Tensor {
        &self.cos
    }

    pub fn sin(&self) -> &Tensor {
        &self.sin
    }

    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, t, _h, _d) = x.dims4()?;
        let cos = self.cos.narrow(1, offset, t)?;
        let sin = self.sin.narrow(1, offset, t)?;
        let (_b, _t, _h, d) = x.dims4()?;
        let half_d = d / 2;
        let x1 = x.narrow(3, 0, half_d)?;
        let x2 = x.narrow(3, half_d, half_d)?;
        let y1 = x1
            .broadcast_mul(&cos)?
            .broadcast_add(&x2.broadcast_mul(&sin)?)?;
        let neg_sin = sin.neg()?;
        let y2 = x1
            .broadcast_mul(&neg_sin)?
            .broadcast_add(&x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[&y1, &y2], 3)
    }
}
