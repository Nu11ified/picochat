use candle_core::{Result, Tensor};

pub fn rms_norm(x: &Tensor) -> Result<Tensor> {
    let dim = x.dims().len() - 1;
    let eps = 1e-6;
    let mean_sq = x.sqr()?.mean_keepdim(dim)?;
    let rsqrt = (mean_sq + eps)?.sqrt()?.recip()?;
    x.broadcast_mul(&rsqrt)
}
