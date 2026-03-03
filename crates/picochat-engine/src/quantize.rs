use anyhow::Result;
use candle_core::{Device, Tensor};

/// Per-channel absmax INT8 quantization result.
pub struct QuantizedTensor {
    /// INT8 weight data stored as u8 (value = int8 + 128)
    pub data: Vec<u8>,
    /// Per-output-channel scale factors (FP32)
    pub scales: Vec<f32>,
    /// Original shape [out_features, in_features]
    pub shape: [usize; 2],
}

/// Quantize a 2D FP32 weight tensor to INT8 using per-channel absmax scaling.
/// For a (out_features, in_features) matrix, computes one scale per output channel.
pub fn quantize_tensor(tensor: &Tensor, _device: &Device) -> Result<QuantizedTensor> {
    let dims = tensor.dims();
    assert_eq!(dims.len(), 2, "quantize_tensor expects 2D tensors");
    let shape = [dims[0], dims[1]];

    let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    let mut scales = Vec::with_capacity(shape[0]);
    let mut data = Vec::with_capacity(shape[0] * shape[1]);

    for r in 0..shape[0] {
        let row = &flat[r * shape[1]..(r + 1) * shape[1]];
        let absmax = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // Avoid division by zero for all-zero rows
        let scale = if absmax < 1e-10 { 1.0 } else { absmax / 127.0 };
        scales.push(scale);

        for &val in row {
            let q = (val / scale).round().clamp(-128.0, 127.0) as i8;
            data.push((q as i16 + 128) as u8);
        }
    }

    Ok(QuantizedTensor { data, scales, shape })
}

/// Dequantize INT8 back to FP32.
pub fn dequantize_tensor(qt: &QuantizedTensor, device: &Device) -> Result<Tensor> {
    let [rows, cols] = qt.shape;
    let mut fp32 = Vec::with_capacity(rows * cols);

    for r in 0..rows {
        let scale = qt.scales[r];
        for c in 0..cols {
            let q = qt.data[r * cols + c] as i16 - 128;
            fp32.push(q as f32 * scale);
        }
    }

    Tensor::new(fp32, device)?.reshape(&[rows, cols])
}

/// Compute maximum quantization error between original and dequantized tensor.
pub fn quantization_error(original: &Tensor, quantized: &QuantizedTensor, device: &Device) -> Result<f32> {
    let reconstructed = dequantize_tensor(quantized, device)?;
    let diff = (original - &reconstructed)?.abs()?;
    let max_err: f32 = diff.max(0)?.max(0)?.to_scalar()?;
    Ok(max_err)
}
