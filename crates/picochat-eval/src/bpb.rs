use anyhow::Result;
use candle_core::{Device, Tensor};
use picochat_core::model::GPT;
use picochat_data::dataloader::PackingDataLoader;
use picochat_data::parquet::ParquetTextReader;
use picochat_tokenizer::Tokenizer;

pub struct BpbResult {
    pub bpb: f64,
    pub num_tokens: usize,
    pub num_bytes: usize,
    pub avg_loss: f64,
}

/// Evaluate BPB on a validation parquet file.
///
/// Tokenizes and packs all validation data, runs forward-only passes to
/// accumulate cross-entropy, then computes `bpb = total_nll / (total_bytes * ln(2))`.
pub fn evaluate_bpb(
    model: &GPT,
    val_path: &str,
    tokenizer: &Tokenizer,
    batch_size: usize,
    seq_len: usize,
    device: &Device,
) -> Result<BpbResult> {
    let bos_id = tokenizer.bos_id();
    let mut loader = PackingDataLoader::new(batch_size, seq_len, bos_id);

    let mut reader = ParquetTextReader::open_fineweb(val_path)?;
    let mut total_bytes: usize = 0;

    loop {
        match reader.next_text()? {
            Some(text) => {
                total_bytes += text.len();
                let tokens = tokenizer.encode(&text)?;
                loader.add_document(&tokens);
            }
            None => break,
        }
    }
    loader.flush();

    let mut total_loss = 0.0f64;
    let mut total_tokens: usize = 0;

    while let Some((input_vecs, target_vecs)) = loader.next_batch() {
        let input = Tensor::new(input_vecs, device)?;
        let target = Tensor::new(target_vecs, device)?;

        let loss = model.forward(&input, Some(&target))?;
        let loss_val: f32 = loss.to_scalar()?;

        let batch_tokens = batch_size * seq_len;
        total_loss += loss_val as f64 * batch_tokens as f64;
        total_tokens += batch_tokens;
    }

    if total_tokens == 0 {
        anyhow::bail!("No validation tokens found in {val_path}");
    }

    let avg_loss = total_loss / total_tokens as f64;
    // bpb = total_nll_nats / (total_bytes * ln(2))
    let bpb = total_loss / (total_bytes as f64 * 2.0f64.ln());

    Ok(BpbResult {
        bpb,
        num_tokens: total_tokens,
        num_bytes: total_bytes,
        avg_loss,
    })
}
