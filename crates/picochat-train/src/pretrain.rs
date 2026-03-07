use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::init::initialize_weights;
use picochat_core::model::GPT;
use picochat_data::dataloader::PackingDataLoader;
use picochat_data::parquet::ParquetTextReader;
use picochat_tokenizer::Tokenizer;
use crate::checkpoint;
use crate::metrics::TrainingMetrics;
use crate::trainer::Trainer;
use picochat_optim::LrSchedule;

pub struct PretrainConfig {
    pub data_dir: String,
    pub val_data: Option<String>,
    pub tokenizer_path: String,
    pub total_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub max_lr: f64,
    pub warmup_steps: usize,
    pub min_lr_ratio: f64,
    pub eval_every: usize,
    pub save_every: usize,
    pub save_dir: String,
    pub depth: usize,
}

impl PretrainConfig {
    pub fn tokens_per_step(&self) -> usize {
        self.batch_size * self.seq_len
    }
}

fn collect_parquet_files(dir: &str) -> Result<Vec<String>> {
    let mut paths: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "parquet") {
                Some(path.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Fill the packing loader from a parquet reader until we have at least
/// `min_ready` sequences. Returns (bytes_consumed, reader_exhausted).
fn fill_loader(
    reader: &mut ParquetTextReader,
    tokenizer: &Tokenizer,
    loader: &mut PackingDataLoader,
    min_ready: usize,
) -> Result<(usize, bool)> {
    let mut total_bytes = 0usize;
    let mut exhausted = false;

    while loader.ready_count() < min_ready {
        match reader.next_text()? {
            Some(text) => {
                total_bytes += text.len();
                let tokens = tokenizer.encode(&text)?;
                loader.add_document(&tokens);
            }
            None => {
                exhausted = true;
                break;
            }
        }
    }

    Ok((total_bytes, exhausted))
}

/// Run the pretraining loop.
pub fn pretrain(config: &PretrainConfig, device: &Device) -> Result<()> {
    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;
    let mut model_config = GPTConfig::from_depth(config.depth);
    // Use the tokenizer's actual vocab size instead of the hardcoded default
    model_config.vocab_size = tokenizer.vocab_size();

    println!("Pretrain: depth={}, seq_len={}, batch={}, steps={}",
        config.depth, config.seq_len, config.batch_size, config.total_steps);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(&model_config, vb)?;
    initialize_weights(&varmap, &model_config)?;

    println!("Parameters: {} ({:.2}M)",
        model.num_parameters(), model.num_parameters() as f64 / 1e6);

    // min_lr_ratio controls the warmdown fraction of total steps
    let schedule = LrSchedule::new(
        config.max_lr,
        config.warmup_steps,
        config.total_steps,
        config.min_lr_ratio,
    );
    let mut trainer = Trainer::with_schedule(&varmap, &model_config, schedule);

    let train_files = collect_parquet_files(&config.data_dir)?;
    if train_files.is_empty() {
        anyhow::bail!("No parquet files found in {}", config.data_dir);
    }
    println!("Training data: {} parquet files", train_files.len());

    let bos_id = tokenizer.bos_id();
    let mut loader = PackingDataLoader::new(config.batch_size, config.seq_len, bos_id);

    let mut total_text_bytes: usize = 0;
    let mut total_tokens: usize = 0;
    let mut file_idx = 0;
    let mut reader = ParquetTextReader::open_fineweb(&train_files[file_idx])?;

    let start = std::time::Instant::now();

    for step in 0..config.total_steps {
        // Ensure enough packed sequences for a batch
        loop {
            let (bytes, exhausted) = fill_loader(
                &mut reader, &tokenizer, &mut loader, config.batch_size,
            )?;
            total_text_bytes += bytes;

            if loader.ready_count() >= config.batch_size {
                break;
            }

            if exhausted {
                file_idx += 1;
                if file_idx >= train_files.len() {
                    file_idx = 0;
                    println!("Epoch boundary: restarting from first file");
                }
                reader = ParquetTextReader::open_fineweb(&train_files[file_idx])?;
            }
        }

        let (input_vecs, target_vecs) = loader.next_batch().unwrap();
        let input = Tensor::new(input_vecs, device)?;
        let target = Tensor::new(target_vecs, device)?;

        let loss = trainer.train_step(&model, &input, &target)?;
        let loss_val: f32 = loss.to_scalar()?;

        total_tokens += config.tokens_per_step();

        if step % 10 == 0 || step == config.total_steps - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let avg_bpt = if total_tokens > 0 {
                total_text_bytes as f64 / total_tokens as f64
            } else {
                4.0
            };
            let bpb = TrainingMetrics::compute_bpb(loss_val as f64, avg_bpt);
            let tok_s = TrainingMetrics::compute_throughput(total_tokens, elapsed);
            println!(
                "step {:>5}/{} | loss: {:.4} | bpb: {:.4} | tok/s: {:.0}",
                step, config.total_steps, loss_val, bpb, tok_s
            );
        }

        if config.save_every > 0 && (step + 1) % config.save_every == 0 {
            let ckpt_dir = format!("{}/step-{}", config.save_dir, step + 1);
            std::fs::create_dir_all(&ckpt_dir)?;
            checkpoint::save_varmap(&varmap, format!("{ckpt_dir}/model.safetensors"))?;
            checkpoint::save_config(&model_config, format!("{ckpt_dir}/config.json"))?;
            println!("Checkpoint saved to {ckpt_dir}/");
        }
    }

    std::fs::create_dir_all(&config.save_dir)?;
    checkpoint::save_varmap(&varmap, format!("{}/model.safetensors", config.save_dir))?;
    checkpoint::save_config(&model_config, format!("{}/config.json", config.save_dir))?;
    println!("Final checkpoint saved to {}/", config.save_dir);

    let elapsed = start.elapsed().as_secs_f64();
    println!("Pretraining complete in {:.1}s ({} steps, {} tokens)",
        elapsed, config.total_steps, total_tokens);

    Ok(())
}
