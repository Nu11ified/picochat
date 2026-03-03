use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::model::GPT;
use picochat_data::mixture::{DatasetMixture, MixtureDataset};
use picochat_data::sft::load_sft_data;
use picochat_tokenizer::Tokenizer;
use picochat_optim::LrSchedule;
use crate::checkpoint;
use crate::trainer::Trainer;

pub struct SftConfig {
    pub checkpoint_dir: String,
    pub tokenizer_path: String,
    /// Vec of (jsonl_path, weight)
    pub datasets: Vec<(String, f64)>,
    pub total_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub max_lr: f64,
    pub warmup_steps: usize,
    pub min_lr_ratio: f64,
    pub save_dir: String,
    pub save_every: usize,
}

/// Masked cross-entropy: only positions where mask=1 contribute to loss.
pub fn masked_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
) -> candle_core::Result<Tensor> {
    let (b, t, vocab) = logits.dims3()?;
    let logits_flat = logits.reshape((b * t, vocab))?;
    let targets_flat = targets.flatten_all()?.to_dtype(DType::U32)?;
    let mask_flat = mask.flatten_all()?;

    let log_sm = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;
    let targets_idx = targets_flat.unsqueeze(1)?;
    let nll_per_token = log_sm.gather(&targets_idx, 1)?.squeeze(1)?.neg()?;

    let masked_nll = (&nll_per_token * &mask_flat)?;
    // 1e-8 prevents division by zero when mask is all zeros
    let mask_sum = (mask_flat.sum_all()?.to_dtype(DType::F32)? + 1e-8)?;
    masked_nll.sum_all()?.to_dtype(DType::F32)?.div(&mask_sum)
}

/// Truncate/pad conversations and masks to seq_len, return tensor batch.
fn prepare_batch(
    conversations: &[Vec<u32>],
    masks: &[Vec<u8>],
    seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let batch_size = conversations.len();
    let mut input_vecs: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut target_vecs: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut mask_vecs: Vec<Vec<f32>> = Vec::with_capacity(batch_size);

    for (tokens, mask) in conversations.iter().zip(masks.iter()) {
        let len = (tokens.len() - 1).min(seq_len);
        let mut input = tokens[..len].to_vec();
        let mut target = tokens[1..len + 1].to_vec();
        let mut m: Vec<f32> = mask[..len].iter().map(|&b| b as f32).collect();

        while input.len() < seq_len {
            input.push(0);
            target.push(0);
            m.push(0.0);
        }

        input_vecs.push(input);
        target_vecs.push(target);
        mask_vecs.push(m);
    }

    let input = Tensor::new(input_vecs, device)?;
    let target = Tensor::new(target_vecs, device)?;
    let mask = Tensor::new(mask_vecs, device)?;

    Ok((input, target, mask))
}

/// Run the SFT training loop.
pub fn sft(config: &SftConfig, device: &Device) -> Result<()> {
    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;

    let model_config = checkpoint::load_config(format!("{}/config.json", config.checkpoint_dir))?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(&model_config, vb)?;
    checkpoint::load_varmap(
        &varmap,
        format!("{}/model.safetensors", config.checkpoint_dir),
        device,
    )?;

    println!("SFT: loaded checkpoint from {}", config.checkpoint_dir);
    println!("Parameters: {} ({:.2}M)",
        model.num_parameters(), model.num_parameters() as f64 / 1e6);

    // Encode tokens+mask together so DatasetMixture can sample them.
    // Format: [token_count, ...tokens, ...mask_as_u32]
    let mut mixture_datasets = Vec::new();
    for (path, weight) in &config.datasets {
        let tokenized = load_sft_data(path, &tokenizer)?;
        println!("SFT dataset '{}': {} conversations, weight={}", path, tokenized.len(), weight);

        let combined: Vec<Vec<u32>> = tokenized.iter().map(|tc| {
            let mut v = Vec::with_capacity(1 + tc.tokens.len() + tc.mask.len());
            v.push(tc.tokens.len() as u32);
            v.extend_from_slice(&tc.tokens);
            v.extend(tc.mask.iter().map(|&b| b as u32));
            v
        }).collect();

        mixture_datasets.push(MixtureDataset {
            name: path.clone(),
            weight: *weight,
            items: combined,
        });
    }

    let mut mixture = DatasetMixture::new(mixture_datasets);

    let schedule = LrSchedule::new(
        config.max_lr,
        config.warmup_steps,
        config.total_steps,
        config.min_lr_ratio,
    );
    let mut trainer = Trainer::with_schedule(&varmap, &model_config, schedule);

    println!("SFT training: {} steps, batch={}, seq_len={}",
        config.total_steps, config.batch_size, config.seq_len);

    let start = std::time::Instant::now();

    for step in 0..config.total_steps {
        let mut batch_tokens: Vec<Vec<u32>> = Vec::with_capacity(config.batch_size);
        let mut batch_masks: Vec<Vec<u8>> = Vec::with_capacity(config.batch_size);

        for _ in 0..config.batch_size {
            let combined = mixture.sample();
            let token_count = combined[0] as usize;
            let tokens = combined[1..1 + token_count].to_vec();
            let mask: Vec<u8> = combined[1 + token_count..].iter().map(|&v| v as u8).collect();
            batch_tokens.push(tokens);
            batch_masks.push(mask);
        }

        let (input, target, mask) = prepare_batch(
            &batch_tokens, &batch_masks, config.seq_len, device,
        )?;

        let logits = model.forward(&input, None)?;
        let loss = masked_cross_entropy(&logits, &target, &mask)?;

        // Manual LR scheduling since we use masked loss instead of train_step
        let sched = trainer.schedule_ref().cloned();
        match sched {
            Some(sched) => {
                let base_lr = sched.base_lr();
                let current_lr = sched.get_lr(step);
                let mult = if base_lr > 0.0 { current_lr / base_lr } else { 1.0 };
                trainer.optimizer_mut().backward_step_with_lr(&loss, mult)?;
            }
            None => {
                trainer.optimizer_mut().backward_step(&loss)?;
            }
        }

        if step % 10 == 0 || step == config.total_steps - 1 {
            let loss_val: f32 = loss.to_scalar()?;
            let elapsed = start.elapsed().as_secs_f64();
            let tok_s = ((step + 1) * config.batch_size * config.seq_len) as f64 / elapsed;
            println!("sft step {:>4}/{} | loss: {:.4} | tok/s: {:.0}",
                step, config.total_steps, loss_val, tok_s);
        }

        if config.save_every > 0 && (step + 1) % config.save_every == 0 {
            let ckpt_dir = format!("{}/step-{}", config.save_dir, step + 1);
            std::fs::create_dir_all(&ckpt_dir)?;
            checkpoint::save_varmap(&varmap, format!("{ckpt_dir}/model.safetensors"))?;
            checkpoint::save_config(&model_config, format!("{ckpt_dir}/config.json"))?;
            println!("SFT checkpoint saved to {ckpt_dir}/");
        }
    }

    std::fs::create_dir_all(&config.save_dir)?;
    checkpoint::save_varmap(&varmap, format!("{}/model.safetensors", config.save_dir))?;
    checkpoint::save_config(&model_config, format!("{}/config.json", config.save_dir))?;
    println!("SFT complete. Checkpoint saved to {}/", config.save_dir);

    Ok(())
}
