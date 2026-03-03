use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use rand::seq::SliceRandom;

use picochat_core::model::GPT;
use picochat_data::arc::{load_arc_jsonl, format_arc_prompt, ArcQuestion};
use picochat_data::tool_data::{load_tool_scenarios, format_tool_prompt};
use picochat_engine::generate_with_logprobs::{generate_with_logprobs, LogprobGenerationConfig};
use picochat_engine::sampling::SamplingParams;
use picochat_eval::gsm8k::{load_gsm8k_jsonl, format_gsm_prompt, GsmQuestion};
use picochat_optim::LrSchedule;
use picochat_tokenizer::Tokenizer;

use crate::checkpoint;
use crate::rewards::{composite_reward, RewardWeights, TaskType};
use crate::trainer::Trainer;

pub struct GrpoConfig {
    pub checkpoint_dir: String,
    pub tokenizer_path: String,
    pub gsm8k_path: Option<String>,
    pub arc_path: Option<String>,
    pub tool_data_path: Option<String>,
    pub group_size: usize,
    pub total_steps: usize,
    pub max_gen_tokens: usize,
    pub clip_eps: f64,
    pub kl_beta: f64,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub save_dir: String,
    pub save_every: usize,
    pub target_len: usize,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: String::new(),
            tokenizer_path: String::new(),
            gsm8k_path: None,
            arc_path: None,
            tool_data_path: None,
            group_size: 16,
            total_steps: 1000,
            max_gen_tokens: 512,
            clip_eps: 0.2,
            kl_beta: 0.04,
            learning_rate: 1e-6,
            warmup_steps: 50,
            save_dir: String::new(),
            save_every: 0,
            target_len: 256,
        }
    }
}

/// Normalize rewards within a group to zero mean and unit variance.
/// When all rewards are identical, returns zeros.
pub fn normalize_advantages(rewards: &[f64]) -> Vec<f64> {
    let n = rewards.len() as f64;
    let mean: f64 = rewards.iter().sum::<f64>() / n;
    let var: f64 = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();

    if std < 1e-8 {
        return vec![0.0; rewards.len()];
    }

    rewards.iter().map(|r| (r - mean) / std).collect()
}

/// PPO-style clipped surrogate objective.
/// Returns min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage).
pub fn compute_clipped_objective(ratio: f64, advantage: f64, clip_eps: f64) -> f64 {
    let clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
    let obj1 = ratio * advantage;
    let obj2 = clipped_ratio * advantage;
    obj1.min(obj2)
}

/// Approximate KL divergence from (policy_logprob, ref_logprob) pairs.
/// KL(policy || ref) ≈ mean(exp(ref - policy) - (ref - policy) - 1)
pub fn compute_kl_penalty(logprob_pairs: &[(f32, f32)]) -> f64 {
    if logprob_pairs.is_empty() {
        return 0.0;
    }
    let sum: f64 = logprob_pairs.iter().map(|&(policy_lp, ref_lp)| {
        let diff = (ref_lp - policy_lp) as f64;
        diff.exp() - diff - 1.0
    }).sum();
    sum / logprob_pairs.len() as f64
}

/// A training prompt with its ground truth and task metadata.
struct TrainingPrompt {
    prompt_text: String,
    ground_truth: String,
    task_type: TaskType,
    requires_tool: bool,
}

/// Build the pool of training prompts from all data sources.
fn build_prompt_pool(
    config: &GrpoConfig,
    gsm_exemplars: &[GsmQuestion],
    arc_exemplars: &[ArcQuestion],
) -> Result<Vec<TrainingPrompt>> {
    let mut prompts = Vec::new();

    if let Some(path) = &config.gsm8k_path {
        let questions = load_gsm8k_jsonl(path)?;
        println!("GRPO: loaded {} GSM8K questions", questions.len());
        for q in &questions {
            let prompt_text = format_gsm_prompt(gsm_exemplars, q);
            let gt = picochat_eval::gsm8k::extract_answer(&q.answer)
                .unwrap_or_default();
            prompts.push(TrainingPrompt {
                prompt_text,
                ground_truth: gt,
                task_type: TaskType::Math,
                requires_tool: false,
            });
        }
    }

    if let Some(path) = &config.arc_path {
        let questions = load_arc_jsonl(path)?;
        println!("GRPO: loaded {} ARC questions", questions.len());
        for q in &questions {
            let prompt_text = format_arc_prompt(arc_exemplars, q);
            prompts.push(TrainingPrompt {
                prompt_text,
                ground_truth: q.answer_key.clone(),
                task_type: TaskType::MultipleChoice,
                requires_tool: false,
            });
        }
    }

    if let Some(path) = &config.tool_data_path {
        let scenarios = load_tool_scenarios(path)?;
        println!("GRPO: loaded {} tool-use scenarios", scenarios.len());
        for s in &scenarios {
            let prompt_text = format_tool_prompt(&s);
            prompts.push(TrainingPrompt {
                prompt_text,
                ground_truth: s.expected_answer.clone(),
                task_type: TaskType::ToolUse,
                requires_tool: s.requires_tool,
            });
        }
    }

    Ok(prompts)
}

/// Compute per-token log-probabilities for a fixed token sequence using a teacher-forcing
/// forward pass. Returns a Vec<f32> with one logprob per generated token.
fn score_sequence(
    model: &GPT,
    prompt_tokens: &[u32],
    generated_tokens: &[u32],
    device: &Device,
) -> Result<Vec<f32>> {
    let mut full = Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
    full.extend_from_slice(prompt_tokens);
    full.extend_from_slice(generated_tokens);

    // Input is everything except the last token (we predict each next token)
    let input = Tensor::new(&full[..full.len() - 1], device)?.unsqueeze(0)?;
    let logits = model.forward(&input, None)?; // (1, T-1, vocab)
    let logits = logits.squeeze(0)?; // (T-1, vocab)

    let prompt_len = prompt_tokens.len();
    let mut logprobs = Vec::with_capacity(generated_tokens.len());

    for (i, &token) in generated_tokens.iter().enumerate() {
        // logits[prompt_len - 1 + i] predicts generated_tokens[i]
        let pos = prompt_len - 1 + i;
        let logit_row: Vec<f32> = logits.get(pos)?.to_vec1()?;
        let lp = compute_log_softmax(&logit_row);
        logprobs.push(lp[token as usize]);
    }

    Ok(logprobs)
}

/// Compute per-token log-probabilities as a differentiable Tensor for the policy gradient.
/// Returns a (gen_len,) tensor of log-probs at the generated token positions.
fn score_sequence_tensor(
    model: &GPT,
    prompt_tokens: &[u32],
    generated_tokens: &[u32],
    device: &Device,
) -> Result<Tensor> {
    let mut full = Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
    full.extend_from_slice(prompt_tokens);
    full.extend_from_slice(generated_tokens);

    let input = Tensor::new(&full[..full.len() - 1], device)?.unsqueeze(0)?;
    let logits = model.forward(&input, None)?; // (1, T-1, vocab)
    let logits = logits.squeeze(0)?; // (T-1, vocab)

    // Extract the logits at generated token positions
    let prompt_len = prompt_tokens.len();
    let gen_len = generated_tokens.len();
    let gen_logits = logits.narrow(0, prompt_len - 1, gen_len)?; // (gen_len, vocab)

    // log_softmax
    let log_probs = candle_nn::ops::log_softmax(&gen_logits, D::Minus1)?; // (gen_len, vocab)

    // Gather at the generated token indices
    let indices = Tensor::new(generated_tokens, device)?
        .to_dtype(DType::U32)?
        .unsqueeze(1)?; // (gen_len, 1)
    let token_lps = log_probs.gather(&indices, 1)?.squeeze(1)?; // (gen_len,)

    Ok(token_lps)
}

fn compute_log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_sum = sum_exp.ln() + max;
    logits.iter().map(|&l| l - log_sum).collect()
}

/// Run the GRPO training loop.
pub fn grpo(config: &GrpoConfig, device: &Device) -> Result<()> {
    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;
    let model_config = checkpoint::load_config(format!("{}/config.json", config.checkpoint_dir))?;

    // Policy model (updated by gradient descent)
    let policy_varmap = VarMap::new();
    let policy_vb = VarBuilder::from_varmap(&policy_varmap, DType::F32, device);
    let policy_model = GPT::new(&model_config, policy_vb)?;
    checkpoint::load_varmap(
        &policy_varmap,
        format!("{}/model.safetensors", config.checkpoint_dir),
        device,
    )?;

    // Reference model (frozen copy for KL penalty)
    let ref_varmap = VarMap::new();
    let ref_vb = VarBuilder::from_varmap(&ref_varmap, DType::F32, device);
    let ref_model = GPT::new(&model_config, ref_vb)?;
    checkpoint::load_varmap(
        &ref_varmap,
        format!("{}/model.safetensors", config.checkpoint_dir),
        device,
    )?;

    println!("GRPO: loaded policy + reference models from {}", config.checkpoint_dir);
    println!("Parameters: {} ({:.2}M)",
        policy_model.num_parameters(), policy_model.num_parameters() as f64 / 1e6);

    // Build GSM8K exemplars (first 5 questions as few-shot examples)
    let gsm_exemplars: Vec<GsmQuestion> = if config.gsm8k_path.is_some() {
        vec![
            GsmQuestion {
                question: "There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?".to_string(),
                answer: "There are 21 - 15 = 6 trees planted today. #### 6".to_string(),
            },
        ]
    } else {
        vec![]
    };

    let arc_exemplars: Vec<ArcQuestion> = if config.arc_path.is_some() {
        vec![
            ArcQuestion {
                question: "Which is a compound?".to_string(),
                choices: vec!["O2".into(), "NaCl".into(), "Fe".into(), "Ar".into()],
                answer_key: "B".to_string(),
            },
        ]
    } else {
        vec![]
    };

    let prompts = build_prompt_pool(config, &gsm_exemplars, &arc_exemplars)?;
    if prompts.is_empty() {
        anyhow::bail!("GRPO: no training data loaded. Provide at least one of --gsm8k-data, --arc-data, --tool-data");
    }
    println!("GRPO: {} total training prompts", prompts.len());

    // Optimizer and LR schedule
    let schedule = LrSchedule::new(
        config.learning_rate,
        config.warmup_steps,
        config.total_steps,
        0.1, // warmdown fraction
    );
    let mut trainer = Trainer::with_schedule(&policy_varmap, &model_config, schedule);

    let reward_weights = RewardWeights::default();
    let mut rng = rand::thread_rng();

    let gen_config = LogprobGenerationConfig {
        max_new_tokens: config.max_gen_tokens,
        sampling: SamplingParams {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
        },
        stop_tokens: vec![],
        tool_call_start_id: tokenizer.encode("<tool_call_start>").ok()
            .and_then(|t| t.first().copied()),
        tool_call_end_id: tokenizer.encode("<tool_call_end>").ok()
            .and_then(|t| t.first().copied()),
        tool_result_start_id: tokenizer.encode("<tool_result_start>").ok()
            .and_then(|t| t.first().copied()),
        tool_result_end_id: tokenizer.encode("<tool_result_end>").ok()
            .and_then(|t| t.first().copied()),
        max_tool_calls: 3,
    };

    let start = std::time::Instant::now();
    let mut total_reward = 0.0f64;
    let mut total_accuracy = 0.0f64;

    println!("GRPO training: {} steps, group_size={}, max_gen={}",
        config.total_steps, config.group_size, config.max_gen_tokens);

    for step in 0..config.total_steps {
        // 1. Sample a random prompt
        let prompt = prompts.choose(&mut rng)
            .expect("prompts pool is non-empty");
        let prompt_tokens = tokenizer.encode(&prompt.prompt_text)?;

        // 2. Generate G completions
        let mut completions: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(config.group_size);
        for _ in 0..config.group_size {
            let (ids, lps) = generate_with_logprobs(
                &policy_model, &prompt_tokens, &gen_config, device, Some(&tokenizer),
            )?;
            completions.push((ids, lps));
        }

        // 3. Compute rewards for each completion
        let mut rewards = Vec::with_capacity(config.group_size);
        for (ids, _) in &completions {
            let text = tokenizer.decode(ids);
            let reward = composite_reward(
                &text,
                &prompt.ground_truth,
                prompt.task_type,
                prompt.requires_tool,
                ids.len(),
                config.target_len,
                &reward_weights,
            );
            rewards.push(reward);
        }

        // 4. Normalize advantages within the group
        let advantages = normalize_advantages(&rewards);

        // Best-of-N approximation: apply gradient only to the highest-advantage completion.
        // The full group is still used for advantage normalization, but updating only one
        // completion per step keeps memory bounded on CPU.
        let best_idx = advantages.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_ids = &completions[best_idx].0;
        let best_advantage = advantages[best_idx];
        let best_old_lps = &completions[best_idx].1;

        // Track accuracy
        let best_text = tokenizer.decode(best_ids);
        let acc = crate::rewards::accuracy_reward(
            &best_text, &prompt.ground_truth, prompt.task_type,
        );
        total_accuracy += acc;
        total_reward += rewards[best_idx];

        // Skip gradient update if advantage is negligible (all completions equally good/bad)
        if best_advantage.abs() < 1e-6 {
            if step % 10 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                println!("grpo step {:>4}/{} | reward: {:.3} | acc: {:.1}% | skipped (uniform) | {:.1}s",
                    step, config.total_steps,
                    total_reward / (step + 1) as f64,
                    total_accuracy / (step + 1) as f64 * 100.0,
                    elapsed);
            }
            continue;
        }

        // 6. Compute reference log-probs for the best completion (frozen model, no grad)
        let ref_lps = score_sequence(&ref_model, &prompt_tokens, best_ids, device)?;

        // 7. Compute policy log-probs as a differentiable tensor
        let policy_lp_tensor = score_sequence_tensor(
            &policy_model, &prompt_tokens, best_ids, device,
        )?;

        // 8. Compute per-token old policy log-probs as a constant tensor
        let old_lps_slice: Vec<f32> = if best_old_lps.len() == best_ids.len() {
            best_old_lps.clone()
        } else {
            // Generation logprobs may differ in length from scored sequence (e.g., tool calls);
            // fall back to scoring through policy model
            score_sequence(&policy_model, &prompt_tokens, best_ids, device)?
        };
        let old_lp_tensor = Tensor::new(&old_lps_slice[..], device)?
            .detach();

        // 9. Compute ratio = exp(new_lp - old_lp) for each token
        let log_ratio = (&policy_lp_tensor - &old_lp_tensor)?;
        let ratio = log_ratio.exp()?;

        // 10. Clipped surrogate objective (as tensor)
        let advantage_t = Tensor::new(&[best_advantage as f32], device)?
            .broadcast_as(ratio.shape())?;
        let obj1 = (&ratio * &advantage_t)?;
        let clipped_ratio = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps)?;
        let obj2 = (&clipped_ratio * &advantage_t)?;
        let clipped_obj = obj1.minimum(&obj2)?;
        let policy_loss = clipped_obj.mean_all()?.neg()?;

        // 11. KL penalty
        let kl_pairs: Vec<(f32, f32)> = old_lps_slice.iter()
            .zip(ref_lps.iter())
            .map(|(&p, &r)| (p, r))
            .collect();
        let kl = compute_kl_penalty(&kl_pairs);
        let kl_tensor = Tensor::new(&[kl as f32], device)?;
        let kl_loss = (&kl_tensor * config.kl_beta)?;

        // 12. Total loss = policy_loss + kl_penalty
        let total_loss = (&policy_loss + &kl_loss.squeeze(0)?)?;

        // 13. Gradient step
        let sched = trainer.schedule_ref().cloned();
        match sched {
            Some(sched) => {
                let base_lr = sched.base_lr();
                let current_lr = sched.get_lr(step);
                let mult = if base_lr > 0.0 { current_lr / base_lr } else { 1.0 };
                trainer.optimizer_mut().backward_step_with_lr(&total_loss, mult)?;
            }
            None => {
                trainer.optimizer_mut().backward_step(&total_loss)?;
            }
        }

        // 14. Logging
        if step % 10 == 0 || step == config.total_steps - 1 {
            let loss_val: f32 = total_loss.to_scalar()?;
            let elapsed = start.elapsed().as_secs_f64();
            let avg_reward = total_reward / (step + 1) as f64;
            let avg_acc = total_accuracy / (step + 1) as f64 * 100.0;
            println!("grpo step {:>4}/{} | loss: {:.4} | reward: {:.3} | acc: {:.1}% | kl: {:.4} | {:.1}s",
                step, config.total_steps, loss_val, avg_reward, avg_acc, kl, elapsed);
        }

        // 15. Checkpoint
        if config.save_every > 0 && (step + 1) % config.save_every == 0 {
            let ckpt_dir = format!("{}/step-{}", config.save_dir, step + 1);
            std::fs::create_dir_all(&ckpt_dir)?;
            checkpoint::save_varmap(&policy_varmap, format!("{ckpt_dir}/model.safetensors"))?;
            checkpoint::save_config(&model_config, format!("{ckpt_dir}/config.json"))?;
            println!("GRPO checkpoint saved to {ckpt_dir}/");
        }
    }

    // Final save
    std::fs::create_dir_all(&config.save_dir)?;
    checkpoint::save_varmap(&policy_varmap, format!("{}/model.safetensors", config.save_dir))?;
    checkpoint::save_config(&model_config, format!("{}/config.json", config.save_dir))?;

    let avg_reward = total_reward / config.total_steps as f64;
    let avg_acc = total_accuracy / config.total_steps as f64 * 100.0;
    println!("GRPO complete. avg_reward={:.3}, avg_acc={:.1}%, saved to {}/",
        avg_reward, avg_acc, config.save_dir);

    Ok(())
}
