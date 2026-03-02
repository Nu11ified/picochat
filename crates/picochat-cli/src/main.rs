use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use picochat_core::config::GPTConfig;
use picochat_core::init::initialize_weights;
use picochat_core::model::GPT;
use picochat_data::dataloader::{DataLoader, TokenDataset};
use picochat_optim::LrSchedule;
use picochat_train::checkpoint;
use picochat_train::trainer::Trainer;

#[derive(Parser)]
#[command(name = "picochat", version, about = "Train and chat with small reasoning LLMs")]
struct Cli {
    /// Model depth (number of transformer layers)
    #[arg(long, default_value_t = 4)]
    depth: usize,

    /// Run a quick smoke test forward pass
    #[arg(long)]
    smoke_test: bool,

    /// Train on synthetic data
    #[arg(long)]
    train: bool,

    /// Number of training steps
    #[arg(long, default_value_t = 50)]
    steps: usize,

    /// Batch size
    #[arg(long, default_value_t = 2)]
    batch_size: usize,

    /// Sequence length
    #[arg(long, default_value_t = 64)]
    seq_len: usize,

    /// Save checkpoint to this directory
    #[arg(long)]
    save: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = Device::Cpu;

    if cli.smoke_test {
        run_smoke_test(&cli, &device)?;
    } else if cli.train {
        run_train(&cli, &device)?;
    } else {
        println!("picochat v0.1.0");
        println!("  --smoke-test  Run forward pass verification");
        println!("  --train       Train on synthetic data");
    }
    Ok(())
}

fn run_smoke_test(cli: &Cli, device: &Device) -> Result<()> {
    println!("picochat smoke test (depth={})", cli.depth);
    let config = GPTConfig::from_depth(cli.depth);
    println!(
        "Config: n_layer={}, n_embd={}, n_head={}, n_kv_head={}",
        config.n_layer, config.n_embd, config.n_head, config.n_kv_head
    );
    println!(
        "Vocab: {} (padded: {})",
        config.vocab_size,
        config.padded_vocab_size()
    );

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(&config, vb)?;

    let num_params = model.num_parameters();
    println!(
        "Total parameters: {num_params} ({:.2}M)",
        num_params as f64 / 1e6
    );

    // Forward pass with dummy tokens
    let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], device)?;
    println!("Running forward pass with {} tokens...", 8);
    let logits = model.forward(&input, None)?;
    println!("Output logits shape: {:?}", logits.shape());
    println!("Smoke test PASSED!");
    Ok(())
}

fn run_train(cli: &Cli, device: &Device) -> Result<()> {
    let config = GPTConfig::from_depth(cli.depth);
    println!("picochat training (depth={})", cli.depth);
    println!(
        "Config: n_layer={}, n_embd={}, n_head={}, n_kv_head={}",
        config.n_layer, config.n_embd, config.n_head, config.n_kv_head
    );

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(&config, vb)?;
    initialize_weights(&varmap, &config)?;
    println!(
        "Parameters: {} ({:.2}M)",
        model.num_parameters(),
        model.num_parameters() as f64 / 1e6
    );

    // Generate synthetic data (sequential tokens for overfitting test)
    let num_tokens = cli.batch_size * cli.seq_len * cli.steps * 2;
    let tokens: Vec<u32> = (0..num_tokens)
        .map(|i| (i as u32) % config.vocab_size as u32)
        .collect();
    let dataset = TokenDataset::new(tokens);
    let mut dataloader = DataLoader::new(dataset, cli.batch_size, cli.seq_len);

    let schedule = LrSchedule::new(0.001, cli.steps / 10, cli.steps, 0.2);
    let mut trainer = Trainer::with_schedule(&varmap, &config, schedule);

    println!(
        "Training for {} steps (batch={}, seq_len={})...",
        cli.steps, cli.batch_size, cli.seq_len
    );
    let start = std::time::Instant::now();

    for step in 0..cli.steps {
        let (input_vecs, target_vecs) = dataloader.next_batch()?;
        let input = Tensor::new(input_vecs, device)?;
        let target = Tensor::new(target_vecs, device)?;
        let loss = trainer.train_step(&model, &input, &target)?;

        if step % 10 == 0 || step == cli.steps - 1 {
            let loss_val: f32 = loss.to_scalar()?;
            let elapsed = start.elapsed().as_secs_f64();
            let tok_s = ((step + 1) * cli.batch_size * cli.seq_len) as f64 / elapsed;
            println!(
                "step {:>4}/{} | loss: {:.4} | tok/s: {:.0}",
                step, cli.steps, loss_val, tok_s
            );
        }
    }

    println!("Training complete in {:.1}s", start.elapsed().as_secs_f64());

    if let Some(ref path) = cli.save {
        std::fs::create_dir_all(path)?;
        checkpoint::save_varmap(&varmap, format!("{path}/model.safetensors"))?;
        checkpoint::save_config(&config, format!("{path}/config.json"))?;
        println!("Checkpoint saved to {path}/");
    }

    Ok(())
}
