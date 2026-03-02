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

    /// Train a BPE tokenizer from a text file
    #[arg(long)]
    tokenizer_train: bool,

    /// Encode text with a trained tokenizer
    #[arg(long)]
    tokenizer_encode: bool,

    /// Path to training data (text file or parquet)
    #[arg(long)]
    data: Option<String>,

    /// Vocabulary size for tokenizer training
    #[arg(long, default_value_t = 32768)]
    vocab_size: usize,

    /// Path to save/load tokenizer JSON
    #[arg(long)]
    tokenizer: Option<String>,

    /// Text to encode (for --tokenizer-encode)
    #[arg(long)]
    text: Option<String>,

    /// Path to save trained tokenizer
    #[arg(long)]
    save_tokenizer: Option<String>,

    /// Interactive chat mode
    #[arg(long)]
    chat: bool,

    /// Path to model checkpoint directory (for chat/inference)
    #[arg(long)]
    load: Option<String>,

    /// Maximum tokens to generate per response
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = Device::Cpu;

    if cli.smoke_test {
        run_smoke_test(&cli, &device)?;
    } else if cli.train {
        run_train(&cli, &device)?;
    } else if cli.tokenizer_train {
        run_tokenizer_train(&cli)?;
    } else if cli.tokenizer_encode {
        run_tokenizer_encode(&cli)?;
    } else if cli.chat {
        run_chat(&cli, &device)?;
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

fn run_tokenizer_train(cli: &Cli) -> Result<()> {
    let data_path = cli.data.as_ref().expect("--data is required for tokenizer training");
    let save_path = cli.save_tokenizer.as_ref().expect("--save-tokenizer is required");

    println!("Training BPE tokenizer (vocab_size={})...", cli.vocab_size);
    println!("Reading data from: {data_path}");

    // Read text: if .parquet, use parquet reader; otherwise read as plain text
    let text = if data_path.ends_with(".parquet") {
        picochat_data::parquet::read_all_text(data_path, "text")?
    } else {
        std::fs::read_to_string(data_path)?
    };

    println!("Training text: {} bytes", text.len());
    let tok = picochat_tokenizer::Tokenizer::train(&text, cli.vocab_size)?;
    println!("Learned {} merges", tok.num_merges());

    tok.save(save_path)?;
    println!("Tokenizer saved to: {save_path}");

    // Quick sample encode
    let sample = &text[..text.len().min(200)];
    let ids = tok.encode(sample)?;
    println!("Sample encode: {} chars -> {} tokens (ratio: {:.2}x)",
        sample.len(), ids.len(), sample.len() as f64 / ids.len() as f64);

    Ok(())
}

fn run_tokenizer_encode(cli: &Cli) -> Result<()> {
    let tok_path = cli.tokenizer.as_ref().expect("--tokenizer is required");
    let text = cli.text.as_ref().expect("--text is required");

    let tok = picochat_tokenizer::Tokenizer::load(tok_path)?;
    let ids = tok.encode(text)?;
    println!("Text: {text}");
    println!("Tokens ({} total): {:?}", ids.len(), ids);

    let decoded = tok.decode(&ids);
    println!("Decoded: {decoded}");

    Ok(())
}

fn run_chat(cli: &Cli, device: &Device) -> Result<()> {
    use std::io::{self, Write, BufRead};

    let ckpt_path = cli.load.as_ref().expect("--load is required for chat");
    let tok_path = cli.tokenizer.as_ref().expect("--tokenizer is required for chat");

    let config = checkpoint::load_config(format!("{ckpt_path}/config.json"))?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(&config, vb)?;
    checkpoint::load_varmap(&varmap, format!("{ckpt_path}/model.safetensors"), device)?;

    let tok = picochat_tokenizer::Tokenizer::load(tok_path)?;

    let assistant_end_id = tok.special()
        .token_id(picochat_tokenizer::special::SpecialToken::AssistantEnd);

    println!("picochat chat (depth={}, params={:.2}M)",
        config.n_layer, model.num_parameters() as f64 / 1e6);
    println!("Type a message and press Enter. 'quit' to exit.\n");

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        let prompt = format!(
            "<|bos|><|user_start|>{input}<|user_end|><|assistant_start|>"
        );
        let prompt_tokens = tok.encode(&prompt)?;

        let gen_config = picochat_engine::generate::GenerationConfig {
            max_new_tokens: cli.max_tokens,
            sampling: picochat_engine::sampling::SamplingParams {
                temperature: cli.temperature,
                top_k: 50,
                top_p: 0.95,
            },
            stop_tokens: vec![assistant_end_id],
        };

        let output_tokens = picochat_engine::generate::generate(
            &model,
            &prompt_tokens,
            &gen_config,
            device,
        )?;

        let response = tok.decode(&output_tokens);
        println!("{response}\n");
    }

    Ok(())
}
