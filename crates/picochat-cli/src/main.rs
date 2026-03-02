use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;

#[derive(Parser)]
#[command(name = "picochat", version, about = "Train and chat with small reasoning LLMs")]
struct Cli {
    /// Model depth (number of transformer layers)
    #[arg(long, default_value_t = 4)]
    depth: usize,

    /// Run a quick smoke test forward pass
    #[arg(long)]
    smoke_test: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.smoke_test {
        println!("picochat smoke test (depth={})", cli.depth);
        let device = Device::Cpu;
        let config = GPTConfig::from_depth(cli.depth);
        println!("Config: n_layer={}, n_embd={}, n_head={}, n_kv_head={}",
            config.n_layer, config.n_embd, config.n_head, config.n_kv_head);
        println!("Vocab: {} (padded: {})", config.vocab_size, config.padded_vocab_size());

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = GPT::new(&config, vb)?;

        let num_params = model.num_parameters();
        println!("Total parameters: {num_params} ({:.2}M)", num_params as f64 / 1e6);

        // Forward pass with dummy tokens
        let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], &device)?;
        println!("Running forward pass with {} tokens...", 8);
        let logits = model.forward(&input, None)?;
        println!("Output logits shape: {:?}", logits.shape());
        println!("Smoke test PASSED!");
    } else {
        println!("picochat v0.1.0 — use --smoke-test to verify setup");
    }

    Ok(())
}
