use candle_core::{DType, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};

use crate::attention::CausalSelfAttention;
use crate::config::GPTConfig;
use crate::mlp::MLP;
use crate::norm::rms_norm;
use crate::rotary::RotaryEmbedding;

/// A single transformer block: attention + MLP with pre-norm residual connections.
struct Block {
    attn: CausalSelfAttention,
    mlp: MLP,
}

impl Block {
    fn new(config: &GPTConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let attn = CausalSelfAttention::new(config, layer_idx, vb.pp("attn"))?;
        let mlp = MLP::new(config.n_embd, vb.pp("mlp"))?;
        Ok(Self { attn, mlp })
    }

    fn forward(
        &self,
        x: &Tensor,
        ve: Option<&Tensor>,
        rope: &RotaryEmbedding,
        rope_offset: usize,
        window_size: (usize, usize),
    ) -> Result<Tensor> {
        // Pre-norm residual: x = x + attn(rms_norm(x))
        let x = (x + self.attn.forward(&rms_norm(x)?, ve, rope, rope_offset, window_size, None)?)?;
        // Pre-norm residual: x = x + mlp(rms_norm(x))
        let x = (&x + self.mlp.forward(&rms_norm(&x)?)?)?;
        Ok(x)
    }
}

/// Full GPT model assembling all components.
pub struct GPT {
    wte: Embedding,
    blocks: Vec<Block>,
    lm_head: Linear,
    resid_lambdas: Tensor,
    x0_lambdas: Tensor,
    value_embeds: Vec<Option<Embedding>>,
    rope: RotaryEmbedding,
    window_sizes: Vec<(usize, usize)>,
    config: GPTConfig,
}

impl GPT {
    pub fn new(config: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        let padded_vocab = config.padded_vocab_size();
        let head_dim = config.head_dim();
        let kv_dim = config.n_kv_head * head_dim;

        // Token embedding: padded_vocab_size x n_embd
        let wte = embedding(padded_vocab, config.n_embd, vb.pp("wte"))?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let block = Block::new(config, i, vb.pp(format!("h.{i}")))?;
            blocks.push(block);
        }

        // LM head: n_embd -> padded_vocab_size (untied from wte)
        let lm_head = linear_no_bias(config.n_embd, padded_vocab, vb.pp("lm_head"))?;

        // Per-layer residual scaling lambdas
        let resid_lambdas = vb.get((config.n_layer,), "resid_lambdas")?;
        let x0_lambdas = vb.get((config.n_layer,), "x0_lambdas")?;

        // Value embeddings for alternating layers
        let mut value_embeds = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            if config.has_value_embedding(i) {
                let ve = embedding(padded_vocab, kv_dim, vb.pp(format!("ve.{i}")))?;
                value_embeds.push(Some(ve));
            } else {
                value_embeds.push(None);
            }
        }

        // Rotary position embeddings
        let device = vb.device();
        let rope = RotaryEmbedding::new(head_dim, config.sequence_len * 10, 10000.0, device)?;

        // Window sizes from config
        let window_sizes = config.compute_window_sizes();

        Ok(Self {
            wte,
            blocks,
            lm_head,
            resid_lambdas,
            x0_lambdas,
            value_embeds,
            rope,
            window_sizes,
            config: config.clone(),
        })
    }

    /// Forward pass. Returns logits (B, T, vocab_size) if targets is None,
    /// or scalar cross-entropy loss if targets are provided.
    pub fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> Result<Tensor> {
        // Embed tokens: (B, T) -> (B, T, n_embd)
        let mut x = self.wte.forward(idx)?;

        // Initial RMS norm on embeddings
        x = rms_norm(&x)?;

        // Save x0 for x0 residual mixing
        let x0 = x.clone();

        // Run through transformer blocks
        for i in 0..self.blocks.len() {
            // Apply per-layer residual scaling: x = resid_lambda * x + x0_lambda * x0
            let resid_lambda = self.resid_lambdas.get(i)?.unsqueeze(0)?.unsqueeze(0)?;
            let x0_lambda = self.x0_lambdas.get(i)?.unsqueeze(0)?.unsqueeze(0)?;
            x = (resid_lambda.broadcast_mul(&x)? + x0_lambda.broadcast_mul(&x0)?)?;

            // Get value embedding for this layer if applicable
            let ve = match &self.value_embeds[i] {
                Some(ve_embed) => Some(ve_embed.forward(idx)?),
                None => None,
            };

            // Run block forward
            x = self.blocks[i].forward(
                &x,
                ve.as_ref(),
                &self.rope,
                0, // rope_offset
                self.window_sizes[i],
            )?;
        }

        // Final RMS norm
        x = rms_norm(&x)?;

        // LM head -> logits: (B, T, padded_vocab_size)
        let logits = self.lm_head.forward(&x)?;

        // Narrow to actual vocab_size (remove padding)
        let logits = logits.narrow(D::Minus1, 0, self.config.vocab_size)?;

        // Cast to f32
        let logits = logits.to_dtype(DType::F32)?;

        // Logit softcap: 15.0 * tanh(logits / 15.0)
        let cap = 15.0f64;
        let logits = ((logits / cap)?.tanh()? * cap)?;

        if let Some(targets) = targets {
            // Compute cross-entropy loss
            let (b, t, vocab) = logits.dims3()?;
            let logits_flat = logits.reshape((b * t, vocab))?;
            let targets_flat = targets.flatten_all()?.to_dtype(DType::U32)?;
            let log_sm = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;
            let loss = candle_nn::loss::nll(&log_sm, &targets_flat)?;
            Ok(loss)
        } else {
            Ok(logits)
        }
    }

    /// Approximate parameter count by summing tensor element counts from the model.
    pub fn num_parameters(&self) -> usize {
        let padded_vocab = self.config.padded_vocab_size();
        let n_embd = self.config.n_embd;
        let head_dim = self.config.head_dim();
        let n_head = self.config.n_head;
        let n_kv_head = self.config.n_kv_head;
        let kv_dim = n_kv_head * head_dim;
        let n_layer = self.config.n_layer;

        let mut count = 0usize;

        // wte: padded_vocab x n_embd
        count += padded_vocab * n_embd;

        // lm_head: n_embd x padded_vocab
        count += n_embd * padded_vocab;

        // resid_lambdas + x0_lambdas
        count += n_layer * 2;

        // Per layer
        for i in 0..n_layer {
            // Attention: c_q, c_k, c_v, c_proj
            count += n_embd * (n_head * head_dim);     // c_q
            count += n_embd * (n_kv_head * head_dim);   // c_k
            count += n_embd * (n_kv_head * head_dim);   // c_v
            count += n_embd * n_embd;                    // c_proj

            // ve_gate if applicable
            if self.config.has_value_embedding(i) {
                count += 32 * n_kv_head; // ve_gate: 32 -> n_kv_head
            }

            // MLP: c_fc + c_proj
            count += n_embd * (4 * n_embd); // c_fc
            count += (4 * n_embd) * n_embd; // c_proj

            // Value embedding if applicable
            if self.config.has_value_embedding(i) {
                count += padded_vocab * kv_dim;
            }
        }

        count
    }
}
