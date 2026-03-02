use candle_core::{Result, Tensor};

/// Per-layer key-value cache for autoregressive inference.
pub struct LayerCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl LayerCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Append new K, V to the cache. Returns the full (cached + new) K and V.
    ///
    /// `k_new` and `v_new` are (B, n_kv_head, T_new, head_dim) -- already transposed.
    pub fn update(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = match self.k.take() {
            Some(cached) => Tensor::cat(&[&cached, k_new], 2)?,
            None => k_new.clone(),
        };
        let v = match self.v.take() {
            Some(cached) => Tensor::cat(&[&cached, v_new], 2)?,
            None => v_new.clone(),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    /// Current number of cached positions (T dimension).
    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dims()[2])
    }

    /// Clear the cache.
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// KV cache for the full model (one LayerCache per transformer layer).
pub struct KVCache {
    pub layers: Vec<LayerCache>,
}

impl KVCache {
    pub fn new(n_layers: usize) -> Self {
        let layers = (0..n_layers).map(|_| LayerCache::new()).collect();
        Self { layers }
    }

    /// Number of cached positions (same across all layers after a forward pass).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.seq_len())
    }

    /// Clear all layer caches.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}
