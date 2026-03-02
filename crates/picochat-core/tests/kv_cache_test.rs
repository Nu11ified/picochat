use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::kv_cache::KVCache;
use picochat_core::model::GPT;

#[test]
fn test_kv_cache_new() {
    let cache = KVCache::new(4);
    assert_eq!(cache.layers.len(), 4);
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_layer_cache_update() {
    use picochat_core::kv_cache::LayerCache;
    let device = Device::Cpu;
    let mut cache = LayerCache::new();
    assert_eq!(cache.seq_len(), 0);

    let k1 = Tensor::zeros((1, 2, 4, 8), DType::F32, &device).unwrap();
    let v1 = Tensor::zeros((1, 2, 4, 8), DType::F32, &device).unwrap();
    let (k, v) = cache.update(&k1, &v1).unwrap();
    assert_eq!(k.dims(), &[1, 2, 4, 8]);
    assert_eq!(v.dims(), &[1, 2, 4, 8]);
    assert_eq!(cache.seq_len(), 4);

    let k2 = Tensor::ones((1, 2, 1, 8), DType::F32, &device).unwrap();
    let v2 = Tensor::ones((1, 2, 1, 8), DType::F32, &device).unwrap();
    let (k, v) = cache.update(&k2, &v2).unwrap();
    assert_eq!(k.dims(), &[1, 2, 5, 8]);
    assert_eq!(v.dims(), &[1, 2, 5, 8]);
    assert_eq!(cache.seq_len(), 5);
}

#[test]
fn test_kv_cache_reset() {
    use picochat_core::kv_cache::LayerCache;
    let device = Device::Cpu;
    let mut cache = LayerCache::new();
    let k = Tensor::zeros((1, 2, 4, 8), DType::F32, &device).unwrap();
    let v = Tensor::zeros((1, 2, 4, 8), DType::F32, &device).unwrap();
    cache.update(&k, &v).unwrap();
    assert_eq!(cache.seq_len(), 4);
    cache.reset();
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_forward_with_cache_prefill() {
    let config = GPTConfig::from_depth(2);
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&config, vb).unwrap();

    let mut cache = KVCache::new(config.n_layer);
    let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
    let logits = model.forward_with_cache(&input, &mut cache).unwrap();

    assert_eq!(logits.dims()[0], 1);
    assert_eq!(logits.dims()[1], 4);
    assert_eq!(logits.dims()[2], config.vocab_size);
    assert_eq!(cache.seq_len(), 4);
}

#[test]
fn test_forward_with_cache_decode() {
    let config = GPTConfig::from_depth(2);
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&config, vb).unwrap();

    let mut cache = KVCache::new(config.n_layer);

    let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
    model.forward_with_cache(&input, &mut cache).unwrap();
    assert_eq!(cache.seq_len(), 4);

    let next_input = Tensor::new(&[[5u32]], &device).unwrap();
    let logits = model.forward_with_cache(&next_input, &mut cache).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
    assert_eq!(cache.seq_len(), 5);

    let next_input = Tensor::new(&[[6u32]], &device).unwrap();
    let logits = model.forward_with_cache(&next_input, &mut cache).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
    assert_eq!(cache.seq_len(), 6);
}

#[test]
fn test_training_forward_unchanged() {
    let config = GPTConfig::from_depth(2);
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&config, vb).unwrap();

    let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], &device).unwrap();
    let logits = model.forward(&input, None).unwrap();
    assert_eq!(logits.dims()[1], 8);
}
