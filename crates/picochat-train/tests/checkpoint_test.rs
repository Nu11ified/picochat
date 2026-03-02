use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::init::initialize_weights;
use picochat_core::model::GPT;
use picochat_train::checkpoint;

#[test]
fn test_save_and_load_roundtrip() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();

    let path = "/tmp/picochat_test_checkpoint.safetensors";
    checkpoint::save_varmap(&varmap, path).unwrap();

    let varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
    let _model2 = GPT::new(&config, vb2).unwrap();
    checkpoint::load_varmap(&varmap2, path, &device).unwrap();

    let data1 = varmap.data().lock().unwrap();
    let data2 = varmap2.data().lock().unwrap();
    for (name, var1) in data1.iter() {
        let var2 = data2.get(name).unwrap_or_else(|| panic!("missing {name}"));
        let diff = (var1.as_tensor() - var2.as_tensor())
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "mismatch in {name}: diff={diff}");
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn test_save_and_load_config() {
    let config = GPTConfig::from_depth(4);
    let path = "/tmp/picochat_test_config.json";
    checkpoint::save_config(&config, path).unwrap();
    let loaded = checkpoint::load_config(path).unwrap();
    assert_eq!(loaded.n_layer, config.n_layer);
    assert_eq!(loaded.n_embd, config.n_embd);
    assert_eq!(loaded.n_head, config.n_head);
    std::fs::remove_file(path).ok();
}
