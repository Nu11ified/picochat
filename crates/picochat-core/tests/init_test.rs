use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_core::init::initialize_weights;

#[test]
fn test_resid_lambdas_init_to_one() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();
    let data = varmap.data().lock().unwrap();
    let lambdas = data.get("resid_lambdas").unwrap();
    let vals: Vec<f32> = lambdas.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-6, "resid_lambdas[{i}] should be 1.0, got {v}");
    }
}

#[test]
fn test_x0_lambdas_init_to_point_one() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();
    let data = varmap.data().lock().unwrap();
    let lambdas = data.get("x0_lambdas").unwrap();
    let vals: Vec<f32> = lambdas.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!((v - 0.1).abs() < 1e-6, "x0_lambdas[{i}] should be 0.1, got {v}");
    }
}

#[test]
fn test_c_proj_init_to_zero() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();
    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        if name.contains("c_proj") {
            let vals: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            for (i, v) in vals.iter().enumerate() {
                assert!(v.abs() < 1e-10, "{name}[{i}] should be 0.0, got {v}");
            }
        }
    }
}

#[test]
fn test_ve_gate_init_to_zero() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();
    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        if name.contains("ve_gate") {
            let vals: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            for (i, v) in vals.iter().enumerate() {
                assert!(v.abs() < 1e-10, "{name}[{i}] should be 0.0, got {v}");
            }
        }
    }
}

#[test]
fn test_wte_init_normal() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();
    let data = varmap.data().lock().unwrap();
    let wte = data.get("wte.weight").unwrap();
    let vals: Vec<f32> = wte.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    // Normal(0, 1.0) should have non-trivial values
    assert!(!vals.is_empty());
    // Check RMS is approximately 1.0 (within reasonable tolerance for finite sample)
    let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
    assert!(
        rms > 0.5 && rms < 2.0,
        "wte RMS should be ~1.0, got {rms}"
    );
    // Check mean is close to 0
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(
        mean.abs() < 0.1,
        "wte mean should be ~0.0, got {mean}"
    );
}

#[test]
fn test_lm_head_init_narrow_normal() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();
    let data = varmap.data().lock().unwrap();
    let lm_head = data.get("lm_head.weight").unwrap();
    let vals: Vec<f32> = lm_head.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    // Normal(0, 0.001) should have very small values
    let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
    assert!(
        rms < 0.01,
        "lm_head RMS should be ~0.001, got {rms}"
    );
}

#[test]
fn test_uniform_weights_in_range() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();
    let s = (3.0 / config.n_embd as f64).sqrt() as f32;
    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        if name.contains("c_q") || name.contains("c_k") || name.contains("c_v")
            || name.contains("c_fc")
        {
            // Skip c_proj names (c_v won't match c_proj anyway)
            if name.contains("c_proj") {
                continue;
            }
            let vals: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            for (i, v) in vals.iter().enumerate() {
                assert!(
                    *v >= -s && *v <= s,
                    "{name}[{i}] = {v} out of range [-{s}, {s}]"
                );
            }
        }
    }
}
