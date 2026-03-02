use anyhow::Result;
use candle_core::{safetensors, Device, Tensor};
use candle_nn::VarMap;
use picochat_core::config::GPTConfig;
use std::collections::HashMap;
use std::path::Path;

pub fn save_varmap<P: AsRef<Path>>(varmap: &VarMap, path: P) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let tensors: HashMap<String, Tensor> = data
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect();
    safetensors::save(&tensors, path)?;
    Ok(())
}

pub fn load_varmap<P: AsRef<Path>>(varmap: &VarMap, path: P, device: &Device) -> Result<()> {
    let saved = safetensors::load(path, device)?;
    let data = varmap.data().lock().unwrap();
    for (name, var) in data.iter() {
        if let Some(saved_tensor) = saved.get(name) {
            var.set(saved_tensor)?;
        }
    }
    Ok(())
}

pub fn save_config<P: AsRef<Path>>(config: &GPTConfig, path: P) -> Result<()> {
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_config<P: AsRef<Path>>(path: P) -> Result<GPTConfig> {
    let json = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}
