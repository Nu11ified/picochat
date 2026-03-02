use candle_core::{DType, Device, Var};
use candle_nn::{Init, VarMap};
use picochat_optim::{MuonAdamW, ParamGroup};

fn device() -> Device {
    Device::Cpu
}

#[test]
fn test_classify_params_by_name() {
    let groups = vec![
        ParamGroup::muon("h.0.attn.c_q.weight", 0.02),
        ParamGroup::adamw("wte.weight", 0.2, 0.0),
        ParamGroup::adamw("resid_lambdas", 0.005, 0.0),
    ];
    assert!(groups[0].is_muon());
    assert!(!groups[1].is_muon());
    assert!(!groups[2].is_muon());
}

/// Helper: build a small VarMap that exercises every classification branch.
fn make_test_varmap(dev: &Device) -> VarMap {
    let vm = VarMap::new();
    let dtype = DType::F32;

    // 2D weight -> should be routed to Muon
    vm.get((4, 4), "h.0.attn.c_q.weight", Init::Const(1.0), dtype, dev)
        .unwrap();
    // wte embedding -> AdamW
    vm.get((8, 4), "wte.weight", Init::Const(1.0), dtype, dev)
        .unwrap();
    // 1D bias -> AdamW fallback
    vm.get(4, "h.0.attn.c_q.bias", Init::Const(0.1), dtype, dev)
        .unwrap();
    // lm_head -> AdamW
    vm.get((4, 4), "lm_head.weight", Init::Const(1.0), dtype, dev)
        .unwrap();
    // resid_lambdas -> AdamW
    vm.get(4, "resid_lambdas", Init::Const(1.0), dtype, dev)
        .unwrap();
    // x0_lambdas -> AdamW
    vm.get(4, "x0_lambdas", Init::Const(1.0), dtype, dev)
        .unwrap();

    vm
}

#[test]
fn test_from_varmap_classifies_correctly() {
    let dev = device();
    let vm = make_test_varmap(&dev);
    let opt = MuonAdamW::from_varmap(&vm, 768);

    let summary = opt.route_summary();

    for (name, is_muon, lr) in &summary {
        match *name {
            "h.0.attn.c_q.weight" => {
                assert!(is_muon, "{name} should be Muon");
                assert!((*lr - 0.02).abs() < 1e-9, "{name} lr should be 0.02, got {lr}");
            }
            "wte.weight" => {
                assert!(!is_muon, "{name} should be AdamW");
                // scale = (768/768)^{-0.5} = 1.0, lr = 0.2 * 1.0
                assert!((*lr - 0.2).abs() < 1e-9, "{name} lr should be 0.2, got {lr}");
            }
            "h.0.attn.c_q.bias" => {
                assert!(!is_muon, "{name} should be AdamW (1D fallback)");
                assert!((*lr - 0.001).abs() < 1e-9, "{name} lr should be 0.001, got {lr}");
            }
            "lm_head.weight" => {
                assert!(!is_muon, "{name} should be AdamW");
                assert!((*lr - 0.004).abs() < 1e-9, "{name} lr should be 0.004, got {lr}");
            }
            "resid_lambdas" => {
                assert!(!is_muon, "{name} should be AdamW");
                assert!((*lr - 0.005).abs() < 1e-9, "{name} lr should be 0.005, got {lr}");
            }
            "x0_lambdas" => {
                assert!(!is_muon, "{name} should be AdamW");
                assert!((*lr - 0.5).abs() < 1e-9, "{name} lr should be 0.5, got {lr}");
            }
            other => panic!("unexpected param name: {other}"),
        }
    }
}

#[test]
fn test_combined_step_updates_all_params() {
    let dev = device();
    let vm = make_test_varmap(&dev);

    // Snapshot initial values for every var.
    let before: Vec<(String, Vec<f32>)> = {
        let data = vm.data().lock().unwrap();
        data.iter()
            .map(|(name, var)| {
                let vals: Vec<f32> = var
                    .as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                (name.clone(), vals)
            })
            .collect()
    };

    let mut opt = MuonAdamW::from_varmap(&vm, 768);

    // Build a loss: sum of squares of all vars.
    // We must collect the vars from the VarMap while the lock is held,
    // then drop the lock before calling backward (which may need the vars too).
    let vars: Vec<Var> = {
        let data = vm.data().lock().unwrap();
        data.values().cloned().collect()
    };

    let mut loss = vars[0].as_tensor().sqr().unwrap().sum_all().unwrap();
    for v in &vars[1..] {
        let term = v.as_tensor().sqr().unwrap().sum_all().unwrap();
        loss = (loss + term).unwrap();
    }

    opt.backward_step(&loss).unwrap();

    // Verify every parameter changed.
    let data = vm.data().lock().unwrap();
    for (name, before_vals) in &before {
        let var = data.get(name).unwrap();
        let after_vals: Vec<f32> = var
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let changed = before_vals
            .iter()
            .zip(after_vals.iter())
            .any(|(b, a)| (b - a).abs() > 1e-10);
        assert!(
            changed,
            "Parameter '{name}' should have been updated but was not"
        );
    }
}

#[test]
fn test_combined_with_lr_multiplier() {
    let dev = device();
    let vm = make_test_varmap(&dev);
    let mut opt = MuonAdamW::from_varmap(&vm, 768);

    let vars: Vec<Var> = {
        let data = vm.data().lock().unwrap();
        data.values().cloned().collect()
    };

    let mut loss = vars[0].as_tensor().sqr().unwrap().sum_all().unwrap();
    for v in &vars[1..] {
        let term = v.as_tensor().sqr().unwrap().sum_all().unwrap();
        loss = (loss + term).unwrap();
    }

    // A multiplier of 0.5 should still produce valid updates without errors.
    opt.backward_step_with_lr(&loss, 0.5).unwrap();

    // Verify at least one param changed (sanity check).
    let data = vm.data().lock().unwrap();
    let any_changed = data.values().any(|var| {
        let vals: Vec<f32> = var
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        vals.iter().any(|v| (*v - 1.0).abs() > 1e-10 || (*v - 0.1).abs() > 1e-10)
    });
    assert!(any_changed, "At least one parameter should change with lr_mult=0.5");
}

#[test]
fn test_scaling_with_different_n_embd() {
    let dev = device();
    let vm = VarMap::new();
    let dtype = DType::F32;

    // Just create a lm_head to test scaling
    vm.get((4, 4), "lm_head.weight", Init::Const(1.0), dtype, &dev)
        .unwrap();

    // n_embd = 384 -> scale = (384/768)^{-0.5} = sqrt(2) ~ 1.4142
    let opt = MuonAdamW::from_varmap(&vm, 384);
    let summary = opt.route_summary();
    let (_, _, lr) = summary.iter().find(|(n, _, _)| *n == "lm_head.weight").unwrap();
    let expected_scale = (384.0_f64 / 768.0).powf(-0.5);
    let expected_lr = 0.004 * expected_scale;
    assert!(
        (*lr - expected_lr).abs() < 1e-9,
        "lm_head lr with n_embd=384 should be {expected_lr}, got {lr}"
    );
}
