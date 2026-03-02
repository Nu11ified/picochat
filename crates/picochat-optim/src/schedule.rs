/// Learning-rate schedule with three phases:
/// 1. Linear warmup: 0 -> base_lr over `warmup_steps`
/// 2. Constant: base_lr
/// 3. Cosine warmdown: base_lr -> 0 over the final `warmdown_frac` of training
pub struct LrSchedule {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    warmdown_start: usize,
}

impl LrSchedule {
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize, warmdown_frac: f64) -> Self {
        let warmdown_steps = (total_steps as f64 * warmdown_frac) as usize;
        let warmdown_start = total_steps.saturating_sub(warmdown_steps);
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            warmdown_start,
        }
    }

    /// Return the learning rate for the given training step.
    pub fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            self.base_lr * (step as f64 / self.warmup_steps as f64)
        } else if step < self.warmdown_start {
            self.base_lr
        } else {
            let progress = (step - self.warmdown_start) as f64
                / (self.total_steps - self.warmdown_start) as f64;
            self.base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }
}
