use candle_core::{Result, Tensor};
use candle_nn::VarMap;
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_optim::LrSchedule;
use picochat_optim::MuonAdamW;
use std::collections::HashMap;

pub struct Trainer {
    optimizer: MuonAdamW,
    schedule: Option<LrSchedule>,
    step_count: usize,
}

impl Trainer {
    pub fn new(varmap: &VarMap, config: &GPTConfig) -> Self {
        let optimizer = MuonAdamW::from_varmap(varmap, config.n_embd);
        Self {
            optimizer,
            schedule: None,
            step_count: 0,
        }
    }

    pub fn with_schedule(varmap: &VarMap, config: &GPTConfig, schedule: LrSchedule) -> Self {
        let optimizer = MuonAdamW::from_varmap(varmap, config.n_embd);
        Self {
            optimizer,
            schedule: Some(schedule),
            step_count: 0,
        }
    }

    /// Execute one training step: forward -> loss -> backward -> optimizer step.
    /// Returns the loss tensor.
    pub fn train_step(&mut self, model: &GPT, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        let loss = model.forward(input, Some(target))?;

        match &self.schedule {
            Some(sched) => {
                let base_lr = sched.base_lr();
                let current_lr = sched.get_lr(self.step_count);
                let mult = if base_lr > 0.0 {
                    current_lr / base_lr
                } else {
                    1.0
                };
                self.optimizer.backward_step_with_lr(&loss, mult)?;
            }
            None => {
                self.optimizer.backward_step(&loss)?;
            }
        }

        self.step_count += 1;
        Ok(loss)
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Set the step count for LR schedule offset when resuming training.
    pub fn set_step_count(&mut self, step: usize) {
        self.step_count = step;
    }

    pub fn schedule_ref(&self) -> Option<&LrSchedule> {
        self.schedule.as_ref()
    }

    pub fn optimizer_mut(&mut self) -> &mut MuonAdamW {
        &mut self.optimizer
    }

    /// Export optimizer state tensors for checkpointing.
    pub fn save_optimizer_state(&self) -> Result<HashMap<String, Tensor>> {
        self.optimizer.save_state()
    }

    /// Restore optimizer state from checkpoint tensors.
    pub fn load_optimizer_state(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        self.optimizer.load_state(tensors)
    }
}
