/// Training metrics: BPB, throughput, MFU.
pub struct TrainingMetrics {
    avg_bytes_per_token: f64,
    last_bpb: f64,
    last_throughput: f64,
}

impl TrainingMetrics {
    pub fn new(avg_bytes_per_token: f64) -> Self {
        Self {
            avg_bytes_per_token,
            last_bpb: 0.0,
            last_throughput: 0.0,
        }
    }

    /// BPB = loss * log2(e) / avg_bytes_per_token
    pub fn compute_bpb(loss: f64, avg_bytes_per_token: f64) -> f64 {
        loss * std::f64::consts::LOG2_E / avg_bytes_per_token
    }

    /// Throughput in tokens/sec.
    pub fn compute_throughput(num_tokens: usize, elapsed_secs: f64) -> f64 {
        num_tokens as f64 / elapsed_secs
    }

    /// MFU = 6 * num_params * tokens_per_step / (elapsed * peak_tflops * 1e12)
    pub fn compute_mfu(
        num_params: usize,
        tokens_per_step: usize,
        elapsed_secs: f64,
        peak_tflops: f64,
    ) -> f64 {
        6.0 * num_params as f64 * tokens_per_step as f64
            / (elapsed_secs * peak_tflops * 1e12)
    }

    pub fn record_step(&mut self, loss: f64, num_tokens: usize, elapsed_secs: f64) {
        self.last_bpb = Self::compute_bpb(loss, self.avg_bytes_per_token);
        self.last_throughput = Self::compute_throughput(num_tokens, elapsed_secs);
    }

    pub fn last_bpb(&self) -> f64 {
        self.last_bpb
    }

    pub fn last_throughput(&self) -> f64 {
        self.last_throughput
    }
}
