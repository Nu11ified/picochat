use rand::Rng;

pub struct MixtureDataset {
    pub name: String,
    pub weight: f64,
    pub items: Vec<Vec<u32>>,
}

/// Weighted random sampling across multiple datasets with epoch cycling.
pub struct DatasetMixture {
    datasets: Vec<MixtureDataset>,
    cumulative_weights: Vec<f64>,
    cursors: Vec<usize>,
    shuffled_indices: Vec<Vec<usize>>,
    rng: rand::rngs::ThreadRng,
}

impl DatasetMixture {
    pub fn new(datasets: Vec<MixtureDataset>) -> Self {
        let total_weight: f64 = datasets.iter().map(|d| d.weight).sum();
        let mut cumulative_weights = Vec::with_capacity(datasets.len());
        let mut cum = 0.0;
        for d in &datasets {
            cum += d.weight / total_weight;
            cumulative_weights.push(cum);
        }

        let mut rng = rand::thread_rng();
        let shuffled_indices: Vec<Vec<usize>> = datasets
            .iter()
            .map(|d| {
                let mut indices: Vec<usize> = (0..d.items.len()).collect();
                shuffle(&mut indices, &mut rng);
                indices
            })
            .collect();

        let cursors = vec![0; datasets.len()];

        Self {
            datasets,
            cumulative_weights,
            cursors,
            shuffled_indices,
            rng,
        }
    }

    /// Sample one item using weighted random selection.
    pub fn sample(&mut self) -> Vec<u32> {
        let r: f64 = self.rng.gen();
        let mut dataset_idx = self.datasets.len() - 1;
        for (i, &cw) in self.cumulative_weights.iter().enumerate() {
            if r < cw {
                dataset_idx = i;
                break;
            }
        }

        if self.cursors[dataset_idx] >= self.datasets[dataset_idx].items.len() {
            // Epoch boundary: reshuffle and reset cursor
            let mut indices: Vec<usize> =
                (0..self.datasets[dataset_idx].items.len()).collect();
            shuffle(&mut indices, &mut self.rng);
            self.shuffled_indices[dataset_idx] = indices;
            self.cursors[dataset_idx] = 0;
        }

        let idx = self.shuffled_indices[dataset_idx][self.cursors[dataset_idx]];
        self.cursors[dataset_idx] += 1;

        self.datasets[dataset_idx].items[idx].clone()
    }
}

fn shuffle(slice: &mut [usize], rng: &mut rand::rngs::ThreadRng) {
    for i in (1..slice.len()).rev() {
        let j = rng.gen_range(0..=i);
        slice.swap(i, j);
    }
}
