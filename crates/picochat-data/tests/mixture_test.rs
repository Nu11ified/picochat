use picochat_data::mixture::{DatasetMixture, MixtureDataset};

#[test]
fn test_mixture_weighted_sampling() {
    let mut mixture = DatasetMixture::new(vec![
        MixtureDataset {
            name: "a".to_string(),
            weight: 0.7,
            items: (0..100).map(|i| vec![i as u32]).collect(),
        },
        MixtureDataset {
            name: "b".to_string(),
            weight: 0.3,
            items: (100..200).map(|i| vec![i as u32]).collect(),
        },
    ]);

    let mut a_count = 0usize;
    for _ in 0..10000 {
        let item = mixture.sample();
        if item[0] < 100 {
            a_count += 1;
        }
    }

    let a_ratio = a_count as f64 / 10000.0;
    assert!(
        (a_ratio - 0.7).abs() < 0.05,
        "a_ratio was {a_ratio}"
    );
}

#[test]
fn test_mixture_single_dataset() {
    let mut mixture = DatasetMixture::new(vec![MixtureDataset {
        name: "only".to_string(),
        weight: 1.0,
        items: vec![vec![42]],
    }]);
    let item = mixture.sample();
    assert_eq!(item, vec![42]);
}

#[test]
fn test_mixture_epoch_cycling() {
    let mut mixture = DatasetMixture::new(vec![MixtureDataset {
        name: "small".to_string(),
        weight: 1.0,
        items: vec![vec![1], vec![2]],
    }]);

    let _ = mixture.sample();
    let _ = mixture.sample();
    let item3 = mixture.sample();
    assert!(item3 == vec![1] || item3 == vec![2]);
}
