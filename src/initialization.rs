use rand::{distributions::{Distribution, WeightedIndex}, Rng};
use rayon::prelude::*;

use crate::distance::earth_movers_distance;

pub fn kmeans_plusplus(data: &[Vec<u32>], k: usize) -> Vec<Vec<u32>> {
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<u32>> = Vec::new();

    // Randomly select the first centroid
    let first_idx = rng.gen_range(0..data.len());
    centroids.push(data[first_idx].clone());

    if k == 1 {
        return centroids;
    }

    let mut min_distances = vec![f64::MAX; data.len()];

    // Select k-1 remaining centroids
    for _ in 1..k {
        // Update the minimum distances in parallel
        min_distances.par_iter_mut()
            .enumerate()
            .for_each(|(idx, min_dist)| {
                let distance = earth_movers_distance(&data[idx], &centroids.last().unwrap());
                *min_dist = (*min_dist).min(distance);
            });

        // Efficient weighted selection using the WeightedIndex distribution
        let dist = WeightedIndex::new(&min_distances).unwrap();
        let next_centroid_idx = dist.sample(&mut rng);
        centroids.push(data[next_centroid_idx].clone());
    }

    centroids
}
