use itertools::Itertools;
use rand::{distributions::{Distribution, WeightedIndex}, Rng};
use rayon::prelude::*;

use crate::distance::{earth_movers_distance, euclidian_distance};

pub fn kmeans_plusplus_euclidian(data: &[Vec<u8>], k: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<f32>> = Vec::new();

    // Randomly select the first centroid
    let first_idx = rng.gen_range(0..data.len());
    centroids.push(data[first_idx].clone().iter().map(|&val| val as f32).collect_vec());

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
                let distance = centroids.iter().map(|centroid| euclidian_distance(
                    &data[idx].iter().map(|&v| v as f64).collect(), &centroid.iter().map(|&v| v as f64).collect()
                )).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                *min_dist = distance;
            });

        // Efficient weighted selection using the WeightedIndex distribution
        let dist = WeightedIndex::new(&min_distances).unwrap();
        let next_centroid_idx = dist.sample(&mut rng);
        centroids.push(data[next_centroid_idx].clone().iter().map(|&val| val as f32).collect_vec());
    }

    centroids
}

pub fn kmeans_plusplus(data: &[Vec<u8>], k: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<f32>> = Vec::new();

    // Randomly select the first centroid
    let first_idx = rng.gen_range(0..data.len());
    centroids.push(data[first_idx].clone().iter().map(|&val| val as f32).collect_vec());

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
                let distance = earth_movers_distance(
                    &data[idx].iter().map(|&v| v as f64).collect(),
                    &centroids.last().unwrap().iter().map(|&v| v as f64).collect()
                );
                *min_dist = (*min_dist).min(distance);
            });

        // Efficient weighted selection using the WeightedIndex distribution
        let dist = WeightedIndex::new(&min_distances).unwrap();
        let next_centroid_idx = dist.sample(&mut rng);
        centroids.push(data[next_centroid_idx].clone().iter().map(|&val| val as f32).collect_vec());
    }

    centroids
}
