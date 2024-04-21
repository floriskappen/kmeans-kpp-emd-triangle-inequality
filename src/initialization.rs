use itertools::Itertools;
use rand::{distributions::{Distribution, WeightedIndex}, Rng};
use rayon::prelude::*;

use crate::distance::{earth_movers_distance, euclidian_distance};

pub fn kmeans_plusplus_euclidian(data: &[u8], histogram_size: usize, k: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<f32>> = Vec::new();
    let num_histograms = data.len() / histogram_size;

    // Randomly select the first centroid
    let first_idx = rng.gen_range(0..num_histograms) * histogram_size;
    centroids.push(data[first_idx..first_idx + histogram_size].iter().map(|&val| val as f32).collect_vec());

    if k == 1 {
        return centroids;
    }

    let mut min_distances = vec![f32::MAX; num_histograms];

    // Select k-1 remaining centroids
    for _ in 1..k {
        // Compute distances to the last added centroid and update min_distances
        let last_centroid = centroids.last().unwrap();
        min_distances.par_iter_mut()
            .enumerate()
            .for_each(|(idx, min_dist)| {
                let start_index = idx * histogram_size;
                let histogram_as_f32: Vec<f64> = data[start_index..start_index + histogram_size].iter().map(|&v| v as f64).collect();
                let distance = euclidian_distance(&histogram_as_f32, &last_centroid.iter().map(|&v| v as f64).collect());
                *min_dist = (*min_dist).min(distance as f32);
            });
    
        // Select next centroid based on updated min_distances
        let dist = WeightedIndex::new(&min_distances).unwrap();
        let next_centroid_idx = dist.sample(&mut rng) * histogram_size;
        centroids.push(data[next_centroid_idx..next_centroid_idx + histogram_size].iter().map(|&val| val as f32).collect_vec());
    }

    centroids
}

pub fn kmeans_plusplus(data: &[u8], histogram_size: usize, k: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<f32>> = Vec::new();
    let num_histograms = data.len() / histogram_size;

    // Randomly select the first centroid
    let first_idx = rng.gen_range(0..num_histograms) * histogram_size;
    centroids.push(data[first_idx..first_idx + histogram_size].iter().map(|&val| val as f32).collect_vec());

    if k == 1 {
        return centroids;
    }

    let mut min_distances = vec![f32::MAX; num_histograms];

    // Select k-1 remaining centroids
    for _ in 1..k {
        // Update the minimum distances in parallel
        min_distances.par_iter_mut()
            .enumerate()
            .for_each(|(idx, min_dist)| {
                let start_index = idx * histogram_size;
                let distance = earth_movers_distance(
                    &data[start_index..start_index + histogram_size].iter().map(|&v| v as f64).collect(),
                    &centroids.last().unwrap().iter().map(|&v| v as f64).collect()
                );
                *min_dist = (*min_dist).min(distance as f32);
            });

        // Efficient weighted selection using the WeightedIndex distribution
        let dist = WeightedIndex::new(&min_distances).unwrap();
        let next_centroid_idx = dist.sample(&mut rng);
        centroids.push(data[next_centroid_idx..next_centroid_idx + histogram_size].iter().map(|&val| val as f32).collect_vec());
    }

    centroids
}
