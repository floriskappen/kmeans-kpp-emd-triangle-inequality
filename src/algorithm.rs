use rand::{seq::SliceRandom, Rng};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;
use std::sync::Mutex;

use crate::inertia::calculate_inertia;
use crate::{distance::earth_movers_distance, initialization::kmeans_plusplus};

fn calculate_frobenius_norm(centroid: &[Vec<u32>], prev_centroid: &[Vec<u32>]) -> f64 {
    centroid.iter().zip(prev_centroid.iter())
        .map(|(c1, c2)| {
            c1.iter().zip(c2.iter())
                .map(|(&val1, &val2)| {
                    let diff = val1 as i64 - val2 as i64;
                    (diff * diff) as f64
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt()
}

pub fn kmeans_triangle_inequality(
    data: &[Vec<u32>],
    k: usize,
    max_iters: usize,
    convergence_threshold: f64,
) -> Result<(Vec<Vec<u32>>, Vec<usize>, f64), &'static str> {
    if k > data.len() {
        return Err("Number of clusters cannot be greater than the number of data points");
    }

    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    let centroids = Arc::new(Mutex::new(kmeans_plusplus(data, k)));
    log::info!("initialized kmeans++");
    let mut prev_centroids = vec![vec![0u32; 30]; k];
    let labels: Vec<_> = (0..data.len()).map(|i| AtomicUsize::new(0)).collect();
    let min_distances = Arc::new(Mutex::new(vec![std::f64::MAX; data.len()]));

    for iter in 0..max_iters {
        if iter > 0 && iter % 10 == 0 {
            log::info!("Finished iteration {}", iter);
        }
        let cluster_sizes = Arc::new(Mutex::new(vec![0; k]));
        let centroids_clone = centroids.clone();
        let cluster_sizes_clone = cluster_sizes.clone();
        let min_distances_clone = min_distances.clone();

        data.par_iter().enumerate().for_each(|(idx, point)| {
            let centroids = centroids_clone.lock().unwrap();
            let mut local_min_distances = min_distances_clone.lock().unwrap();
            let mut thread_rng = rand::thread_rng();

            // Use triangle inequality
            let reference_centroid_idx = thread_rng.gen_range(0..k);
            let reference_distances: Vec<f64> = centroids.iter()
                .map(|c| earth_movers_distance(c, &centroids[reference_centroid_idx]))
                .collect();

            let reference_distance = earth_movers_distance(point, &centroids[reference_centroid_idx]);

            centroids.iter().enumerate().for_each(|(centroid_idx, centroid)| {
                if reference_distance + reference_distances[centroid_idx] < local_min_distances[idx] {
                    let distance = earth_movers_distance(point, centroid);
                    if distance < local_min_distances[idx] {
                        local_min_distances[idx] = distance;
                        labels[idx].store(centroid_idx, Ordering::Relaxed);
                    }
                }
            });

            let mut sizes = cluster_sizes_clone.lock().unwrap();
            sizes[labels[idx].load(Ordering::Relaxed)] += 1;
        });

        let new_centroids = vec![vec![0u32; 30]; k];
        let new_centroids_arc = Arc::new(Mutex::new(new_centroids));

        data.par_iter().enumerate().for_each(|(idx, point)| {
            let label = labels[idx].load(Ordering::Relaxed);
            let mut new_centroids = new_centroids_arc.lock().unwrap();
            for (bin, value) in new_centroids[label].iter_mut().zip(point.iter()) {
                *bin += *value;
            }
        });

        let mut centroids = centroids.lock().unwrap();
        let new_centroids = new_centroids_arc.lock().unwrap();
        for (idx, centroid) in centroids.iter_mut().enumerate() {
            let size = cluster_sizes.lock().unwrap()[idx];
            if size > 0 {
                centroid.iter_mut().zip(new_centroids[idx].iter()).for_each(|(c, new_c)| {
                    *c = *new_c / size as u32;
                });
            } else {
                // Reassign to a random data point if cluster is empty
                *centroid = data.choose(&mut rng).unwrap().clone();
            }
        }

        // Check for convergence
        let frobenius_norm = calculate_frobenius_norm(&centroids, &prev_centroids);
        let new_centroids_norm = centroids.iter()
            .map(|centroid| centroid.iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>())
            .sum::<f64>()
            .sqrt();

        if frobenius_norm / new_centroids_norm < convergence_threshold {
            log::info!("Converged after {} iterations", iter + 1);
            break;
        }

        prev_centroids = centroids.clone();
    }

    let final_labels = labels.iter().map(|x| x.load(Ordering::Relaxed)).collect::<Vec<_>>();

    let inertia = calculate_inertia(data, &centroids.lock().unwrap(), &final_labels);

    return Ok((Arc::try_unwrap(centroids).unwrap().into_inner().unwrap(), final_labels, inertia))
}


pub fn kmeans_default(
    data: &[Vec<u32>],
    k: usize,
    max_iters: usize,
    convergence_threshold: f64,
) -> Result<(Vec<Vec<u32>>, Vec<usize>, f64), &'static str> {
    if k > data.len() {
        return Err("Number of clusters cannot be greater than the number of data points");
    }

    let mut rng = rand::thread_rng();
    let centroids = Arc::new(Mutex::new(kmeans_plusplus(data, k)));
    log::info!("initialized kmeans++");
    let labels: Vec<_> = (0..data.len()).map(|_| AtomicUsize::new(0)).collect();
    let mut prev_centroids = vec![vec![0u32; 30]; k];
    let min_distances = Arc::new(Mutex::new(vec![std::f64::MAX; data.len()]));

    for iter in 0..max_iters {
        if iter > 0 && iter % 10 == 0 {
            log::info!("Finished iteration {}", iter);
        }
        let cluster_sizes = Arc::new(Mutex::new(vec![0; k]));
        let centroids_clone = centroids.clone();
        let cluster_sizes_clone = cluster_sizes.clone();
        let min_distances_clone = min_distances.clone();

        data.par_iter().enumerate().for_each(|(idx, point)| {
            let centroids = centroids_clone.lock().unwrap();
            let mut local_min_distances = min_distances_clone.lock().unwrap();
            
            centroids.iter().enumerate().for_each(|(centroid_idx, centroid)| {
                let distance = earth_movers_distance(point, centroid);
                if distance < local_min_distances[idx] {
                    local_min_distances[idx] = distance;
                    labels[idx].store(centroid_idx, Ordering::Relaxed);
                }
            });

            let mut sizes = cluster_sizes_clone.lock().unwrap();
            sizes[labels[idx].load(Ordering::Relaxed)] += 1;
        });

        let new_centroids = vec![vec![0u32; 30]; k];
        let new_centroids_arc = Arc::new(Mutex::new(new_centroids));

        data.par_iter().enumerate().for_each(|(idx, point)| {
            let label = labels[idx].load(Ordering::Relaxed);
            let mut new_centroids = new_centroids_arc.lock().unwrap();
            for (bin, value) in new_centroids[label].iter_mut().zip(point.iter()) {
                *bin += *value;
            }
        });

        let mut centroids = centroids.lock().unwrap();
        let new_centroids = new_centroids_arc.lock().unwrap();
        for (idx, centroid) in centroids.iter_mut().enumerate() {
            let size = cluster_sizes.lock().unwrap()[idx];
            if size > 0 {
                centroid.iter_mut().zip(new_centroids[idx].iter()).for_each(|(c, new_c)| {
                    *c = *new_c / size as u32;
                });
            } else {
                // Reassign to a random data point if cluster is empty
                *centroid = data.choose(&mut rng).unwrap().clone();
            }
        }

        // Check for convergence
        let frobenius_norm = calculate_frobenius_norm(&centroids, &prev_centroids);
        let new_centroids_norm = centroids.iter()
            .map(|centroid| centroid.iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>())
            .sum::<f64>()
            .sqrt();

        if frobenius_norm / new_centroids_norm < convergence_threshold {
            log::info!("Converged after {} iterations", iter + 1);
            break;
        }

        prev_centroids = centroids.clone();
    }

    let final_labels = labels.iter().map(|x| x.load(Ordering::Relaxed)).collect::<Vec<_>>();

    let inertia = calculate_inertia(data, &centroids.lock().unwrap(), &final_labels);

    Ok((Arc::try_unwrap(centroids).unwrap().into_inner().unwrap(), final_labels, inertia))
}
