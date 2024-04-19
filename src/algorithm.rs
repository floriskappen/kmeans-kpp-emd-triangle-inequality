use itertools::Itertools;
use rand::{seq::SliceRandom, Rng};
use std::sync::atomic::{AtomicI32, Ordering};
use rayon::prelude::*;
use std::mem;

use crate::distance::{earth_movers_distance_centroid, earth_movers_distance_centroid_centroid};
use crate::inertia::calculate_inertia;
use crate::{distance::earth_movers_distance, initialization::kmeans_plusplus};

fn calculate_frobenius_norm(centroid: &[Vec<f32>], prev_centroid: &[Vec<f32>]) -> f64 {
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
    data: &[Vec<u8>],
    k: usize,
    max_iters: usize,
    convergence_threshold: f64,
) -> Result<(Vec<Vec<f32>>, Vec<u32>, f64), &'static str> {
    if k > data.len() {
        return Err("Number of clusters cannot be greater than the number of data points");
    }
    if max_iters % 2 == 0 {
        return Err("Please use an odd number of iterations")
    }
    let mut centroids = kmeans_plusplus(data, k);
    log::info!("initialized kmeans++");
    let mut labels: Vec<u32> = vec![0; data.len()];
    let mut prev_centroids = centroids.clone();
    let mut min_distances = vec![std::f64::MAX; data.len()];
    for iter in 0..max_iters {
        let cluster_sizes: Vec<AtomicI32> = (0..k).map(|_| AtomicI32::new(0)).collect_vec();

        // Use triangle inequality optimization 
        data.par_iter().zip(&mut min_distances).zip(&mut labels).for_each(|((point, min_distance), label)| {
            let mut thread_rng = rand::thread_rng();
            let reference_centroid_idx = thread_rng.gen_range(0..k);
            let reference_distances: Vec<f64> = centroids.par_iter()
                .map(|c| earth_movers_distance_centroid_centroid(c, &centroids[reference_centroid_idx]))
                .collect();
            let reference_distance = earth_movers_distance_centroid(point, &centroids[reference_centroid_idx]);
            let (best_centroid_idx, best_distance) = centroids.par_iter().enumerate()
                .map(|(centroid_idx, centroid)| {
                    if reference_distance + reference_distances[centroid_idx] < *min_distance {
                        let distance = earth_movers_distance_centroid(point, centroid);
                        (centroid_idx, distance)
                    } else {
                        (centroid_idx, std::f64::MAX)
                    }
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            *min_distance = best_distance;
            *label = best_centroid_idx as u32;
            cluster_sizes[best_centroid_idx].fetch_add(1, Ordering::Relaxed);
        });

        let cluster_sizes: Vec<i32> = cluster_sizes.iter()
            .map(|atomic| atomic.load(Ordering::SeqCst))
            .collect();
        let new_centroids: Vec<Vec<f32>> = data.par_iter()
            .zip(&labels)
            .fold(
                || vec![vec![0f32; 30]; k],
                |mut local_centroids, (point, &label)| {
                    for (c, &p) in local_centroids[label as usize].iter_mut().zip(point) {
                        *c += p as f32;
                    }
                    local_centroids
                }
            )
            .reduce(
                || vec![vec![0f32; 30]; k],
                |mut a, b| {
                    for (a_centroid, b_centroid) in a.iter_mut().zip(b) {
                        for (a_val, b_val) in a_centroid.iter_mut().zip(b_centroid) {
                            *a_val += b_val;
                        }
                    }
                    a
                }
            );
        centroids.par_iter_mut().zip(new_centroids.par_iter()).zip(&cluster_sizes).for_each(|((centroid, new_centroid), &size)| {
            if size > 0 {
                centroid.par_iter_mut().zip(new_centroid.par_iter()).for_each(|(c, &new_c)| {
                    *c = new_c / size as f32;
                });
            } else {
                let mut nested_thread_rng = rand::thread_rng();
                *centroid = data.choose(&mut nested_thread_rng).unwrap().iter().map(|&val| val as f32).collect_vec();
            }
        });

        // Only even iterations will be good
        if iter > 0 && iter % 2 != 0 {
            let frobenius_norm = calculate_frobenius_norm(&centroids, &prev_centroids);
            let new_centroids_norm = centroids.par_iter()
                .map(|centroid| centroid.par_iter().map(|&x| (x as f64).powi(2)).sum::<f64>())
                .sum::<f64>()
                .sqrt();
            if frobenius_norm / new_centroids_norm < convergence_threshold {
                log::info!("Converged after {} iterations", iter + 1);
                break;
            }
            prev_centroids = centroids.clone();
            let inertia = calculate_inertia(data, &centroids, &labels);
            
            if (iter-1) % 10 == 0 {
                log::info!("Finished iteration {} with an inertia of {}", iter, inertia);
            }
        }
    }
    let inertia = calculate_inertia(data, &centroids, &labels);
    Ok((centroids, labels, inertia))
}


pub fn kmeans_default(
    data: &[Vec<u8>],
    k: usize,
    max_iters: usize,
    convergence_threshold: f64,
) -> Result<(Vec<Vec<f32>>, Vec<u32>, f64), &'static str> {
    if k > data.len() {
        return Err("Number of clusters cannot be greater than the number of data points");
    }
    let mut centroids = kmeans_plusplus(data, k);
    println!("initial centroids: {:?}", centroids);
    log::info!("initialized kmeans++");
    let mut labels: Vec<u32> = vec![0; data.len()];
    let mut prev_centroids = centroids.clone();
    for iter in 0..max_iters {
        if iter > 0 && iter % 10 == 0 {
            log::info!("Finished iteration {}", iter);
        }
        let cluster_sizes: Vec<AtomicI32> = (0..k).map(|_| AtomicI32::new(0)).collect_vec();
        data.par_iter().zip(&mut labels).for_each(|(point, label)| {
            let (best_centroid_idx, _) = centroids.par_iter().enumerate()
                .map(|(centroid_idx, centroid)| {
                    let distance = earth_movers_distance_centroid(point, centroid);
                    (centroid_idx, distance)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            *label = best_centroid_idx as u32;
            cluster_sizes[best_centroid_idx].fetch_add(1, Ordering::Relaxed);
        });
        let cluster_sizes: Vec<i32> = cluster_sizes.iter()
            .map(|atomic| atomic.load(Ordering::SeqCst))
            .collect();
        let new_centroids: Vec<Vec<f32>> = data.par_iter()
            .zip(&labels)
            .fold(
                || vec![vec![0f32; 30]; k],
                |mut local_centroids, (point, &label)| {
                    for (c, &p) in local_centroids[label as usize].iter_mut().zip(point) {
                        *c += p as f32;
                    }
                    local_centroids
                }
            )
            .reduce(
                || vec![vec![0f32; 30]; k],
                |mut a, b| {
                    for (a_centroid, b_centroid) in a.iter_mut().zip(b) {
                        for (a_val, b_val) in a_centroid.iter_mut().zip(b_centroid) {
                            *a_val += b_val;
                        }
                    }
                    a
                }
            );
        centroids.par_iter_mut().zip(new_centroids.par_iter()).zip(&cluster_sizes).for_each(|((centroid, new_centroid), &size)| {
            if size > 0 {
                centroid.par_iter_mut().zip(new_centroid.par_iter()).for_each(|(c, &new_c)| {
                    *c = new_c / size as f32;
                });
            } else {
                let mut nested_thread_rng = rand::thread_rng();
                *centroid = data.choose(&mut nested_thread_rng).unwrap().iter().map(|&val| val as f32).collect_vec();
            }
        });
        // println!("centroids[1]: {:?}", centroids[1]);
        let frobenius_norm = calculate_frobenius_norm(&centroids, &prev_centroids);
        let new_centroids_norm = centroids.par_iter()
            .map(|centroid| centroid.par_iter().map(|&x| (x as f64).powi(2)).sum::<f64>())
            .sum::<f64>()
            .sqrt();
        if frobenius_norm / new_centroids_norm < convergence_threshold {
            log::info!("Converged after {} iterations", iter + 1);
            break;
        }
        prev_centroids = centroids.clone();
        let inertia = calculate_inertia(data, &centroids, &labels);
        println!("inertia: {}", inertia);
    }
    let inertia = calculate_inertia(data, &centroids, &labels);
    Ok((centroids, labels, inertia))
}