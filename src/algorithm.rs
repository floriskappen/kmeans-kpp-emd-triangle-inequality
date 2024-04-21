use itertools::Itertools;
use rand::{seq::SliceRandom, Rng};
use std::sync::atomic::{AtomicI32, Ordering};
use rayon::prelude::*;

use crate::distance::{euclidian_distance, earth_movers_distance};
use crate::inertia::{calculate_inertia, calculate_inertia_euclidian};
use crate::initialization::{kmeans_plusplus, kmeans_plusplus_euclidian};

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

pub fn kmeans_euclidian(
    data: &[u8],
    histogram_size: usize,
    k: usize,
    max_iters: usize,
    convergence_threshold: f64,
) -> Result<(Vec<Vec<f32>>, Vec<u32>, f64), &'static str> {
    if k > data.len() / histogram_size {
        return Err("Number of clusters cannot be greater than the number of data points");
    }

    let mut centroids = kmeans_plusplus_euclidian(data, histogram_size, k);
    log::info!("initialized kmeans++");
    let num_histograms = data.len() / histogram_size;
    let mut labels = vec![0u32; num_histograms];
    let mut prev_centroids = centroids.clone();

    for iter in 0..max_iters {
        // Assign points to the nearest centroid
        (0..num_histograms).into_par_iter().zip(&mut labels).for_each(|(i, label)| {
            let point_start = i * histogram_size;
            let point = &data[point_start..point_start + histogram_size];
            let (best_centroid_idx, _) = centroids.par_iter().enumerate()
                .map(|(centroid_idx, centroid)| {
                    let distance = euclidian_distance(&point.iter().map(|&v| v as f64).collect::<Vec<f64>>(), &centroid.iter().map(|&v| v as f64).collect::<Vec<f64>>());
                    (centroid_idx, distance)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            *label = best_centroid_idx as u32;
        });

        let mut new_centroids = vec![vec![0f32; histogram_size]; k];
        let mut cluster_sizes = vec![0usize; k];

        for (i, &label) in labels.iter().enumerate() {
            let point_start = i * histogram_size;
            let point = &data[point_start..point_start + histogram_size];
            let centroid = &mut new_centroids[label as usize];
            for (c, &p) in centroid.iter_mut().zip(point.iter()) {
                *c += p as f32;
            }
            cluster_sizes[label as usize] += 1;
        }

        for (centroid, (new_centroid, &size)) in centroids.iter_mut().zip(new_centroids.iter().zip(&cluster_sizes)) {
            if size > 0 {
                for (c, &new_c) in centroid.iter_mut().zip(new_centroid.iter()) {
                    *c = new_c / size as f32;
                }
            } else {
                let mut rng = rand::thread_rng();
                let random_point_start = rng.gen_range(0..num_histograms) * histogram_size;
                let random_point = &data[random_point_start..random_point_start + histogram_size];
                *centroid = random_point.iter().map(|&val| val as f32).collect();
            }
        }

        let frobenius_norm = calculate_frobenius_norm(&centroids, &prev_centroids);
        let new_centroids_norm = centroids.par_iter()
            .map(|centroid| centroid.iter().map(|&x| (x as f64).powi(2)).sum::<f64>())
            .sum::<f64>()
            .sqrt();
        if frobenius_norm / new_centroids_norm < convergence_threshold {
            log::info!("Converged after {} iterations", iter + 1);
            break;
        }
        prev_centroids = centroids.clone();

        if iter > 0 && iter % 10 == 0 {
            let inertia = calculate_inertia_euclidian(data, histogram_size, &centroids, &labels);
            log::info!("Finished iteration {} with an inertia of {}", iter, inertia);
        }
    }
    let inertia = calculate_inertia_euclidian(data, histogram_size, &centroids, &labels);
    Ok((centroids, labels, inertia))
}

pub fn kmeans_euclidian_triangle_inequality(
    data: &[u8],
    histogram_size: usize,
    k: usize,
    max_iters: usize,
    convergence_threshold: f64,
) -> Result<(Vec<Vec<f32>>, Vec<u32>, f64), &'static str> {
    if k > data.len() / histogram_size {
        return Err("Number of clusters cannot be greater than the number of data points");
    }

    let mut centroids = kmeans_plusplus_euclidian(data, histogram_size, k);
    log::info!("initialized kmeans++");

    let num_histograms = data.len() / histogram_size;
    let mut labels: Vec<u32> = vec![0; num_histograms];
    let mut prev_centroids = centroids.clone();
    let mut min_distances_squared = vec![std::f64::MAX; num_histograms];
    let mut lower_bounds = vec![0.0; num_histograms * k];
    let mut upper_bounds = vec![std::f64::MAX; num_histograms];

    for iter in 0..max_iters {
        let cluster_sizes: Vec<AtomicI32> = (0..k).map(|_| AtomicI32::new(0)).collect_vec();

        // Update lower bounds and upper bounds
        (0..num_histograms).into_par_iter()
            .zip(lower_bounds.par_chunks_mut(k))
            .zip(upper_bounds.par_iter_mut())
            .for_each(|((point_idx, lower_bounds_chunk), upper_bound)| {
                let point_start = point_idx * histogram_size;
                let point = &data[point_start..point_start + histogram_size];
                for (centroid_idx, lower_bound) in lower_bounds_chunk.iter_mut().enumerate() {
                    let distance_squared = euclidian_distance(
                        &point.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
                        &centroids[centroid_idx].iter().map(|&v| v as f64).collect::<Vec<f64>>(),
                    )
                    .powi(2);
                    *lower_bound = distance_squared;
                }
                *upper_bound = lower_bounds_chunk[labels[point_idx] as usize];
            });

        // Assign points to clusters
        (0..num_histograms).into_par_iter()
            .zip(min_distances_squared.par_iter_mut())
            .zip(labels.par_iter_mut())
            .zip(upper_bounds.par_iter_mut())
            .for_each(|(((point_idx, min_distance_squared), label), upper_bound)| {
                let point_start = point_idx * histogram_size;
                let point = &data[point_start..point_start + histogram_size];
                let mut best_centroid_idx = *label as usize;
                let mut best_distance_squared = *min_distance_squared;
                let start_idx = point_idx * k;
                let end_idx = start_idx + k;
                for (centroid_idx, &lower_bound) in lower_bounds[start_idx..end_idx].iter().enumerate() {
                    if *upper_bound > lower_bound {
                        let distance_squared = euclidian_distance(
                            &point.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
                            &centroids[centroid_idx].iter().map(|&v| v as f64).collect::<Vec<f64>>(),
                        )
                        .powi(2);
                        if distance_squared < best_distance_squared {
                            best_centroid_idx = centroid_idx;
                            best_distance_squared = distance_squared;
                        }
                    }
                }
                if best_centroid_idx != *label as usize {
                    *label = best_centroid_idx as u32;
                    *min_distance_squared = best_distance_squared;
                    *upper_bound = best_distance_squared;
                }
                cluster_sizes[best_centroid_idx].fetch_add(1, Ordering::Relaxed);
            });

        let cluster_sizes: Vec<i32> = cluster_sizes.iter().map(|atomic| atomic.load(Ordering::SeqCst)).collect();

        let mut new_centroids = vec![vec![0f32; histogram_size]; k];

        for (i, &label) in labels.iter().enumerate() {
            let point_start = i * histogram_size;
            let point = &data[point_start..point_start + histogram_size];
            for (c, &p) in new_centroids[label as usize].iter_mut().zip(point.iter()) {
                *c += p as f32;
            }
        }

        for (centroid, (new_centroid, &size)) in centroids.iter_mut().zip(new_centroids.iter().zip(&cluster_sizes)) {
            if size > 0 {
                for (c, &new_c) in centroid.iter_mut().zip(new_centroid.iter()) {
                    *c = new_c / size as f32;
                }
            } else {
                // If a cluster becomes empty, reinitialize its centroid randomly
                let mut rng = rand::thread_rng();
                let random_point_start = rng.gen_range(0..num_histograms) * histogram_size;
                let random_point = &data[random_point_start..random_point_start + histogram_size];
                *centroid = random_point.iter().map(|&val| val as f32).collect();
            }
        }

        // Only even iterations will be good
        if iter > 0 && iter % 2 != 0 {
            let frobenius_norm = calculate_frobenius_norm(&centroids, &prev_centroids);
            let new_centroids_norm = centroids
                .par_iter()
                .map(|centroid| centroid.par_iter().map(|&x| (x as f64).powi(2)).sum::<f64>())
                .sum::<f64>()
                .sqrt();
            if frobenius_norm / new_centroids_norm < convergence_threshold {
                log::info!("Converged after {} iterations", iter + 1);
                break;
            }
            prev_centroids = centroids.clone();
            if (iter - 1) % 10 == 0 {
                let inertia = calculate_inertia_euclidian(data, histogram_size, &centroids, &labels);
                log::info!("Finished iteration {} with an inertia of {}", iter, inertia);
            }
        }
    }

    let inertia = calculate_inertia_euclidian(data, histogram_size, &centroids, &labels);
    Ok((centroids, labels, inertia))
}

pub fn kmeans_emd_triangle_inequality(
    data: &[u8],
    histogram_size: usize,
    k: usize,
    max_iters: usize,
    convergence_threshold: f64,
) -> Result<(Vec<Vec<f32>>, Vec<u32>, f64), &'static str> {
    if k > data.len() / histogram_size {
        return Err("Number of clusters cannot be greater than the number of data points");
    }
    if max_iters % 2 == 0 {
        return Err("Please use an odd number of iterations");
    }
    let mut centroids = kmeans_plusplus(data, histogram_size, k);  // Assumes updated to work with flat data
    log::info!("initialized kmeans++");
    let num_histograms = data.len() / histogram_size;
    let mut labels: Vec<u32> = vec![0; num_histograms];
    let mut prev_centroids = centroids.clone();
    let mut min_distances = vec![std::f64::MAX; num_histograms];
    for iter in 0..max_iters {
        let cluster_sizes: Vec<AtomicI32> = (0..k).map(|_| AtomicI32::new(0)).collect_vec();

        // Use triangle inequality optimization
        (0..num_histograms).into_par_iter().zip(&mut min_distances).zip(&mut labels).for_each(|((i, min_distance), label)| {
            let point_start = i * histogram_size;
            let point = &data[point_start..point_start + histogram_size];
            let mut thread_rng = rand::thread_rng();
            let reference_centroid_idx = thread_rng.gen_range(0..k);
            let reference_distances: Vec<f64> = centroids.par_iter()
                .map(|c| earth_movers_distance(&c.iter().map(|&v| v as f64).collect(), &centroids[reference_centroid_idx].iter().map(|&v| v as f64).collect()))
                .collect();
            let reference_distance = earth_movers_distance(&point.iter().map(|&v| v as f64).collect(), &centroids[reference_centroid_idx].iter().map(|&v| v as f64).collect());
            let (best_centroid_idx, best_distance) = centroids.par_iter().enumerate()
                .map(|(centroid_idx, centroid)| {
                    if reference_distance + reference_distances[centroid_idx] < *min_distance {
                        let distance = earth_movers_distance(&point.iter().map(|&v| v as f64).collect(), &centroid.iter().map(|&v| v as f64).collect());
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
        let new_centroids: Vec<Vec<f32>> = (0..num_histograms).into_par_iter()
            .zip(&labels)
            .fold(
                || vec![vec![0f32; histogram_size]; k],
                |mut local_centroids, (i, &label)| {
                    let point_start = i * histogram_size;
                    let point = &data[point_start..point_start + histogram_size];
                    for (c, &p) in local_centroids[label as usize].iter_mut().zip(point.iter()) {
                        *c += p as f32;
                    }
                    local_centroids
                }
            )
            .reduce(
                || vec![vec![0f32; histogram_size]; k],
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
                let random_point_start = nested_thread_rng.gen_range(0..num_histograms) * histogram_size;
                let random_point = &data[random_point_start..random_point_start + histogram_size];
                *centroid = random_point.iter().map(|&val| val as f32).collect_vec();
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
            
            if (iter-1) % 10 == 0 {
                let inertia = calculate_inertia(data, histogram_size, &centroids, &labels);
                log::info!("Finished iteration {} with an inertia of {}", iter, inertia);
            }
        }
    }
    let inertia = calculate_inertia(data, histogram_size, &centroids, &labels);
    Ok((centroids, labels, inertia))
}


pub fn kmeans_emd(
    data: &[u8],
    histogram_size: usize,
    k: usize,
    max_iters: usize,
    convergence_threshold: f64,
) -> Result<(Vec<Vec<f32>>, Vec<u32>, f64), &'static str> {
    if k > data.len() / histogram_size {
        return Err("Number of clusters cannot be greater than the number of data points");
    }
    let mut centroids = kmeans_plusplus(data, histogram_size, k); // Ensure this function is also adapted to use flat data.
    println!("initial centroids: {:?}", centroids);
    log::info!("initialized kmeans++");
    let num_histograms = data.len() / histogram_size;
    let mut labels: Vec<u32> = vec![0; num_histograms];
    let mut prev_centroids = centroids.clone();

    for iter in 0..max_iters {
        if iter > 0 && iter % 10 == 0 {
            log::info!("Finished iteration {}", iter);
        }
        let cluster_sizes: Vec<AtomicI32> = (0..k).map(|_| AtomicI32::new(0)).collect();

        (0..num_histograms).into_par_iter().zip(&mut labels).for_each(|(i, label)| {
            let point_start = i * histogram_size;
            let point = &data[point_start..point_start + histogram_size];
            let (best_centroid_idx, _) = centroids.par_iter().enumerate()
                .map(|(centroid_idx, centroid)| {
                    let distance = earth_movers_distance(
                        &point.iter().map(|&v| v as f64).collect(),
                        &centroid.iter().map(|&v| v as f64).collect()
                    );
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

        let new_centroids: Vec<Vec<f32>> = (0..num_histograms).into_par_iter()
            .zip(&labels)
            .fold(
                || vec![vec![0f32; histogram_size]; k],
                |mut local_centroids, (i, &label)| {
                    let point_start = i * histogram_size;
                    let point = &data[point_start..point_start + histogram_size];
                    for (c, &p) in local_centroids[label as usize].iter_mut().zip(point.iter()) {
                        *c += p as f32;
                    }
                    local_centroids
                }
            )
            .reduce(
                || vec![vec![0f32; histogram_size]; k],
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
                let mut rng = rand::thread_rng();
                let random_point_start = rng.gen_range(0..num_histograms) * histogram_size;
                let random_point = &data[random_point_start..random_point_start + histogram_size];
                *centroid = random_point.iter().map(|&val| val as f32).collect();
            }
        });

        let frobenius_norm = calculate_frobenius_norm(&centroids, &prev_centroids);
        let new_centroids_norm = centroids.par_iter()
            .map(|centroid| centroid.iter().map(|&x| (x as f64).powi(2)).sum::<f64>())
            .sum::<f64>()
            .sqrt();
        if frobenius_norm / new_centroids_norm < convergence_threshold {
            log::info!("Converged after {} iterations", iter + 1);
            break;
        }
        prev_centroids = centroids.clone();
        let inertia = calculate_inertia(data, histogram_size, &centroids, &labels); // Ensure inertia calculation uses flat data.
        println!("inertia: {}", inertia);
    }
    let inertia = calculate_inertia(data, histogram_size, &centroids, &labels); // Again, adjust inertia function.
    Ok((centroids, labels, inertia))
}
