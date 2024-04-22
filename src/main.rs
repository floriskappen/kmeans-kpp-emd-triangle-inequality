mod proto {
    include!("proto/build/_.rs");
}
mod logger;
mod load;
mod distance;
mod initialization;
mod algorithm;
mod inertia;

use load::{load_potential_aware_data, save_data, HistogramLoader};
use algorithm::{kmeans_emd, kmeans_emd_triangle_inequality, kmeans_euclidian, kmeans_euclidian_triangle_inequality};
use crate::{algorithm::{kmeans_emd_f64, kmeans_emd_triangle_inequality_f64}, load::save_data_f64, logger::init_logger};

fn kmeans(data: &Vec<u8>, histogram_size: usize, round: usize, k: usize, max_iters: usize, convergence_threshold: f64, num_initializations: usize, triangle_inequality: bool, euclidian_distance: bool, only_save_best: bool) /*-> Result<(Vec<Vec<u32>>, Vec<usize>), &'static str>*/ {
    let mut best_centroids: Vec<Vec<f32>> = vec![];
    let mut best_labels: Vec<u32> = vec![];
    let mut best_inertia = std::f64::MAX;
    let mut inertia_per_initialization: Vec<f64> = vec![];
    let mut best_initialization_index = 0;

    for initialization_index in 0..num_initializations {
        let centroids;
        let labels;
        let calculated_inertia;
        if triangle_inequality {
            if euclidian_distance {
                log::info!("Starting KMeans with L2 (Euclidian) distance & triangle inequality");
                (centroids, labels, calculated_inertia) = kmeans_euclidian_triangle_inequality(
                    data,
                    histogram_size,
                    k,
                    max_iters,
                    convergence_threshold
                ).expect("error during kmeans");
            } else {
                log::info!("Starting KMeans with Earth Mover's Distance & triangle inequality");
                (centroids, labels, calculated_inertia) = kmeans_emd_triangle_inequality(
                    data,
                    histogram_size,
                    k,
                    max_iters,
                    convergence_threshold
                ).expect("error during kmeans");
            }
        } else {
            if euclidian_distance {
                log::info!("Starting KMeans with L2 (Euclidian) distance");
                (centroids, labels, calculated_inertia) = kmeans_euclidian(
                    data,
                    histogram_size,
                    k,
                    max_iters,
                    convergence_threshold
                ).expect("error during kmeans");
            } else {
                log::info!("Starting KMeans with Earth Mover's Distance");
                (centroids, labels, calculated_inertia) = kmeans_emd(
                    data,
                    histogram_size,
                    k,
                    max_iters,
                    convergence_threshold
                ).expect("error during kmeans");
            }
        }

        if !only_save_best {
            save_data(labels, centroids, round, initialization_index).expect("Error saving labels... :(");
            if calculated_inertia < best_inertia {
                best_inertia = calculated_inertia;
                best_initialization_index = initialization_index;
            };
        } else {
            if calculated_inertia < best_inertia {
                best_inertia = calculated_inertia;
                best_initialization_index = initialization_index;
                // Keep track of the best one
                if only_save_best {
                    best_labels = labels;
                    best_centroids = centroids;
                }
            };
        }
        inertia_per_initialization.push(calculated_inertia);

        log::info!("Finished KMeans for initialization #{} - Inertia: {}", initialization_index, calculated_inertia);

    }

    // println!("labels: {:?}", best_labels);

    log::info!("Finished all initializations!");
    log::info!("Inertia per initialization: {:?}", inertia_per_initialization);
    log::info!("Best initialization is index #{} with {} inertia", best_initialization_index, best_inertia);

    if only_save_best {
        save_data(best_labels, best_centroids, round, best_initialization_index).expect("Error saving labels... :(");
    }
}

// High precision version of kmeans
fn kmeans_f64(data: &Vec<f64>, histogram_size: usize, round: usize, k: usize, max_iters: usize, convergence_threshold: f64, num_initializations: usize, triangle_inequality: bool, only_save_best: bool) {
    let mut best_centroids: Vec<Vec<f64>> = vec![];
    let mut best_labels: Vec<u32> = vec![];
    let mut best_inertia = std::f64::MAX;
    let mut inertia_per_initialization: Vec<f64> = vec![];
    let mut best_initialization_index = 0;

    for initialization_index in 0..num_initializations {
        let centroids;
        let labels;
        let calculated_inertia;
        if triangle_inequality {
            log::info!("Starting KMeans with Earth Mover's Distance & triangle inequality");
            (centroids, labels, calculated_inertia) = kmeans_emd_triangle_inequality_f64(
                data,
                histogram_size,
                k,
                max_iters,
                convergence_threshold
            ).expect("error during kmeans");
        } else {
            log::info!("Starting KMeans with Earth Mover's Distance");
            (centroids, labels, calculated_inertia) = kmeans_emd_f64(
                data,
                histogram_size,
                k,
                max_iters,
                convergence_threshold
            ).expect("error during kmeans");
        }

        if !only_save_best {
            save_data_f64(labels, centroids, round, initialization_index).expect("Error saving labels... :(");
            if calculated_inertia < best_inertia {
                best_inertia = calculated_inertia;
                best_initialization_index = initialization_index;
            };
        } else {
            if calculated_inertia < best_inertia {
                best_inertia = calculated_inertia;
                best_initialization_index = initialization_index;
                // Keep track of the best one
                if only_save_best {
                    best_labels = labels;
                    best_centroids = centroids;
                }
            };
        }
        inertia_per_initialization.push(calculated_inertia);

        log::info!("Finished KMeans for initialization #{} - Inertia: {}", initialization_index, calculated_inertia);

    }

    // println!("labels: {:?}", best_labels);

    log::info!("Finished all initializations!");
    log::info!("Inertia per initialization: {:?}", inertia_per_initialization);
    log::info!("Best initialization is index #{} with {} inertia", best_initialization_index, best_inertia);

    if only_save_best {
        save_data_f64(best_labels, best_centroids, round, best_initialization_index).expect("Error saving labels... :(");
    }
}

fn main() {
    init_logger().expect("Failed to initialize logger");

    let test_dataset = vec![
        vec![34, 118],
        vec![34, 118],
        vec![34, 118],
        vec![34, 118],

        vec![40, 74],
        vec![40, 74],
        vec![40, 74],
        vec![40, 74],

        vec![37, 122],
        vec![37, 122],
        vec![37, 122],
        vec![37, 122],
    ];

    // let round = 3;
    // let histogram_size = 50;
    // let histogram_loader = HistogramLoader::new(round, histogram_size).expect("Failed to initialize HandLoader");

    // kmeans(
    //     &histogram_loader.histograms,
    //     histogram_size,
    //     round,
    //     200,
    //     250,
    //     0.0001,
    //     5,
    //     true,
    //     true,
    //     false
    // );


    let round = 1;
    let histogram_size = 200;

    let potential_aware_emd_matrix = load_potential_aware_data().expect("Failed to load potential aware EMD matrix data");
    kmeans_f64(
        &potential_aware_emd_matrix,
        histogram_size,
        round,
        200,
        251,
        0.0001,
        5,
        true,
        false
    );
}
