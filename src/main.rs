mod logger;
mod load;
mod distance;
mod centroid;
mod initialization;
mod algorithm;

use load::{save_data, HistogramLoader};
use algorithm::{kmeans_triangle_inequality, kmeans_default};
use crate::logger::init_logger;

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


    let round = 1;
    let mut histogram_loader = HistogramLoader::new(round).expect("Failed to initialize HandLoader");
    let (centroids, labels) = kmeans_triangle_inequality(
        &histogram_loader.histograms,
        200,
        1000,
        0.01
    ).expect("error during kmeans");
    save_data(&labels, round).expect("Error saving labels... :(");
    // println!("In case labels weren't saved correctly: {:?}", labels);
}