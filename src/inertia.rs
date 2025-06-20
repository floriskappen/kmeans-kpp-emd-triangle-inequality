use crate::distance::{earth_movers_distance, euclidian_distance};


pub fn calculate_inertia_emd(data: &[u8], histogram_size: usize, centroids: &[Vec<f64>], labels: &[u16]) -> f64 {
    let result = labels.iter().enumerate().map(|(i, &label)| {
        let point_start = i * histogram_size;
        let point = &data[point_start..point_start + histogram_size];
        let centroid = &centroids[label as usize];
        let emd = earth_movers_distance(
            &point.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
            &centroid
        );
        emd
    }).sum();

    result
}

pub fn calculate_inertia_emd_f64(data: &[f64], histogram_size: usize, centroids: &[Vec<f64>], labels: &[u16]) -> f64 {
    let result = labels.iter().enumerate().map(|(i, &label)| {
        let point_start = i * histogram_size;
        let point = &data[point_start..point_start + histogram_size].to_vec();
        let centroid = &centroids[label as usize];
        let emd = earth_movers_distance(
            &point,
            &centroid
        );
        emd
    }).sum();

    result
}

pub fn calculate_inertia_euclidian(data: &[u8], histogram_size: usize, centroids: &[Vec<f64>], labels: &[u16]) -> f64 {
    let result = labels.iter().enumerate().map(|(i, &label)| {
        let point_start = i * histogram_size;
        let point = &data[point_start..point_start + histogram_size];
        let centroid = &centroids[label as usize];
        let emd = euclidian_distance(
            &point.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
            &centroid.iter().map(|&v| v as f64).collect::<Vec<f64>>()
        );
        emd
    }).sum();

    result
}
