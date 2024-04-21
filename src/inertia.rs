use crate::distance::{earth_movers_distance, euclidian_distance};


pub fn calculate_inertia(data: &[Vec<u8>], centroids: &[Vec<f32>], labels: &[u32]) -> f64 {
    let result = data.iter().enumerate().map(|(i, point)| {
        let centroid = &centroids[labels[i] as usize];
        let emd = earth_movers_distance(&point.iter().map(|&v| v as f64).collect(), &centroid.iter().map(|&v| v as f64).collect());
        return emd;
    }).sum();

    return result;
}

pub fn calculate_inertia_euclidian(data: &[Vec<u8>], centroids: &[Vec<f32>], labels: &[u32]) -> f64 {
    let result = data.iter().enumerate().map(|(i, point)| {
        let centroid = &centroids[labels[i] as usize];
        let emd = euclidian_distance(&point.iter().map(|&v| v as f64).collect(), &centroid.iter().map(|&v| v as f64).collect());
        return emd;
    }).sum();

    return result;
}
