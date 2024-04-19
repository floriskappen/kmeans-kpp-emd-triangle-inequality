use crate::distance::earth_movers_distance_centroid;


pub fn calculate_inertia(data: &[Vec<u8>], centroids: &[Vec<f32>], labels: &[u32]) -> f64 {
    let result = data.iter().enumerate().map(|(i, point)| {
        let centroid = &centroids[labels[i] as usize];
        let emd = earth_movers_distance_centroid(point, centroid);
        return emd;
    }).sum();

    return result;
}