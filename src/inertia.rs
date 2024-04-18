use crate::distance::earth_movers_distance;


pub fn calculate_inertia(data: &[Vec<u32>], centroids: &[Vec<u32>], labels: &[usize]) -> f64 {
    return data.iter().enumerate().map(|(i, point)| {
        let centroid = &centroids[labels[i]];
        earth_movers_distance(point, centroid)
    }).sum();
}