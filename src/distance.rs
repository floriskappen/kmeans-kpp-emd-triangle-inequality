pub trait Distance<T> {
    fn euclidian_distance(&self, other: &T) -> f64;
    fn earth_movers_distance(&self, other: &T) -> f64;
}

pub fn euclidian_distance(us: &Vec<u32>, them: &Vec<u32>) -> f64 {
    us.iter()
        .zip(them.iter())
        .map(|(a, b)| {
            let diff = *a as i32 - *b as i32;
            (diff * diff) as f64
        })
        .sum::<f64>()
        .sqrt()
}

pub fn earth_movers_distance(us: &Vec<u8>, them: &Vec<u8>) -> f64 {
    let mut cum_us = 0i64;
    let mut cum_them = 0i64;
    let mut emd = 0.0;

    // Iterating over both vectors simultaneously since they are guaranteed to be of the same length
    for (&s, &o) in us.iter().zip(them.iter()) {
        cum_us += s as i64;
        cum_them += o as i64;
        emd += (cum_us - cum_them).abs() as f64;
    }

    return emd;
}

pub fn earth_movers_distance_centroid(us: &Vec<u8>, them: &Vec<f32>) -> f64 {
    let mut cum_us = 0f64;
    let mut cum_them = 0f64;
    let mut emd = 0.0;

    // Iterating over both vectors simultaneously since they are guaranteed to be of the same length
    for (&s, &o) in us.iter().zip(them.iter()) {
        cum_us += s as f64;
        cum_them += o as f64;
        emd += (cum_us - cum_them).abs() as f64;
    }

    return emd;
}

pub fn earth_movers_distance_centroid_centroid(us: &Vec<f32>, them: &Vec<f32>) -> f64 {
    let mut cum_us = 0f64;
    let mut cum_them = 0f64;
    let mut emd = 0.0;

    // Iterating over both vectors simultaneously since they are guaranteed to be of the same length
    for (&s, &o) in us.iter().zip(them.iter()) {
        cum_us += s as f64;
        cum_them += o as f64;
        emd += (cum_us - cum_them).abs() as f64;
    }

    return emd;
}
