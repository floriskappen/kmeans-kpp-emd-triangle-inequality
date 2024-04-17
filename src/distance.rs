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

pub fn earth_movers_distance(us: &Vec<u32>, them: &Vec<u32>) -> f64 {
    // Assume self and other are of the same length.
    let mut cum_us = 0u32;
    let mut cum_them = 0u32;
    let mut emd = 0.0;

    // Iterating over both vectors simultaneously since they are guaranteed to be of the same length
    for (&s, &o) in us.iter().zip(them.iter()) {
        cum_us += s;
        cum_them += o;
        emd += (cum_us as i64 - cum_them as i64).abs() as f64;
    }

    emd
}
