pub fn euclidian_distance(us: &Vec<f64>, them: &Vec<f64>) -> f64 {
    us.iter()
        .zip(them.iter())
        .map(|(a, b)| {
            let diff = *a - *b;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

pub fn earth_movers_distance(us: &Vec<f64>, them: &Vec<f64>) -> f64 {
    let mut cum_us = 0f64;
    let mut cum_them = 0f64;
    let mut emd = 0.0;

    // Iterating over both vectors simultaneously since they are guaranteed to be of the same length
    for (&s, &o) in us.iter().zip(them.iter()) {
        cum_us += s;
        cum_them += o;
        emd += (cum_us - cum_them).abs();
    }

    return emd;
}
