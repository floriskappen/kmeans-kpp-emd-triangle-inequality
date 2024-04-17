use std::fmt::Debug;


pub trait Centroid: Sized + Clone + Debug + Default {
    fn update_centroid(&mut self, data_point: &Self);
    fn finalize_centroid(&mut self, count: usize);
}


impl Centroid for Vec<u32> {
    fn update_centroid(&mut self, data_point: &Self) {
        if self.is_empty() {
            *self = vec![0; data_point.len()];
        }

        for (a, b) in self.iter_mut().zip(data_point.iter()) {
            *a += *b;
        }
    }

    fn finalize_centroid(&mut self, count: usize) {
        for a in self.iter_mut() {
            *a /= count as u32;
        }
    }
}