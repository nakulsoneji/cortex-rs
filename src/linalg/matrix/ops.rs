use crate::prelude::*;

impl Matrix {
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Matrix {
        let data = self.data.iter().copied().map(f).collect::<Vec<f32>>();
        Matrix::new_with_strides(data, self.shape, self.strides)
    }

    pub fn apply<F: Fn(f32) -> f32>(&mut self, f: F) {
        self.data_mut().iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn clip(&self, min: f32, max: f32) -> Matrix {
        self.map(|v| v.clamp(min, max))
    }

    pub fn clip_inplace(&mut self, min: f32, max: f32) {
        self.apply(|v| v.clamp(min, max));
    }
    
    pub fn normalize(&self) -> Matrix {
        let data = self.normalize_vec();
        Matrix::new_with_strides(data, self.shape, self.strides)
    }

    pub fn normalize_inplace(&mut self) {
        let min = self.min();
        let range = self.max() - min;
        if range == 0.0 { return; }
        self.apply(|v| (v - min) / range);
    }

    pub fn standardize(&self) -> Matrix {
        let data = self.standardize_vec();
        Matrix::new_with_strides(data, self.shape, self.strides)
    }

    pub fn standardize_inplace(&mut self) {
        let mean = self.mean();
        let std = self.std_dev();
        if std == 0.0 { return; }
        self.apply(|v| (v - mean) / std);
    }
}