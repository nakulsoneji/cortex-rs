use crate::prelude::*;

impl Vector {
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Vector {
        Vector::from(self.data.iter().copied().map(f).collect::<Vec<f32>>())
    }

    pub fn apply<F: Fn(f32) -> f32>(&mut self, f: F) {
        self.data_mut().iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn clip(&self, min: f32, max: f32) -> Vector {
        self.map(|v| v.clamp(min, max))
    }

    pub fn clip_inplace(&mut self, min: f32, max: f32) {
        self.apply(|v| v.clamp(min, max));
    }

    pub fn normalize(&self) -> Vector {
        let max = self.max();
        let min = self.min();
        let range = max - min;
        if range == 0.0 { return self.clone(); }
        self.map(|v| (v - min) / range)
    }

    pub fn normalize_inplace(&mut self) {
        let max = self.max();
        let min = self.min();
        let range = max - min;
        if range == 0.0 { return; }
        self.apply(|v| (v - min) / range);
    }

    pub fn standardize(&self) -> Vector {
        let mean = self.sum() / self.len() as f32;
        let std = (self.map(|v| (v - mean).powi(2)).sum() / self.len() as f32).sqrt();
        if std == 0.0 { return self.clone(); }
        self.map(|v| (v - mean) / std)
    }

    pub fn standardize_inplace(&mut self) {
        let mean = self.sum() / self.len() as f32;
        let std = (self.map(|v| (v - mean).powi(2)).sum() / self.len() as f32).sqrt();
        if std == 0.0 { return; }
        self.apply(|v| (v - mean) / std);
    }
}