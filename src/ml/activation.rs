use crate::prelude::*;

#[derive(Clone, Debug)]
pub enum Activation {
    ReLU,
    LeakyReLU(f32), // alpha parameter
    Sigmoid,
    Tanh,
    Linear,
    Softmax,
}

impl Activation {
    pub fn apply(&self, x: &Vector) -> Vector {
        match self {
            Activation::ReLU => x.map(|v| v.max(0.0)),
            Activation::LeakyReLU(alpha) => x.map(|v| if v > 0.0 { v } else { alpha * v }),
            Activation::Sigmoid => x.map(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.map(|v| v.tanh()),
            Activation::Linear => x.clone(),
            Activation::Softmax => {
                let max = x.max();
                let exp = x.map(|v| (v - max).exp());
                let sum = exp.sum();
                exp.map(|v| v / sum)
            }
        }
    }

    pub fn derivative(&self, x: &Vector) -> Vector {
        match self {
            Activation::ReLU => x.map(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyReLU(alpha) => x.map(|v| if v > 0.0 { 1.0 } else { *alpha }),
            Activation::Sigmoid => x.map(|v| {
                let s = 1.0 / (1.0 + (-v).exp());
                s * (1.0 - s)
            }),
            Activation::Tanh => x.map(|v| 1.0 - v.tanh().powi(2)),
            Activation::Linear => Vector::ones(x.len()),
            Activation::Softmax => Vector::ones(x.len()),
        }
    }
}