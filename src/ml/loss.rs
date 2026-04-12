use crate::prelude::*;

#[derive(Clone)]
pub enum Loss {
    MSE,
    CrossEntropy,
}

impl Loss {
    pub fn compute(&self, predicted: &Vector, actual: &Vector) -> f32 {
        match self {
            Loss::MSE => {
                let diff = predicted - actual;
                diff.dot(&diff) / predicted.len() as f32
            },
            Loss::CrossEntropy => {
                let eps = 1e-7f32;
                actual.iter().zip(predicted.iter())
                    .map(|(a, p)| -a * (p + eps).ln() - (1.0 - a) * (1.0 - p + eps).ln())
                    .sum::<f32>()
            }
        }
    }

    pub fn gradient(&self, predicted: &Vector, actual: &Vector) -> Vector {
        match self {
            Loss::MSE => {
                let n = predicted.len() as f32;
                (predicted.clone() - actual) * (2.0 / n)
            },
            Loss::CrossEntropy => {
                // combined softmax + cross entropy gradient
                predicted.clone() - actual
            }
        }
    }
}