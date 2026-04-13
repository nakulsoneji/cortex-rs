use crate::{ml::activation::Activation, prelude::*};
use rand_distr::{Distribution, Normal};
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Matrix,
    pub bias: Vector,
    pub activation: Activation,

    pub(crate) input: Vector,
    pub(crate) z: Vector,
    pub(crate) a: Vector,
}


impl Layer {
    pub fn forward(&mut self, input: &Vector) -> Vector {
        self.input = input.clone();
        self.z = self.weights.dot(input) + &self.bias;

        self.a = self.activation.apply(&self.z);

        self.a.clone()
    }

    pub fn predict(&self, input: &Vector) -> Vector {
        let z = self.weights.dot(input) + &self.bias;
        self.activation.apply(&z)
    }
    pub fn compute_gradients(&self, delta: &Vector, grad_clip: Option<f32>) -> (Matrix, Vector, Vector) {
        let act_deriv = self.activation.derivative(&self.z);
        let mut delta = delta * act_deriv.clone();

        if let Some(clip) = grad_clip {
            delta.clip_inplace(-clip, clip);
        }

        let prev_delta = self.weights.transpose().dot(&delta);
        let dw = delta.outer(&self.input);
        let db = delta.clone();

        (dw, db, prev_delta)
    }

    pub fn backward(&mut self, delta: &Vector, grad_clip: Option<f32>) -> (Matrix, Vector, Vector) {
        let (dw, db, prev_delta) = self.compute_gradients(delta, grad_clip);
        // note: don't apply here — let FeedForward::backward handle it
        (dw, db, prev_delta)
    }

    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let std_dev = match activation {
            Activation::ReLU | Activation::LeakyReLU(_) => (2.0 / input_size as f32).sqrt(),
            _ => (2.0 / (input_size + output_size) as f32).sqrt(),       // Xavier
        };

        let normal = Normal::new(0.0f32, std_dev).unwrap();
        let mut rng = rand::rng();
        // let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let weights = Matrix::new(
            (0..input_size * output_size)
                .map(|_| normal.sample(&mut rng))
                .collect(),
            [output_size, input_size],
        );

        let bias = Vector::from(vec![0.0f32; output_size]);

        Self {
            weights,
            bias,
            activation,
            input: Vector::from(vec![0.0f32; input_size]),
            z: Vector::from(vec![0.0f32; output_size]),
            a: Vector::from(vec![0.0f32; output_size]),
        }
    }
}
