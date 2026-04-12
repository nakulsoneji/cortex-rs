use crate::ml::{layer::Layer, loss::Loss};
use crate::prelude::*;

pub mod core;

pub struct FeedForward {
    pub layers: Vec<Layer>,
    pub loss: Loss,
    pub optimizer: Optimizer,
    pub weight_clip: Option<f32>,  // None = no clipping, Some(v) = clip to [-v, v]
    pub grad_clip: Option<f32>,
}


