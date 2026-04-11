#[macro_use]
mod macros;
pub mod index;
pub mod core;
pub mod from;
pub mod decompositions;
pub mod reductions;
pub mod math;
pub mod slice;

use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub data: Arc<Vec<f32>>,
    pub shape: [usize; 2],
    pub strides: [usize; 2]
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Axis {
    Row = 0,
    Col = 1,
}
