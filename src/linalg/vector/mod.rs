#[macro_use]
pub mod macros;
pub mod index;
pub mod from;
pub mod core;
pub mod math;
pub mod reductions;
pub mod ops;

use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub data: Arc<Vec<f32>>
}

