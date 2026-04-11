use std::{ops::Deref, sync::Arc};

use super::Vector;

impl From<Vec<f32>> for Vector {
    fn from(value: Vec<f32>) -> Vector {
        Vector {
            data: Arc::new(value),
        }
    }
}

impl From<Vec<i32>> for Vector {
    fn from(value: Vec<i32>) -> Vector {
        Vector::from(value
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>()
        )
    }
}

impl<const N: usize> From<[f32; N]> for Vector {
    fn from(value: [f32; N]) -> Self {
        Vector::from(Vec::from(value))
    }
}

impl From<&[f32]> for Vector {
    fn from(value: &[f32]) -> Self {
        Vector::from(Vec::from(value))
    }
}

impl<const N: usize> From<Box<[f32; N]>> for Vector {
    fn from(value: Box<[f32; N]>) -> Self {
        Vector::from(Vec::from(*value))
    }
}


impl<const N: usize> From<[i32; N]> for Vector {
    fn from(value: [i32; N]) -> Self {
        Vector::from(value
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>()
        )
    }
}


impl Deref for Vector {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.data
    }
}
