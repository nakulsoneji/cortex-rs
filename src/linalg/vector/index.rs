use std::ops::{Index, IndexMut};
use crate::prelude::*;

impl Index<usize> for Vector {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let len = self.len();
        self.get(index).unwrap_or_else(|| panic!("{}", ValidationError::VectorIndexOutOfBounds {
            index, 
            len
        }))
    }
}

impl IndexMut<usize> for Vector {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let len = self.len();
        self.get_mut(index).unwrap_or_else(|| panic!("{}", ValidationError::VectorIndexOutOfBounds {
            index, 
            len
        }))
    }
}