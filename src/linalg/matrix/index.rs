use std::ops::{Index, IndexMut};
use crate::prelude::*;

impl Index<[usize; 2]> for Matrix {
    type Output = f32;

    #[inline]
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let [row, col] = index;
        let [rows, cols] = self.shape;
        self.get(row, col).unwrap_or_else(|| panic!("{}", ValidationError::MatrixIndexOutOfBounds {
            row,
            col,
            rows,
            cols,
        }))
    }
}

impl IndexMut<[usize; 2]> for Matrix {
    #[inline]
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let [row, col] = index;
        let [rows, cols] = self.shape;
        self.get_mut(row, col).unwrap_or_else(|| panic!("{}", ValidationError::MatrixIndexOutOfBounds {
            row,
            col,
            rows,
            cols
        }))
    }
}