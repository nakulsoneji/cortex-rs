use std::sync::Arc;

use crate::prelude::*;

use super::Vector;

impl Vector {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn print(&self) {
        print!("[ ");
        for x in self.data.iter() {
            print!("{:.4} ", (x * 10000.0).round() / 10000.0);
        }
        println!("]");
    }

    pub fn zeros(size: usize) -> Self {
        Vector::from(vec![0; size])
    }

    pub fn ones(size: usize) -> Self {
        Vector::from(vec![1; size])
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &f32 {
        unsafe { self.data.get_unchecked(index) }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&f32> {
        if index < self.len() {
            unsafe { Some(self.get_unchecked(index)) }
        } else {
            None
        }
    }


    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut f32 {
        let data = self.data_mut();
        unsafe { data.get_unchecked_mut(index) }
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut f32> {
        if index < self.len() {
            unsafe { Some(self.get_unchecked_mut(index)) }
        } else {
            None
        }
    }

        pub fn push(&mut self, val: f32) {
        self.data_mut().push(val);
    }

    pub fn extend(&mut self, vals: &[f32]) {
        self.data_mut().extend_from_slice(vals);
    }

    pub fn insert(&mut self, index: usize, val: f32) {
        assert!(index <= self.len(), "{}", ValidationError::VectorIndexOutOfBounds {
            index, len: self.len(),
        });
        self.data_mut().insert(index, val);
    }

    pub fn remove(&mut self, index: usize) -> f32 {
        assert!(index < self.len(), "{}", ValidationError::VectorIndexOutOfBounds {
            index, len: self.len(),
        });
        self.data_mut().remove(index)
    }
}

impl DataSlice for Vector {
    fn as_slice(&self) -> &[f32] { &self.data }
    fn len(&self) -> usize { self.data.len() }
}

impl LinearStorage for Vector {
    fn data_arc(&self) -> &Arc<Vec<f32>> { &self.data }
    fn data_arc_mut(&mut self) -> &mut Arc<Vec<f32>> { &mut self.data }
    fn len(&self) -> usize { self.len() }
    fn assert_same_shape(&self, other: &Self) {
        assert!(self.len() == other.len(), "{}", ValidationError::VectorDimensionMismatch { 
            a: self.len(),
            b: other.len()
        });
    }
}