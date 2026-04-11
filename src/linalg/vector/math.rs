use crate::prelude::*;

impl Vector {
    pub fn sum(&self) -> f32 {
        let ones = vec![1.0; self.len()];
        unsafe {
            blas::sdot(
                self.len() as i32,
                &self.data, 1,
                &ones, 1,
            )
        }
    }

    pub fn abs_sum(&self) -> f32 {
        unsafe {
            blas::sasum(
                self.len() as i32,
                &self.data, 1,
            )
        }
    }

    pub fn l2_norm(&self) -> f32 {
        unsafe {
            blas::snrm2(
                self.len() as i32,
                &self.data,
                1
            )
        }
    }

    pub fn l1_norm(&self) -> f32 {
        unsafe {
            blas::sasum(
                self.len() as i32,
                &self.data,
                1
            )
        }
    }

    pub fn inf_norm_index(&self) -> usize {
        let idx = unsafe {
            blas::isamax(
                self.len() as i32, 
                &self.data, 
                1)
        };

        if idx == 0 { 0 } else { (idx - 1) as usize }
    }

    pub fn inf_norm(&self) -> f32 {
        assert!(self.len() != 0, "{}", ValidationError::VectorEmpty);

        let index = self.inf_norm_index();
        unsafe { self.get_unchecked(index).abs() }
    }

    pub fn max_abs_element(&self) -> f32 {
        assert!(!self.is_empty(), "{}", ValidationError::VectorEmpty);

        unsafe { *self.get_unchecked(self.inf_norm_index()) }
    }

}