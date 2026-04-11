use std::sync::Arc;
use crate::prelude::*;

use super::Matrix;

impl Matrix {
    pub fn try_new(data: Vec<f32>, shape: [usize; 2]) -> LinAlgResult<Matrix> {
        if data.len() != shape[0] * shape[1] {
            return Err(ValidationError::ShapeSizeMismatch {
                expected: shape[0] * shape[1],
                actual: data.len(),
            })?;
        }

        Ok(Self::new_unchecked(data, shape))
    }

    pub fn new(data: Vec<f32>, shape: [usize; 2]) -> Self {
        Self::try_new(data, shape).expect("Matrix creation failed")
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn new_unchecked(data: Vec<f32>, shape: [usize; 2]) -> Self {
        Self {
            data: Arc::new(data),
            shape,
            strides: [shape[1], 1], // Default Row-Major
        }
    }

    pub fn try_new_with_strides(data: Vec<f32>, shape: [usize; 2], strides: [usize; 2]) -> LinAlgResult<Matrix> {
        if strides[0] == 0 || strides[1] == 0 {
            return Err(ValidationError::InvalidStride {
                stride_0: strides[0],
                stride_1: strides[1],
            })?;
        }

        let required = (shape[0] - 1) * strides[0] + (shape[1] - 1) * strides[1] + 1;
        if data.len() < required {
            return Err(ValidationError::ShapeSizeMismatch {
                expected: required,
                actual: data.len(),
            })?;
        }

        Ok(Self {
            data: Arc::new(data),
            shape,
            strides,
        })
    }

    pub fn new_with_strides(data: Vec<f32>, shape: [usize; 2], strides: [usize; 2]) -> Self {
        Self::try_new_with_strides(data, shape, strides)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    pub unsafe fn new_with_strides_unchecked(data: Vec<f32>, shape: [usize; 2], strides: [usize; 2]) -> Self {
        Self {
            data: Arc::new(data),
            shape,
            strides,
        }
    }

    pub fn zeros(shape: [usize; 2]) -> Self {
        Self::new(vec![0.0; shape[0] * shape[1]], shape)
    }

    pub fn full(val: f32, shape: [usize; 2]) -> Self {
        Self::new( vec![val; shape[0] * shape[1]], shape)
    }

    pub fn eye(dim: usize) -> Self {
        let mut data = vec![0.0; dim * dim];
        let stride = dim + 1;

        for i in 0..dim {
            data[i * stride] = 1.0;
        }

        Self::new(data, [dim, dim])
    }

    pub fn to_contiguous(&self) -> Vec<f32> {
        (0..self.shape[0])
            .flat_map(|r| (0..self.shape[1])
                .map(move |c| unsafe { *self.get_unchecked(r, c) }))
            .collect()
    }

    pub fn to_contiguous_col_major(&self) -> Vec<f32> {
        (0..self.shape[1])
            .flat_map(|c| (0..self.shape[0])
                .map(move |r| unsafe { *self.get_unchecked(r, c) }))
            .collect()
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, r: usize, c: usize) -> &f32 {
        let idx = r * self.strides[0] + c * self.strides[1];
        unsafe { self.data.get_unchecked(idx) }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Option<&f32> {
        if r < self.shape[0] && c < self.shape[1] {
            unsafe { Some(self.get_unchecked(r, c)) }
        } else {
            None
        }
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, r: usize, c: usize) -> &mut f32 {
        let idx = r * self.strides[0] + c * self.strides[1];
        let data = self.data_mut();
        unsafe { data.get_unchecked_mut(idx) }
    }

    #[inline]
    pub fn get_mut(&mut self, r: usize, c: usize) -> Option<&mut f32> {
        if r < self.shape[0] && c < self.shape[1] {
            unsafe { Some(self.get_unchecked_mut(r, c)) }
        } else {
            None
        }
    }

    pub fn print(&self) {
        println!("[");
        for r in 0..self.shape[0] {
            print!("[");
            for c in 0..self.shape[1] {
                if c > 0 { print!(", "); }
                print!("{:.4}", unsafe { self.get_unchecked(r, c) });
            }
            println!("]");
        }
        println!("]");
    }

    pub fn try_push_row<V: AsRef<[f32]>>(&mut self, values: V) -> LinAlgResult<()> {
        let values = values.as_ref();
        if values.len() != self.shape[1] {
            return Err(ValidationError::MatrixVectorDimensionMismatch {
                rows_a: 1, cols_a: self.shape[1], b: values.len(),
            }.into());
        }
        unsafe { self.push_row_unchecked(values) };
        Ok(())
    }

    pub fn push_row<V: AsRef<[f32]>>(&mut self, values: V) {
        self.try_push_row(values).unwrap_or_else(|e| panic!("{}", e));
    }

    pub unsafe fn push_row_unchecked<V: AsRef<[f32]>>(&mut self, values: V) {
        let values = values.as_ref();
        self.data_mut().extend_from_slice(values);
        self.shape[0] += 1;
        self.strides = [self.shape[1], 1];
    }

    pub fn try_push_col<V: AsRef<[f32]>>(&mut self, values: V) -> LinAlgResult<()> {
        let values = values.as_ref();
        if values.len() != self.shape[0] {
            return Err(ValidationError::MatrixVectorDimensionMismatch {
                rows_a: self.shape[0], cols_a: 1, b: values.len(),
            }.into());
        }
        unsafe { self.push_col_unchecked(values) };
        Ok(())
    }

    pub fn push_col<V: AsRef<[f32]>>(&mut self, values: V) {
        self.try_push_col(values).unwrap_or_else(|e| panic!("{}", e));
    }

    pub unsafe fn push_col_unchecked<V: AsRef<[f32]>>(&mut self, values: V) {
        let values = values.as_ref();
        let new_cols = self.shape[1] + 1;
        let mut new_data = Vec::with_capacity(self.shape[0] * new_cols);
        for r in 0..self.shape[0] {
            for c in 0..self.shape[1] {
                new_data.push(unsafe { *self.get_unchecked(r, c) });
            }
            new_data.push(values[r]);
        }
        *self.data_mut() = new_data;
        self.shape[1] = new_cols;
        self.strides = [new_cols, 1];
    }
}


impl LinearStorage for Matrix {
    fn data_arc(&self) -> &Arc<Vec<f32>> { &self.data }
    fn data_arc_mut(&mut self) -> &mut Arc<Vec<f32>> { &mut self.data }
    fn len(&self) -> usize { self.data.len() }
    fn assert_same_shape(&self, other: &Self) { 
        assert!(self.shape == other.shape, "{}", ValidationError::MatrixDimensionMismatch {
            rows_a: self.shape[0],
            cols_a: self.shape[1],
            rows_b: other.shape[0],
            cols_b: other.shape[1],
        });
 
    }
}