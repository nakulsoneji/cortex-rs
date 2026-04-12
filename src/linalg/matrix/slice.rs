use crate::prelude::*;

impl Matrix {
    pub fn col(&self, c: usize) -> Vector {
        assert!(c < self.shape[1], "{}", ValidationError::MatrixIndexOutOfBounds {
            row: 0, col: c,
            rows: self.shape[0], cols: self.shape[1],
        });
        let data = (0..self.shape[0])
            .map(|r| unsafe { *self.get_unchecked(r, c) })
            .collect::<Vec<f32>>();
        Vector::from(data)
    }
    pub fn col_iter_mut(&mut self, c: usize) -> impl Iterator<Item = &mut f32> {
        assert!(c < self.shape[1], "{}", ValidationError::MatrixIndexOutOfBounds {
            row: 0, col: c, rows: self.shape[0], cols: self.shape[1],
        });
        let rows = self.shape[0];
        let row_stride = self.strides[0]; // stride between rows
        let col_stride = self.strides[1]; // stride between cols
        let start = c * col_stride;       // start of this column
        let data = self.data_mut();
        (0..rows).map(move |r| unsafe { &mut *data.as_mut_ptr().add(start + r * row_stride) })
    }

    pub fn row_iter_mut(&mut self, r: usize) -> impl Iterator<Item = &mut f32> {
        assert!(r < self.shape[0], "{}", ValidationError::MatrixIndexOutOfBounds {
            row: r, col: 0, rows: self.shape[0], cols: self.shape[1],
        });
        let cols = self.shape[1];
        let col_stride = self.strides[1]; // stride between cols
        let row_stride = self.strides[0]; // stride between rows
        let start = r * row_stride;       // start of this row
        let data = self.data_mut();
        (0..cols).map(move |c| unsafe { &mut *data.as_mut_ptr().add(start + c * col_stride) })
    }

    pub fn apply_col<F: Fn(f32) -> f32>(&mut self, c: usize, f: F) {
        self.col_iter_mut(c).for_each(|x| *x = f(*x));
    }

    pub fn apply_row<F: Fn(f32) -> f32>(&mut self, r: usize, f: F) {
        self.row_iter_mut(r).for_each(|x| *x = f(*x));
    }
    
    pub fn row(&self, r: usize) -> Vector {
        assert!(r < self.shape[0], "{}", ValidationError::MatrixIndexOutOfBounds {
            row: r, col: 0,
            rows: self.shape[0], cols: self.shape[1],
        });
        let data = (0..self.shape[1])
            .map(|c| unsafe { *self.get_unchecked(r, c) })
            .collect::<Vec<f32>>();
        Vector::from(data)
    }

    pub fn try_set_row(&mut self, row: usize, values: &[f32]) -> LinAlgResult<()> {
        let [rows, cols] = self.shape;

        if row >= rows {
            return Err(ValidationError::MatrixIndexOutOfBounds {
                row,
                col: 0,
                rows,
                cols,
            }
            .into());
        }

        if values.len() != cols {
            return Err(ValidationError::MatrixVectorDimensionMismatch {
                rows_a: 1,
                cols_a: cols,
                b: values.len(),
            }
            .into());
        }

        unsafe { self.set_row_unchecked(row, values) };
        Ok(())
    }

    pub fn set_row(&mut self, row: usize, values: &[f32]) {
        self.try_set_row(row, values)
            .unwrap_or_else(|e| panic!("{}", e));
    }

    pub unsafe fn set_row_unchecked(&mut self, row: usize, values: &[f32]) {
        for c in 0..self.shape[1] {
            unsafe {
                *self.get_unchecked_mut(row, c) = *values.get_unchecked(c);
            }
        }
    }

    pub fn try_set_col(&mut self, col: usize, values: &[f32]) -> LinAlgResult<()> {
        let [rows, cols] = self.shape;

        if col >= cols {
            return Err(ValidationError::MatrixIndexOutOfBounds {
                row: 0,
                col,
                rows,
                cols,
            }
            .into());
        }

        if values.len() != rows {
            return Err(ValidationError::MatrixVectorDimensionMismatch {
                rows_a: rows,
                cols_a: 1,
                b: values.len(),
            }
            .into());
        }

        unsafe { self.set_col_unchecked(col, values) };
        Ok(())
    }

    pub fn set_col(&mut self, col: usize, values: &[f32]) {
        self.try_set_col(col, values)
            .unwrap_or_else(|e| panic!("{}", e));
    }

    pub unsafe fn set_col_unchecked(&mut self, col: usize, values: &[f32]) {
        for r in 0..self.shape[0] {
            unsafe {
                *self.get_unchecked_mut(r, col) = *values.get_unchecked(r);
            }
        }
    }

    pub fn row_iter(&self, r: usize) -> impl Iterator<Item = f32> + '_ {
        assert!(r < self.shape[0], "{}", ValidationError::MatrixIndexOutOfBounds {
            row: r, col: 0,
            rows: self.shape[0], cols: self.shape[1],
        });
        (0..self.shape[1]).map(move |c| unsafe { *self.get_unchecked(r, c) })
    }

    pub fn col_iter(&self, c: usize) -> impl Iterator<Item = f32> + '_ {
        assert!(c < self.shape[1], "{}", ValidationError::MatrixIndexOutOfBounds {
            row: 0, col: c,
            rows: self.shape[0], cols: self.shape[1],
        });
        (0..self.shape[0]).map(move |r| unsafe { *self.get_unchecked(r, c) })
    }

    pub fn row_view(&self, r: usize) -> VectorView<'_> {
        assert!(r < self.shape[0], "{}", ValidationError::MatrixIndexOutOfBounds {
            row: r, col: 0,
            rows: self.shape[0], cols: self.shape[1],
        });
        let start = r * self.strides[0];
        VectorView {
            data: &self.data[start..],
            stride: self.strides[1],
            len: self.shape[1],
        }
    }

    pub fn col_view(&self, c: usize) -> VectorView<'_> {
        assert!(c < self.shape[1], "{}", ValidationError::MatrixIndexOutOfBounds {
            row: 0, col: c,
            rows: self.shape[0], cols: self.shape[1],
        });
        let start = c * self.strides[1];
        VectorView {
            data: &self.data[start..],
            stride: self.strides[0],
            len: self.shape[0],
        }
    }

    pub fn map_col<F: Fn(f32) -> f32>(&self, c: usize, f: F) -> Vector {
        Vector::from(self.col_iter(c).map(f).collect::<Vec<f32>>())
    }

    pub fn map_row<F: Fn(f32) -> f32>(&self, r: usize, f: F) -> Vector {
        Vector::from(self.row_iter(r).map(f).collect::<Vec<f32>>())
    }
}