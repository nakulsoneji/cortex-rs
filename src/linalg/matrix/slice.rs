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

    pub fn try_map_col<F: Fn(f32) -> f32>(&mut self, c: usize, f: F) -> LinAlgResult<()> {
        let [rows, cols] = self.shape;
        if c >= cols {
            return Err(ValidationError::MatrixIndexOutOfBounds {
                row: 0, col: c, rows, cols,
            }.into());
        }
        unsafe { self.map_col_unchecked(c, f) };
        Ok(())
    }

    pub fn try_map_row<F: Fn(f32) -> f32>(&mut self, r: usize, f: F) -> LinAlgResult<()> {
        let [rows, cols] = self.shape;
        if r >= rows {
            return Err(ValidationError::MatrixIndexOutOfBounds {
                row: r, col: 0, rows, cols,
            }.into());
        }
        unsafe { self.map_row_unchecked(r, f) };
        Ok(())
    }

    pub fn map_col<F: Fn(f32) -> f32>(&mut self, c: usize, f: F) {
        self.try_map_col(c, f).unwrap_or_else(|e| panic!("{}", e));
    }

    pub fn map_row<F: Fn(f32) -> f32>(&mut self, r: usize, f: F) {
        self.try_map_row(r, f).unwrap_or_else(|e| panic!("{}", e));
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

    pub unsafe fn map_col_unchecked<F: Fn(f32) -> f32>(&mut self, c: usize, f: F) {
        for r in 0..self.shape[0] {
            unsafe { *self.get_unchecked_mut(r, c) = f(*self.get_unchecked(r, c)) };
        }
    }

    pub unsafe fn map_row_unchecked<F: Fn(f32) -> f32>(&mut self, r: usize, f: F) {
        for c in 0..self.shape[1] {
            unsafe { *self.get_unchecked_mut(r, c) = f(*self.get_unchecked(r, c)) };
        }
    }
}