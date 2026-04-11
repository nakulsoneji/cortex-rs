use std::sync::Arc;

use crate::linalg::Matrix;

impl From<&[&[f32]]> for Matrix {
    fn from(nested: &[&[f32]]) -> Self {
        let rows = nested.len();
        if rows == 0 {
            return Matrix::zeros([0, 0]);
        }

        let cols = nested[0].len();

        let mut data = Vec::with_capacity(rows * cols);

        for row in nested {
            assert_eq!(row.len(), cols, "All rows must have the same number of columns");
            data.extend_from_slice(row);
        }

        Matrix {
            data: Arc::new(data),
            shape: [rows, cols],
            strides: [cols, 1],
        }
    }
}