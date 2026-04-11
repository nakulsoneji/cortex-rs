use crate::prelude::*;

// ============================================================
// vDSP
// ============================================================

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn vDSP_dotpr(__A: *const f32, __IA: i64, __B: *const f32, __IB: i64, __C: *mut f32, __N: u64);
}

// ============================================================
// Helpers
// ============================================================

fn blas_layout(strides: [usize; 2]) -> u8 {
    if strides[1] == 1 { b'T' } else { b'N' }
}

#[inline]
fn compute_dot<A: DataSlice, B: DataSlice>(a: &A, b: &B) -> f32 {
    #[cfg(target_os = "macos")]
    {
        let mut result = 0.0f32;
        unsafe {
            vDSP_dotpr(
                a.as_slice().as_ptr(), a.stride() as i64,
                b.as_slice().as_ptr(), b.stride() as i64,
                &mut result,
                a.len() as u64,
            );
        }
        return result;
    }
    #[cfg(not(target_os = "macos"))]
    unsafe {
        blas::sdot(
            a.len() as i32,
            a.as_slice(), a.stride() as i32,
            b.as_slice(), b.stride() as i32,
        )
    }
}

#[inline]
fn gemv_mat_vec(lhs: &Matrix, rhs_slice: &[f32], rhs_stride: i32) -> Vector {
    let m = lhs.shape[0] as i32;
    let n = lhs.shape[1] as i32;
    let mut result = vec![0.0f32; lhs.shape[0]];
    let lhs_op = blas_layout(lhs.strides);
    let (blas_m, blas_n, lda) = if lhs_op == b'T' {
        (n, m, n)
    } else {
        (m, n, m)
    };
    unsafe {
        blas::sgemv(lhs_op, blas_m, blas_n, 1.0, &lhs.data, lda, rhs_slice, rhs_stride, 0.0, &mut result, 1);
    }
    Vector::from(result)
}

#[inline]
fn gemv_vec_mat(lhs_slice: &[f32], lhs_stride: i32, rhs: &Matrix) -> Vector {
    let m = rhs.shape[0] as i32;
    let n = rhs.shape[1] as i32;
    let mut result = vec![0.0f32; rhs.shape[1]];
    let rhs_op = blas_layout(rhs.strides);
    let op = if rhs_op == b'T' { b'N' } else { b'T' };
    let (blas_m, blas_n, lda) = if rhs_op == b'T' {
        (n, m, n)
    } else {
        (m, n, m)
    };
    unsafe {
        blas::sgemv(op, blas_m, blas_n, 1.0, &rhs.data, lda, lhs_slice, lhs_stride, 0.0, &mut result, 1);
    }
    Vector::from(result)
}

#[inline]
fn gemm(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    let m = lhs.shape[0] as i32;
    let k = lhs.shape[1] as i32;
    let n = rhs.shape[1] as i32;
    let mut result = vec![0.0f32; (m * n) as usize];
    let lhs_op = blas_layout(lhs.strides);
    let rhs_op = blas_layout(rhs.strides);
    let lda_lhs = if lhs_op == b'T' { k } else { m };
    let lda_rhs = if rhs_op == b'T' { n } else { k };
    unsafe {
        blas::sgemm(
            lhs_op, rhs_op,
            m, n, k,
            1.0,
            &lhs.data, lda_lhs,
            &rhs.data, lda_rhs,
            0.0,
            &mut result, m,
        );
    }
    Matrix::new_with_strides(result, [m as usize, n as usize], [1, m as usize])
}

// ============================================================
// Matrix · DataSlice
// ============================================================

impl<T: DataSlice> Dot<T> for Matrix {
    type Output = Vector;
    fn assert_dot_compat(&self, rhs: &T) {
        assert!(self.shape[1] == rhs.len(), "{}", ValidationError::MatrixVectorDimensionMismatch {
            rows_a: self.shape[0], cols_a: self.shape[1], b: rhs.len(),
        });
    }
    fn try_dot(&self, rhs: T) -> LinAlgResult<Vector> {
        self.assert_dot_compat(&rhs);
        Ok(gemv_mat_vec(self, rhs.as_slice(), rhs.stride() as i32))
    }
    unsafe fn dot_unchecked(&self, rhs: T) -> Vector {
        gemv_mat_vec(self, rhs.as_slice(), rhs.stride() as i32)
    }
}

impl<T: DataSlice> Dot<T> for &Matrix {
    type Output = Vector;
    fn assert_dot_compat(&self, rhs: &T) { (**self).assert_dot_compat(rhs) }
    fn try_dot(&self, rhs: T) -> LinAlgResult<Vector> { (**self).try_dot(rhs) }
    unsafe fn dot_unchecked(&self, rhs: T) -> Vector { unsafe { (**self).dot_unchecked(rhs) } }
}

// ============================================================
// DataSlice · Matrix
// ============================================================

impl<T: DataSlice> Dot<Matrix> for T {
    type Output = Vector;
    fn assert_dot_compat(&self, rhs: &Matrix) {
        assert!(self.len() == rhs.shape[0], "{}", ValidationError::MatrixVectorDimensionMismatch {
            rows_a: rhs.shape[0], cols_a: rhs.shape[1], b: self.len(),
        });
    }
    fn try_dot(&self, rhs: Matrix) -> LinAlgResult<Vector> {
        self.assert_dot_compat(&rhs);
        Ok(gemv_vec_mat(self.as_slice(), self.stride() as i32, &rhs))
    }
    unsafe fn dot_unchecked(&self, rhs: Matrix) -> Vector {
        gemv_vec_mat(self.as_slice(), self.stride() as i32, &rhs)
    }
}

impl<T: DataSlice> Dot<&Matrix> for T {
    type Output = Vector;
    fn assert_dot_compat(&self, rhs: &&Matrix) {
        assert!(self.len() == rhs.shape[0], "{}", ValidationError::MatrixVectorDimensionMismatch {
            rows_a: rhs.shape[0], cols_a: rhs.shape[1], b: self.len(),
        });
    }
    fn try_dot(&self, rhs: &Matrix) -> LinAlgResult<Vector> {
        self.assert_dot_compat(&rhs);
        Ok(gemv_vec_mat(self.as_slice(), self.stride() as i32, rhs))
    }
    unsafe fn dot_unchecked(&self, rhs: &Matrix) -> Vector {
        gemv_vec_mat(self.as_slice(), self.stride() as i32, rhs)
    }
}

// ============================================================
// Matrix · Matrix
// ============================================================

impl Dot<Matrix> for Matrix {
    type Output = Matrix;
    fn assert_dot_compat(&self, rhs: &Matrix) {
        assert!(self.shape[1] == rhs.shape[0], "{}", ValidationError::MatrixDimensionMismatch {
            rows_a: self.shape[0], cols_a: self.shape[1],
            rows_b: rhs.shape[0], cols_b: rhs.shape[1],
        });
    }
    fn try_dot(&self, rhs: Matrix) -> LinAlgResult<Matrix> {
        self.assert_dot_compat(&rhs);
        Ok(gemm(self, &rhs))
    }
    unsafe fn dot_unchecked(&self, rhs: Matrix) -> Matrix { gemm(self, &rhs) }
}

impl Dot<&Matrix> for Matrix {
    type Output = Matrix;
    fn assert_dot_compat(&self, rhs: &&Matrix) { self.assert_dot_compat(*rhs) }
    fn try_dot(&self, rhs: &Matrix) -> LinAlgResult<Matrix> {
        self.assert_dot_compat(rhs);
        Ok(gemm(self, rhs))
    }
    unsafe fn dot_unchecked(&self, rhs: &Matrix) -> Matrix { gemm(self, rhs) }
}

impl Dot<Matrix> for &Matrix {
    type Output = Matrix;
    fn assert_dot_compat(&self, rhs: &Matrix) { (**self).assert_dot_compat(rhs) }
    fn try_dot(&self, rhs: Matrix) -> LinAlgResult<Matrix> { (**self).try_dot(rhs) }
    unsafe fn dot_unchecked(&self, rhs: Matrix) -> Matrix { unsafe { (**self).dot_unchecked(rhs) } }
}

impl Dot<&Matrix> for &Matrix {
    type Output = Matrix;
    fn assert_dot_compat(&self, rhs: &&Matrix) { (**self).assert_dot_compat(*rhs) }
    fn try_dot(&self, rhs: &Matrix) -> LinAlgResult<Matrix> { (**self).try_dot(rhs.clone()) }
    unsafe fn dot_unchecked(&self, rhs: &Matrix) -> Matrix { unsafe { (**self).dot_unchecked(rhs.clone()) } }
}

// ============================================================
// DataSlice · DataSlice
// ============================================================

impl<A: DataSlice, B: DataSlice> Dot<B> for A {
    type Output = f32;
    fn assert_dot_compat(&self, rhs: &B) {
        assert!(self.len() == rhs.len(), "{}", ValidationError::VectorDimensionMismatch {
            a: self.len(), b: rhs.len(),
        });
    }
    fn try_dot(&self, rhs: B) -> LinAlgResult<f32> {
        self.assert_dot_compat(&rhs);
        Ok(compute_dot(self, &rhs))
    }
    unsafe fn dot_unchecked(&self, rhs: B) -> f32 {
        compute_dot(self, &rhs)
    }
}