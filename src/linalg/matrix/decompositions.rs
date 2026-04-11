// decompositions.rs
use crate::prelude::*;

pub struct QRResult {
    pub q: Matrix,
    pub r: Matrix,
}

pub struct SVDResult {
    pub u: Matrix,
    pub s: Vector,
    pub vt: Matrix,
}

pub struct LUResult {
    pub l: Matrix,
    pub u: Matrix,
    pub p: Vec<i32>,
}

impl Matrix {
    pub fn lu_factorize(&self) -> (Vec<f32>, Vec<i32>, i32) {
        assert!(self.shape[0] == self.shape[1], "{}", ValidationError::NonSquare {
            rows: self.shape[0],
            cols: self.shape[1],
        });

        let n = self.shape[0] as i32;
        let mut a = self.to_contiguous_col_major();
        let mut ipiv = vec![0i32; self.shape[0]];
        let mut info = 0i32;

        unsafe {
            lapack::sgetrf(n, n, &mut a, n, &mut ipiv, &mut info);
        }

        (a, ipiv, info)
    }

    pub fn lu(&self) -> LinAlgResult<LUResult> {
        assert!(self.shape[0] == self.shape[1], "{}", ValidationError::NonSquare {
            rows: self.shape[0],
            cols: self.shape[1],
        });

        let n = self.shape[0];
        let (lu_data, ipiv, info) = self.lu_factorize();

        if info != 0 {
            return Err(DecompositionError::SingularMatrix)?;
        }

        // lu_data is col-major — extract L and U
        // col-major index: element (i, j) is at i + j*n
        let mut l_data = vec![0.0f32; n * n];
        let mut u_data = vec![0.0f32; n * n];

        for i in 0..n {
            for j in 0..n {
                let val = lu_data[i + j * n]; // col-major read
                let row_major_idx = i * n + j; // row-major write
                if i == j {
                    l_data[row_major_idx] = 1.0;
                    u_data[row_major_idx] = val;
                } else if i > j {
                    l_data[row_major_idx] = val;
                } else {
                    u_data[row_major_idx] = val;
                }
            }
        }

        Ok(LUResult {
            l: Matrix::new(l_data, [n, n]),
            u: Matrix::new(u_data, [n, n]),
            p: ipiv,
        })
    }

    pub fn qr(&self) -> LinAlgResult<QRResult> {
        let m = self.shape[0] as i32;
        let n = self.shape[1] as i32;
        let k = m.min(n) as usize;
        let mut a = self.to_contiguous_col_major();
        let mut tau = vec![0.0f32; k];
        let lwork = (4 * n) as usize;
        let mut work = vec![0.0f32; lwork];
        let mut info = 0i32;

        unsafe {
            lapack::sgeqrf(m, n, &mut a, m, &mut tau, &mut work, lwork as i32, &mut info);
        }
        if info != 0 {
            return Err(DecompositionError::ConvergenceFailed { iterations: 0 })?;
        }

        // extract R from upper triangle of col-major a
        // col-major: element (i,j) is at i + j*m
        let mut r_data = vec![0.0f32; k * n as usize];
        for i in 0..k {
            for j in i..n as usize {
                r_data[i * n as usize + j] = a[i + j * m as usize];
            }
        }
        let r = Matrix::new(r_data, [k, n as usize]);

        let mut work2 = vec![0.0f32; lwork];
        unsafe {
            lapack::sorgqr(m, n, n, &mut a, m, &tau, &mut work2, lwork as i32, &mut info);
        }
        if info != 0 {
            return Err(DecompositionError::ConvergenceFailed { iterations: 0 })?;
        }

        // Q comes back col-major
        let q = Matrix::new_with_strides(a, [m as usize, n as usize], [1, m as usize]);
        Ok(QRResult { q, r })
    }

    pub fn svd(&self) -> LinAlgResult<SVDResult> {
        let m = self.shape[0] as i32;
        let n = self.shape[1] as i32;
        let k = m.min(n) as usize;
        let mut a = self.to_contiguous_col_major();
        let mut s = vec![0.0f32; k];
        let mut u = vec![0.0f32; (m * m) as usize];
        let mut vt = vec![0.0f32; (n * n) as usize];
        let mut info = 0i32;

        // workspace query
        let mut work_query = vec![0.0f32; 1];
        unsafe {
            lapack::sgesvd(
                b'A', b'A',
                m, n,
                &mut a, m,
                &mut s,
                &mut u, m,
                &mut vt, n,
                &mut work_query, -1,
                &mut info,
            );
        }

        let lwork = work_query[0] as usize;
        let mut work = vec![0.0f32; lwork];

        // reset a since workspace query may modify it
        a = self.to_contiguous_col_major();

        unsafe {
            lapack::sgesvd(
                b'A', b'A',
                m, n,
                &mut a, m,
                &mut s,
                &mut u, m,
                &mut vt, n,
                &mut work, lwork as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(SVDResult {
                u: Matrix::new_with_strides(u, [m as usize, m as usize], [1, m as usize]),
                s: Vector::from(s),
                vt: Matrix::new_with_strides(vt, [n as usize, n as usize], [1, n as usize]),
            }),
            i if i > 0 => Err(DecompositionError::ConvergenceFailed { iterations: i as usize })?,
            _ => Err(DecompositionError::ConvergenceFailed { iterations: 0 })?,
        }
    }

    pub fn cholesky(&self) -> LinAlgResult<Matrix> {
        assert!(self.shape[0] == self.shape[1], "{}", ValidationError::NonSquare {
            rows: self.shape[0],
            cols: self.shape[1],
        });

        let n = self.shape[0] as i32;
        let mut a = self.to_contiguous_col_major();
        let mut info = 0i32;

        unsafe {
            lapack::spotrf(b'L', n, &mut a, n, &mut info);
        }

        match info {
            0 => {
                // zero out upper triangle in col-major
                // col-major: element (i,j) is at i + j*n
                for j in 0..self.shape[1] {
                    for i in 0..j {
                        a[i + j * self.shape[0]] = 0.0;
                    }
                }
                Ok(Matrix::new_with_strides(a, self.shape, [1, self.shape[0]]))
            },
            i if i > 0 => Err(DecompositionError::SingularMatrix)?,
            _ => Err(DecompositionError::ConvergenceFailed { iterations: 0 })?,
        }
    }
}