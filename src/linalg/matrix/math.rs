use crate::prelude::*;
use super::decompositions::SVDResult;

impl Matrix {
    pub fn transpose(&self) -> Self {
        Matrix {
            data: self.data.clone(),
            shape: [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
        }
    }

    pub fn trace(&self) -> f32 {
        assert!(self.shape[0] == self.shape[1], "{}", ValidationError::NonSquare {
            rows: self.shape[0],
            cols: self.shape[1],
        });
        (0..self.shape[0])
            .map(|i| unsafe { *self.get_unchecked(i, i) })
            .sum()
    }

    pub fn det(&self) -> LinAlgResult<f32> {
        let (lu, ipiv, info) = self.lu_factorize();

        if info != 0 { return Ok(0.0); }

        let n = self.shape[0];
        let mut det = 1.0f32;
        for i in 0..n {
            det *= lu[i + i * n];
        }

        let swaps = ipiv.iter().copied().enumerate()
            .filter(|(i, p)| *p != (*i as i32 + 1))
            .count();

        if swaps % 2 != 0 { Ok(-det) } else { Ok(det) }
    }

    pub fn inv(&self) -> LinAlgResult<Matrix> {
        assert!(self.shape[0] == self.shape[1], "{}", ValidationError::NonSquare {
            rows: self.shape[0],
            cols: self.shape[1],
        });

        let n = self.shape[0] as i32;
        let (mut lu, mut ipiv, info) = self.lu_factorize();

        if info != 0 {
            return Err(DecompositionError::SingularMatrix)?;
        }

        // workspace query
        let mut work_query = vec![0.0f32; 1];
        let mut info = 0i32;
        unsafe {
            lapack::sgetri(n, &mut lu, n, &mut ipiv, &mut work_query, -1, &mut info);
        }

        let lwork = work_query[0] as usize;
        let mut work = vec![0.0f32; lwork];
        let mut info = 0i32;

        unsafe {
            lapack::sgetri(n, &mut lu, n, &mut ipiv, &mut work, lwork as i32, &mut info);
        }

        match info {
            0 => Ok(Matrix::new_with_strides(lu, [n as usize, n as usize], [1, n as usize])),
            _ => Err(DecompositionError::SingularMatrix)?,
        }
    }

    pub fn rank(&self) -> usize {
        let m = self.shape[0] as i32;
        let n = self.shape[1] as i32;
        let k = m.min(n) as usize;
        let mut a = self.to_contiguous_col_major();
        let mut s = vec![0.0f32; k];
        let mut u = vec![0.0f32; 1];
        let mut vt = vec![0.0f32; 1];
        let mut info = 0i32;

        // workspace query
        let mut work_query = vec![0.0f32; 1];
        unsafe {
            lapack::sgesvd(
                b'N', b'N',
                m, n,
                &mut a, m,
                &mut s,
                &mut u, 1,
                &mut vt, 1,
                &mut work_query, -1,
                &mut info,
            );
        }

        let lwork = work_query[0] as usize;
        let mut work = vec![0.0f32; lwork];
        a = self.to_contiguous_col_major();

        unsafe {
            lapack::sgesvd(
                b'N', b'N',
                m, n,
                &mut a, m,
                &mut s,
                &mut u, 1,
                &mut vt, 1,
                &mut work, lwork as i32,
                &mut info,
            );
        }

        let threshold = 1e-6f32 * s[0];
        s.iter().filter(|&&sv| sv > threshold).count()
    }

    pub fn solve(&self, b: &Vector) -> LinAlgResult<Vector> {
        assert!(self.shape[0] == self.shape[1], "{}", ValidationError::NonSquare {
            rows: self.shape[0],
            cols: self.shape[1],
        });
        assert!(self.shape[0] == b.len(), "{}", ValidationError::MatrixVectorDimensionMismatch {
            rows_a: self.shape[0],
            cols_a: self.shape[1],
            b: b.len(),
        });

        let n = self.shape[0] as i32;
        let mut a = self.to_contiguous_col_major();
        let mut b_data = b.data.to_vec();
        let mut ipiv = vec![0i32; self.shape[0]];
        let mut info = 0i32;

        unsafe {
            lapack::sgesv(n, 1, &mut a, n, &mut ipiv, &mut b_data, n, &mut info);
        }

        match info {
            0 => Ok(Vector::from(b_data)),
            i if i > 0 => Err(DecompositionError::SingularMatrix)?,
            _ => Err(DecompositionError::ConvergenceFailed { iterations: 0 })?,
        }
    }

    pub fn pseudo_inv(&self) -> LinAlgResult<Matrix> {
        let SVDResult { u, s, vt } = self.svd()?;

        let m = self.shape[0];
        let n = self.shape[1];
        let k = s.len();

        let threshold = 1e-6 * s.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // precompute scales once
        let scales: Vec<f32> = (0..k)
            .map(|c| {
                let sv = unsafe { *s.get_unchecked(c) };
                if sv > threshold { 1.0 / sv } else { 0.0 }
            })
            .collect();

        // build vs_t as [n x k] row-major directly
        let mut vs_t = Vec::with_capacity(n * k);
        for r in 0..n {
            for c in 0..k {
                vs_t.push(unsafe { *vt.get_unchecked(c, r) } * scales[c]);
            }
        }
        let vs_t = Matrix::new(vs_t, [n, k]);

        // build u_k_t as [k x m] row-major directly
        let mut u_k_t = Vec::with_capacity(k * m);
        for r in 0..k {
            for c in 0..m {
                u_k_t.push(unsafe { *u.get_unchecked(c, r) });
            }
        }
        let u_k_t = Matrix::new(u_k_t, [k, m]);

        // A⁺ = vs_t · u_k_t = [n x k] · [k x m] = [n x m]
        Ok(vs_t.dot(&u_k_t))
    }
}