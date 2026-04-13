use crate::prelude::*;

#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Vector,
}

impl LinearModel {
    pub fn try_fit(x: &Matrix, y: &Vector) -> LinAlgResult<Self> {
        let m = x.shape[0] as i32;
        let n = x.shape[1] as i32;
        let nrhs = 1i32;

        let mut a = x.to_contiguous_col_major();
        let mut b = vec![0.0f32; m.max(n) as usize];
        b[..m as usize].copy_from_slice(&y.data);

        let mut s = vec![0.0f32; m.min(n) as usize];
        let mut rank = 0i32;
        let mut info = 0i32;
        let rcond = -1.0f32;
        let liwork = (3 * m.min(n) * 11 + 3 * m.min(n)).max(1) as usize;
        let mut iwork = vec![0i32; liwork];

        // workspace query — pass lwork=-1 to get optimal size
        let mut work_query = vec![0.0f32; 1];
        unsafe {
            lapack::sgelsd(
                m, n, nrhs,
                &mut a, m,
                &mut b, m.max(n),
                &mut s, rcond,
                &mut rank,
                &mut work_query, -1,
                &mut iwork,
                &mut info,
            );
        }

        // allocate optimal workspace
        let lwork = work_query[0] as i32;
        let mut work = vec![0.0f32; lwork as usize];

        // reset a and b since workspace query may have modified them
        a = x.to_contiguous_col_major();
        b = vec![0.0f32; m.max(n) as usize];
        b[..m as usize].copy_from_slice(&y.data);

        unsafe {
            lapack::sgelsd(
                m, n, nrhs,
                &mut a, m,
                &mut b, m.max(n),
                &mut s, rcond,
                &mut rank,
                &mut work, lwork,
                &mut iwork,
                &mut info,
            );
        }

        match info {
            0 => Ok(Self { weights: Vector::from(b[..n as usize].to_vec()) }),
            _ => Err(DecompositionError::ConvergenceFailed { iterations: 0 }.into()),
        }
    }

    pub fn fit(x: &Matrix, y: &Vector) -> Self {
        Self::try_fit(x, y).unwrap_or_else(|e| panic!("{}", e))
    }
    pub fn predict(&self, x: &Matrix) -> Vector {
        x.dot(&self.weights)
    }

    pub fn weights(&self) -> &Vector {
        &self.weights
    }
}
