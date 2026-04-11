use crate::prelude::*;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn vDSP_maxvi(__A: *const f32, __IA: i64, __C: *mut f32, __I: *mut u64, __N: u64);
    fn vDSP_minvi(__A: *const f32, __IA: i64, __C: *mut f32, __I: *mut u64, __N: u64);
}

impl Reducible for Matrix {
    fn reducible_data(&self) -> (&[f32], usize, usize) {
        (&self.data, 1, self.shape[0] * self.shape[1])
    }
    fn sum(&self) -> f32 {
        let ones = Vector::from(vec![1.0f32; self.shape[1]]);
        self.dot(&ones).sum()
    }
}

impl Matrix {
    pub fn argmax(&self) -> [usize; 2] {
        assert!(!self.is_empty(), "{}", ValidationError::MatrixEmpty);
        #[cfg(target_os = "macos")]
        {
            let mut val = 0.0f32;
            let mut idx = 0u64;
            unsafe { vDSP_maxvi(self.data.as_ptr(), 1, &mut val, &mut idx, self.data.len() as u64); }
            let idx = idx as usize;
            return [idx / self.shape[1], idx % self.shape[1]];
        }
        #[cfg(not(target_os = "macos"))]
        {
            let idx = self.data.iter().copied().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            [idx / self.shape[1], idx % self.shape[1]]
        }
    }

    pub fn argmin(&self) -> [usize; 2] {
        assert!(!self.is_empty(), "{}", ValidationError::MatrixEmpty);
        #[cfg(target_os = "macos")]
        {
            let mut val = 0.0f32;
            let mut idx = 0u64;
            unsafe { vDSP_minvi(self.data.as_ptr(), 1, &mut val, &mut idx, self.data.len() as u64); }
            let idx = idx as usize;
            return [idx / self.shape[1], idx % self.shape[1]];
        }
        #[cfg(not(target_os = "macos"))]
        {
            let idx = self.data.iter().copied().enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            [idx / self.shape[1], idx % self.shape[1]]
        }
    }

    pub fn sum_axis(&self, axis: Axis) -> Vector {
        match axis {
            Axis::Row => self.dot(&Vector::from(vec![1.0f32; self.shape[1]])),
            Axis::Col => Vector::from(vec![1.0f32; self.shape[0]]).dot(self),
        }
    }

    pub fn mean_axis(&self, axis: Axis) -> Vector {
        let n = match axis {
            Axis::Row => self.shape[1],
            Axis::Col => self.shape[0],
        };
        let mut result = self.sum_axis(axis);
        result /= n as f32;
        result
    }

    pub fn max_axis(&self, axis: Axis) -> Vector {
        match axis {
            Axis::Row => Vector::from((0..self.shape[1]).map(|c| self.col_view(c).max()).collect::<Vec<f32>>()),
            Axis::Col => Vector::from((0..self.shape[0]).map(|r| self.row_view(r).max()).collect::<Vec<f32>>()),
        }
    }

    pub fn min_axis(&self, axis: Axis) -> Vector {
        match axis {
            Axis::Row => Vector::from((0..self.shape[1]).map(|c| self.col_view(c).min()).collect::<Vec<f32>>()),
            Axis::Col => Vector::from((0..self.shape[0]).map(|r| self.row_view(r).min()).collect::<Vec<f32>>()),
        }
    }

    pub fn argmax_axis(&self, axis: Axis) -> Vec<usize> {
        match axis {
            Axis::Row => (0..self.shape[1]).map(|c| self.col_view(c).argmax()).collect(),
            Axis::Col => (0..self.shape[0]).map(|r| self.row_view(r).argmax()).collect(),
        }
    }

    pub fn argmin_axis(&self, axis: Axis) -> Vec<usize> {
        match axis {
            Axis::Row => (0..self.shape[1]).map(|c| self.col_view(c).argmin()).collect(),
            Axis::Col => (0..self.shape[0]).map(|r| self.row_view(r).argmin()).collect(),
        }
    }
}