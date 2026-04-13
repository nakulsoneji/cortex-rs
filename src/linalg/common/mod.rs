use std::sync::Arc;
use crate::prelude::*;
pub mod ops;
pub mod dot;
pub mod view;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn vDSP_sve(__A: *const f32, __IA: i64, __C: *mut f32, __N: u64);
    fn vDSP_maxv(__A: *const f32, __IA: i64, __C: *mut f32, __N: u64);
    fn vDSP_minv(__A: *const f32, __IA: i64, __C: *mut f32, __N: u64);
    fn vDSP_maxvi(__A: *const f32, __IA: i64, __C: *mut f32, __I: *mut u64, __N: u64);
    fn vDSP_minvi(__A: *const f32, __IA: i64, __C: *mut f32, __I: *mut u64, __N: u64);
}

pub trait LinearStorage {
    fn data_arc(&self) -> &Arc<Vec<f32>>;
    fn data_arc_mut(&mut self) -> &mut Arc<Vec<f32>>;
    fn data_mut(&mut self) -> &mut Vec<f32> {
        Arc::make_mut(self.data_arc_mut())
    }
    fn len(&self) -> usize;
    fn strides(&self) -> [usize; 2] { [1, 1] }
    fn shape(&self) -> [usize; 2] { [self.len(), 1] }
    fn assert_same_shape(&self, other: &Self);
}

pub trait Dot<T> {
    type Output;
    fn assert_dot_compat(&self, rhs: &T);
    fn try_dot(&self, rhs: T) -> LinAlgResult<Self::Output>;
    fn dot(&self, rhs: T) -> Self::Output {
        self.try_dot(rhs).unwrap_or_else(|e| panic!("{}", e))
    }
    unsafe fn dot_unchecked(&self, rhs: T) -> Self::Output;
}

pub trait Reducible {
    fn reducible_data(&self) -> (&[f32], usize, usize); // (data, stride, len)

    fn mean(&self) -> f32 {
        let (_, _, len) = self.reducible_data();
        self.sum() / len as f32
    }

    fn variance(&self) -> f32 {
        let (data, stride, len) = self.reducible_data();
        let mean = self.mean();
        (0..len)
            .map(|i| {
                let v = unsafe { *data.get_unchecked(i * stride) };
                (v - mean).powi(2)
            })
            .sum::<f32>() / len as f32
    }

    fn std_dev(&self) -> f32 {
        self.variance().sqrt()
    }

    fn sum(&self) -> f32 {
        let (data, stride, len) = self.reducible_data();
        #[cfg(target_os = "macos")]
        {
            let mut result = 0.0f32;
            unsafe { vDSP_sve(data.as_ptr(), stride as i64, &mut result, len as u64); }
            result
        }
        #[cfg(not(target_os = "macos"))]
        { (0..len).map(|i| unsafe { *data.get_unchecked(i * stride) }).sum() }
    }

    fn max(&self) -> f32 {
        let (data, stride, len) = self.reducible_data();
        #[cfg(target_os = "macos")]
        {
            let mut result = 0.0f32;
            unsafe { vDSP_maxv(data.as_ptr(), stride as i64, &mut result, len as u64); }
            result
        }
        #[cfg(not(target_os = "macos"))]
        { (0..len).map(|i| unsafe { *data.get_unchecked(i * stride) }).fold(f32::NEG_INFINITY, f32::max) }
    }

    fn min(&self) -> f32 {
        let (data, stride, len) = self.reducible_data();
        #[cfg(target_os = "macos")]
        {
            let mut result = 0.0f32;
            unsafe { vDSP_minv(data.as_ptr(), stride as i64, &mut result, len as u64); }
            result
        }
        #[cfg(not(target_os = "macos"))]
        { (0..len).map(|i| unsafe { *data.get_unchecked(i * stride) }).fold(f32::INFINITY, f32::min) }
    }

    fn normalize_vec(&self) -> Vec<f32> {
        let (data, stride, len) = self.reducible_data();
        let min = self.min();
        let range = self.max() - min;
        if range == 0.0 {
            return (0..len).map(|i| unsafe { *data.get_unchecked(i * stride) }).collect();
        }
        (0..len).map(|i| {
            let v = unsafe { *data.get_unchecked(i * stride) };
            (v - min) / range
        }).collect()
    }

    fn standardize_vec(&self) -> Vec<f32> {
        let (data, stride, len) = self.reducible_data();
        let mean = self.mean();
        let std = self.std_dev();
        if std == 0.0 {
            return (0..len).map(|i| unsafe { *data.get_unchecked(i * stride) }).collect();
        }
        (0..len).map(|i| {
            let v = unsafe { *data.get_unchecked(i * stride) };
            (v - mean) / std
        }).collect()
    }
}

pub trait DataSlice : Reducible {
    fn as_slice(&self) -> &[f32];
    fn stride(&self) -> usize { 1 } // default stride 1 for contiguous
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }

    fn normalize(&self) -> Vector {
        Vector::from(self.normalize_vec())
    }

    fn standardize(&self) -> Vector {
        Vector::from(self.standardize_vec())
    }
    fn outer(&self, other: &impl DataSlice) -> Matrix {
        let m = self.len() as i32;
        let n = other.len() as i32;
        let mut data = vec![0.0f32; (m * n) as usize];

        unsafe {
            blas::sger(
                m, n,
                1.0,
                self.as_slice(), self.stride() as i32,
                other.as_slice(), other.stride() as i32,
                &mut data, m,
            );
        }

        Matrix::new_with_strides(data, [m as usize, n as usize], [1, m as usize])
    }

    fn argmax_with_val(&self) -> (usize, f32) {
        assert!(!self.is_empty());
        #[cfg(target_os = "macos")]
        {
            let mut val = 0.0f32;
            let mut idx = 0u64;
            unsafe {
                vDSP_maxvi(
                    self.as_slice().as_ptr(), self.stride() as i64,
                    &mut val, &mut idx,
                    self.len() as u64,
                );
            }
            return (idx as usize, val);
        }
        #[cfg(not(target_os = "macos"))]
        {
            (0..self.len())
                .map(|i| (i, unsafe { *self.as_slice().get_unchecked(i * self.stride()) }))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, f32::NEG_INFINITY))
        }
    }

    fn argmin_with_val(&self) -> (usize, f32) {
        assert!(!self.is_empty());
        #[cfg(target_os = "macos")]
        {
            let mut val = 0.0f32;
            let mut idx = 0u64;
            unsafe {
                vDSP_minvi(
                    self.as_slice().as_ptr(), self.stride() as i64,
                    &mut val, &mut idx,
                    self.len() as u64,
                );
            }
            return (idx as usize, val);
        }
        #[cfg(not(target_os = "macos"))]
        {
            (0..self.len())
                .map(|i| (i, unsafe { *self.as_slice().get_unchecked(i * self.stride()) }))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, f32::INFINITY))
        }
    }

    fn argmax(&self) -> usize { self.argmax_with_val().0 }
    fn argmin(&self) -> usize { self.argmin_with_val().0 }
}

