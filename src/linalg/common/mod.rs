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
}

pub trait DataSlice {
    fn as_slice(&self) -> &[f32];
    fn stride(&self) -> usize { 1 } // default stride 1 for contiguous
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    
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

