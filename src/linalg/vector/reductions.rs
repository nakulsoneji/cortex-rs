use crate::prelude::*;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn vDSP_maxvi(__A: *const f32, __IA: i64, __C: *mut f32, __I: *mut u64, __N: u64);
    fn vDSP_minvi(__A: *const f32, __IA: i64, __C: *mut f32, __I: *mut u64, __N: u64);
}

impl Reducible for Vector {
    fn reducible_data(&self) -> (&[f32], usize, usize) { (&self.data, 1, self.data.len()) }
}

impl Vector {
    #[cfg(target_os = "macos")]
    pub fn argmax(&self) -> usize {
        assert!(!self.is_empty(), "{}", ValidationError::VectorEmpty);
        let mut val = 0.0f32;
        let mut idx = 0u64;
        unsafe {
            vDSP_maxvi(
                self.data.as_ptr(), 1,
                &mut val,
                &mut idx,
                self.data.len() as u64,
            );
        }
        idx as usize
    }

    #[cfg(not(target_os = "macos"))]
    pub fn argmax(&self) -> usize {
        assert!(!self.is_empty(), "{}", ValidationError::VectorEmpty);
        let mut best_idx = 0;
        let mut best_val = unsafe { *self.get_unchecked(0) };
        for i in 1..self.len() {
            let v = unsafe { *self.get_unchecked(i) };
            if v > best_val { best_val = v; best_idx = i; }
        }
        best_idx
    }

    #[cfg(target_os = "macos")]
    pub fn argmin(&self) -> usize {
        assert!(!self.is_empty(), "{}", ValidationError::VectorEmpty);
        let mut val = 0.0f32;
        let mut idx = 0u64;
        unsafe {
            vDSP_minvi(
                self.data.as_ptr(), 1,
                &mut val,
                &mut idx,
                self.data.len() as u64,
            );
        }
        idx as usize
    }

    #[cfg(not(target_os = "macos"))]
    pub fn argmin(&self) -> usize {
        assert!(!self.is_empty(), "{}", ValidationError::VectorEmpty);
        let mut best_idx = 0;
        let mut best_val = unsafe { *self.get_unchecked(0) };
        for i in 1..self.len() {
            let v = unsafe { *self.get_unchecked(i) };
            if v < best_val { best_val = v; best_idx = i; }
        }
        best_idx
    }
}