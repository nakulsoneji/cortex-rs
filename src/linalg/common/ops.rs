use crate::prelude::*;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn vDSP_vadd(__A: *const f32, __IA: i64, __B: *const f32, __IB: i64, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vsub(__A: *const f32, __IA: i64, __B: *const f32, __IB: i64, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vsadd(__A: *const f32, __IA: i64, __B: *const f32, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vsmul(__A: *const f32, __IA: i64, __B: *const f32, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vsdiv(__A: *const f32, __IA: i64, __B: *const f32, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vmul(__A: *const f32, __IA: i64, __B: *const f32, __IB: i64, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vdiv(__B: *const f32, __IB: i64, __A: *const f32, __IA: i64, __C: *mut f32, __IC: i64, __N: u64);
}

#[inline]
fn axpy_assign<T: LinearStorage>(lhs: &mut T, rhs: &T, alpha: f32) {
    lhs.assert_same_shape(rhs);
    let [rows, cols] = lhs.shape();
    let lhs_strides = lhs.strides();
    let rhs_strides = rhs.strides();
    let rhs_data = rhs.data_arc().clone();
    let data = lhs.data_mut();

    // fast path — both row-major contiguous
    if lhs_strides[1] == 1 && rhs_strides[1] == 1 {
        let n = (rows * cols) as i32;
        #[cfg(target_os = "macos")]
        unsafe {
            if alpha == 1.0 {
                vDSP_vadd(rhs_data.as_ptr(), 1, data.as_ptr(), 1, data.as_mut_ptr(), 1, n as u64);
            } else if alpha == -1.0 {
                vDSP_vsub(rhs_data.as_ptr(), 1, data.as_ptr(), 1, data.as_mut_ptr(), 1, n as u64);
            } else {
                blas::saxpy(n, alpha, &rhs_data, 1, data, 1);
            }
        }
        #[cfg(not(target_os = "macos"))]
        unsafe { blas::saxpy(n, alpha, &rhs_data, 1, data, 1); }
        return;
    }

    // strided path — vDSP/saxpy per row with correct strides
    for r in 0..rows {
        let lhs_offset = r * lhs_strides[0];
        let rhs_offset = r * rhs_strides[0];
        let n = cols as i32;
        #[cfg(target_os = "macos")]
        unsafe {
            if alpha == 1.0 {
                vDSP_vadd(
                    rhs_data[rhs_offset..].as_ptr(), rhs_strides[1] as i64,
                    data[lhs_offset..].as_ptr(), lhs_strides[1] as i64,
                    data[lhs_offset..].as_mut_ptr(), lhs_strides[1] as i64,
                    n as u64,
                );
            } else if alpha == -1.0 {
                vDSP_vsub(
                    rhs_data[rhs_offset..].as_ptr(), rhs_strides[1] as i64,
                    data[lhs_offset..].as_ptr(), lhs_strides[1] as i64,
                    data[lhs_offset..].as_mut_ptr(), lhs_strides[1] as i64,
                    n as u64,
                );
            } else {
                blas::saxpy(n, alpha, &rhs_data[rhs_offset..], rhs_strides[1] as i32, &mut data[lhs_offset..], lhs_strides[1] as i32);
            }
        }
        #[cfg(not(target_os = "macos"))]
        unsafe {
            blas::saxpy(n, alpha, &rhs_data[rhs_offset..], rhs_strides[1] as i32, &mut data[lhs_offset..], lhs_strides[1] as i32);
        }
    }
}

#[inline]
fn axpy_add_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) { axpy_assign(lhs, rhs, 1.0); }

#[inline]
fn axpy_sub_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) { axpy_assign(lhs, rhs, -1.0); }

#[inline]
fn scalar_add_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    let [rows, cols] = lhs.shape();
    let strides = lhs.strides();

    if strides[1] == 1 {
        let n = (rows * cols) as u64;
        let data = lhs.data_mut();
        #[cfg(target_os = "macos")]
        unsafe { vDSP_vsadd(data.as_ptr(), 1, &alpha, data.as_mut_ptr(), 1, n); }
        #[cfg(not(target_os = "macos"))]
        { let ones = vec![1.0f32; rows * cols]; unsafe { blas::saxpy((rows * cols) as i32, alpha, &ones, 1, data, 1); } }
        return;
    }

    let data = lhs.data_mut();
    for r in 0..rows {
        let offset = r * strides[0];
        #[cfg(target_os = "macos")]
        unsafe { vDSP_vsadd(data[offset..].as_ptr(), strides[1] as i64, &alpha, data[offset..].as_mut_ptr(), strides[1] as i64, cols as u64); }
        #[cfg(not(target_os = "macos"))]
        { let ones = vec![1.0f32; cols]; unsafe { blas::saxpy(cols as i32, alpha, &ones, 1, &mut data[offset..], strides[1] as i32); } }
    }
}

#[inline]
fn scalar_sub_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) { scalar_add_assign(lhs, -alpha); }

#[inline]
fn scalar_mul_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    let [rows, cols] = lhs.shape();
    let strides = lhs.strides();

    if strides[1] == 1 {
        let n = (rows * cols) as i32;
        let data = lhs.data_mut();
        #[cfg(target_os = "macos")]
        unsafe { vDSP_vsmul(data.as_ptr(), 1, &alpha, data.as_mut_ptr(), 1, n as u64); }
        #[cfg(not(target_os = "macos"))]
        unsafe { blas::sscal(n, alpha, data, 1); }
        return;
    }

    let data = lhs.data_mut();
    for r in 0..rows {
        let offset = r * strides[0];
        #[cfg(target_os = "macos")]
        unsafe { vDSP_vsmul(data[offset..].as_ptr(), strides[1] as i64, &alpha, data[offset..].as_mut_ptr(), strides[1] as i64, cols as u64); }
        #[cfg(not(target_os = "macos"))]
        unsafe { blas::sscal(cols as i32, alpha, &mut data[offset..], strides[1] as i32); }
    }
}

#[inline]
fn scalar_div_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    assert_ne!(alpha, 0.0, "{}", ArithmeticError::DivisionByZero);
    let [rows, cols] = lhs.shape();
    let strides = lhs.strides();

    if strides[1] == 1 {
        let n = (rows * cols) as i32;
        let data = lhs.data_mut();
        #[cfg(target_os = "macos")]
        unsafe { vDSP_vsdiv(data.as_ptr(), 1, &alpha, data.as_mut_ptr(), 1, n as u64); }
        #[cfg(not(target_os = "macos"))]
        unsafe { blas::sscal(n, 1.0 / alpha, data, 1); }
        return;
    }

    let data = lhs.data_mut();
    for r in 0..rows {
        let offset = r * strides[0];
        #[cfg(target_os = "macos")]
        unsafe { vDSP_vsdiv(data[offset..].as_ptr(), strides[1] as i64, &alpha, data[offset..].as_mut_ptr(), strides[1] as i64, cols as u64); }
        #[cfg(not(target_os = "macos"))]
        unsafe { blas::sscal(cols as i32, 1.0 / alpha, &mut data[offset..], strides[1] as i32); }
    }
}

#[inline]
fn elementwise_mul_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    lhs.assert_same_shape(rhs);
    let [rows, cols] = lhs.shape();
    let lhs_strides = lhs.strides();
    let rhs_strides = rhs.strides();

    if lhs_strides[1] == 1 && rhs_strides[1] == 1 {
        let n = (rows * cols) as u64;
        let rhs_data = rhs.data_arc().clone();
        let data = lhs.data_mut();
        #[cfg(target_os = "macos")]
        unsafe { vDSP_vmul(data.as_ptr(), 1, rhs_data.as_ptr(), 1, data.as_mut_ptr(), 1, n); }
        #[cfg(not(target_os = "macos"))]
        { data.iter_mut().zip(rhs_data.iter()).for_each(|(a, b)| *a *= b); }
        return;
    }

    let rhs_data = rhs.data_arc().clone();
    let data = lhs.data_mut();
    for r in 0..rows {
        let lhs_offset = r * lhs_strides[0];
        let rhs_offset = r * rhs_strides[0];
        #[cfg(target_os = "macos")]
        unsafe {
            vDSP_vmul(
                data[lhs_offset..].as_ptr(), lhs_strides[1] as i64,
                rhs_data[rhs_offset..].as_ptr(), rhs_strides[1] as i64,
                data[lhs_offset..].as_mut_ptr(), lhs_strides[1] as i64,
                cols as u64,
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            for c in 0..cols {
                let li = lhs_offset + c * lhs_strides[1];
                let ri = rhs_offset + c * rhs_strides[1];
                unsafe { *data.get_unchecked_mut(li) *= *rhs_data.get_unchecked(ri); }
            }
        }
    }
}

#[inline]
fn elementwise_div_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    lhs.assert_same_shape(rhs);
    let [rows, cols] = lhs.shape();
    let lhs_strides = lhs.strides();
    let rhs_strides = rhs.strides();

    if lhs_strides[1] == 1 && rhs_strides[1] == 1 {
        let n = (rows * cols) as u64;
        let rhs_data = rhs.data_arc().clone();
        let data = lhs.data_mut();
        #[cfg(target_os = "macos")]
        unsafe { vDSP_vdiv(rhs_data.as_ptr(), 1, data.as_ptr(), 1, data.as_mut_ptr(), 1, n); }
        #[cfg(not(target_os = "macos"))]
        { data.iter_mut().zip(rhs_data.iter()).for_each(|(a, b)| *a /= b); }
        return;
    }

    let rhs_data = rhs.data_arc().clone();
    let data = lhs.data_mut();
    for r in 0..rows {
        let lhs_offset = r * lhs_strides[0];
        let rhs_offset = r * rhs_strides[0];
        #[cfg(target_os = "macos")]
        unsafe {
            vDSP_vdiv(
                rhs_data[rhs_offset..].as_ptr(), rhs_strides[1] as i64,
                data[lhs_offset..].as_ptr(), lhs_strides[1] as i64,
                data[lhs_offset..].as_mut_ptr(), lhs_strides[1] as i64,
                cols as u64,
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            for c in 0..cols {
                let li = lhs_offset + c * lhs_strides[1];
                let ri = rhs_offset + c * rhs_strides[1];
                unsafe { *data.get_unchecked_mut(li) /= *rhs_data.get_unchecked(ri); }
            }
        }
    }
}

macro_rules! impl_binary_op {
    ($T:ty, $trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl std::ops::$trait<&$T> for $T {
            type Output = $T;
            #[inline]
            fn $method(mut self, rhs: &$T) -> $T {
                std::ops::$assign_trait::$assign_method(&mut self, rhs);
                self
            }
        }
        impl std::ops::$trait<&$T> for &$T where $T: Clone {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: &$T) -> $T { self.clone().$method(rhs) }
        }
        impl std::ops::$trait<$T> for $T {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: $T) -> $T { self.$method(&rhs) }
        }
        impl std::ops::$trait<$T> for &$T where $T: Clone {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: $T) -> $T { self.clone().$method(&rhs) }
        }
    };
}

macro_rules! impl_scalar_op {
    ($T:ty, $trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $assign_fn:ident) => {
        impl std::ops::$assign_trait<f32> for $T {
            #[inline]
            fn $assign_method(&mut self, rhs: f32) { $assign_fn(self, rhs); }
        }
        impl std::ops::$assign_trait<&f32> for $T {
            #[inline]
            fn $assign_method(&mut self, rhs: &f32) { $assign_fn(self, *rhs); }
        }
        impl std::ops::$trait<f32> for $T {
            type Output = $T;
            #[inline]
            fn $method(mut self, rhs: f32) -> $T {
                std::ops::$assign_trait::$assign_method(&mut self, rhs);
                self
            }
        }
        impl std::ops::$trait<&f32> for $T {
            type Output = $T;
            #[inline]
            fn $method(mut self, rhs: &f32) -> $T {
                std::ops::$assign_trait::$assign_method(&mut self, *rhs);
                self
            }
        }
        impl std::ops::$trait<f32> for &$T where $T: Clone {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: f32) -> $T { self.clone().$method(rhs) }
        }
        impl std::ops::$trait<&f32> for &$T where $T: Clone {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: &f32) -> $T { self.clone().$method(*rhs) }
        }
    };
}

macro_rules! impl_scalar_left_op {
    ($T:ty, $trait:ident, $method:ident, $assign_fn:ident) => {
        impl std::ops::$trait<$T> for f32 {
            type Output = $T;
            #[inline]
            fn $method(self, mut rhs: $T) -> $T { $assign_fn(&mut rhs, self); rhs }
        }
        impl std::ops::$trait<&$T> for f32 where $T: Clone {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: &$T) -> $T { self.$method(rhs.clone()) }
        }
        impl std::ops::$trait<$T> for &f32 {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: $T) -> $T { (*self).$method(rhs) }
        }
        impl std::ops::$trait<&$T> for &f32 where $T: Clone {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: &$T) -> $T { (*self).$method(rhs.clone()) }
        }
    };
}

macro_rules! impl_scalar_left_sub {
    ($T:ty) => {
        impl std::ops::Sub<$T> for f32 where $T: Clone {
            type Output = $T;
            #[inline]
            fn sub(self, mut rhs: $T) -> $T {
                scalar_mul_assign(&mut rhs, -1.0);
                scalar_add_assign(&mut rhs, self);
                rhs
            }
        }
        impl std::ops::Sub<&$T> for f32 where $T: Clone {
            type Output = $T;
            #[inline]
            fn sub(self, rhs: &$T) -> $T { self.sub(rhs.clone()) }
        }
        impl std::ops::Sub<$T> for &f32 {
            type Output = $T;
            #[inline]
            fn sub(self, rhs: $T) -> $T { (*self).sub(rhs) }
        }
        impl std::ops::Sub<&$T> for &f32 where $T: Clone {
            type Output = $T;
            #[inline]
            fn sub(self, rhs: &$T) -> $T { (*self).sub(rhs.clone()) }
        }
    };
}

macro_rules! impl_scalar_left_div {
    ($T:ty) => {
        impl std::ops::Div<$T> for f32 where $T: Clone {
            type Output = $T;
            #[inline]
            fn div(self, rhs: $T) -> $T {
                // use map to respect strides
                rhs.map(|x| self / x)
            }
        }
        impl std::ops::Div<&$T> for f32 where $T: Clone {
            type Output = $T;
            #[inline]
            fn div(self, rhs: &$T) -> $T { self.div(rhs.clone()) }
        }
        impl std::ops::Div<$T> for &f32 {
            type Output = $T;
            #[inline]
            fn div(self, rhs: $T) -> $T { (*self).div(rhs) }
        }
        impl std::ops::Div<&$T> for &f32 where $T: Clone {
            type Output = $T;
            #[inline]
            fn div(self, rhs: &$T) -> $T { (*self).div(rhs.clone()) }
        }
    };
}

macro_rules! impl_assign_op {
    ($T:ty, $assign_trait:ident, $assign_method:ident, $assign_fn:ident) => {
        impl std::ops::$assign_trait<&$T> for $T {
            #[inline]
            fn $assign_method(&mut self, rhs: &$T) { $assign_fn(self, rhs); }
        }
        impl std::ops::$assign_trait<$T> for $T {
            #[inline]
            fn $assign_method(&mut self, rhs: $T) { $assign_fn(self, &rhs); }
        }
    };
}

macro_rules! impl_all_ops {
    ($T:ty) => {
        impl_assign_op!($T, AddAssign, add_assign, axpy_add_assign);
        impl_assign_op!($T, SubAssign, sub_assign, axpy_sub_assign);
        impl_assign_op!($T, MulAssign, mul_assign, elementwise_mul_assign);
        impl_assign_op!($T, DivAssign, div_assign, elementwise_div_assign);

        impl_binary_op!($T, Add, add, AddAssign, add_assign);
        impl_binary_op!($T, Sub, sub, SubAssign, sub_assign);
        impl_binary_op!($T, Mul, mul, MulAssign, mul_assign);
        impl_binary_op!($T, Div, div, DivAssign, div_assign);

        impl_scalar_op!($T, Add, add, AddAssign, add_assign, scalar_add_assign);
        impl_scalar_op!($T, Sub, sub, SubAssign, sub_assign, scalar_sub_assign);
        impl_scalar_op!($T, Mul, mul, MulAssign, mul_assign, scalar_mul_assign);
        impl_scalar_op!($T, Div, div, DivAssign, div_assign, scalar_div_assign);

        impl_scalar_left_op!($T, Add, add, scalar_add_assign);
        impl_scalar_left_op!($T, Mul, mul, scalar_mul_assign);
        impl_scalar_left_sub!($T);
        impl_scalar_left_div!($T);
    };
}

impl_all_ops!(Vector);
impl_all_ops!(Matrix);