use crate::prelude::*;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn vDSP_vadd(__A: *const f32, __IA: i64, __B: *const f32, __IB: i64, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vsub(__A: *const f32, __IA: i64, __B: *const f32, __IB: i64, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vsadd(__A: *const f32, __IA: i64, __B: *const f32, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vsmul(__A: *const f32, __IA: i64, __B: *const f32, __C: *mut f32, __IC: i64, __N: u64);
    fn vDSP_vsdiv(__A: *const f32, __IA: i64, __B: *const f32, __C: *mut f32, __IC: i64, __N: u64);
}

#[cfg(target_os = "macos")]
#[inline]
fn axpy_assign<T: LinearStorage>(lhs: &mut T, rhs: &T, alpha: f32) {
    lhs.assert_same_shape(rhs);
    let n = lhs.len() as u64;
    let rhs_data = rhs.data_arc();
    let data = lhs.data_mut();
    if alpha == 1.0 {
        unsafe {
            vDSP_vadd(
                rhs_data.as_ptr(), 1,
                data.as_ptr(), 1,
                data.as_mut_ptr(), 1,
                n,
            );
        }
    } else if alpha == -1.0 {
        unsafe {
            vDSP_vsub(
                rhs_data.as_ptr(), 1,
                data.as_ptr(), 1,
                data.as_mut_ptr(), 1,
                n,
            );
        }
    } else {
        // fallback to saxpy for arbitrary alpha
        let n = n as i32;
        unsafe { blas::saxpy(n, alpha, rhs_data, 1, data, 1) };
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn scalar_add_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    let n = lhs.len() as u64;
    let data = lhs.data_mut();
    unsafe {
        vDSP_vsadd(
            data.as_ptr(), 1,
            &alpha,
            data.as_mut_ptr(), 1,
            n,
        );
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn scalar_sub_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    scalar_add_assign(lhs, -alpha);
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn axpy_assign<T: LinearStorage>(lhs: &mut T, rhs: &T, alpha: f32) {
    lhs.assert_same_shape(rhs);
    let n = lhs.len() as i32;
    let data = lhs.data_mut();
    unsafe { blas::saxpy(n, alpha, rhs.data_arc(), 1, data, 1) };
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn scalar_add_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    let ones = vec![1.0f32; lhs.len()];
    let n = lhs.len() as i32;
    let data = lhs.data_mut();
    unsafe { blas::saxpy(n, alpha, &ones, 1, data, 1) };
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn scalar_sub_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    scalar_add_assign(lhs, -alpha);
}

// these don't change
#[inline]
fn axpy_add_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    axpy_assign(lhs, rhs, 1.0);
}

#[inline]
fn axpy_sub_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    axpy_assign(lhs, rhs, -1.0);
}

#[cfg(target_os = "macos")]
#[inline]
fn scalar_mul_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    let n = lhs.len();
    let data = lhs.data_mut();
    unsafe {
        vDSP_vsmul(
            data.as_ptr(), 1,
            &alpha,
            data.as_mut_ptr(), 1,
            n as u64,
        );
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn scalar_div_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    assert_ne!(alpha, 0.0, "{}", ArithmeticError::DivisionByZero);
    let n = lhs.len();
    let data = lhs.data_mut();
    unsafe {
        vDSP_vsdiv(
            data.as_ptr(), 1,
            &alpha,
            data.as_mut_ptr(), 1,
            n as u64,
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn scalar_mul_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    let n = lhs.len() as i32;
    let data = lhs.data_mut();
    unsafe { blas::sscal(n, alpha, data, 1) };
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn scalar_div_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    assert_ne!(alpha, 0.0, "{}", ArithmeticError::DivisionByZero);
    scalar_mul_assign(lhs, 1.0 / alpha);
}

// #[inline]
// fn elementwise_mul_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
//     lhs.assert_same_shape(rhs);
//     let data = lhs.data_mut();
//     data.iter_mut()
//         .zip(rhs.data_arc().iter())
//         .for_each(|(a, b)| *a *= b);
// }

// #[inline]
// fn elementwise_mul_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
//     lhs.assert_same_shape(rhs);
//     let data = lhs.data_mut();
//     let rhs_data = rhs.data_arc();

//     let len = data.len();
//     let chunks = len / 8;
//     let remainder = len % 8;

//     for i in 0..chunks {
//         let offset = i * 8;
//         let a = f32x8::from_slice(&data[offset..]);
//         let b = f32x8::from_slice(&rhs_data[offset..]);
//         (a * b).copy_to_slice(&mut data[offset..]);
//     }

//     // handle remainder
//     let offset = chunks * 8;
//     for i in 0..remainder {
//         data[offset + i] *= rhs_data[offset + i];
//     }
// }

// #[inline]
// fn elementwise_div_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
//     lhs.assert_same_shape(rhs);
//     let data = lhs.data_mut();
//     data.iter_mut()
//         .zip(rhs.data_arc().iter())
//         .for_each(|(a, b)| *a /= b);
// }

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn vDSP_vmul(
        __A: *const f32, __IA: i64,
        __B: *const f32, __IB: i64,
        __C: *mut f32,   __IC: i64,
        __N: u64,
    );

    fn vDSP_vdiv(
        __B: *const f32, __IB: i64,
        __A: *const f32, __IA: i64,
        __C: *mut f32,   __IC: i64,
        __N: u64,
    );
}

#[cfg(target_os = "macos")]
#[inline]
fn elementwise_mul_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    lhs.assert_same_shape(rhs);
    let n = lhs.len();
    let rhs_data = rhs.data_arc().clone();
    let data = lhs.data_mut();
    unsafe {
        vDSP_vmul(
            data.as_ptr(), 1,
            rhs_data.as_ptr(), 1,
            data.as_mut_ptr(), 1,
            n as u64,
        );
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn elementwise_div_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    lhs.assert_same_shape(rhs);
    let n = lhs.len();
    let rhs_data = rhs.data_arc().clone();
    let data = lhs.data_mut();
    unsafe {
        // note: vDSP_vdiv has B and A swapped — it computes C = A/B
        vDSP_vdiv(
            rhs_data.as_ptr(), 1,
            data.as_ptr(), 1,
            data.as_mut_ptr(), 1,
            n as u64,
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn elementwise_mul_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    lhs.assert_same_shape(rhs);
    let data = lhs.data_mut();
    data.iter_mut().zip(rhs.data_arc().iter()).for_each(|(a, b)| *a *= b);
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn elementwise_div_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    lhs.assert_same_shape(rhs);
    let data = lhs.data_mut();
    data.iter_mut().zip(rhs.data_arc().iter()).for_each(|(a, b)| *a /= b);
}

// #[inline]
// fn elementwise_div_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
//     lhs.assert_same_shape(rhs);
//     let data = lhs.data_mut();
//     let rhs_data = rhs.data_arc();

//     let len = data.len();
//     let chunks = len / 8;
//     let remainder = len % 8;

//     for i in 0..chunks {
//         let offset = i * 8;
//         let a = f32x8::from_slice(&data[offset..]);
//         let b = f32x8::from_slice(&rhs_data[offset..]);
//         (a / b).copy_to_slice(&mut data[offset..]);
//     }

//     let offset = chunks * 8;
//     for i in 0..remainder {
//         data[offset + i] /= rhs_data[offset + i];
//     }
// }

macro_rules! impl_binary_op {
    ($T:ty, $trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        // owned op &
        impl std::ops::$trait<&$T> for $T {
            type Output = $T;
            #[inline]
            fn $method(mut self, rhs: &$T) -> $T {
                std::ops::$assign_trait::$assign_method(&mut self, rhs);
                self
            }
        }
        // & op &
        impl std::ops::$trait<&$T> for &$T where $T: Clone {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: &$T) -> $T {
                self.clone().$method(rhs)
            }
        }
        // owned op owned
        impl std::ops::$trait<$T> for $T {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: $T) -> $T {
                self.$method(&rhs)
            }
        }
        // & op owned
        impl std::ops::$trait<$T> for &$T where $T: Clone {
            type Output = $T;
            #[inline]
            fn $method(self, rhs: $T) -> $T {
                self.clone().$method(&rhs)
            }
        }
    };
}

macro_rules! impl_scalar_op {
    ($T:ty, $trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $assign_fn:ident) => {
        // AssignOps
        impl std::ops::$assign_trait<f32> for $T {
            #[inline]
            fn $assign_method(&mut self, rhs: f32) { $assign_fn(self, rhs); }
        }
        impl std::ops::$assign_trait<&f32> for $T {
            #[inline]
            fn $assign_method(&mut self, rhs: &f32) { $assign_fn(self, *rhs); }
        }
        // owned op scalar
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
        // & op scalar
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

// scalar op T  (only makes sense for +, *, for - and / we need special handling)
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

// scalar - T  =  -T + scalar  (negate then add)
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

// scalar / T  =  T.map(|x| scalar / x)
macro_rules! impl_scalar_left_div {
    ($T:ty) => {
        impl std::ops::Div<$T> for f32 where $T: Clone {
            type Output = $T;
            #[inline]
            fn div(self, mut rhs: $T) -> $T {
                let data = rhs.data_mut();
                data.iter_mut().for_each(|x| *x = self / *x);
                rhs
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

// ============================================================
// AssignOps for T op T
// ============================================================

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
        // T + T, T - T
        impl_assign_op!($T, AddAssign, add_assign, axpy_add_assign);
        impl_assign_op!($T, SubAssign, sub_assign, axpy_sub_assign);
        impl_assign_op!($T, MulAssign, mul_assign, elementwise_mul_assign);
        impl_assign_op!($T, DivAssign, div_assign, elementwise_div_assign);

        impl_binary_op!($T, Add, add, AddAssign, add_assign);
        impl_binary_op!($T, Sub, sub, SubAssign, sub_assign);
        impl_binary_op!($T, Mul, mul, MulAssign, mul_assign);
        impl_binary_op!($T, Div, div, DivAssign, div_assign);

        // T op scalar
        impl_scalar_op!($T, Add, add, AddAssign, add_assign, scalar_add_assign);
        impl_scalar_op!($T, Sub, sub, SubAssign, sub_assign, scalar_sub_assign);
        impl_scalar_op!($T, Mul, mul, MulAssign, mul_assign, scalar_mul_assign);
        impl_scalar_op!($T, Div, div, DivAssign, div_assign, scalar_div_assign);

        // scalar op T
        impl_scalar_left_op!($T, Add, add, scalar_add_assign);
        impl_scalar_left_op!($T, Mul, mul, scalar_mul_assign);
        impl_scalar_left_sub!($T);
        impl_scalar_left_div!($T);
    };
}



impl_all_ops!(Vector);
impl_all_ops!(Matrix);