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

// ============================================================
// Core helpers
// ============================================================

#[inline]
fn is_contiguous(strides: [usize; 2]) -> bool {
    strides[0] == 1 || strides[1] == 1
}

 #[inline]
fn for_each_row_binary<T: LinearStorage>(
    lhs: &mut T,
    rhs_data: &[f32],
    rhs_strides: [usize; 2],
    f_fast: impl Fn(*mut f32, *const f32, u64),
    f_strided: impl Fn(*mut f32, i64, *const f32, i64, u64),
) {
    let [rows, cols] = lhs.shape();
    let lhs_strides = lhs.strides();
    let data = lhs.data_mut();

    // fast path — identical strides (same layout)
    if lhs_strides == rhs_strides {
        f_fast(data.as_mut_ptr(), rhs_data.as_ptr(), (rows * cols) as u64);
        return;
    }

    // strided path — different layouts
    for r in 0..rows {
        let lo = r * lhs_strides[0];
        let ro = r * rhs_strides[0];
        f_strided(
            data[lo..].as_mut_ptr(), lhs_strides[1] as i64,
            rhs_data[ro..].as_ptr(), rhs_strides[1] as i64,
            cols as u64,
        );
    }
}

#[inline]
fn for_each_row_unary<T: LinearStorage>(
    lhs: &mut T,
    f_fast: impl Fn(*mut f32, u64),
    f_strided: impl Fn(*mut f32, i64, u64),
) {
    let [rows, cols] = lhs.shape();
    let strides = lhs.strides();
    let data = lhs.data_mut();

    // fast path — always valid for unary since only one matrix
    if is_contiguous(strides) {
        f_fast(data.as_mut_ptr(), (rows * cols) as u64);
        return;
    }

    // strided path
    for r in 0..rows {
        let offset = r * strides[0];
        f_strided(data[offset..].as_mut_ptr(), strides[1] as i64, cols as u64);
    }
}

// ============================================================
// Operations
// ============================================================

#[inline]
fn axpy_assign<T: LinearStorage>(lhs: &mut T, rhs: &T, alpha: f32) {
    lhs.assert_same_shape(rhs);
    let rhs_data = rhs.data_arc().clone();
    let rhs_strides = rhs.strides();

    for_each_row_binary(lhs, &rhs_data, rhs_strides,
        |d, r, n| unsafe {
            #[cfg(target_os = "macos")]
            {
                if alpha == 1.0 { vDSP_vadd(r, 1, d, 1, d, 1, n); }
                else if alpha == -1.0 { vDSP_vsub(r, 1, d, 1, d, 1, n); }
                else { blas::saxpy(n as i32, alpha, std::slice::from_raw_parts(r, n as usize), 1, std::slice::from_raw_parts_mut(d, n as usize), 1); }
            }
            #[cfg(not(target_os = "macos"))]
            blas::saxpy(n as i32, alpha, std::slice::from_raw_parts(r, n as usize), 1, std::slice::from_raw_parts_mut(d, n as usize), 1);
        },
        |d, ds, r, rs, n| unsafe {
            #[cfg(target_os = "macos")]
            {
                if alpha == 1.0 { vDSP_vadd(r, rs, d, ds, d, ds, n); }
                else if alpha == -1.0 { vDSP_vsub(r, rs, d, ds, d, ds, n); }
                else { blas::saxpy(n as i32, alpha, std::slice::from_raw_parts(r, (n as i64 * rs) as usize), rs as i32, std::slice::from_raw_parts_mut(d, (n as i64 * ds) as usize), ds as i32); }
            }
            #[cfg(not(target_os = "macos"))]
            blas::saxpy(n as i32, alpha, std::slice::from_raw_parts(r, (n as i64 * rs) as usize), rs as i32, std::slice::from_raw_parts_mut(d, (n as i64 * ds) as usize), ds as i32);
        },
    );
}

#[inline]
fn axpy_add_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) { axpy_assign(lhs, rhs, 1.0); }

#[inline]
fn axpy_sub_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) { axpy_assign(lhs, rhs, -1.0); }

#[inline]
fn scalar_add_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    for_each_row_unary(lhs,
        |d, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vsadd(d, 1, &alpha, d, 1, n);
            #[cfg(not(target_os = "macos"))]
            { let ones = vec![1.0f32; n as usize]; blas::saxpy(n as i32, alpha, &ones, 1, std::slice::from_raw_parts_mut(d, n as usize), 1); }
        },
        |d, ds, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vsadd(d, ds, &alpha, d, ds, n);
            #[cfg(not(target_os = "macos"))]
            { let ones = vec![1.0f32; n as usize]; blas::saxpy(n as i32, alpha, &ones, 1, std::slice::from_raw_parts_mut(d, (n as i64 * ds) as usize), ds as i32); }
        },
    );
}

#[inline]
fn scalar_sub_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) { scalar_add_assign(lhs, -alpha); }

#[inline]
fn scalar_mul_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    for_each_row_unary(lhs,
        |d, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vsmul(d, 1, &alpha, d, 1, n);
            #[cfg(not(target_os = "macos"))]
            blas::sscal(n as i32, alpha, std::slice::from_raw_parts_mut(d, n as usize), 1);
        },
        |d, ds, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vsmul(d, ds, &alpha, d, ds, n);
            #[cfg(not(target_os = "macos"))]
            blas::sscal(n as i32, alpha, std::slice::from_raw_parts_mut(d, (n as i64 * ds) as usize), ds as i32);
        },
    );
}

#[inline]
fn scalar_div_assign<T: LinearStorage>(lhs: &mut T, alpha: f32) {
    assert_ne!(alpha, 0.0, "{}", ArithmeticError::DivisionByZero);
    for_each_row_unary(lhs,
        |d, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vsdiv(d, 1, &alpha, d, 1, n);
            #[cfg(not(target_os = "macos"))]
            blas::sscal(n as i32, 1.0 / alpha, std::slice::from_raw_parts_mut(d, n as usize), 1);
        },
        |d, ds, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vsdiv(d, ds, &alpha, d, ds, n);
            #[cfg(not(target_os = "macos"))]
            blas::sscal(n as i32, 1.0 / alpha, std::slice::from_raw_parts_mut(d, (n as i64 * ds) as usize), ds as i32);
        },
    );
}

#[inline]
fn elementwise_mul_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    lhs.assert_same_shape(rhs);
    let rhs_data = rhs.data_arc().clone();
    let rhs_strides = rhs.strides();

    for_each_row_binary(lhs, &rhs_data, rhs_strides,
        |d, r, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vmul(d, 1, r, 1, d, 1, n);
            #[cfg(not(target_os = "macos"))]
            std::slice::from_raw_parts_mut(d, n as usize).iter_mut()
                .zip(std::slice::from_raw_parts(r, n as usize))
                .for_each(|(a, b)| *a *= b);
        },
        |d, ds, r, rs, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vmul(d, ds, r, rs, d, ds, n);
            #[cfg(not(target_os = "macos"))]
            for i in 0..n as usize {
                *d.offset(i as isize * ds as isize) *= *r.offset(i as isize * rs as isize);
            }
        },
    );
}

#[inline]
fn elementwise_div_assign<T: LinearStorage>(lhs: &mut T, rhs: &T) {
    lhs.assert_same_shape(rhs);
    let rhs_data = rhs.data_arc().clone();
    let rhs_strides = rhs.strides();

    for_each_row_binary(lhs, &rhs_data, rhs_strides,
        |d, r, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vdiv(r, 1, d, 1, d, 1, n);
            #[cfg(not(target_os = "macos"))]
            std::slice::from_raw_parts_mut(d, n as usize).iter_mut()
                .zip(std::slice::from_raw_parts(r, n as usize))
                .for_each(|(a, b)| *a /= b);
        },
        |d, ds, r, rs, n| unsafe {
            #[cfg(target_os = "macos")]
            vDSP_vdiv(r, rs, d, ds, d, ds, n);
            #[cfg(not(target_os = "macos"))]
            for i in 0..n as usize {
                *d.offset(i as isize * ds as isize) /= *r.offset(i as isize * rs as isize);
            }
        },
    );
}

// ============================================================
// Macros
// ============================================================

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
            fn div(self, rhs: $T) -> $T { rhs.map(|x| self / x) }
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