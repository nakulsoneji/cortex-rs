use crate::prelude::*;

pub struct VectorView<'a> {
    data: &'a [f32],
    stride: usize,
    len: usize,
}

impl<'a> VectorView<'a> {
    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> f32 {
        unsafe { *self.data.get_unchecked(i * self.stride) }
    }

    #[inline]
    pub fn try_get(&self, i: usize) -> LinAlgResult<f32> {
        if i < self.len {
            unsafe { Ok(self.get_unchecked(i)) }
        } else {
            Err(ValidationError::VectorIndexOutOfBounds {
                index: i,
                len: self.len,
            }.into())
        }
    }

    #[inline]
    pub fn get(&self, i: usize) -> f32 {
        self.try_get(i).unwrap_or_else(|e| panic!("{}", e))
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.len).map(|i| unsafe { self.get_unchecked(i) })
    }

    pub fn to_vector(&self) -> Vector {
        Vector::from(self.iter().collect::<Vec<f32>>())
    }
}
