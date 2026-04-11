#[macro_export]
macro_rules! vector {
    ($($x:expr),* $(,)?) => {
        $crate::linalg::vector::Vector::from(vec![$($x),*])
    };
}