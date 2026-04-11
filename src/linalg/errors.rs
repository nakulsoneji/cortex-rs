use thiserror::Error;

pub type LinAlgResult<T> = Result<T, LinAlgError>;

#[derive(Debug, Error)]
pub enum LinAlgError {
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("Arithmetic error: {0}")]
    Arithmetic(#[from] ArithmeticError),

    #[error("Decomposition error: {0}")]
    Decomposition(#[from] DecompositionError),
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Vector dimension mismatch: got {a} and {b}")]
    VectorDimensionMismatch { a: usize, b: usize },

    #[error("Matrix dimension mismatch: got [{rows_a} x {cols_a}] and [{rows_b} x {cols_b}]")]
    MatrixDimensionMismatch { rows_a: usize, cols_a: usize, rows_b: usize, cols_b: usize },

    #[error("Matrix Vector dimension mismatch: got [{rows_a} x {cols_a}] and {b}")]
    MatrixVectorDimensionMismatch { rows_a: usize, cols_a: usize, b: usize },

    #[error("Index {index} out of bounds for vector of length {len}")]
    VectorIndexOutOfBounds { index: usize, len: usize },

    #[error("Index ({row}, {col}) out of bounds for matrix of shape [{rows} x {cols}]")]
    MatrixIndexOutOfBounds { row: usize, col: usize, rows: usize, cols: usize },

    #[error("Expected {expected} elements but got {actual}")]
    ShapeSizeMismatch { expected: usize, actual: usize },

    #[error("Strides must be non-zero, got [{stride_0}, {stride_1}]")]
    InvalidStride { stride_0: usize, stride_1: usize },

    #[error("Dimension [({rows}) x ({cols})] is not square")]
    NonSquare { rows: usize, cols: usize },

    #[error("Vector must not be empty")]
    VectorEmpty,

    #[error("Matrix must not be empty")]
    MatrixEmpty,
}

#[derive(Debug, Error)]
pub enum ArithmeticError {
    #[error("Division by zero")]
    DivisionByZero,

    #[error("Numerical overflow occurred")]
    Overflow,

    #[error("NaN or Inf encountered")]
    NotFinite,
}

#[derive(Debug, Error)]
pub enum DecompositionError {
    #[error("Matrix is singular")]
    SingularMatrix,

    #[error("Algorithm failed to converge after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    #[error("Matrix is rank deficient: rank {rank}, expected {expected}")]
    RankDeficient { rank: usize, expected: usize },
}