use thiserror::Error;

#[derive(Debug, Error)]
pub enum CsvMatrixError {
    #[error("failed to read CSV '{path}': {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("ragged CSV at row {row}: expected {expected} columns, got {actual}")]
    RaggedRow {
        row: usize,
        expected: usize,
        actual: usize,
    },

    #[error("non-integer value '{value}' at row {row}, column {col}")]
    NonInteger {
        value: String,
        row: usize,
        col: usize,
    },
}