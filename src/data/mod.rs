use crate::linalg::Matrix;
use self::error::*;
pub type CsvMatrixResult<T> = Result<T, CsvMatrixError>;

pub mod error;


pub fn csv_to_matrix(path: &str) -> Matrix {
    try_csv_to_matrix(path).expect("csv_to_matrix failed")
}

pub fn csv_str_to_matrix(csv: &str) -> Matrix {
    try_csv_str_to_matrix(csv).expect("csv_str_to_matrix failed")
}

pub fn try_csv_to_matrix(path: &str) -> CsvMatrixResult<Matrix> {
    let csv = std::fs::read_to_string(path).map_err(|source| CsvMatrixError::Read {
        path: path.to_string(),
        source,
    })?;

    try_csv_str_to_matrix(&csv)
}

pub fn try_csv_str_to_matrix(csv: &str) -> CsvMatrixResult<Matrix> {
    let mut data: Vec<f32> = Vec::new();
    let mut rows = 0usize;
    let mut cols: Option<usize> = None;

    for (row_idx, line) in csv.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let values: Vec<&str> = line.split(',').map(|v| v.trim()).collect();

        match cols {
            Some(expected) if values.len() != expected => {
                return Err(CsvMatrixError::RaggedRow {
                    row: row_idx + 1,
                    expected,
                    actual: values.len(),
                });
            }
            None => cols = Some(values.len()),
            _ => {}
        }

        for (col_idx, raw) in values.iter().enumerate() {
            let parsed = raw.parse::<i32>().map_err(|_| CsvMatrixError::NonInteger {
                value: (*raw).to_string(),
                row: row_idx + 1,
                col: col_idx + 1,
            });
            let parsed = parsed?;
            data.push(parsed as f32);
        }

        rows += 1;
    }

    let cols = cols.unwrap_or(0);
    Ok(Matrix::new(data, [rows, cols]))
}
