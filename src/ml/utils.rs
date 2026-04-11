use crate::prelude::*;

pub fn residuals(y_true: &Vector, y_pred: &Vector) -> Vector {
    y_true - y_pred
}

pub fn mse(y_true: &Vector, y_pred: &Vector) -> f32 {
    let e = residuals(y_true, y_pred);
    e.dot(&e) / e.len() as f32
}

pub fn rmse(y_true: &Vector, y_pred: &Vector) -> f32 {
    mse(y_true, y_pred).sqrt()
}

pub fn mae(y_true: &Vector, y_pred: &Vector) -> f32 {
    residuals(y_true, y_pred).l1_norm() / y_true.len() as f32
}

pub fn r2(y_true: &Vector, y_pred: &Vector) -> f32 {
    let e = residuals(y_true, y_pred);
    let ss_res = e.dot(&e);

    let mean_y = y_true.sum() / y_true.len() as f32;
    let centered = y_true - &(Vector::ones(y_true.len()) * mean_y);
    let ss_tot = centered.dot(&centered);

    if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - (ss_res / ss_tot)
    }
}
