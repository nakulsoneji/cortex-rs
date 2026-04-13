// src/data/mnist.rs
use crate::prelude::*;
use mnist::{Mnist, MnistBuilder};

pub fn load_mnist() -> (Matrix, Matrix, Matrix, Matrix) {
    let Mnist {
        trn_img, trn_lbl,
        tst_img, tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data")
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    // normalize pixel values to [0, 1]
    let x_train = Matrix::new(
        trn_img.iter().map(|&p| p as f32 / 255.0).collect(),
        [60_000, 784],
    );

    // one-hot encode labels
    let y_train = Matrix::new(
        trn_lbl.iter().flat_map(|&l| {
            let mut one_hot = vec![0.0f32; 10];
            one_hot[l as usize] = 1.0;
            one_hot
        }).collect(),
        [60_000, 10],
    );

    let x_test = Matrix::new(
        tst_img.iter().map(|&p| p as f32 / 255.0).collect(),
        [10_000, 784],
    );

    let y_test = Matrix::new(
        tst_lbl.iter().flat_map(|&l| {
            let mut one_hot = vec![0.0f32; 10];
            one_hot[l as usize] = 1.0;
            one_hot
        }).collect(),
        [10_000, 10],
    );

    (x_train, y_train, x_test, y_test)
}