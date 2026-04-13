use cortex::prelude::*;
use cortex::data::mnist::load_mnist;
use rand::seq::SliceRandom;


fn check_ops() {
    // row-major + col-major
    let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let b = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).transpose();
    let mut c = a.clone();
    c += &b;
    assert!((unsafe { *c.get_unchecked(0, 0) } - 2.0).abs() < 1e-5, "row+col (0,0)");
    assert!((unsafe { *c.get_unchecked(0, 1) } - 5.0).abs() < 1e-5, "row+col (0,1)");
    assert!((unsafe { *c.get_unchecked(1, 0) } - 5.0).abs() < 1e-5, "row+col (1,0)");
    assert!((unsafe { *c.get_unchecked(1, 1) } - 8.0).abs() < 1e-5, "row+col (1,1)");
    println!("✓ row-major + col-major");

    // outer product accumulation
    let delta = Vector::from(vec![1.0, 2.0]);
    let input = Vector::from(vec![3.0, 4.0, 5.0]);
    let dw = delta.outer(&input);
    // expected [2x3]: [[3,4,5],[6,8,10]]
    assert!((unsafe { *dw.get_unchecked(0, 0) } - 3.0).abs() < 1e-5, "outer (0,0)");
    assert!((unsafe { *dw.get_unchecked(0, 1) } - 4.0).abs() < 1e-5, "outer (0,1)");
    assert!((unsafe { *dw.get_unchecked(1, 0) } - 6.0).abs() < 1e-5, "outer (1,0)");
    assert!((unsafe { *dw.get_unchecked(1, 1) } - 8.0).abs() < 1e-5, "outer (1,1)");
    println!("✓ outer product");

    // outer product += row-major accumulator
    let mut acc = Matrix::zeros([2, 3]);
    acc += &dw;
    assert!((unsafe { *acc.get_unchecked(0, 0) } - 3.0).abs() < 1e-5, "acc (0,0)");
    assert!((unsafe { *acc.get_unchecked(1, 1) } - 8.0).abs() < 1e-5, "acc (1,1)");
    println!("✓ outer product accumulation");

    // scalar ops on col-major
    let mut m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).transpose();
    m *= 2.0;
    assert!((unsafe { *m.get_unchecked(0, 0) } - 2.0).abs() < 1e-5, "scalar mul col-major (0,0)");
    assert!((unsafe { *m.get_unchecked(0, 1) } - 6.0).abs() < 1e-5, "scalar mul col-major (0,1)");
    println!("✓ scalar ops on col-major");

    println!("✓ all ops checks passed");
}


fn main() {
    check_ops();
    println!("Loading MNIST...");
    let (x_train, y_train, x_test, y_test) = load_mnist();
    println!("train: {}x{} test: {}x{}",
        x_train.shape[0], x_train.shape[1],
        x_test.shape[0], x_test.shape[1]);

    let mut net = FeedForward::new(
        vec![
            Layer::new(784, 128, Activation::LeakyReLU(0.01)),
            Layer::new(128, 64, Activation::LeakyReLU(0.01)),
            Layer::new(64, 10, Activation::Softmax),
        ],
        Loss::CrossEntropy,
        Optimizer::adam(0.001),
    ).with_batch_size(32);

    let epochs = 10;

    for epoch in 0..epochs {
        let n_train = x_train.shape[0];
        let batch_size = net.batch_size;

        let mut indices: Vec<usize> = (0..n_train).collect();
        indices.shuffle(&mut rand::rng());

        let mut total_loss = 0.0;
        let n_batches = n_train / batch_size;

        for batch in 0..n_batches {
            let mut dw_acc: Vec<Matrix> = net.layers.iter()
                .map(|l| Matrix::zeros(l.weights.shape))
                .collect();
            let mut db_acc: Vec<Vector> = net.layers.iter()
                .map(|l| Vector::from(vec![0.0f32; l.bias.len()]))
                .collect();

            let mut batch_loss = 0.0;
            for j in 0..batch_size {
                let i = indices[batch * batch_size + j];
                let input = x_train.row(i);
                let target = y_train.row(i);
                batch_loss += net.accumulate(&input, &target, &mut dw_acc, &mut db_acc);
            }

            net.apply_gradients(&mut dw_acc, &mut db_acc, batch_size);
            total_loss += batch_loss / batch_size as f32;

            if batch % 200 == 0 {
                println!("  epoch {:>2} [{:>6}/{}] loss = {:.4}",
                    epoch, batch * batch_size, n_train, total_loss / (batch + 1) as f32);
            }
        }

    let accuracy = net.accuracy(&x_test, &y_test);
    println!("epoch {:>2}: loss = {:.4} test_acc = {:.2}%",
        epoch, total_loss / n_batches as f32, accuracy);
    }
}
