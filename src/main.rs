use cortex::prelude::*;
use cortex::data::mnist::load_mnist;

fn main() {
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
    )
    .with_batch_size(32)
    .with_grad_clip(1.0)
    .with_weight_clip(5.0);

    let epochs = 5;
    println!("Training for {} epochs...", epochs);
    net.fit(&x_train, &y_train, epochs);

    let test_loss = net.evaluate(&x_test, &y_test);
    let test_acc = net.accuracy(&x_test, &y_test);
    println!("test_loss = {:.4}", test_loss);
    println!("test_acc  = {:.2}%", test_acc);
}
