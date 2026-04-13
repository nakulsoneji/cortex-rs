# Cortex (rawml)

Cortex is a full-stack machine learning project in Rust that implements core linear algebra and neural network training from the ground up.

It is designed to be both practical and educational: you can inspect, modify, and extend every layer of the stack.

## Why Cortex Is Fast

Cortex is built around optimized low-level numeric kernels:

- BLAS-backed vector and matrix operations
- LAPACK-backed decompositions and solvers
- Apple Accelerate integration on macOS
- Native routines implemented in low-level Fortran/C++/assembly beneath the Rust API surface

The result is a system that remains lightweight at the Rust layer while delegating heavy numerical work to highly optimized native implementations.

## Core Capabilities

### Linear Algebra

- `Matrix` and `Vector` core types
- Dot operations:
  - vector · vector
  - matrix · vector
  - vector · matrix
  - matrix · matrix
- Elementwise operators: `+`, `-`, `*`, `/`
- Scalar operations and in-place arithmetic
- Slicing, row/column extraction, and views
- Reductions: sum, mean, variance, std, min/max, argmin/argmax
- Outer product and norm utilities

### Decomposition and Solvers

Cortex includes decomposition-focused APIs for practical numerical tasks:

- LU decomposition (`lu`)
- QR decomposition (`qr`)
- SVD (`svd`)
- Cholesky decomposition (`cholesky`)
- Determinant (`det`)
- Inverse (`inv`)
- Rank estimation (`rank`)
- Linear solve (`solve`)
- Pseudo-inverse (`pseudo_inv`)

These operations are executed through LAPACK-backed workflows.

### ML Stack

- Fully connected `Layer` abstraction
- Activations:
  - `ReLU`
  - `LeakyReLU(alpha)`
  - `Sigmoid`
  - `Tanh`
  - `Linear`
  - `Softmax`
- Losses:
  - `MSE`
  - `CrossEntropy`
- Optimizers:
  - `SGD`
  - `Adam`
- Feed-forward training flow:
  - forward pass
  - backward gradients
  - mini-batch accumulation and updates
  - optional gradient and weight clipping

### Data Utilities

- CSV to matrix loaders
- MNIST loader with:
  - normalization
  - one-hot encoding

## Project Structure

- `src/linalg`: matrix/vector math, ops, dot products, decompositions, shared traits
- `src/ml`: layers, activations, losses, optimizers, feed-forward network
- `src/data`: CSV and MNIST utilities
- `src/main.rs`: runnable end-to-end training example

## Example 1: Linear Algebra and Decompositions

```rust
use cortex::prelude::*;

fn main() {
    let a = Matrix::new(vec![
        4.0, 1.0,
        2.0, 3.0,
    ], [2, 2]);

    let b = vector![1.0, 2.0];

    let y = a.dot(&b);
    let det = a.det().unwrap();
    let inv = a.inv().unwrap();
    let x = a.solve(&b).unwrap();

    println!("A * b = {:?}", y);
    println!("det(A) = {}", det);
    println!("inv(A) = {:?}", inv);
    println!("solve(Ax=b) => x = {:?}", x);
}
```

## Example 2: Feed-Forward Network Setup

```rust
use cortex::prelude::*;

fn main() {
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

    // Example training usage:
    // net.fit(&x_train, &y_train, 10);
    // let acc = net.accuracy(&x_test, &y_test);

    let _ = &mut net;
}
```

## Example 3: MNIST Training Flow

```rust
use cortex::prelude::*;
use cortex::data::mnist::load_mnist;

fn main() {
  let (x_train, y_train, x_test, y_test) = load_mnist();

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

  net.fit(&x_train, &y_train, 5);

  let test_loss = net.evaluate(&x_test, &y_test);
  let test_acc = net.accuracy(&x_test, &y_test);

  println!("test_loss = {:.4}", test_loss);
  println!("test_acc  = {:.2}%", test_acc);
}
```

## Build and Run

```bash
cargo build --release
cargo run --release
```

The default executable includes operator checks plus an end-to-end MNIST training workflow.

## Performance Notes

- Performance-critical paths are delegated to BLAS/LAPACK kernels.
- macOS builds use Apple Accelerate-backed routines.
- Core compute benefits from mature low-level native code (Fortran/C++/assembly) underneath the Rust API.
- The codebase focuses on balancing speed, clarity, and full-stack ownership.

## Engineering Goals

- Keep the numerics path fast and transparent
- Keep the ML stack understandable and hackable
- Keep the architecture modular for extension

## Future Work

- More decomposition/solver variants
- Expanded benchmarking and profiling harnesses
- Additional layer types and training utilities
- Serialization and model export tooling

## License

Add a license file and update this section with the selected license.
