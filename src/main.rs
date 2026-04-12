use cortex::prelude::*;

fn main() {
    // same data as tensorflow example
    let inputs = vec![
        Vector::from(vec![ 1.0,  1.0]),
        Vector::from(vec![-1.0,  1.0]),
        Vector::from(vec![-1.0, -1.0]),
        Vector::from(vec![ 1.0, -1.0]),
        Vector::from(vec![ 2.0,  2.0]),
        Vector::from(vec![-2.0,  2.0]),
        Vector::from(vec![-2.0, -2.0]),
        Vector::from(vec![ 2.0, -2.0]),
    ];

    // one-hot encoded targets
    let targets = vec![
        Vector::from(vec![1.0, 0.0, 0.0, 0.0]), // Q1
        Vector::from(vec![0.0, 1.0, 0.0, 0.0]), // Q2
        Vector::from(vec![0.0, 0.0, 1.0, 0.0]), // Q3
        Vector::from(vec![0.0, 0.0, 0.0, 1.0]), // Q4
        Vector::from(vec![1.0, 0.0, 0.0, 0.0]), // Q1
        Vector::from(vec![0.0, 1.0, 0.0, 0.0]), // Q2
        Vector::from(vec![0.0, 0.0, 1.0, 0.0]), // Q3
        Vector::from(vec![0.0, 0.0, 0.0, 1.0]), // Q4
    ];

    let mut net = FeedForward::new(
        vec![
            Layer::new(2, 8, Activation::ReLU),
            Layer::new(8, 4, Activation::Softmax),
        ],
        Loss::CrossEntropy,
        Optimizer::adam(0.001),
    );

    for epoch in 0..1000 {
        let mut total_loss = 0.0;
        for (x, y) in inputs.iter().zip(targets.iter()) {
            total_loss += net.train(x, y);
        }
        if epoch % 10 == 0 {
            println!("epoch {:>3}: loss = {:.6}", epoch, total_loss / inputs.len() as f32);
        }
    }

    println!("\n=== Results ===");
    let quadrant_names = ["Q1", "Q2", "Q3", "Q4"];
    let mut correct = 0;

    for (x, y) in inputs.iter().zip(targets.iter()) {
        let output = net.predict(x);
        let predicted = output.argmax();
        let actual = y.argmax();
        if predicted == actual { correct += 1; }
        println!(
            "input: [{:>5.1}, {:>5.1}] → predicted: {} actual: {} {}",
            unsafe { *x.get_unchecked(0) },
            unsafe { *x.get_unchecked(1) },
            quadrant_names[predicted],
            quadrant_names[actual],
            if predicted == actual { "✓" } else { "✗" }
        );
    }

    println!("\nAccuracy: {}/{}", correct, inputs.len());
    assert_eq!(correct, inputs.len(), "expected 100% accuracy");
    println!("✓ quadrant classification works");
}