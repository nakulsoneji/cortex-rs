use crate::prelude::*;

impl FeedForward {
    pub fn new(layers: Vec<Layer>, loss: Loss, optimizer: Optimizer) -> Self {
        let mut optimizer = optimizer;
        optimizer.init(&layers);
        Self { layers, loss, optimizer, weight_clip: None, grad_clip: None, batch_size: 1 }
    }

    pub fn with_weight_clip(mut self, clip: f32) -> Self {
        self.weight_clip = Some(clip);
        self
    }

    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = Some(clip);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn forward(&mut self, input: &Vector) -> Vector {
        self.layers.iter_mut().fold(input.clone(), |x, layer| layer.forward(&x))
    }

    pub fn backward(&mut self, delta: &Vector) {
        let mut delta = delta.clone();
        let n = self.layers.len();
        for i in (0..n).rev() {
            let (dw, db, prev_delta) = self.layers[i].backward(&delta, self.grad_clip);
            let layer = &mut self.layers[i];
            self.optimizer.apply(i, &mut layer.weights, &mut layer.bias, &dw, &db);
            if let Some(clip) = self.weight_clip {
                layer.weights.clip_inplace(-clip, clip);
                layer.bias.clip_inplace(-clip, clip);
            }
            delta = prev_delta;
        }
    }

    pub fn accumulate(&mut self, input: &Vector, target: &Vector,
                    dw_acc: &mut Vec<Matrix>, db_acc: &mut Vec<Vector>) -> f32 {
        let output = self.forward(input);
        let loss = self.loss.compute(&output, target);
        let mut delta = self.loss.gradient(&output, target);

        let n = self.layers.len();
        for i in (0..n).rev() {
            let (dw, db, prev_delta) = self.layers[i].compute_gradients(&delta, self.grad_clip);
            dw_acc[i] += &dw;
            db_acc[i] += &db;
            delta = prev_delta;
        }
        loss
    }

    pub fn apply_gradients(&mut self, dw_acc: &mut Vec<Matrix>, db_acc: &mut Vec<Vector>, batch_size: usize) {
        let n = self.layers.len();
        for i in 0..n {
            dw_acc[i] /= batch_size as f32;
            db_acc[i] /= batch_size as f32;
            let layer = &mut self.layers[i];
            self.optimizer.apply(i, &mut layer.weights, &mut layer.bias, &dw_acc[i], &db_acc[i]);
            
            if let Some(clip) = self.weight_clip {
                layer.weights.clip_inplace(-clip, clip);
                layer.bias.clip_inplace(-clip, clip);
            }
        }
    }

    pub fn train(&mut self, input: &Vector, target: &Vector) -> f32 {
        let output = self.forward(input);
        let loss = self.loss.compute(&output, target);
        let delta = self.loss.gradient(&output, target);
        self.backward(&delta);
        loss
    }

    pub fn predict(&self, input: &Vector) -> Vector {
        self.layers.iter().fold(input.clone(), |x, layer| layer.predict(&x))
    }

    pub fn fit(&mut self, x: &Matrix, y: &Matrix, epochs: usize) {
        let n = x.shape[0];
        let batch_size = self.batch_size;
        let n_batches = n / batch_size;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for batch in 0..n_batches {
                let start = batch * batch_size;
                let end = start + batch_size;

                let mut dw_acc: Vec<Matrix> = self.layers.iter()
                    .map(|l| Matrix::zeros(l.weights.shape))
                    .collect();
                let mut db_acc: Vec<Vector> = self.layers.iter()
                    .map(|l| Vector::from(vec![0.0f32; l.bias.len()]))
                    .collect();

                let mut batch_loss = 0.0;
                for i in start..end {
                    let input = x.row(i);
                    let target = y.row(i);
                    batch_loss += self.accumulate(&input, &target, &mut dw_acc, &mut db_acc);
                }

                self.apply_gradients(&mut dw_acc, &mut db_acc, batch_size);
                total_loss += batch_loss / batch_size as f32;
            }

            println!("epoch {:>5}: loss = {:.6}", epoch, total_loss / n_batches as f32);
        }
    }

    pub fn evaluate(&self, x: &Matrix, y: &Matrix) -> f32 {
        let n = x.shape[0];
        let total_loss: f32 = (0..n)
            .map(|i| {
                let input = x.row(i);
                let target = y.row(i);
                let output = self.predict(&input);
                self.loss.compute(&output, &target)
            })
            .sum();
        total_loss / n as f32
    }

    pub fn accuracy(&self, x: &Matrix, y: &Matrix) -> f32 {
        let n = x.shape[0];
        let correct = (0..n)
            .filter(|&i| {
                let output = self.predict(&x.row(i));
                output.argmax() == y.row(i).argmax()
            })
            .count();
        correct as f32 / n as f32 * 100.0
    }
}