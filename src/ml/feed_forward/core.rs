use crate::prelude::*;

impl FeedForward {
    pub fn new(layers: Vec<Layer>, loss: Loss, optimizer: Optimizer) -> Self {
        let mut optimizer = optimizer;
        optimizer.init(&layers);
        Self { layers, loss, optimizer, weight_clip: None, grad_clip: None }
    }

    pub fn with_weight_clip(mut self, clip: f32) -> Self {
        self.weight_clip = Some(clip);
        self
    }

    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = Some(clip);
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

    // pub fn fit(&mut self, x: &Matrix, y: &Matrix, epochs: usize, lr: f32) {
    //     let n = x.shape[0];
    //     for epoch in 0..epochs {
    //         let mut total_loss = 0.0;
    //         for i in 0..n {
    //             let input = x.row(i);
    //             let target = y.row(i);
    //             total_loss += self.train(&input, &target, lr);
    //         }
    //         println!("epoch {:>5}: loss = {:.6}", epoch, total_loss / n as f32);
    //     }
    // }

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

}
