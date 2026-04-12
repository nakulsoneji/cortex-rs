use crate::prelude::*;



#[derive(Clone, Debug)]
pub enum Optimizer {
    SGD { lr: f32 },
    Adam { lr: f32, states: Vec<AdamState> },
}


#[derive(Clone, Debug)]
pub struct AdamState {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m_w: Matrix,
    v_w: Matrix,
    m_b: Vector,
    v_b: Vector,
}

impl AdamState {
    pub fn new(lr: f32, weight_shape: [usize; 2], bias_len: usize) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m_w: Matrix::zeros(weight_shape),
            v_w: Matrix::zeros(weight_shape),
            m_b: Vector::from(vec![0.0f32; bias_len]),
            v_b: Vector::from(vec![0.0f32; bias_len]),
        }
    }

    pub fn update(&mut self, weights: &mut Matrix, bias: &mut Vector, dw: &Matrix, db: &Vector) {
        self.t += 1;
        let t = self.t as f32;

        // update first moment
        self.m_w = &self.m_w * self.beta1 + dw * (1.0 - self.beta1);
        self.m_b = &self.m_b * self.beta1 + db * (1.0 - self.beta1);

        // update second moment
        self.v_w = &self.v_w * self.beta2 + dw.map(|v| v * v) * (1.0 - self.beta2);
        self.v_b = &self.v_b * self.beta2 + db.map(|v| v * v) * (1.0 - self.beta2);

        // bias corrected learning rate
        let lr_t = self.lr * (1.0 - self.beta2.powf(t)).sqrt() / (1.0 - self.beta1.powf(t));

        // update weights and bias
        *weights -= self.m_w.map(|m| lr_t * m) / self.v_w.map(|v| v.sqrt() + self.epsilon);
        *bias -= self.m_b.map(|m| lr_t * m) / self.v_b.map(|v| v.sqrt() + self.epsilon);
    }
}

impl Optimizer {
    pub fn sgd(lr: f32) -> Self {
        Optimizer::SGD { lr }
    }

    pub fn adam(lr: f32) -> Self {
        Optimizer::Adam { lr, states: Vec::new() }
    }

    // called once after layers are known
    pub fn init(&mut self, layers: &[Layer]) {
        match self {
            Optimizer::SGD { .. } => {},
            Optimizer::Adam { lr, states } => {
                *states = layers.iter().map(|l| {
                    AdamState::new(*lr, l.weights.shape, l.bias.len())
                }).collect();
            }
        }
    }

    pub fn apply(&mut self, layer_idx: usize, weights: &mut Matrix, bias: &mut Vector, dw: &Matrix, db: &Vector) {
        match self {
            Optimizer::SGD { lr } => {
                *weights -= dw * *lr;
                *bias -= db * *lr;
            },
            Optimizer::Adam { states, .. } => {
                states[layer_idx].update(weights, bias, dw, db);
            }
        }
    }
}