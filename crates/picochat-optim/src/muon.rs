use candle_core::{Result, Tensor, TensorId, Var};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// Coefficients for Polar Express iterative orthogonalization
/// (Amsel et al., 2025 "The Polar Express").
///
/// Each tuple is (a, b, c) for one degree-5 Newton-Schulz iteration:
///   A  = X @ X^T
///   B  = b*A + c*A@A
///   X' = a*X + B@X
///
/// The first 5 coefficients handle rapid initial convergence while
/// the final 3 refine to near machine-precision orthogonality.
const POLAR_EXPRESS_COEFFS: [(f64, f64, f64); 8] = [
    (8.287212, -23.595887, 17.300387),
    (4.107059, -2.947850, 0.544843),
    (3.948691, -2.908902, 0.551819),
    (3.318420, -2.488488, 0.510049),
    (2.300652, -1.668904, 0.418807),
    (1.891301, -1.267996, 0.376804),
    (1.875001, -1.250002, 0.375000),
    (1.875000, -1.250000, 0.375000),
];

/// Compute the polar factor of a matrix via iterative Newton-Schulz
/// ("Polar Express"). The result approximates the closest orthogonal
/// matrix to `g`.
///
/// For tall matrices (rows > cols) the input is internally transposed
/// so that the iteration always operates on a wide or square matrix.
pub fn polar_express(g: &Tensor) -> Result<Tensor> {
    let (rows, cols) = g.shape().dims2()?;
    let transposed = rows > cols;
    let mut x = if transposed { g.t()? } else { g.clone() };

    // Normalize by Frobenius norm with safety factor (per Polar Express paper)
    let norm = x.sqr()?.sum_all()?.sqrt()?;
    x = x.broadcast_div(&((norm * 1.01)? + 1e-7)?)?;

    for &(a, b, c) in &POLAR_EXPRESS_COEFFS {
        let a_mat = x.matmul(&x.t()?)?; // X @ X^T
        let b_mat = ((&a_mat * b)? + (a_mat.matmul(&a_mat)? * c)?)?; // b*A + c*A@A
        x = ((&x * a)? + b_mat.matmul(&x)?)?; // a*X + B@X
    }

    if transposed {
        x.t()
    } else {
        Ok(x)
    }
}

/// Per-variable state for the Muon optimizer.
struct MuonState {
    buf: Tensor, // momentum buffer
}

/// Muon optimizer: Nesterov momentum + Polar Express orthogonalization.
///
/// Designed for weight matrices where an orthogonal update direction
/// can improve conditioning. The momentum buffer accumulates gradients
/// and the Polar Express step projects the update onto the nearest
/// orthogonal matrix.
pub struct Muon {
    default_lr: f64,
    beta: f64,
    states: HashMap<TensorId, MuonState>,
}

impl Muon {
    pub fn new(lr: f64, beta: f64) -> Self {
        Self {
            default_lr: lr,
            beta,
            states: HashMap::new(),
        }
    }

    /// Return the default learning rate.
    pub fn default_lr(&self) -> f64 {
        self.default_lr
    }

    /// Perform one Muon update for a single variable.
    ///
    /// 1. Nesterov momentum: `buf = beta * buf + grad`,
    ///    `nesterov = grad + beta * buf`
    /// 2. Orthogonalize: `update = polar_express(nesterov)`
    /// 3. Apply: `theta = theta - lr * update`
    pub fn step_var(&mut self, var: &Var, grad: &Tensor, lr: f64) -> Result<()> {
        let var_id = var.as_tensor().id();

        // Initialize state on first call for this variable.
        if let Entry::Vacant(e) = self.states.entry(var_id) {
            e.insert(MuonState {
                buf: Tensor::zeros_like(var.as_tensor())?,
            });
        }

        let state = self.states.get_mut(&var_id).unwrap();

        // 1. Nesterov momentum
        state.buf = ((&state.buf * self.beta)? + grad)?;
        let nesterov = (grad + (&state.buf * self.beta)?)?;

        // 2. Orthogonalize via Polar Express
        let update = polar_express(&nesterov)?;

        // 3. Apply update
        let new_theta = (var.as_tensor() - (update * lr)?)?;
        var.set(&new_theta)?;
        Ok(())
    }
}
