//! A minimal stateful GRU (gated recurrent unit) for single-observation,
//! sequential inference.
//!
//! # Gate convention
//!
//! This module implements exactly one convention
//! (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §14.3):
//!
//! ```text
//! z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)
//! r_t = sigmoid(W_r x_t + U_r h_{t-1} + b_r)
//! h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)
//! h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
//! ```
//!
//! Do not mix it with a different reference.
//!
//! # Atomicity
//!
//! [`Gru`] implements the foundation [`StateModel`], so a step computes a
//! candidate and commits it only on success. A failed step leaves the
//! previously committed hidden state untouched. [`Gru::step`] is a thin
//! stateful wrapper over that transition.

use crate::foundation::numerical::{DimensionMismatch, Matrix, Vector, sigmoid, tanh};
use crate::foundation::observation::Observation;
use crate::foundation::state::StateModel;

/// Failure classes reported by a GRU step or construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GruError {
    /// A configured dimension is zero.
    ZeroDimension,
    /// A parameter or observation has the wrong length.
    DimensionMismatch {
        /// The required length.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// A model parameter is not finite.
    NonFiniteParameter,
    /// An observation is not finite.
    NonFiniteInput,
    /// The computed candidate state is not finite.
    NonFiniteCandidate,
}

impl From<DimensionMismatch> for GruError {
    fn from(error: DimensionMismatch) -> Self {
        Self::DimensionMismatch {
            expected: error.expected,
            actual: error.actual,
        }
    }
}

/// All parameters required by the gate convention above.
#[derive(Debug, Clone, PartialEq)]
pub struct GruParameters {
    /// Update-gate input weights.
    pub w_z: Matrix,
    /// Update-gate recurrent weights.
    pub u_z: Matrix,
    /// Update-gate bias.
    pub b_z: Vector,
    /// Reset-gate input weights.
    pub w_r: Matrix,
    /// Reset-gate recurrent weights.
    pub u_r: Matrix,
    /// Reset-gate bias.
    pub b_r: Vector,
    /// Candidate input weights.
    pub w_h: Matrix,
    /// Candidate recurrent weights.
    pub u_h: Matrix,
    /// Candidate bias.
    pub b_h: Vector,
}

impl GruParameters {
    /// Deterministic pseudo-random parameters for examples, tests, and benches.
    ///
    /// ponytail: this is a fixed LCG, not a trained or statistically sound
    /// initialization. It exists so a GRU can be constructed without adding a
    /// randomness dependency; trained parameters arrive from outside the core.
    /// however once we finalize and validate the algorithm; consider
    /// including hull-dobell therom we may need to take this apart
    /// Bit-Truncation Efficiency included wrapping_* will overflow automatically
    /// discard low-order bits and extracting only most significant bits
    pub fn deterministic(input_dim: usize, hidden_dim: usize) -> Self {
        let mut state = 0x5eed_5eed_5eed_5eed_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1_u64 << 24) as f32) - 0.5
        };
        Self {
            w_z: Matrix::from_fn(hidden_dim, input_dim, |_, _| next()),
            u_z: Matrix::from_fn(hidden_dim, hidden_dim, |_, _| next()),
            b_z: Vector::from_fn(hidden_dim, |_| next()),
            w_r: Matrix::from_fn(hidden_dim, input_dim, |_, _| next()),
            u_r: Matrix::from_fn(hidden_dim, hidden_dim, |_, _| next()),
            b_r: Vector::from_fn(hidden_dim, |_| next()),
            w_h: Matrix::from_fn(hidden_dim, input_dim, |_, _| next()),
            u_h: Matrix::from_fn(hidden_dim, hidden_dim, |_, _| next()),
            b_h: Vector::from_fn(hidden_dim, |_| next()),
        }
    }
}

/// A stateful GRU with a committed hidden state.
#[derive(Debug)]
pub struct Gru {
    input_dim: usize,
    hidden_dim: usize,
    parameters: GruParameters,
    hidden: Vector,
}

impl Gru {
    /// Builds a GRU, validating dimensions and parameter finiteness.
    ///
    /// The hidden state starts at zeros.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        parameters: GruParameters,
    ) -> Result<Self, GruError> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(GruError::ZeroDimension);
        }
        validate_matrix(&parameters.w_z, hidden_dim, input_dim)?;
        validate_matrix(&parameters.u_z, hidden_dim, hidden_dim)?;
        validate_vector(&parameters.b_z, hidden_dim)?;
        validate_matrix(&parameters.w_r, hidden_dim, input_dim)?;
        validate_matrix(&parameters.u_r, hidden_dim, hidden_dim)?;
        validate_vector(&parameters.b_r, hidden_dim)?;
        validate_matrix(&parameters.w_h, hidden_dim, input_dim)?;
        validate_matrix(&parameters.u_h, hidden_dim, hidden_dim)?;
        validate_vector(&parameters.b_h, hidden_dim)?;

        Ok(Self {
            input_dim,
            hidden_dim,
            parameters,
            hidden: Vector::zeros(hidden_dim),
        })
    }

    /// The configured input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// The configured hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// The current committed hidden state.
    pub fn hidden(&self) -> &Vector {
        &self.hidden
    }

    /// Advances the committed hidden state by one observation.
    ///
    /// On failure the hidden state is unchanged.
    pub fn step(&mut self, observation: &Observation<Vector>) -> Result<&Vector, GruError> {
        let next = self.update(&self.hidden, observation)?;
        self.hidden = next;
        Ok(&self.hidden)
    }

    /// Resets the hidden state to zeros, starting a new sequence.
    pub fn reset(&mut self) {
        self.hidden = Vector::zeros(self.hidden_dim);
    }

    fn compute(
        parameters: &GruParameters,
        previous: &Vector,
        input: &Vector,
    ) -> Result<Vector, GruError> {
        if input.len() != parameters.w_z.cols() {
            return Err(GruError::DimensionMismatch {
                expected: parameters.w_z.cols(),
                actual: input.len(),
            });
        }
        if previous.len() != parameters.u_z.rows() {
            return Err(GruError::DimensionMismatch {
                expected: parameters.u_z.rows(),
                actual: previous.len(),
            });
        }
        if !input.as_slice().iter().all(|value| value.is_finite()) {
            return Err(GruError::NonFiniteInput);
        }

        let update_gate = parameters
            .w_z
            .mul_vector(input)?
            .add(&parameters.u_z.mul_vector(previous)?)?
            .add(&parameters.b_z)?
            .map(sigmoid);
        let reset_gate = parameters
            .w_r
            .mul_vector(input)?
            .add(&parameters.u_r.mul_vector(previous)?)?
            .add(&parameters.b_r)?
            .map(sigmoid);
        let candidate = parameters
            .w_h
            .mul_vector(input)?
            .add(&parameters.u_h.mul_vector(&reset_gate.multiply(previous)?)?)?
            .add(&parameters.b_h)?
            .map(tanh);

        let next = update_gate
            .complement()
            .multiply(previous)?
            .add(&update_gate.multiply(&candidate)?)?;

        if !next.as_slice().iter().all(|value| value.is_finite()) {
            return Err(GruError::NonFiniteCandidate);
        }
        Ok(next)
    }
}

impl StateModel for Gru {
    type State = Vector;
    type Observation = Observation<Vector>;
    type Error = GruError;

    fn update(
        &self,
        state: &Vector,
        observation: &Observation<Vector>,
    ) -> Result<Vector, GruError> {
        Self::compute(&self.parameters, state, observation.value())
    }
}

fn validate_matrix(matrix: &Matrix, rows: usize, cols: usize) -> Result<(), GruError> {
    if matrix.rows() != rows {
        return Err(GruError::DimensionMismatch {
            expected: rows,
            actual: matrix.rows(),
        });
    }
    if matrix.cols() != cols {
        return Err(GruError::DimensionMismatch {
            expected: cols,
            actual: matrix.cols(),
        });
    }
    if !matrix.as_slice().iter().all(|value| value.is_finite()) {
        return Err(GruError::NonFiniteParameter);
    }
    Ok(())
}

fn validate_vector(vector: &Vector, len: usize) -> Result<(), GruError> {
    if vector.len() != len {
        return Err(GruError::DimensionMismatch {
            expected: len,
            actual: vector.len(),
        });
    }
    if !vector.as_slice().iter().all(|value| value.is_finite()) {
        return Err(GruError::NonFiniteParameter);
    }
    Ok(())
}
