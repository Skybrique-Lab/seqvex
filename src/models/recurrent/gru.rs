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
//!
//! # Reference and production paths
//!
//! [`Gru::step`] and the [`StateModel::update`] implementation are the
//! **reference** path: value-returning, allocating one `Vector` per operation,
//! and the semantic oracle every optimized path must be tested against.
//!
//! [`Gru::step_in_place`] and [`Gru::update_in_place`] are an **experimental
//! production** path. They reuse a private workspace, allocate nothing in
//! steady state, and are bit-identical to the reference path in the tested
//! range. The design is **provisional**: the workspace is owned by the model and
//! reached through `&mut Gru`, which is under a CRITICAL architecture review
//! because it couples immutable, shareable weights with per-stream scratch and
//! prevents multiple executors from sharing one model. Do not treat this API as
//! settled, and do not generalize the workspace to other models yet.

use crate::execution::streaming::StreamingExecutor;
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

/// Reusable scratch storage for the allocation-free production path.
///
/// Allocated once per model and reused every observation, so the production
/// step allocates nothing in steady state. It never aliases the committed
/// hidden state.
#[derive(Debug)]
struct Workspace {
    /// Holds `z_t`, then `(1 − z_t) ⊙ h`, then the committed candidate `h_t`.
    acc: Vec<f32>,
    /// Holds `r_t`, then `r_t ⊙ h`, then the candidate `h̃_t`.
    gate: Vec<f32>,
    /// Mat-vec partials and the `z_t ⊙ h̃_t` blend term.
    scratch: Vec<f32>,
}

impl Workspace {
    fn new(hidden_dim: usize) -> Self {
        Self {
            acc: vec![0.0; hidden_dim],
            gate: vec![0.0; hidden_dim],
            scratch: vec![0.0; hidden_dim],
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
    workspace: Workspace,
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
            workspace: Workspace::new(hidden_dim),
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

    /// Advances the committed hidden state using the reusable workspace.
    ///
    /// This production path allocates nothing in steady state and commits the
    /// candidate only after validation, so a failed step leaves the committed
    /// state unchanged. It is numerically equivalent to [`Gru::step`]; the test
    /// suite verifies bitwise agreement.
    ///
    /// **Provisional:** the workspace is model-owned and reached through
    /// `&mut self`; this API shape is under a CRITICAL architecture review and
    /// is not settled.
    pub fn step_in_place(
        &mut self,
        observation: &Observation<Vector>,
    ) -> Result<&Vector, GruError> {
        {
            let Self {
                parameters,
                workspace,
                hidden,
                ..
            } = self;
            compute_in_place(parameters, workspace, hidden, observation)?;
        }
        Ok(&self.hidden)
    }

    /// Computes the candidate for `observation` into `state` using the reusable
    /// workspace, committing it only after validation.
    ///
    /// Exposes the production path over an explicit state so streaming execution
    /// can drive it without owning the model's hidden state.
    ///
    /// **Provisional:** the workspace is model-owned and reached through
    /// `&mut self`; this API shape is under a CRITICAL architecture review and
    /// is not settled.
    pub fn update_in_place(
        &mut self,
        state: &mut Vector,
        observation: &Observation<Vector>,
    ) -> Result<(), GruError> {
        let Self {
            parameters,
            workspace,
            ..
        } = self;
        compute_in_place(parameters, workspace, state, observation)
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

/// Computes one GRU step into `state` using only reusable workspace storage.
///
/// The operation order matches [`Gru::compute`] exactly, so the result is
/// bit-identical to the reference path. `state` is read as the previous hidden
/// state and overwritten only after the candidate is validated.
fn compute_in_place(
    parameters: &GruParameters,
    workspace: &mut Workspace,
    state: &mut Vector,
    observation: &Observation<Vector>,
) -> Result<(), GruError> {
    let input = observation.value();
    if input.len() != parameters.w_z.cols() {
        return Err(GruError::DimensionMismatch {
            expected: parameters.w_z.cols(),
            actual: input.len(),
        });
    }
    if state.len() != parameters.u_z.rows() {
        return Err(GruError::DimensionMismatch {
            expected: parameters.u_z.rows(),
            actual: state.len(),
        });
    }
    if !input.as_slice().iter().all(|value| value.is_finite()) {
        return Err(GruError::NonFiniteInput);
    }

    {
        let h = state.as_slice();
        let x = input.as_slice();

        // acc = sigmoid(W_z x + U_z h + b_z)
        matvec_into(&parameters.w_z, x, &mut workspace.acc);
        matvec_into(&parameters.u_z, h, &mut workspace.scratch);
        add_assign(&mut workspace.acc, &workspace.scratch);
        add_assign(&mut workspace.acc, parameters.b_z.as_slice());
        sigmoid_assign(&mut workspace.acc);

        // gate = sigmoid(W_r x + U_r h + b_r)
        matvec_into(&parameters.w_r, x, &mut workspace.gate);
        matvec_into(&parameters.u_r, h, &mut workspace.scratch);
        add_assign(&mut workspace.gate, &workspace.scratch);
        add_assign(&mut workspace.gate, parameters.b_r.as_slice());
        sigmoid_assign(&mut workspace.gate);

        // gate = tanh(W_h x + U_h (gate ⊙ h) + b_h)
        mul_assign(&mut workspace.gate, h);
        matvec_into(&parameters.u_h, &workspace.gate, &mut workspace.scratch);
        matvec_into(&parameters.w_h, x, &mut workspace.gate);
        add_assign(&mut workspace.gate, &workspace.scratch);
        add_assign(&mut workspace.gate, parameters.b_h.as_slice());
        tanh_assign(&mut workspace.gate);

        // scratch = z_t ⊙ h̃_t
        for (scratch, (z, candidate)) in workspace
            .scratch
            .iter_mut()
            .zip(workspace.acc.iter().zip(workspace.gate.iter()))
        {
            *scratch = *z * *candidate;
        }
        // acc = (1 − z_t) ⊙ h + z_t ⊙ h̃_t
        for value in workspace.acc.iter_mut() {
            *value = 1.0 - *value;
        }
        mul_assign(&mut workspace.acc, h);
        add_assign(&mut workspace.acc, &workspace.scratch);
    }

    if !workspace.acc.iter().all(|value| value.is_finite()) {
        return Err(GruError::NonFiniteCandidate);
    }
    state.as_mut_slice().copy_from_slice(&workspace.acc);
    Ok(())
}

/// Writes `matrix · input` into `out` with the same per-row accumulation order
/// as [`Matrix::mul_vector`].
fn matvec_into(matrix: &Matrix, input: &[f32], out: &mut [f32]) {
    let cols = matrix.cols();
    let data = matrix.as_slice();
    for (row, out_row) in out.iter_mut().enumerate() {
        let start = row * cols;
        let mut sum = 0.0_f32;
        for (weight, value) in data[start..start + cols].iter().zip(input) {
            sum += weight * value;
        }
        *out_row = sum;
    }
}

fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += *src;
    }
}

fn mul_assign(dst: &mut [f32], src: &[f32]) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst *= *src;
    }
}

fn sigmoid_assign(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = sigmoid(*value);
    }
}

fn tanh_assign(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = tanh(*value);
    }
}

impl<'m> StreamingExecutor<'m, Gru> {
    /// Production streaming step: allocation-free steady state.
    ///
    /// Equivalent to the reference [`StateModel::update`] path; the test suite
    /// verifies bitwise agreement. On failure the committed state is unchanged.
    ///
    /// **Provisional:** this is a Gru-specific entry point on the generic
    /// executor type, reached because the workspace is model-owned. It is under
    /// a CRITICAL architecture review and is not a settled API.
    pub fn process_one_optimized(
        &mut self,
        observation: &Observation<Vector>,
    ) -> Result<&Vector, GruError> {
        let state = self
            .state
            .as_mut()
            .expect("executor state is present between calls");
        self.model.update_in_place(state, observation)?;
        Ok(self.state())
    }
}
