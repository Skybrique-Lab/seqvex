//! GRU behavior tests.
//!
//! The reference values are computed by an independent scalar implementation
//! (plain `Vec` loops and standard-library `exp`/`tanh`) written directly from
//! the documented gate convention in `src/models/recurrent/gru.rs`. It does not
//! call the library's kernels, so agreement is evidence the implementation is
//! correct rather than evidence the two paths share a bug.

use seqvex::foundation::numerical::{Matrix, Vector};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::recurrent::gru::{Gru, GruError, GruParameters};

// --- independent scalar reference -------------------------------------------

#[derive(Clone)]
struct RefParams {
    w_z: Vec<Vec<f32>>,
    u_z: Vec<Vec<f32>>,
    b_z: Vec<f32>,
    w_r: Vec<Vec<f32>>,
    u_r: Vec<Vec<f32>>,
    b_r: Vec<f32>,
    w_h: Vec<Vec<f32>>,
    u_h: Vec<Vec<f32>>,
    b_h: Vec<f32>,
}

fn ref_matvec(matrix: &[Vec<f32>], input: &[f32]) -> Vec<f32> {
    matrix
        .iter()
        .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
        .collect()
}

fn ref_add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

fn ref_mul(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a * b).collect()
}

fn ref_map(values: &[f32], f: impl Fn(f32) -> f32) -> Vec<f32> {
    values.iter().map(|&value| f(value)).collect()
}

fn ref_sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn ref_tanh(value: f32) -> f32 {
    value.tanh()
}

fn ref_step(params: &RefParams, previous: &[f32], input: &[f32]) -> Vec<f32> {
    let update_gate = ref_map(
        &ref_add(
            &ref_add(
                &ref_matvec(&params.w_z, input),
                &ref_matvec(&params.u_z, previous),
            ),
            &params.b_z,
        ),
        ref_sigmoid,
    );
    let reset_gate = ref_map(
        &ref_add(
            &ref_add(
                &ref_matvec(&params.w_r, input),
                &ref_matvec(&params.u_r, previous),
            ),
            &params.b_r,
        ),
        ref_sigmoid,
    );
    let candidate = ref_map(
        &ref_add(
            &ref_add(
                &ref_matvec(&params.w_h, input),
                &ref_matvec(&params.u_h, &ref_mul(&reset_gate, previous)),
            ),
            &params.b_h,
        ),
        ref_tanh,
    );
    ref_add(
        &ref_mul(&ref_map(&update_gate, |value| 1.0 - value), previous),
        &ref_mul(&update_gate, &candidate),
    )
}

fn matrix(rows: &[Vec<f32>]) -> Matrix {
    let slices: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    Matrix::from_rows(&slices).unwrap()
}

fn to_parameters(params: &RefParams) -> GruParameters {
    GruParameters {
        w_z: matrix(&params.w_z),
        u_z: matrix(&params.u_z),
        b_z: Vector::from_slice(&params.b_z),
        w_r: matrix(&params.w_r),
        u_r: matrix(&params.u_r),
        b_r: Vector::from_slice(&params.b_r),
        w_h: matrix(&params.w_h),
        u_h: matrix(&params.u_h),
        b_h: Vector::from_slice(&params.b_h),
    }
}

fn sample_params() -> RefParams {
    RefParams {
        w_z: vec![vec![0.5, -0.25], vec![0.1, 0.3]],
        u_z: vec![vec![-0.2, 0.4], vec![0.05, -0.15]],
        b_z: vec![0.1, -0.05],
        w_r: vec![vec![0.15, 0.2], vec![-0.3, 0.05]],
        u_r: vec![vec![0.25, -0.1], vec![0.2, 0.35]],
        b_r: vec![0.0, 0.05],
        w_h: vec![vec![0.4, -0.35], vec![0.2, 0.1]],
        u_h: vec![vec![0.3, 0.15], vec![-0.25, 0.45]],
        b_h: vec![-0.1, 0.2],
    }
}

fn observation(values: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(values))
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn assert_bitwise_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "index {index}");
    }
}

// --- construction -----------------------------------------------------------

#[test]
fn construction_rejects_zero_dimensions() {
    assert_eq!(
        Gru::new(0, 1, GruParameters::deterministic(1, 1)).unwrap_err(),
        GruError::ZeroDimension
    );
    assert_eq!(
        Gru::new(1, 0, GruParameters::deterministic(1, 1)).unwrap_err(),
        GruError::ZeroDimension
    );
}

#[test]
fn construction_rejects_wrong_parameter_dimensions() {
    assert_eq!(
        Gru::new(3, 2, GruParameters::deterministic(2, 2)).unwrap_err(),
        GruError::DimensionMismatch {
            expected: 3,
            actual: 2,
        }
    );
}

#[test]
fn construction_rejects_non_finite_parameters() {
    let parameters = GruParameters {
        w_z: Matrix::from_rows(&[&[f32::NAN]]).unwrap(),
        ..GruParameters::deterministic(1, 1)
    };
    assert_eq!(
        Gru::new(1, 1, parameters).unwrap_err(),
        GruError::NonFiniteParameter
    );
}

// --- one step and reference values ------------------------------------------

#[test]
fn single_step_matches_hand_computed_value() {
    // z = r = sigmoid(0) = 0.5; the reset gate zeroes h_0 in the candidate, so
    // h̃ = tanh(1); h_1 = (1 - 0.5) * 0 + 0.5 * tanh(1).
    let parameters = GruParameters {
        w_z: Matrix::from_rows(&[&[0.0]]).unwrap(),
        u_z: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_z: Vector::zeros(1),
        w_r: Matrix::from_rows(&[&[0.0]]).unwrap(),
        u_r: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_r: Vector::zeros(1),
        w_h: Matrix::from_rows(&[&[1.0]]).unwrap(),
        u_h: Matrix::from_rows(&[&[1.0]]).unwrap(),
        b_h: Vector::zeros(1),
    };
    let mut gru = Gru::new(1, 1, parameters).unwrap();
    gru.step(&observation(&[1.0])).unwrap();
    assert_close(gru.hidden().as_slice(), &[0.5 * 1.0_f32.tanh()], 1e-6);
}

#[test]
fn repeated_steps_match_independent_reference() {
    let reference = sample_params();
    let mut gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let mut hidden = vec![0.0, 0.0];
    for input in [
        vec![0.5, -0.5],
        vec![1.0, 0.25],
        vec![-0.75, 0.1],
        vec![0.0, 0.9],
    ] {
        gru.step(&observation(&input)).unwrap();
        hidden = ref_step(&reference, &hidden, &input);
        assert_close(gru.hidden().as_slice(), &hidden, 1e-5);
    }
}

#[test]
fn step_uses_previous_committed_state() {
    let reference = sample_params();
    let mut gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    gru.step(&observation(&[0.5, -0.5])).unwrap();
    let after_first = gru.hidden().clone();
    gru.step(&observation(&[1.0, 0.25])).unwrap();
    let expected = ref_step(&reference, after_first.as_slice(), &[1.0, 0.25]);
    assert_close(gru.hidden().as_slice(), &expected, 1e-5);
}

#[test]
fn state_model_update_matches_step() {
    let reference = sample_params();
    let mut gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let observation = observation(&[0.3, -0.2]);
    let expected = gru.update(&Vector::zeros(2), &observation).unwrap();
    gru.step(&observation).unwrap();
    assert_close(gru.hidden().as_slice(), expected.as_slice(), 0.0);
}

#[test]
fn sequence_context_does_not_change_the_computation() {
    let reference = sample_params();
    let mut plain = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let mut ordered = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    plain.step(&observation(&[0.5, -0.5])).unwrap();
    ordered
        .step(
            &observation(&[0.5, -0.5])
                .with_sequence(seqvex::foundation::observation::SequenceNumber::new(7)),
        )
        .unwrap();
    assert_eq!(plain.hidden(), ordered.hidden());
}

// --- long sequences, reset ---------------------------------------------------

#[test]
fn long_sequence_matches_independent_reference() {
    let reference = sample_params();
    let mut gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let mut hidden = vec![0.0, 0.0];
    for step in 0..500 {
        let input = vec![(step as f32 * 0.37).sin(), (step as f32 * 0.11).cos()];
        gru.step(&observation(&input)).unwrap();
        hidden = ref_step(&reference, &hidden, &input);
    }
    assert_close(gru.hidden().as_slice(), &hidden, 1e-4);
}

#[test]
fn reset_starts_a_new_sequence() {
    let reference = sample_params();
    let mut gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    gru.step(&observation(&[0.5, -0.5])).unwrap();
    gru.reset();
    assert_close(gru.hidden().as_slice(), &[0.0, 0.0], 0.0);

    let input = [0.2, -0.7];
    gru.step(&observation(&input)).unwrap();

    let mut fresh = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    fresh.step(&observation(&input)).unwrap();
    assert_close(gru.hidden().as_slice(), fresh.hidden().as_slice(), 1e-6);

    let mut continued = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    continued.step(&observation(&[0.5, -0.5])).unwrap();
    continued.step(&observation(&input)).unwrap();
    assert_ne!(gru.hidden().as_slice(), continued.hidden().as_slice());
}

// --- failure behavior --------------------------------------------------------

#[test]
fn non_finite_input_is_rejected_and_state_is_preserved() {
    let reference = sample_params();
    let mut gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    gru.step(&observation(&[0.5, -0.5])).unwrap();
    let committed = gru.hidden().clone();
    assert_eq!(
        gru.step(&observation(&[f32::NAN, 0.0])).unwrap_err(),
        GruError::NonFiniteInput
    );
    assert_close(gru.hidden().as_slice(), committed.as_slice(), 0.0);
}

#[test]
fn wrong_input_length_is_rejected_and_state_is_preserved() {
    let reference = sample_params();
    let mut gru = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    gru.step(&observation(&[0.5, -0.5])).unwrap();
    let committed = gru.hidden().clone();
    assert_eq!(
        gru.step(&observation(&[1.0])).unwrap_err(),
        GruError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
    assert_close(gru.hidden().as_slice(), committed.as_slice(), 0.0);
}

#[test]
fn non_finite_candidate_is_rejected_and_state_is_preserved() {
    // inf + (-inf) in the update gate produces NaN from finite inputs and
    // finite parameters, exercising the candidate validation boundary.
    let parameters = GruParameters {
        w_z: Matrix::from_rows(&[&[f32::MAX, f32::MAX]]).unwrap(),
        u_z: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_z: Vector::zeros(1),
        w_r: Matrix::from_rows(&[&[0.0, 0.0]]).unwrap(),
        u_r: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_r: Vector::zeros(1),
        w_h: Matrix::from_rows(&[&[0.0, 0.0]]).unwrap(),
        u_h: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_h: Vector::zeros(1),
    };
    let mut gru = Gru::new(2, 1, parameters).unwrap();
    assert_eq!(
        gru.step(&observation(&[f32::MAX, -f32::MAX])).unwrap_err(),
        GruError::NonFiniteCandidate
    );
    assert_close(gru.hidden().as_slice(), &[0.0], 0.0);
}

// --- determinism -------------------------------------------------------------

#[test]
fn deterministic_parameters_are_reproducible() {
    assert_eq!(
        GruParameters::deterministic(3, 2),
        GruParameters::deterministic(3, 2)
    );

    let mut left = Gru::new(2, 2, GruParameters::deterministic(2, 2)).unwrap();
    let mut right = Gru::new(2, 2, GruParameters::deterministic(2, 2)).unwrap();
    left.step(&observation(&[0.4, 0.6])).unwrap();
    right.step(&observation(&[0.4, 0.6])).unwrap();
    assert_eq!(left.hidden(), right.hidden());
}

// --- production path equivalence --------------------------------------------

#[test]
fn production_step_matches_reference_over_long_sequence() {
    let mut reference = Gru::new(8, 16, GruParameters::deterministic(8, 16)).unwrap();
    let mut production = Gru::new(8, 16, GruParameters::deterministic(8, 16)).unwrap();

    for step in 0..500 {
        let input = Vector::from_fn(8, |i| ((step as f32) * 0.13 + (i as f32) * 0.07).sin());
        let observation = Observation::new(input);
        reference.step(&observation).unwrap();
        production.step_in_place(&observation).unwrap();
        assert_bitwise_eq(
            production.hidden().as_slice(),
            reference.hidden().as_slice(),
        );

        if step == 250 {
            reference.reset();
            production.reset();
            assert_bitwise_eq(
                production.hidden().as_slice(),
                reference.hidden().as_slice(),
            );
        }
    }
}

#[test]
fn production_update_in_place_matches_update_bit_for_bit() {
    let reference = sample_params();
    let mut model = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    let mut update_state = Vector::zeros(2);
    let mut in_place_state = Vector::zeros(2);

    for step in 0..64 {
        let input = vec![(step as f32 * 0.23).sin(), (step as f32 * 0.07).cos()];
        let observation = observation(&input);
        update_state = model.update(&update_state, &observation).unwrap();
        model
            .update_in_place(&mut in_place_state, &observation)
            .unwrap();
        assert_bitwise_eq(in_place_state.as_slice(), update_state.as_slice());
    }
}

#[test]
fn production_step_rejects_non_finite_input_and_preserves_state() {
    let reference = sample_params();
    let mut model = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    model.step_in_place(&observation(&[0.5, -0.5])).unwrap();
    let committed = model.hidden().clone();

    assert_eq!(
        model
            .step_in_place(&observation(&[f32::NAN, 0.0]))
            .unwrap_err(),
        GruError::NonFiniteInput
    );
    assert_bitwise_eq(model.hidden().as_slice(), committed.as_slice());
}

#[test]
fn production_step_rejects_wrong_input_length_and_preserves_state() {
    let reference = sample_params();
    let mut model = Gru::new(2, 2, to_parameters(&reference)).unwrap();
    model.step_in_place(&observation(&[0.5, -0.5])).unwrap();
    let committed = model.hidden().clone();

    assert_eq!(
        model.step_in_place(&observation(&[1.0])).unwrap_err(),
        GruError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
    assert_bitwise_eq(model.hidden().as_slice(), committed.as_slice());
}

#[test]
fn production_step_rejects_non_finite_candidate_and_preserves_state() {
    let parameters = GruParameters {
        w_z: Matrix::from_rows(&[&[f32::MAX, f32::MAX]]).unwrap(),
        u_z: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_z: Vector::zeros(1),
        w_r: Matrix::from_rows(&[&[0.0, 0.0]]).unwrap(),
        u_r: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_r: Vector::zeros(1),
        w_h: Matrix::from_rows(&[&[0.0, 0.0]]).unwrap(),
        u_h: Matrix::from_rows(&[&[0.0]]).unwrap(),
        b_h: Vector::zeros(1),
    };
    let mut model = Gru::new(2, 1, parameters).unwrap();

    assert_eq!(
        model
            .step_in_place(&observation(&[f32::MAX, -f32::MAX]))
            .unwrap_err(),
        GruError::NonFiniteCandidate
    );
    assert_bitwise_eq(model.hidden().as_slice(), &[0.0]);
}
