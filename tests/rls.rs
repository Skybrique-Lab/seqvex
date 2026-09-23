//! RLS behavior tests.
//!
//! The independent oracle is a scalar `Vec<f32>` implementation written from
//! the documented equations and accumulating in the same order as the library.
//! All state comparisons against it are bitwise, because both paths use the same
//! operations in the same order.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::{Matrix, Vector};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::{StateModel, process_batch, process_one, process_stream};
use seqvex::models::online::rls::{Rls, RlsError, RlsSample, RlsState};

fn model(dimension: usize) -> Rls {
    Rls::new(dimension, 0.99, 1.0e4).unwrap()
}

fn sample(features: &[f32], target: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn state(w: &[f32], p: &[&[f32]]) -> RlsState {
    RlsState {
        w: Vector::from_slice(w),
        p: Matrix::from_rows(p).unwrap(),
    }
}

fn assert_f32_bits_eq(actual: f32, expected: f32) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "expected bitwise equality: {actual} vs {expected}"
    );
}

fn assert_state_bitwise_eq(actual: &RlsState, expected: &RlsState) {
    assert_eq!(actual.w.len(), expected.w.len(), "w length mismatch");
    for (actual, expected) in actual.w.as_slice().iter().zip(expected.w.as_slice()) {
        assert_f32_bits_eq(*actual, *expected);
    }
    assert_eq!(
        actual.p.as_slice().len(),
        expected.p.as_slice().len(),
        "P length mismatch"
    );
    for (actual, expected) in actual.p.as_slice().iter().zip(expected.p.as_slice()) {
        assert_f32_bits_eq(*actual, *expected);
    }
}

// --- independent scalar reference -------------------------------------------

struct RefState {
    w: Vec<f32>,
    p: Vec<Vec<f32>>,
}

fn ref_initial(dimension: usize, delta: f32) -> RefState {
    RefState {
        w: vec![0.0; dimension],
        p: (0..dimension)
            .map(|row| {
                (0..dimension)
                    .map(|col| if row == col { delta } else { 0.0 })
                    .collect()
            })
            .collect(),
    }
}

fn ref_step(state: &RefState, x: &[f32], y: f32, lambda: f32) -> RefState {
    let v: Vec<f32> = state
        .p
        .iter()
        .map(|row| row.iter().zip(x).map(|(p, x)| p * x).sum())
        .collect();
    let denominator = lambda + x.iter().zip(&v).map(|(x, v)| x * v).sum::<f32>();
    let prediction = state.w.iter().zip(x).map(|(w, x)| w * x).sum::<f32>();
    let error = y - prediction;
    let next_w: Vec<f32> = state
        .w
        .iter()
        .zip(&v)
        .map(|(w, v)| w + (v / denominator) * error)
        .collect();
    let next_p: Vec<Vec<f32>> = state
        .p
        .iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .map(|(col, p)| (p - (v[row] * v[col]) / denominator) / lambda)
                .collect()
        })
        .collect();
    RefState {
        w: next_w,
        p: next_p,
    }
}

fn to_state(reference: &RefState) -> RlsState {
    let rows: Vec<&[f32]> = reference.p.iter().map(Vec::as_slice).collect();
    RlsState {
        w: Vector::from_slice(&reference.w),
        p: Matrix::from_rows(&rows).unwrap(),
    }
}

// --- mathematical correctness -------------------------------------------------

#[test]
fn hand_computed_two_feature_updates_are_exact() {
    let model = Rls::new(2, 1.0, 1.0).unwrap();
    let mut state = model.initial_state();

    state = model.update(&state, &sample(&[1.0, 0.0], 1.0)).unwrap();
    assert_eq!(state.w.as_slice(), &[0.5, 0.0]);
    assert_eq!(state.p.as_slice(), &[0.5, 0.0, 0.0, 1.0]);

    state = model.update(&state, &sample(&[0.0, 1.0], 3.0)).unwrap();
    assert_eq!(state.w.as_slice(), &[0.5, 1.5]);
    assert_eq!(state.p.as_slice(), &[0.5, 0.0, 0.0, 0.5]);

    assert_eq!(
        model.predict(&state, &Vector::from_slice(&[1.0, 1.0])),
        Ok(2.0)
    );
}

#[test]
fn sequence_matches_independent_scalar_reference_bitwise() {
    let model = Rls::new(3, 0.99, 1.0e4).unwrap();
    let mut state = model.initial_state();
    let mut reference = ref_initial(3, 1.0e4);

    for step in 0..200 {
        let features = [
            (step as f32 * 0.37).sin(),
            (step as f32 * 0.11).cos(),
            (step as f32 * 0.53).sin(),
        ];
        let target = (step as f32 * 0.29).cos();
        let observation = sample(&features, target);

        state = model.update(&state, &observation).unwrap();
        reference = ref_step(&reference, &features, target, 0.99);
        assert_state_bitwise_eq(&state, &to_state(&reference));
    }
}

#[test]
fn predict_uses_committed_weights() {
    let model = Rls::new(1, 1.0, 1.0).unwrap();
    let mut state = model.initial_state();
    state = model.update(&state, &sample(&[2.0], 4.0)).unwrap();
    let prediction = model.predict(&state, &Vector::from_slice(&[3.0])).unwrap();
    assert_f32_bits_eq(prediction, state.w.as_slice()[0] * 3.0);
}

// --- dimension validation -----------------------------------------------------

#[test]
fn zero_dimension_is_rejected() {
    assert_eq!(Rls::new(0, 0.99, 1.0), Err(RlsError::ZeroDimension));
}

#[test]
fn dimension_mismatch_is_reported_for_features_weights_and_covariance() {
    let model = Rls::new(2, 0.99, 1.0).unwrap();
    let good = model.initial_state();

    assert_eq!(
        model.update(&good, &sample(&[1.0], 1.0)).unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let short_w = state(&[0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
    assert_eq!(
        model
            .update(&short_w, &sample(&[1.0, 2.0], 1.0))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let short_rows = state(&[0.0, 0.0], &[&[1.0, 0.0]]);
    assert_eq!(
        model
            .update(&short_rows, &sample(&[1.0, 2.0], 1.0))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let short_cols = state(&[0.0, 0.0], &[&[1.0], &[0.0]]);
    assert_eq!(
        model
            .update(&short_cols, &sample(&[1.0, 2.0], 1.0))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );

    assert_eq!(
        model
            .predict(&good, &Vector::from_slice(&[1.0]))
            .unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
}

// --- parameter validation -----------------------------------------------------

#[test]
fn invalid_forgetting_factor_is_rejected() {
    for lambda in [0.0, -0.5, 1.5, f32::NAN, f32::INFINITY] {
        assert_eq!(
            Rls::new(1, lambda, 1.0),
            Err(RlsError::InvalidForgettingFactor),
            "lambda = {lambda}"
        );
    }
    assert!(Rls::new(1, 1.0, 1.0).is_ok());
}

#[test]
fn invalid_initial_covariance_is_rejected() {
    for delta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            Rls::new(1, 0.5, delta),
            Err(RlsError::InvalidInitialCovariance),
            "delta = {delta}"
        );
    }
}

// --- finite and denominator validation ---------------------------------------

#[test]
fn non_finite_observation_is_rejected() {
    let model = Rls::new(2, 1.0, 1.0).unwrap();
    let state = model.initial_state();
    assert_eq!(
        model
            .update(&state, &sample(&[f32::NAN, 0.0], 0.0))
            .unwrap_err(),
        RlsError::NonFiniteInput
    );
    assert_eq!(
        model
            .update(&state, &sample(&[1.0, 2.0], f32::INFINITY))
            .unwrap_err(),
        RlsError::NonFiniteInput
    );
    assert_eq!(
        model
            .predict(&state, &Vector::from_slice(&[f32::INFINITY, 0.0]))
            .unwrap_err(),
        RlsError::NonFiniteInput
    );
}

#[test]
fn non_positive_denominator_is_rejected() {
    let model = Rls::new(1, 1.0, 1.0).unwrap();
    // P = [-1] makes d = λ + xᵀPx = 1 - 1 = 0.
    let corrupted = state(&[0.0], &[&[-1.0]]);
    assert_eq!(
        model.update(&corrupted, &sample(&[1.0], 1.0)).unwrap_err(),
        RlsError::InvalidDenominator
    );

    // A non-finite P makes d non-finite.
    let nan_p = state(&[0.0], &[&[f32::NAN]]);
    assert_eq!(
        model.update(&nan_p, &sample(&[1.0], 1.0)).unwrap_err(),
        RlsError::InvalidDenominator
    );
}

#[test]
fn non_finite_candidate_is_rejected() {
    // A tiny δ keeps d finite while w = f32::MAX overflows the prediction error.
    let model = Rls::new(1, 1.0, 1.0e-30).unwrap();
    let explosive = state(&[f32::MAX], &[&[1.0e-30]]);
    assert_eq!(
        model
            .update(&explosive, &sample(&[1.0], -f32::MAX))
            .unwrap_err(),
        RlsError::NonFiniteCandidate
    );
}

// --- atomicity ----------------------------------------------------------------

#[test]
fn failed_update_leaves_committed_state_bitwise_unchanged() {
    let model = Rls::new(2, 0.99, 10.0).unwrap();
    let mut committed = model
        .update(&model.initial_state(), &sample(&[1.0, 2.0], 1.0))
        .unwrap();
    let snapshot = committed.clone();

    assert!(model.update(&committed, &sample(&[1.0], 0.0)).is_err());
    assert_state_bitwise_eq(&committed, &snapshot);

    assert!(
        model
            .update(&committed, &sample(&[f32::NAN, 0.0], 0.0))
            .is_err()
    );
    assert_state_bitwise_eq(&committed, &snapshot);

    // The next valid observation continues from the untouched committed state.
    committed = model
        .update(&committed, &sample(&[0.5, -0.5], 1.0))
        .unwrap();
    let expected = model.update(&snapshot, &sample(&[0.5, -0.5], 1.0)).unwrap();
    assert_state_bitwise_eq(&committed, &expected);
}

#[test]
fn executor_failure_preserves_state_and_the_stream_continues() {
    let mut streaming_model = Rls::new(2, 0.99, 10.0).unwrap();
    let reference = streaming_model.clone();
    let initial = streaming_model.initial_state();
    let mut executor = StreamingExecutor::new(&mut streaming_model, initial);

    executor.process_one(&sample(&[1.0, 2.0], 1.0)).unwrap();
    let committed = executor.state().clone();

    assert_eq!(
        executor.process_one(&sample(&[1.0], 0.0)).unwrap_err(),
        RlsError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
    assert_state_bitwise_eq(executor.state(), &committed);

    executor.process_one(&sample(&[0.5, -0.5], 1.0)).unwrap();
    let expected = reference
        .update(&committed, &sample(&[0.5, -0.5], 1.0))
        .unwrap();
    assert_state_bitwise_eq(executor.state(), &expected);
}

// --- sequential equivalence, reset, process_stream ----------------------------

#[test]
fn process_stream_matches_repeated_update_bitwise() {
    let model = Rls::new(2, 0.5, 4.0).unwrap();
    let observations = vec![
        sample(&[1.0, 0.0], 1.0),
        sample(&[0.0, 1.0], -2.0),
        sample(&[0.5, 0.5], 3.0),
    ];

    let mut direct = model.initial_state();
    for observation in &observations {
        direct = model.update(&direct, observation).unwrap();
    }

    let streamed = process_stream(&model, model.initial_state(), observations, |_| {
        panic!("unexpected failure")
    });
    assert_state_bitwise_eq(&direct, &streamed);
}

#[test]
fn process_stream_continues_after_a_failure() {
    let model = Rls::new(2, 1.0, 4.0).unwrap();
    let observations = vec![
        sample(&[1.0, 0.0], 1.0),
        sample(&[f32::NAN, 0.0], 1.0),
        sample(&[0.0, 1.0], 2.0),
    ];
    let mut failures = 0;

    let final_state = process_stream(&model, model.initial_state(), observations, |error| {
        assert_eq!(*error, RlsError::NonFiniteInput);
        failures += 1;
    });

    assert_eq!(failures, 1);
    let mut expected = model.initial_state();
    expected = model.update(&expected, &sample(&[1.0, 0.0], 1.0)).unwrap();
    expected = model.update(&expected, &sample(&[0.0, 1.0], 2.0)).unwrap();
    assert_state_bitwise_eq(&final_state, &expected);
}

#[test]
fn reset_starts_a_new_sequence() {
    let mut streaming_model = Rls::new(2, 0.9, 2.0).unwrap();
    let reference = streaming_model.clone();
    let initial = streaming_model.initial_state();
    let mut executor = StreamingExecutor::new(&mut streaming_model, initial.clone());

    executor.process_one(&sample(&[1.0, 2.0], 1.0)).unwrap();
    executor.reset(initial.clone());
    assert_state_bitwise_eq(executor.state(), &initial);

    executor.process_one(&sample(&[0.5, 0.5], 2.0)).unwrap();
    let expected = reference
        .update(&reference.initial_state(), &sample(&[0.5, 0.5], 2.0))
        .unwrap();
    assert_state_bitwise_eq(executor.state(), &expected);
}

// --- bounded micro-batch (foundation ordered fold) ----------------------------

#[test]
fn bounded_fold_matches_repeated_process_one_bitwise() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let batch = vec![
        sample(&[1.0, 0.0], 1.0),
        sample(&[0.0, 1.0], -2.0),
        sample(&[0.5, 0.5], 3.0),
    ];

    let folded = process_batch(&model, model.initial_state(), batch.clone()).unwrap();

    let mut manual = model.initial_state();
    for observation in &batch {
        manual = process_one(&model, &manual, observation).unwrap();
    }
    assert_state_bitwise_eq(&folded, &manual);
}

#[test]
fn bounded_fold_stops_on_first_failure_and_keeps_last_valid_state() {
    let model = Rls::new(2, 0.95, 5.0).unwrap();
    let first = sample(&[1.0, 0.0], 1.0);
    let bad = sample(&[f32::NAN, 0.0], 1.0);
    let third = sample(&[0.0, 1.0], 2.0);

    let failure = process_batch(
        &model,
        model.initial_state(),
        vec![first.clone(), bad, third],
    )
    .unwrap_err();

    assert_eq!(failure.error, RlsError::NonFiniteInput);
    let expected = model.update(&model.initial_state(), &first).unwrap();
    assert_state_bitwise_eq(&failure.state, &expected);
}

// --- shared model, independent states (architecture evidence) -----------------

#[test]
fn one_shared_model_drives_independent_stream_states() {
    let model = Rls::new(2, 0.9, 3.0).unwrap();
    let sequence_a = vec![sample(&[1.0, 0.0], 1.0), sample(&[0.0, 1.0], 2.0)];
    let sequence_b = vec![sample(&[0.0, 1.0], 3.0), sample(&[1.0, 0.5], -1.0)];

    let mut state_a = model.initial_state();
    for observation in &sequence_a {
        state_a = process_one(&model, &state_a, observation).unwrap();
    }
    let state_b = process_stream(&model, model.initial_state(), sequence_b.clone(), |_| {
        panic!("unexpected failure")
    });

    assert_ne!(state_a, state_b);

    let mut reference_a = model.initial_state();
    for observation in &sequence_a {
        reference_a = model.update(&reference_a, observation).unwrap();
    }
    assert_state_bitwise_eq(&state_a, &reference_a);

    let mut reference_b = model.initial_state();
    for observation in &sequence_b {
        reference_b = model.update(&reference_b, observation).unwrap();
    }
    assert_state_bitwise_eq(&state_b, &reference_b);
}

// --- construction and sequence context ----------------------------------------

#[test]
fn initial_state_is_zero_weights_and_scaled_identity() {
    let state = model(3).initial_state();
    assert_eq!(state.w.as_slice(), &[0.0, 0.0, 0.0]);
    assert_eq!(
        state.p.as_slice(),
        &[1.0e4, 0.0, 0.0, 0.0, 1.0e4, 0.0, 0.0, 0.0, 1.0e4]
    );
}

#[test]
fn sequence_context_does_not_change_the_computation() {
    let model = Rls::new(1, 1.0, 1.0).unwrap();
    let plain = model
        .update(&model.initial_state(), &sample(&[2.0], 4.0))
        .unwrap();
    let ordered = model
        .update(
            &model.initial_state(),
            &sample(&[2.0], 4.0)
                .with_sequence(seqvex::foundation::observation::SequenceNumber::new(7)),
        )
        .unwrap();
    assert_state_bitwise_eq(&plain, &ordered);
}

// --- long-run numerical stability ---------------------------------------------

#[test]
fn long_run_stays_finite_symmetric_and_matches_reference() {
    let model = Rls::new(4, 0.999, 1.0e3).unwrap();
    let dimension = model.dimension();
    let mut state = model.initial_state();
    let mut reference = ref_initial(dimension, 1.0e3);
    let mut max_prediction = 0.0_f32;

    for step in 0..10_000 {
        let features = [
            (step as f32 * 0.17).sin(),
            (step as f32 * 0.07).cos(),
            (step as f32 * 0.31).sin(),
            (step as f32 * 0.23).cos(),
        ];
        let target = (step as f32 * 0.13).sin();
        let observation = sample(&features, target);

        state = model.update(&state, &observation).unwrap();
        reference = ref_step(&reference, &features, target, 0.999);

        assert!(
            state.w.as_slice().iter().all(|value| value.is_finite()),
            "non-finite w at step {step}"
        );
        assert!(
            state.p.as_slice().iter().all(|value| value.is_finite()),
            "non-finite P at step {step}"
        );
        let prediction = model
            .predict(&state, &observation.value().features)
            .unwrap();
        max_prediction = max_prediction.max(prediction.abs());
    }

    assert_state_bitwise_eq(&state, &to_state(&reference));

    // Symmetry is preserved exactly: v_i * v_j == v_j * v_i in IEEE f32.
    for row in 0..dimension {
        for col in 0..dimension {
            assert_f32_bits_eq(
                state.p.get(row, col).unwrap(),
                state.p.get(col, row).unwrap(),
            );
        }
    }
    println!("RLS long run: max |prediction| = {max_prediction}");
    assert!(max_prediction.is_finite());
}
