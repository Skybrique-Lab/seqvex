//! Integration: `observation -> streaming execution -> GRU -> output`.
//!
//! The execution layer is tested first against a trivial foundation fixture
//! (`SumModel`) to show it is model-agnostic, then against the GRU to show the
//! full path preserves streaming state and failure atomicity.

mod common;

use common::SumModel;
use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::recurrent::gru::{Gru, GruError, GruParameters};

fn observation(values: &[f32]) -> Observation<Vector> {
    Observation::new(Vector::from_slice(values))
}

fn params() -> GruParameters {
    GruParameters::deterministic(2, 2)
}

#[test]
fn executor_drives_a_non_gru_model() {
    let model = SumModel::new(100);
    let mut executor = StreamingExecutor::new(&model, 0_i64);
    executor.process_one(&Observation::new(3)).unwrap();
    executor.process_one(&Observation::new(4)).unwrap();
    assert_eq!(*executor.state(), 7);
}

#[test]
fn streaming_execution_matches_direct_stepping() {
    let model = Gru::new(2, 2, params()).unwrap();
    let mut reference = Gru::new(2, 2, params()).unwrap();
    let mut executor = StreamingExecutor::new(&model, Vector::zeros(2));

    for input in [[0.1, 0.2], [0.3, -0.4], [-0.5, 0.6]] {
        let observation = observation(&input);
        executor.process_one(&observation).unwrap();
        reference.step(&observation).unwrap();
        assert_eq!(executor.state(), reference.hidden());
    }
}

#[test]
fn failed_update_preserves_state_and_the_stream_continues() {
    let model = Gru::new(2, 2, params()).unwrap();
    let mut executor = StreamingExecutor::new(&model, Vector::zeros(2));
    executor.process_one(&observation(&[0.1, 0.2])).unwrap();
    let committed = executor.state().clone();

    assert_eq!(
        executor
            .process_one(&observation(&[f32::NAN, 0.0]))
            .unwrap_err(),
        GruError::NonFiniteInput
    );
    assert_eq!(executor.state(), &committed);

    // The next valid observation continues from the last committed state.
    let mut reference = Gru::new(2, 2, params()).unwrap();
    reference.step(&observation(&[0.1, 0.2])).unwrap();
    executor.process_one(&observation(&[0.4, -0.1])).unwrap();
    reference.step(&observation(&[0.4, -0.1])).unwrap();
    assert_eq!(executor.state(), reference.hidden());
}

#[test]
fn process_stream_reports_failures_without_stopping() {
    let model = Gru::new(2, 2, params()).unwrap();
    let mut executor = StreamingExecutor::new(&model, Vector::zeros(2));
    let mut failures = 0;

    let final_state = executor
        .process_stream(
            [
                observation(&[0.1, 0.2]),
                observation(&[f32::NAN, 0.0]),
                observation(&[0.3, -0.4]),
            ],
            |error| {
                assert_eq!(*error, GruError::NonFiniteInput);
                failures += 1;
            },
        )
        .clone();

    assert_eq!(failures, 1);
    let mut reference = Gru::new(2, 2, params()).unwrap();
    reference.step(&observation(&[0.1, 0.2])).unwrap();
    reference.step(&observation(&[0.3, -0.4])).unwrap();
    assert_eq!(&final_state, reference.hidden());
}

#[test]
fn reset_starts_a_new_sequence_at_the_execution_layer() {
    let model = Gru::new(2, 2, params()).unwrap();
    let mut executor = StreamingExecutor::new(&model, Vector::zeros(2));
    executor.process_one(&observation(&[0.1, 0.2])).unwrap();
    executor.reset(Vector::zeros(2));
    executor.process_one(&observation(&[0.3, -0.4])).unwrap();

    let mut fresh = Gru::new(2, 2, params()).unwrap();
    fresh.step(&observation(&[0.3, -0.4])).unwrap();
    assert_eq!(executor.state(), fresh.hidden());
}
