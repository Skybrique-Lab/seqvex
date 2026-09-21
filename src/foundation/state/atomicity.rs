//! Failure-preserving streaming execution.
//!
//! A failed update must never silently commit invalid or partial state. Because
//! transitions return a candidate rather than mutating, the last committed
//! valid state remains the active state after a failure, and processing can
//! continue from it.

use crate::foundation::state::transition::StateModel;

/// Replays `observations` in order, starting from `initial`.
///
/// A successful update advances the state. A failed update preserves the last
/// committed valid state, is reported to `on_failure`, and does not stop the
/// stream. The next observation therefore continues from the last valid state,
/// which is also the value returned.
///
/// Failures are handed to `on_failure` as they occur instead of being retained,
/// so a long stream with recurring failures cannot grow memory without bound.
/// Any retention, retry, or recovery policy belongs to the caller.
pub fn process_stream<M, I, F>(
    model: &M,
    initial: M::State,
    observations: I,
    mut on_failure: F,
) -> M::State
where
    M: StateModel,
    I: IntoIterator<Item = M::Observation>,
    F: FnMut(&M::Error),
{
    let mut state = initial;
    for observation in observations {
        match model.update(&state, &observation) {
            Ok(next) => state = next,
            Err(error) => on_failure(&error),
        }
    }
    state
}
