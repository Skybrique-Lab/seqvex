//! State transition contract and single / bounded micro-batch execution.
//!
//! The transition mechanism here is deliberately provisional. A candidate next
//! state is represented as a returned value rather than by mutating the current
//! state, which makes failure atomicity structural. Whether production updates
//! are transactional, in-place, copy-on-write, or otherwise remains a deferred
//! decision (see `docs/FAILURE_AND_RECOVERY.md` §16).

/// A stateful computation.
///
/// Given the current valid state and one observation, an implementation
/// produces a candidate next state or a classified failure. It must not mutate
/// the current state, so a failed update cannot commit partial state.
pub trait StateModel {
    /// The committed state carried between observations.
    type State;
    /// The input accepted by this model.
    type Observation;
    /// The failure reported by this model.
    type Error;

    /// Computes the candidate next state for `observation`.
    fn update(
        &self,
        state: &Self::State,
        observation: &Self::Observation,
    ) -> Result<Self::State, Self::Error>;
}

/// Single-observation execution: the smallest semantic execution unit.
pub fn process_one<M: StateModel>(
    model: &M,
    state: &M::State,
    observation: &M::Observation,
) -> Result<M::State, M::Error> {
    model.update(state, observation)
}

/// A rejected micro-batch: the failure and the last committed valid state.
///
/// Carrying the state back keeps the last committed valid state identifiable
/// after a failed micro-batch, so processing can continue from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchFailure<S, E> {
    /// The last state committed before the failing observation.
    pub state: S,
    /// The failure that rejected the batch.
    pub error: E,
}

/// Bounded micro-batch execution.
///
/// The batch is applied in observation order. Execution granularity may change
/// for efficiency, but stream and state semantics must not. On success the
/// committed state is returned; on failure the last committed valid state is
/// returned with the failure so it remains identifiable.
///
/// ponytail: this is an ordered sequential fold, not a vectorized
/// implementation. It is the reference semantic any future vectorized or
/// hardware-accelerated micro-batch path must preserve.
pub fn process_batch<M, I>(
    model: &M,
    state: M::State,
    batch: I,
) -> Result<M::State, BatchFailure<M::State, M::Error>>
where
    M: StateModel,
    I: IntoIterator<Item = M::Observation>,
{
    let mut current = state;
    for observation in batch {
        match model.update(&current, &observation) {
            Ok(next) => current = next,
            Err(error) => {
                return Err(BatchFailure {
                    state: current,
                    error,
                });
            }
        }
    }
    Ok(current)
}
