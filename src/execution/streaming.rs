//! Streaming execution over an ordered sequence of observations.
//!
//! This layer owns *when and how a model is stepped*, not how it computes. It
//! composes the foundation state semantics (`process_one`, `process_stream`)
//! rather than reimplementing the fold, and adds only what a model does not
//! own: retaining the committed state across calls and an explicit reset
//! boundary (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §15).
//!
//! # Provisional model ownership
//!
//! The executor borrows its model mutably so a model-owned production workspace
//! can be reached. That is **provisional** and under a CRITICAL architecture
//! review: it prevents several executors from sharing one model and couples
//! immutable, shareable weights with per-stream scratch. Do not treat this
//! signature as settled.

use crate::foundation::state::{StateModel, process_one, process_stream};

/// Drives one stateful model, holding its committed state between calls.
///
/// The state lives in an `Option` only so [`StreamingExecutor::process_stream`]
/// can satisfy the foundation's consume-and-return contract without cloning.
/// It is `Some` at every point a caller can observe it.
pub struct StreamingExecutor<'m, M: StateModel> {
    pub(crate) model: &'m mut M,
    pub(crate) state: Option<M::State>,
}

impl<'m, M: StateModel> StreamingExecutor<'m, M> {
    /// Creates an executor over `model`, starting from `initial`.
    ///
    /// The model is borrowed mutably so production-oriented paths that reuse
    /// model-owned scratch storage can be driven through the executor. That
    /// borrow is **provisional** and under a CRITICAL architecture review; see
    /// the module documentation.
    pub fn new(model: &'m mut M, initial: M::State) -> Self {
        Self {
            model,
            state: Some(initial),
        }
    }

    /// The current committed state.
    pub fn state(&self) -> &M::State {
        self.state
            .as_ref()
            .expect("executor state is present between calls")
    }

    /// Processes one observation, committing the next state on success.
    ///
    /// On failure the previous committed state is left intact.
    pub fn process_one(&mut self, observation: &M::Observation) -> Result<&M::State, M::Error> {
        let next = process_one(&*self.model, self.state(), observation)?;
        self.state = Some(next);
        Ok(self.state())
    }

    /// Processes `observations` in order, reporting failures to `on_failure`,
    /// and returns the last committed state.
    ///
    /// A failed update preserves the committed state and does not stop the
    /// stream; the next observation continues from the last valid state.
    pub fn process_stream<I, F>(&mut self, observations: I, on_failure: F) -> &M::State
    where
        I: IntoIterator<Item = M::Observation>,
        F: FnMut(&M::Error),
    {
        let initial = self
            .state
            .take()
            .expect("executor state is present between calls");
        self.state = Some(process_stream(
            &*self.model,
            initial,
            observations,
            on_failure,
        ));
        self.state()
    }

    /// Resets to an explicit initial state, so the next observation starts a
    /// new sequence rather than continuing the previous one.
    pub fn reset(&mut self, initial: M::State) {
        self.state = Some(initial);
    }
}
