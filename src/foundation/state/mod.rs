//! State, transitions, and failure-preserving execution.

pub mod atomicity;
pub mod transition;

pub use atomicity::process_stream;
pub use transition::{BatchFailure, StateModel, process_batch, process_one};
