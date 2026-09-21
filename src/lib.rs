//! Seqvex — a streaming-first ML/RL framework for sequential, non-IID data.
//!
//! This crate contains the foundational semantics (observation, ordering,
//! streaming state, state transitions, and failure atomicity) plus the first
//! vertical slice built on them: a small `f32` numerical substrate, streaming
//! execution, and a stateful GRU (`models::recurrent::gru`).
//!
//! The GRU is the slice's first model, not the framework's model abstraction;
//! hardware placement and storage design remain deliberately out of scope.
//!
//! The canonical development contract is `docs/DEVELOPMENT.md`.

#![forbid(unsafe_code)]

pub mod execution;
pub mod foundation;
pub mod models;
