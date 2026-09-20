//! Seqvex — a streaming-first ML/RL framework for sequential, non-IID data.
//!
//! This crate currently contains only the foundational semantics scaffold:
//! observation, ordering, streaming state, state transitions, and failure
//! atomicity. ML/RL algorithms, hardware placement, and storage design are
//! deliberately out of scope.
//!
//! The canonical development contract is `docs/DEVELOPMENT.md`.

#![forbid(unsafe_code)]

pub mod foundation;
