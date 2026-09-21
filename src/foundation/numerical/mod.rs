//! Minimal numerical foundation.
//!
//! `statistics` holds the established `f64` primitives. `vector`, `linalg`, and
//! `activations` are the `f32` substrate required by the GRU. The two numeric
//! types are kept separate on purpose: the `statistics` contracts are `f64`,
//! the substrate is `f32` (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §9–13).

pub mod activations;
pub mod linalg;
pub mod statistics;
pub mod vector;

pub use activations::{sigmoid, tanh};
pub use linalg::Matrix;
pub use statistics::{dot, mean, norm, online_mean, online_variance, sum, variance};
pub use vector::{DimensionMismatch, Vector};
