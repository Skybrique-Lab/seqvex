//! Minimal numerical foundation (contract only).
//!
//! The functions below are contract points specified by ignored tests in
//! `tests/numerical.rs`. They are intentionally not implemented in the
//! foundation scaffold.

pub mod statistics;

pub use statistics::{dot, mean, norm, online_mean, online_variance, sum, variance};
