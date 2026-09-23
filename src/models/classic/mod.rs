//! Classic ML model namespace.
//!
//! The first classic slice is [linear regression](linear_regression), used as
//! the architectural control case: an immutable, shareable model with no
//! model-owned scratch. See `docs/ML_VERTICAL_SLICES.md`.

pub mod linear_regression;

pub use linear_regression::{LinearRegression, RegressionError};
