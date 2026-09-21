//! Scalar activation functions required by the GRU.
//!
//! Two functions, implemented over `f32` and relying on the standard library's
//! `exp`/`tanh`. This is not an activation framework
//! (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §13).

/// Logistic sigmoid: `1 / (1 + e^{-x})`.
///
/// Saturates to `0.0`/`1.0` for large magnitudes and propagates `NaN`.
pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Hyperbolic tangent.
///
/// Saturates to `-1.0`/`1.0` for large magnitudes and propagates `NaN`.
pub fn tanh(value: f32) -> f32 {
    value.tanh()
}
