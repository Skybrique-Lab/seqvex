//! Shared fixtures for the foundation integration tests.
//!
//! Each integration test binary compiles this module independently, so not
//! every fixture is used by every binary.
#![allow(dead_code)]

use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;

/// Failure classification used by the test fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureError {
    InvalidInput,
    NumericalFailure,
    InvariantViolation,
}

/// Additive model over non-negative values with an upper state bound.
pub struct SumModel {
    /// Largest state the model may commit.
    pub max: i64,
}

impl SumModel {
    pub fn new(max: i64) -> Self {
        Self { max }
    }
}

impl StateModel for SumModel {
    type State = i64;
    type Observation = Observation<i64>;
    type Error = FixtureError;

    fn update(&self, state: &i64, observation: &Observation<i64>) -> Result<i64, FixtureError> {
        let value = *observation.value();
        if value < 0 {
            return Err(FixtureError::InvalidInput);
        }
        let candidate = *state + value;
        if candidate > self.max {
            return Err(FixtureError::InvariantViolation);
        }
        Ok(candidate)
    }
}

/// Order-sensitive model that concatenates observations as decimal digits.
pub struct DigitsModel;

impl StateModel for DigitsModel {
    type State = i64;
    type Observation = Observation<i64>;
    type Error = FixtureError;

    fn update(&self, state: &i64, observation: &Observation<i64>) -> Result<i64, FixtureError> {
        let digit = *observation.value();
        if !(0..=9).contains(&digit) {
            return Err(FixtureError::InvalidInput);
        }
        Ok(*state * 10 + digit)
    }
}

/// Floating-point model that rejects non-finite values.
pub struct FloatModel;

impl StateModel for FloatModel {
    type State = f64;
    type Observation = Observation<f64>;
    type Error = FixtureError;

    fn update(&self, state: &f64, observation: &Observation<f64>) -> Result<f64, FixtureError> {
        let value = *observation.value();
        if !value.is_finite() {
            return Err(FixtureError::NumericalFailure);
        }
        let candidate = *state + value;
        if !candidate.is_finite() {
            return Err(FixtureError::NumericalFailure);
        }
        Ok(candidate)
    }
}
