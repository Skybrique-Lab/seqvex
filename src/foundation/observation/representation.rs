//! Observation representation.

use crate::foundation::observation::ordering::SequenceNumber;

/// A single unit of input to a stateful computation.
///
/// An observation carries the input a computation requires. Sequence context is
/// optional: not every observation needs a position, and observations do not
/// require a timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation<T> {
    value: T,
    sequence: Option<SequenceNumber>,
}

impl<T> Observation<T> {
    /// Creates an observation carrying `value` with no sequence context.
    pub fn new(value: T) -> Self {
        Self {
            value,
            sequence: None,
        }
    }

    /// Attaches sequence context to this observation.
    pub fn with_sequence(mut self, sequence: SequenceNumber) -> Self {
        self.sequence = Some(sequence);
        self
    }

    /// The input carried by this observation.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// The optional sequence position of this observation.
    pub fn sequence(&self) -> Option<SequenceNumber> {
        self.sequence
    }
}
