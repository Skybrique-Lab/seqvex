//! Sequence ordering context.

/// The position of an observation within an ordered sequence.
///
/// Comparing `SequenceNumber` values expresses sequence ordering explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SequenceNumber(u64);

impl SequenceNumber {
    /// Creates a sequence position from a zero-based index.
    pub fn new(index: u64) -> Self {
        Self(index)
    }

    /// The underlying position.
    pub fn get(self) -> u64 {
        self.0
    }
}
