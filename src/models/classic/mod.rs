//! Classic ML model namespace.
//!
//! The first classic slice is [linear regression](linear_regression), used as
//! the architectural control case: an immutable, shareable model with no
//! model-owned scratch. The second is [decision tree](decision_tree) prediction,
//! read-only traversal of a caller-supplied trained tree, also immutable and
//! scratch-free. See `docs/ML_VERTICAL_SLICES.md`.

pub mod decision_tree;
pub mod linear_regression;

pub use decision_tree::{DecisionTree, DecisionTreeError, DecisionTreeNode};
pub use linear_regression::{LinearRegression, RegressionError};
