//! Classic ML model namespace.
//!
//! The first classic slice is [linear regression](linear_regression), used as
//! the architectural control case: an immutable, shareable model with no
//! model-owned scratch. The second is [decision tree](decision_tree) prediction,
//! read-only traversal of a caller-supplied trained tree, also immutable and
//! scratch-free. The third is [KNN](knn) prediction over stored reference
//! observations, immutable and shareable with only transient per-query scratch.
//! See `docs/ML_VERTICAL_SLICES.md`.

pub mod decision_tree;
pub mod knn;
pub mod linear_regression;

pub use decision_tree::{DecisionTree, DecisionTreeError, DecisionTreeNode};
pub use knn::{Knn, KnnError, Reference};
pub use linear_regression::{LinearRegression, RegressionError};
