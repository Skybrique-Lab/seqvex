//! Numerical contracts, specified but not implemented in the scaffold.
//!
//! `docs/KILO_SCAFFOLD.md` requires these contracts to be specified before they
//! are implemented. Each function documents its intended behavior and is
//! exercised by an ignored test in `tests/numerical.rs`.
//!
//! The signatures are provisional: they are the simplest interface that
//! expresses a stable requirement, not a frozen API.
#![allow(unused_variables)] // signatures are contract specifications awaiting implementation

use core::error;
use std::{arch::x86_64, collections::btree_map::Values, fmt::Alignment::Left};

/// Sum of all values; the empty slice sums to `0.0`.
pub fn sum(values: &[f64]) -> f64 {
    //todo!("contract: sum of values; empty slice = 0.0")
    if values.is_empty() {
        0.0 // Contract requirement
    } else {
        values.iter().sum()
    }
    
}

/// Arithmetic mean of `values`; undefined for an empty slice.
pub fn mean(values: &[f64]) -> f64 {
    //todo!("contract: arithmetic mean; empty slice undefined")
    assert!(!values.is_empty(), "empty slice undefined");
        return sum(values)/ values.len() as f64
}

/// Population variance of `values`; must be non-negative.
pub fn variance(values: &[f64]) -> f64 {
    //todo!("contract: population variance; >= 0")
    assert!(!values.is_empty(), "empty slice undefined");
        let mean = mean(values) as f64;
        // Auto-vectorization targets this cleanly
        let square_diff_sum: f64 = values.iter()
        .map(|&x|{ 
            let diff = x - mean;
            diff * diff
        }).sum();
        let result = square_diff_sum / values.len() as f64;
        //non-negative validation
        assert!(result >= 0.0, "must be non-negative");
        return result;
    
}

/// Dot product of two equal-length slices.
pub fn dot(left: &[f64], right: &[f64]) -> f64 {
    //todo!("contract: dot product of equal-length slices")
    assert_eq!(left.len(), right.len(), "Vectors must be of equal length");

         // Highly vectorizable Fused Multiply-Add structure
        left.iter()
        .zip(right)
        .map(|(x,y)| x * y)
        .sum()
    

}

/// Euclidean (L2) norm of a vector.
pub fn norm(vector: &[f64]) -> f64 {
    //todo!("contract: Euclidean norm; >= 0")
    assert!(!vector.is_empty(), "empty slice undefined");
    let sum_of_squares: f64 = vector.iter().map(|x| x * x).sum();
    let result = sum_of_squares.sqrt() as f64;
    assert!(result >= 0.0, "must be non-negative");
    return result;
    }


/// Online mean update from a running mean, a prior count, and the next value.
pub fn online_mean(previous_mean: f64, count: u64, value: f64) -> f64 {
    //todo!("contract: online mean; count is the count before the new value")
    
    assert!(
        !(count == 0 && previous_mean != 0.0 ),
        "count is the count before the new value");
        previous_mean + (value - previous_mean)/ (count + 1) as f64
    }


/// Online population variance update from running statistics and the next value.
pub fn online_variance(previous_variance: f64, previous_mean: f64, count: u64, value: f64) -> f64 {
    //todo!("contract: online variance; >= 0")
    // Contract Validation: Ensure count matches state history
    assert!(
        !(count == 0 && (previous_mean != 0.0 || previous_variance != 0.0)),
        "count is the count before the new value"
    );

    let next_count = count + 1;
    let next_count_f = next_count as f64;

    // 1. Reconstruct the prior sum of squares (M2) from the population variance
    let previous_m2 = count as f64 * previous_variance;

    // 2. Calculate the difference from the old mean
    let delta = value - previous_mean;

    // 3. Compute the updated mean
    let next_mean = previous_mean + (delta / next_count_f);

    // 4. Compute the updated sum of squares (M2)
    let next_m2 = previous_m2 + delta * (value - next_mean);

    // 5. Compute and return the new population variance (divided by N)
    next_m2 / next_count_f
    
}
