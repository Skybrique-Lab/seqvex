//! Scalar-reference benchmarks for the operations in the GRU hot path.
//!
//! Run with `cargo bench --bench numerical`.
//!
//! `harness = false` with `std::time::Instant` keeps the sprint dependency-free.
//! These are indicative timings, not a statistically rigorous comparison.
//!
//! ponytail: single-run timings answer "which operation dominates and at which
//! size", which is the sprint question. Add `criterion` only if a decision
//! needs statistical confidence.

use std::hint::black_box;
use std::time::Instant;

use seqvex::foundation::numerical::{Matrix, Vector, sigmoid, tanh};

const ITERATIONS: u32 = 20_000;

fn measure(label: &str, iterations: u32, mut body: impl FnMut()) {
    for _ in 0..(iterations / 10).max(1) {
        body();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        body();
    }
    println!(
        "  {label:<20} {:>10.1} ns/op",
        start.elapsed().as_nanos() as f64 / f64::from(iterations)
    );
}

fn main() {
    println!("scalar-reference f32 hot path ({ITERATIONS} iterations/measurement)\n");

    for size in [8usize, 32, 128, 512] {
        println!("size {size}");
        let left = Vector::from_fn(size, |i| i as f32 * 0.5);
        let right = Vector::from_fn(size, |i| 1.0 - i as f32 * 0.25);

        measure("dot", ITERATIONS, || {
            black_box(black_box(&left).dot(black_box(&right)).unwrap());
        });
        measure("elementwise add", ITERATIONS, || {
            black_box(black_box(&left).add(black_box(&right)).unwrap());
        });
        measure("elementwise multiply", ITERATIONS, || {
            black_box(black_box(&left).multiply(black_box(&right)).unwrap());
        });

        let weights = Matrix::from_fn(size, size, |r, c| ((r + c) % 7) as f32 * 0.125);
        measure("matvec", ITERATIONS.min(2_000), || {
            black_box(black_box(&weights).mul_vector(black_box(&left)).unwrap());
        });
        measure("map sigmoid", ITERATIONS.min(2_000), || {
            black_box(black_box(&left).map(sigmoid));
        });
        measure("scalar sigmoid", ITERATIONS, || {
            black_box(sigmoid(black_box(0.5)));
        });
        measure("scalar tanh", ITERATIONS, || {
            black_box(tanh(black_box(0.5)));
        });
        println!();
    }
}
