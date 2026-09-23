//! Linear-regression prediction benchmarks: reference, streaming, and bounded
//! micro-batch, with allocations and bytes per observation.
//!
//! Run with `cargo bench --bench linear_regression`.
//!
//! `harness = false` with `std::time::Instant` keeps the project dependency-free.
//! Each path is measured over repeated runs and reported as median / min / max /
//! IQR, because a micro-batch is often assumed to be faster and that assumption
//! must be measured rather than asserted.
//!
//! Under the debug test profile (`cargo test --all-targets` executes
//! `harness = false` benchmarks) the run count and step counts are shortened so
//! the test command stays usable; those numbers are never used for decisions.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::classic::LinearRegression;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Repeated timing runs per path and configuration.
const RUNS: usize = if cfg!(debug_assertions) { 1 } else { 20 };

/// Shortens the measurement under the debug test profile.
fn timed_steps(steps: u32) -> u32 {
    if cfg!(debug_assertions) { 200 } else { steps }
}

fn model(features: usize) -> LinearRegression {
    LinearRegression::new(
        Vector::from_fn(features, |index| (index as f32 * 0.37).sin()),
        0.125,
    )
    .unwrap()
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = (((sorted.len() as f64 - 1.0) * fraction).round() as usize).min(sorted.len() - 1);
    sorted[index]
}

/// Runs `body` for a warmup plus `RUNS` timed runs and reports per-observation
/// latency, through-steps, and allocations.
///
/// `work_per_call` is the number of observations each `body` call processes, so
/// a bounded micro-batch can be compared with single-observation paths on the
/// same per-observation scale.
fn measure(label: &str, steps: u32, work_per_call: usize, mut body: impl FnMut()) {
    let work_per_call = work_per_call as f64;
    let observations = f64::from(steps) * work_per_call;

    for _ in 0..(steps / 10).max(if cfg!(debug_assertions) { 50 } else { 1000 }) {
        body();
    }

    let mut samples = Vec::with_capacity(RUNS);
    let mut allocations = 0.0;
    let mut bytes = 0.0;
    for _ in 0..RUNS {
        let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
        let bytes_before = ALLOCATED_BYTES.load(Ordering::Relaxed);
        let start = Instant::now();
        for _ in 0..steps {
            body();
        }
        let elapsed = start.elapsed();
        allocations =
            (ALLOCATIONS.load(Ordering::Relaxed) - allocations_before) as f64 / observations;
        bytes = (ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes_before) as f64 / observations;
        samples.push(elapsed.as_nanos() as f64 / observations);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = percentile(&samples, 0.5);
    let min = samples[0];
    let max = samples[samples.len() - 1];
    let q1 = percentile(&samples, 0.25);
    let q3 = percentile(&samples, 0.75);
    let iqr = q3 - q1;
    let throughput = 1e9 / median;

    println!(
        "{label:<20} {median:>9.1} ns/obs median (min {min:.1}, max {max:.1}, iqr {iqr:.1})  \
         {throughput:.0} obs/s  {allocations:.3} allocs/obs  {bytes:.1} bytes/obs"
    );
}

fn main() {
    println!("Linear-regression prediction benchmark ({RUNS} runs/measurement)\n");

    for (features, micro_batch) in [(8_usize, 32_usize), (32, 32), (128, 16)] {
        let steps = timed_steps(if features <= 32 { 200_000 } else { 50_000 });
        println!("features={features} micro_batch={micro_batch} steps={steps}");

        let observation = Observation::new(Vector::from_fn(features, |i| (i as f32 * 0.31).sin()));
        let batch = vec![observation.clone(); micro_batch];

        let reference = model(features);
        measure("reference", steps, 1, || {
            black_box(reference.predict(black_box(observation.value())).unwrap());
        });

        let mut streaming_model = model(features);
        let mut executor = StreamingExecutor::new(&mut streaming_model, 0.0);
        measure("streaming", steps, 1, || {
            black_box(executor.process_one(black_box(&observation)).unwrap());
        });

        let micro_batch_model = model(features);
        measure("micro-batch", steps, micro_batch, || {
            black_box(micro_batch_model.predict_batch(black_box(&batch)).unwrap());
        });

        println!(
            "    weights {} bytes; micro-batch output {} bytes/prediction\n",
            features * size_of::<f32>(),
            size_of::<f32>(),
        );
    }
}
