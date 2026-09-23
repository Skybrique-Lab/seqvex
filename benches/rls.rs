//! RLS benchmarks: reference, streaming, and bounded foundation fold, with
//! allocations and bytes per observation.
//!
//! Run with `cargo bench --bench rls`.
//!
//! `harness = false` with `std::time::Instant` keeps the project dependency-free.
//! Each path is measured over repeated runs and reported as median / min / max /
//! IQR, because micro-batching is often assumed to be faster and that assumption
//! must be measured rather than asserted.
//!
//! Unlike linear-regression prediction, the RLS reference is **not**
//! allocation-free: the value-returning `StateModel` contract produces a new
//! `D×D` candidate `P'` every observation, which dominates the allocation. That
//! is the RLS-specific evidence this benchmark exists to surface; it is reported,
//! not optimized.
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
use seqvex::foundation::state::{StateModel, process_batch};
use seqvex::models::online::rls::{Rls, RlsSample};

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

/// Forgetting factor used by every path.
///
/// `λ = 1` (no forgetting) is deliberate: the benchmark replays a fixed
/// observation, and a fixed input with `λ < 1` is not persistently exciting —
/// `P` inflates by `1/λ` in the directions the input never excites and
/// eventually trips the denominator guard. The update path, its `O(D²)` work,
/// and its allocation profile are identical to the `λ < 1` case, which the
/// correctness tests cover over varying inputs.
const LAMBDA: f32 = 1.0;
/// Initial covariance scale used by every path.
const DELTA: f32 = 1.0e4;
/// Bounded fold size; the fold is a local experiment, not a speed claim.
const BATCH: usize = 8;

/// Repeated timing runs per path and configuration.
const RUNS: usize = if cfg!(debug_assertions) { 1 } else { 20 };

/// Shortens the measurement under the debug test profile.
fn timed_steps(steps: u32) -> u32 {
    if cfg!(debug_assertions) { 200 } else { steps }
}

fn model(features: usize) -> Rls {
    Rls::new(features, LAMBDA, DELTA).unwrap()
}

fn observation(features: usize) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_fn(features, |index| (index as f32 * 0.31).sin()),
        target: 0.25,
    })
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = (((sorted.len() as f64 - 1.0) * fraction).round() as usize).min(sorted.len() - 1);
    sorted[index]
}

/// Runs `body` for a warmup plus `RUNS` timed runs and reports per-observation
/// latency, throughput, and allocations.
///
/// `work_per_call` is the number of observations each `body` call processes, so
/// the bounded fold can be compared with single-observation paths on the same
/// per-observation scale.
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
    println!("RLS benchmark ({RUNS} runs/measurement, lambda={LAMBDA}, delta={DELTA})\n");

    for (features, steps) in [
        (8_usize, 200_000_u32),
        (32, 50_000),
        (128, 5_000),
        (256, 2_000),
    ] {
        let steps = timed_steps(steps);
        println!("features={features} bounded_batch={BATCH} steps={steps}");

        let observation = observation(features);
        let batch = vec![observation.clone(); BATCH];

        let reference = model(features);
        let mut reference_state = reference.initial_state();
        measure("reference", steps, 1, || {
            reference_state = reference
                .update(black_box(&reference_state), black_box(&observation))
                .unwrap();
        });

        let mut streaming_model = model(features);
        let initial = streaming_model.initial_state();
        let mut executor = StreamingExecutor::new(&mut streaming_model, initial);
        measure("streaming", steps, 1, || {
            black_box(executor.process_one(black_box(&observation)).unwrap());
        });

        // The bounded fold consumes owned observations, so feeding it clones the
        // batch; that clone cost is included and reported rather than hidden.
        let fold_model = model(features);
        let mut fold_state = Some(fold_model.initial_state());
        measure("bounded-fold", steps, BATCH, || {
            fold_state = Some(
                process_batch(
                    &fold_model,
                    fold_state.take().unwrap(),
                    batch.iter().cloned(),
                )
                .unwrap(),
            );
        });

        let committed = (features + features * features) * size_of::<f32>();
        let transient = (features + features + features * features) * size_of::<f32>();
        println!(
            "    committed state {committed} bytes (w {features} + P {}); \
             candidate + intermediates ~{transient} bytes\n",
            features * features,
        );
    }
}
