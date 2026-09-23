//! Per-step GRU benchmarks: latency, throughput, sustained sequences, and
//! allocation behavior in the complete step path.
//!
//! Run with `cargo bench --bench gru`.
//!
//! `harness = false` with `std::time::Instant` keeps the sprint dependency-free.
//! The counting allocator measures allocations per step. Each path is measured
//! over repeated runs and reported as median / min / max / IQR, because the
//! Stage 2 decision compares reference and production paths whose differences may
//! be small.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::recurrent::gru::{Gru, GruParameters};

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
///
/// The release `bench` profile (used by `cargo bench --bench gru`) runs the full
/// count. The debug test profile, which `cargo test --all-targets` executes for
/// `harness = false` benchmarks, uses a single short run so the test command
/// stays usable; those numbers are never used for decisions.
const RUNS: usize = if cfg!(debug_assertions) { 1 } else { 20 };
/// Workspace scratch buffers on the production path (three `H`-sized buffers).
const WORKSPACE_BUFFERS: usize = 3;

/// Shortens the measurement under the debug test profile.
fn timed_steps(steps: u32) -> u32 {
    if cfg!(debug_assertions) { 200 } else { steps }
}

fn parameter_bytes(parameters: &GruParameters) -> usize {
    let matrices = [
        &parameters.w_z,
        &parameters.u_z,
        &parameters.w_r,
        &parameters.u_r,
        &parameters.w_h,
        &parameters.u_h,
    ];
    let biases = [&parameters.b_z, &parameters.b_r, &parameters.b_h];
    let elements = matrices.iter().map(|m| m.as_slice().len()).sum::<usize>()
        + biases.iter().map(|b| b.len()).sum::<usize>();
    elements * size_of::<f32>()
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = (((sorted.len() as f64 - 1.0) * fraction).round() as usize).min(sorted.len() - 1);
    sorted[index]
}

/// Runs `body` for a warmup plus `RUNS` timed runs and reports per-observation
/// allocations, bytes, and the latency distribution.
fn measure(label: &str, steps: u32, mut body: impl FnMut()) {
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
            (ALLOCATIONS.load(Ordering::Relaxed) - allocations_before) as f64 / f64::from(steps);
        bytes = (ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes_before) as f64 / f64::from(steps);
        samples.push(elapsed.as_nanos() as f64 / f64::from(steps));
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
        "{label:<20} {median:>9.1} ns/step median (min {min:.1}, max {max:.1}, iqr {iqr:.1})  \
         {throughput:.0} obs/s  {allocations:.2} allocs/step  {bytes:.0} bytes/step"
    );
}

fn main() {
    println!("GRU per-step benchmark ({RUNS} runs/measurement)\n");

    for (input_dim, hidden_dim, steps) in [
        (8_usize, 16_usize, 100_000_u32),
        (32, 64, 50_000),
        (128, 256, 5_000),
        (256, 512, 2_000),
    ] {
        let steps = timed_steps(steps);
        println!("input={input_dim} hidden={hidden_dim} steps={steps}");
        let observation = Observation::new(Vector::from_fn(input_dim, |i| (i as f32 * 0.31).sin()));

        let mut reference = Gru::new(
            input_dim,
            hidden_dim,
            GruParameters::deterministic(input_dim, hidden_dim),
        )
        .unwrap();
        measure("direct-reference", steps, || {
            black_box(reference.step(black_box(&observation)).unwrap());
        });

        let mut production = Gru::new(
            input_dim,
            hidden_dim,
            GruParameters::deterministic(input_dim, hidden_dim),
        )
        .unwrap();
        measure("direct-production", steps, || {
            black_box(production.step_in_place(black_box(&observation)).unwrap());
        });

        let mut reference_model = Gru::new(
            input_dim,
            hidden_dim,
            GruParameters::deterministic(input_dim, hidden_dim),
        )
        .unwrap();
        let mut reference_executor =
            StreamingExecutor::new(&mut reference_model, Vector::zeros(hidden_dim));
        measure("streaming-reference", steps, || {
            black_box(
                reference_executor
                    .process_one(black_box(&observation))
                    .unwrap(),
            );
        });

        let mut production_model = Gru::new(
            input_dim,
            hidden_dim,
            GruParameters::deterministic(input_dim, hidden_dim),
        )
        .unwrap();
        let mut production_executor =
            StreamingExecutor::new(&mut production_model, Vector::zeros(hidden_dim));
        measure("streaming-production", steps, || {
            black_box(
                production_executor
                    .process_one_optimized(black_box(&observation))
                    .unwrap(),
            );
        });

        println!(
            "    hidden footprint {} bytes; workspace scratch {} bytes; parameters {} bytes\n",
            hidden_dim * size_of::<f32>(),
            WORKSPACE_BUFFERS * hidden_dim * size_of::<f32>(),
            parameter_bytes(&GruParameters::deterministic(input_dim, hidden_dim)),
        );
    }
}
