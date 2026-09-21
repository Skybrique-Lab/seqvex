//! Per-step GRU benchmarks: latency, throughput, sustained sequences, and
//! allocation behavior in the complete step path.
//!
//! Run with `cargo bench --bench gru`.
//!
//! `harness = false` with `std::time::Instant` keeps the sprint dependency-free.
//! The counting allocator measures allocations per step; timings are
//! indicative, not statistically rigorous (see `benches/numerical.rs`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

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

const STEPS: u32 = 100_000;

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

fn main() {
    println!("GRU per-step benchmark ({STEPS} steps/measurement)\n");

    for (input_dim, hidden_dim) in [(8_usize, 16_usize), (32, 64), (128, 256)] {
        let parameter_bytes = parameter_bytes(&GruParameters::deterministic(input_dim, hidden_dim));
        let mut gru = Gru::new(
            input_dim,
            hidden_dim,
            GruParameters::deterministic(input_dim, hidden_dim),
        )
        .unwrap();
        let observation = Observation::new(Vector::from_fn(input_dim, |i| (i as f32 * 0.31).sin()));

        for _ in 0..STEPS / 10 {
            gru.step(&observation).unwrap();
        }

        let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
        let bytes_before = ALLOCATED_BYTES.load(Ordering::Relaxed);
        let start = Instant::now();
        for _ in 0..STEPS {
            black_box(gru.step(black_box(&observation)).unwrap());
        }
        let elapsed = start.elapsed();

        let allocation_count = ALLOCATIONS.load(Ordering::Relaxed) - allocations_before;
        let allocated_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes_before;
        let nanoseconds = elapsed.as_nanos() as f64 / f64::from(STEPS);

        println!(
            "input={input_dim:<4} hidden={hidden_dim:<4} {nanoseconds:>9.1} ns/step \
             {:>10.0} obs/s  {:.2} allocs/step  {:.0} bytes/step",
            1e9 / nanoseconds,
            allocation_count as f64 / f64::from(STEPS),
            allocated_bytes as f64 / f64::from(STEPS),
        );
        println!(
            "    hidden footprint {} bytes; parameters {} bytes",
            hidden_dim * size_of::<f32>(),
            parameter_bytes,
        );
        println!();
    }
}
