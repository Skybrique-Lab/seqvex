//! GRU per-step benchmarks: control (fixed observation) and decision-grade
//! sequential workloads A/B/C, comparing the reference implementation against
//! the current production implementation.
//!
//! Run with `cargo bench --bench gru`.
//!
//! This is strictly "reference implementation vs current production
//! implementation", never "workspace vs no workspace"; no causal claim about
//! workspace ownership is made. The decision-grade matrix runs in the optimized
//! profile only; the debug test profile runs the control and
//! `verify_workloads()`.

use std::hint::black_box;
use std::mem::size_of;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::process_one;
use seqvex::models::recurrent::gru::{Gru, GruParameters};

#[allow(dead_code)]
mod common;

/// Workspace scratch buffers on the production path (three `H`-sized buffers).
const WORKSPACE_BUFFERS: usize = 3;

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

fn control_steps(hidden_dim: usize) -> u32 {
    common::timed_steps(match hidden_dim {
        16 => 100_000,
        64 => 50_000,
        256 => 5_000,
        _ => 2_000,
    })
}

fn build(input_dim: usize, hidden_dim: usize) -> Gru {
    Gru::new(
        input_dim,
        hidden_dim,
        GruParameters::deterministic(input_dim, hidden_dim),
    )
    .unwrap()
}

fn observations_from(xs: &[Vec<f32>]) -> Vec<Observation<Vector>> {
    xs.iter()
        .map(|x| Observation::new(Vector::from_slice(x)))
        .collect()
}

/// Evidence-phase step counts (release only); modest at large hidden sizes
/// because per-step cost grows as `O(IH + H^2)`.
fn evidence_steps(hidden_dim: usize) -> u32 {
    match hidden_dim {
        16 => 50_000,
        64 => 10_000,
        256 => 2_000,
        512 => 500,
        _ => 100,
    }
}

/// Category B evidence: GRU reference vs current production, 30 runs, p95.
///
/// Measures exactly the two existing paths; no implementation is added and the
/// comparison is labelled "reference implementation vs current production
/// implementation" (never "workspace vs no workspace").
fn run_evidence() {
    if cfg!(debug_assertions) {
        println!("GRU evidence comparison skipped in the debug profile");
        return;
    }
    const RUNS: usize = 30;
    println!("\n== evidence: GRU reference vs production ({RUNS} runs) ==");
    for (input_dim, hidden_dim) in [
        (8_usize, 16_usize),
        (32, 64),
        (128, 256),
        (256, 512),
        (512, 1024),
    ] {
        let steps = evidence_steps(hidden_dim);
        let xs = common::persistent_excitation_features(input_dim, 2048);
        let observations = observations_from(&xs);
        println!("GRU evidence input={input_dim} hidden={hidden_dim} steps={steps}");

        let mut reference = build(input_dim, hidden_dim);
        let mut reference_index = 0_usize;
        let reference_stats = common::measure_runs("direct-reference", steps, 1, RUNS, || {
            let observation = &observations[reference_index % observations.len()];
            reference_index += 1;
            black_box(reference.step(black_box(observation)).unwrap());
        });

        let mut production = build(input_dim, hidden_dim);
        let mut production_index = 0_usize;
        let production_stats = common::measure_runs("direct-production", steps, 1, RUNS, || {
            let observation = &observations[production_index % observations.len()];
            production_index += 1;
            black_box(production.step_in_place(black_box(observation)).unwrap());
        });

        println!(
            "    direct (I,H)=({input_dim},{hidden_dim}) delta={:.1} ns/step -> {}",
            (reference_stats.median - production_stats.median).abs(),
            common::materiality(
                reference_stats.median,
                reference_stats.iqr,
                production_stats.median,
                production_stats.iqr,
            ),
        );

        let mut reference_model = build(input_dim, hidden_dim);
        let mut reference_executor =
            StreamingExecutor::new(&mut reference_model, Vector::zeros(hidden_dim));
        let mut stream_reference_index = 0_usize;
        let stream_reference_stats =
            common::measure_runs("streaming-reference", steps, 1, RUNS, || {
                let observation = &observations[stream_reference_index % observations.len()];
                stream_reference_index += 1;
                black_box(
                    reference_executor
                        .process_one(black_box(observation))
                        .unwrap(),
                );
            });

        let mut production_model = build(input_dim, hidden_dim);
        let mut production_executor =
            StreamingExecutor::new(&mut production_model, Vector::zeros(hidden_dim));
        let mut stream_production_index = 0_usize;
        let stream_production_stats =
            common::measure_runs("streaming-production", steps, 1, RUNS, || {
                let observation = &observations[stream_production_index % observations.len()];
                stream_production_index += 1;
                black_box(
                    production_executor
                        .process_one_optimized(black_box(observation))
                        .unwrap(),
                );
            });

        println!(
            "    streaming (I,H)=({input_dim},{hidden_dim}) delta={:.1} ns/step -> {}",
            (stream_reference_stats.median - stream_production_stats.median).abs(),
            common::materiality(
                stream_reference_stats.median,
                stream_reference_stats.iqr,
                stream_production_stats.median,
                stream_production_stats.iqr,
            ),
        );
    }
}

fn main() {
    println!("GRU per-step benchmark ({} runs/measurement)", common::RUNS);
    common::verify_workloads();
    println!("{}", common::env_summary());

    // Evidence-phase convenience: `SEQVEX_EVIDENCE_ONLY=1` runs only the
    // reference-vs-production evidence section. The default run still executes
    // everything and also calls `run_evidence()` at the end.
    if std::env::var_os("SEQVEX_EVIDENCE_ONLY").is_some() {
        run_evidence();
        return;
    }

    // Control: existing fixed observation, four paths.
    println!("\n== control (fixed observation) ==");
    for (input_dim, hidden_dim, steps) in [
        (8_usize, 16_usize, 100_000_u32),
        (32, 64, 50_000),
        (128, 256, 5_000),
        (256, 512, 2_000),
    ] {
        let steps = common::timed_steps(steps);
        println!("input={input_dim} hidden={hidden_dim} steps={steps}");
        let observation = Observation::new(Vector::from_fn(input_dim, |i| (i as f32 * 0.31).sin()));

        let mut reference = build(input_dim, hidden_dim);
        common::measure("direct-reference", steps, 1, || {
            black_box(reference.step(black_box(&observation)).unwrap());
        });

        let mut production = build(input_dim, hidden_dim);
        common::measure("direct-production", steps, 1, || {
            black_box(production.step_in_place(black_box(&observation)).unwrap());
        });

        let mut reference_model = build(input_dim, hidden_dim);
        let mut reference_executor =
            StreamingExecutor::new(&mut reference_model, Vector::zeros(hidden_dim));
        common::measure("streaming-reference", steps, 1, || {
            black_box(
                reference_executor
                    .process_one(black_box(&observation))
                    .unwrap(),
            );
        });

        let mut production_model = build(input_dim, hidden_dim);
        let mut production_executor =
            StreamingExecutor::new(&mut production_model, Vector::zeros(hidden_dim));
        common::measure("streaming-production", steps, 1, || {
            black_box(
                production_executor
                    .process_one_optimized(black_box(&observation))
                    .unwrap(),
            );
        });

        println!(
            "    hidden footprint {} bytes; workspace scratch {} bytes; parameters {} bytes",
            hidden_dim * size_of::<f32>(),
            WORKSPACE_BUFFERS * hidden_dim * size_of::<f32>(),
            parameter_bytes(&GruParameters::deterministic(input_dim, hidden_dim)),
        );
    }

    if cfg!(debug_assertions) {
        println!("\ndecision-grade workload matrix skipped in the debug profile");
        return;
    }

    // Decision-grade workloads A/B/C over the required `(input, hidden)` matrix.
    let dimensions = [(8_usize, 16_usize), (32, 64), (128, 256), (256, 512)];
    for workload in ["A", "B", "C"] {
        println!("\n== workload {workload} ==");
        let mut scaling: Vec<(f64, f64)> = Vec::new();
        for &(input_dim, hidden_dim) in &dimensions {
            let steps = control_steps(hidden_dim);
            let xs = match workload {
                "A" => common::persistent_excitation_features(input_dim, 2048),
                "B" => common::structured_features(
                    input_dim,
                    2048,
                    common::AR_COEFFICIENT,
                    common::STRUCTURED_SEED,
                ),
                _ => common::persistent_excitation_features(input_dim, steps as usize),
            };
            let observations = observations_from(&xs);
            println!(
                "GRU workload={workload} input={input_dim} hidden={hidden_dim} steps={steps} stream={}",
                observations.len()
            );

            common::measure_startup("init", common::startup_reps(), || {
                black_box(build(input_dim, hidden_dim));
            });

            let mut reference = build(input_dim, hidden_dim);
            let mut index = 0_usize;
            let reference_stats = common::measure("direct-reference", steps, 1, || {
                let observation = &observations[index % observations.len()];
                index += 1;
                black_box(reference.step(black_box(observation)).unwrap());
            });

            let mut production = build(input_dim, hidden_dim);
            let mut production_index = 0_usize;
            let production_stats = common::measure("direct-production", steps, 1, || {
                let observation = &observations[production_index % observations.len()];
                production_index += 1;
                black_box(production.step_in_place(black_box(observation)).unwrap());
            });
            println!(
                "    direct delta={:.1} ns/step -> {}",
                (reference_stats.median - production_stats.median).abs(),
                common::materiality(
                    reference_stats.median,
                    reference_stats.iqr,
                    production_stats.median,
                    production_stats.iqr,
                )
            );

            let mut reference_model = build(input_dim, hidden_dim);
            let mut reference_executor =
                StreamingExecutor::new(&mut reference_model, Vector::zeros(hidden_dim));
            let mut stream_reference_index = 0_usize;
            let stream_reference_stats = common::measure("streaming-reference", steps, 1, || {
                let observation = &observations[stream_reference_index % observations.len()];
                stream_reference_index += 1;
                black_box(
                    reference_executor
                        .process_one(black_box(observation))
                        .unwrap(),
                );
            });

            let mut production_model = build(input_dim, hidden_dim);
            let mut production_executor =
                StreamingExecutor::new(&mut production_model, Vector::zeros(hidden_dim));
            let mut stream_production_index = 0_usize;
            let stream_production_stats = common::measure("streaming-production", steps, 1, || {
                let observation = &observations[stream_production_index % observations.len()];
                stream_production_index += 1;
                black_box(
                    production_executor
                        .process_one_optimized(black_box(observation))
                        .unwrap(),
                );
            });
            println!(
                "    streaming delta={:.1} ns/step -> {}",
                (stream_reference_stats.median - stream_production_stats.median).abs(),
                common::materiality(
                    stream_reference_stats.median,
                    stream_reference_stats.iqr,
                    stream_production_stats.median,
                    stream_production_stats.iqr,
                )
            );

            if workload == "A" {
                // Model sharing through the existing foundation `&M` path
                // (reference semantics), one model with two hidden states versus
                // two models.
                let shared = build(input_dim, hidden_dim);
                let mut shared_a = Vector::zeros(hidden_dim);
                let mut shared_b = Vector::zeros(hidden_dim);
                let mut shared_index = 0_usize;
                let shared_stats = common::measure("sharing-1model", steps, 1, || {
                    let observation = &observations[shared_index % observations.len()];
                    shared_index += 1;
                    shared_a = process_one(&shared, &shared_a, black_box(observation)).unwrap();
                    shared_b = process_one(&shared, &shared_b, black_box(observation)).unwrap();
                });
                let model_a = build(input_dim, hidden_dim);
                let model_b = build(input_dim, hidden_dim);
                let mut separate_a = Vector::zeros(hidden_dim);
                let mut separate_b = Vector::zeros(hidden_dim);
                let mut separate_index = 0_usize;
                let separate_stats = common::measure("sharing-2models", steps, 1, || {
                    let observation = &observations[separate_index % observations.len()];
                    separate_index += 1;
                    separate_a =
                        process_one(&model_a, &separate_a, black_box(observation)).unwrap();
                    separate_b =
                        process_one(&model_b, &separate_b, black_box(observation)).unwrap();
                });
                println!(
                    "    sharing delta={:.1} ns/step -> {}",
                    (shared_stats.median - separate_stats.median).abs(),
                    common::materiality(
                        shared_stats.median,
                        shared_stats.iqr,
                        separate_stats.median,
                        separate_stats.iqr,
                    )
                );
            }

            if workload != "B" {
                scaling.push((hidden_dim as f64, reference_stats.median));
            }
        }
        if scaling.len() >= 2 {
            println!(
                "GRU workload={workload} scaling exponent versus hidden={:.3} (expected ~1..2)",
                common::log_log_slope(&scaling)
            );
        }
    }

    run_evidence();
}
