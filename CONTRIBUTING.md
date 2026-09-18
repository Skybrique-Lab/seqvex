# Contributing to Seqvex

Thank you for your interest in Seqvex.

Seqvex is an early-stage project. At this stage, careful design, experimentation, testing, and learning are more important than rapidly increasing the amount of code.

Contributions should help the project become more correct, understandable, measurable, and maintainable.

## Before Contributing

For significant changes, especially changes involving architecture, execution, memory, numerical representation, or dependencies, please open a discussion or issue before implementing the change.

Small fixes and clearly scoped improvements can generally proceed directly.

Please read the below documents, before contributing, it will greatly help you in understanding seqvex:

1. [`ARCHITECTURE.md`](ARCHITECTURE.md)
2. [`DEVELOPMENT.md`](docs/DEVELOPMENT.md)
3. [`FAILURE_AND_RECOVERY.md`](docs/FAILURE_AND_RECOVERY.md)

## Do

### Understand before optimizing

Understand the behavior and bottleneck before introducing an optimization.

Prefer:

1. Baseline
2. Benchmark
3. Profile
4. Identify bottleneck
5. Optimize
6. Benchmark again

Do not assume that a lower-level implementation is automatically faster.

### Keep components focused

Keep functionality close to the component responsible for it.

Avoid introducing dependencies between unrelated components simply for convenience.

### Write tests

New functionality should include appropriate tests.

Tests should verify behavior and correctness, not merely increase coverage numbers.

### Benchmark performance-sensitive code

If a change is intended to improve performance, provide a benchmark where practical.

Important measurements may include:

- throughput
- latency
- tail latency
- allocations
- memory usage
- CPU utilization
- accelerator utilization

### Document important decisions

If a change introduces a significant architectural decision, explain:

- the problem
- alternatives considered
- why the chosen approach was selected
- what trade-offs it introduces

### Prefer simple designs

Start with the simplest design that correctly solves the demonstrated problem.

Complexity should have a reason.

### Preserve hardware neutrality

Do not introduce unnecessary assumptions about a particular CPU, GPU, vendor, or accelerator.

Hardware-specific optimizations should be isolated where practical.

### Respect real-time constraints

If code is intended for a real-time or latency-sensitive execution path, make its resource behavior explicit.

Do not introduce hidden allocations, unnecessary synchronization, or unbounded work without documenting the implications.

## Don't

### Don't optimize without evidence

Do not add SIMD, unsafe code, custom allocators, prefetching, cache tricks, or architecture-specific code merely because they sound faster.

Measure first.

### Don't prematurely abstract

Do not create a generic abstraction for a problem that Seqvex has not actually encountered.

A useful abstraction should solve a demonstrated problem.

### Don't prematurely freeze architecture

Seqvex is intentionally evolving.

Avoid making broad architectural changes based solely on theoretical requirements.

### Don't assume the GPU is always faster

GPU execution can introduce transfer, synchronization, and launch overhead.

Choose execution based on workload characteristics rather than assuming accelerator execution is universally superior.

### Don't hide expensive operations

Avoid APIs that make expensive allocation, copying, synchronization, or device transfers look indistinguishable from inexpensive operations when that distinction matters for performance.

### Don't add large dependencies casually

Before adding a dependency, consider:

- Is it necessary?
- What functionality does it provide?
- What are its performance implications?
- Does it impose an architectural model on Seqvex?
- Could a smaller dependency or internal implementation be more appropriate?
- Does the maintenance cost justify it?

### Don't duplicate existing functionality without a reason

Seqvex should not reimplement functionality simply for ideological reasons.

However, foundational components may be implemented internally when doing so is necessary to achieve Seqvex's architectural, performance, portability, or learning objectives.

### Don't submit large generated changes without understanding them

Code should be understandable and maintainable by the contributors responsible for it.

Automated tools and AI may be used during development, but generated code should be reviewed, understood, tested, and justified before being committed.

## Rust Guidelines

Prefer:

- safe Rust by default
- clear ownership
- explicit lifetimes where they improve correctness or performance
- focused traits
- predictable allocation behavior
- small, testable components
- idiomatic Rust where it does not conflict with measured requirements

`unsafe` code is permitted when justified by a measurable or fundamental systems requirement.

Unsafe code should be:

- narrowly scoped
- documented
- tested
- reviewed carefully

## Architectural Changes

Changes affecting the following areas should receive additional scrutiny:

- execution semantics
- memory ownership
- memory layout
- device abstraction
- synchronization
- numerical representation
- determinism
- public API design
- serialization/checkpoint formats
- concurrency
- dependency architecture

When proposing such a change, explain the problem and alternatives before committing to a particular implementation.

## A Simple Rule

When in doubt:

> Understand the problem → build the smallest experiment → measure → then generalize.

Seqvex should grow from demonstrated requirements rather than from speculative complexity.
