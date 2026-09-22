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

## Issue Classification and Labels

Seqvex uses several independent classification dimensions for GitHub Issues. These
dimensions should not be treated as interchangeable.

### Issue Type

Seqvex currently uses exactly three GitHub Issue Types:

- **Bug** — an unexpected problem or behavior.
- **Feature** — a request, idea, or new functionality.
- **Task** — a specific piece of work.

Do not introduce additional Issue Types without first discussing and approving
the change.

### Area labels

Labels using the `area::*` namespace identify the functional or architectural
area to which the Issue belongs.

Area labels should normally correspond to meaningful functional or architectural
areas of the repository, rather than reproducing every level of the filesystem
hierarchy.

The current area labels are:

- `area::benchmark` — `/benches`
- `area::numerical` — `src/foundation/numerical`
- `area::observation` — `src/foundation/observation`
- `area::ordering` — `src/foundation/ordering`
- `area::state` — `src/foundation/state`
- `area::streaming` — `src/foundation/streaming`
- `area::test` — testing work as a primary area

The exact file, module, function, or deeper subdirectory should be identified
in the Issue body when relevant.

`area::test` should be used when the work itself belongs to testing. It should
not be added merely because tests are used to validate another Issue.

Do not create area labels for individual files, functions, or one-off tasks.

### Nature labels

Nature labels describe the broad characteristic of the work. They are
independent of Issue Type and Area and should be used when they provide useful
additional classification.

Current nature-oriented labels include:

- `optimization` — optimization work
- `documentation` — improvements or additions to documentation
- `experimental` — work intended to investigate, validate, or falsify a hypothesis through a minimal experiment
- `integration` — work that combines validated components or behaviors into a coherent end-to-end flow

Other existing labels may also be used when their established meaning applies:

- `accessibility` — barrier affecting people with disabilities
- `bug` — something isn't working
- `duplicate` — this issue or pull request already exists
- `enhancement` — new feature or request
- `good first issue` — good for newcomers
- `help wanted` — extra attention is needed
- `invalid` — this doesn't seem right
- `question` — further information is requested
- `wontfix` — this will not be worked on

These labels should be applied according to their established meaning rather
than used as substitutes for the Issue Type.

### Label selection

When applicable, an Issue should normally have:

- one appropriate GitHub Issue Type;
- one primary `area::*` label;
- a nature label when it provides useful additional classification.

Do not create a new label merely because a particular file, function, or
one-off task does not have an existing label.

Before introducing a new label, confirm that an existing label cannot express
the same classification and discuss the proposed addition before creating it.

### Milestones

A milestone identifies the development objective to which an Issue belongs.

Milestones are separate from labels and should not be used as substitutes for
Area or Nature classification.

The current milestones are:

- **Phase 1 — Foundations**
- **First Sequential ML Slice**

Do not create a new milestone without first discussing and approving it.

### Project status

Project status describes the workflow state of the Issue and is separate from
Issue Type, Area, Nature, and Milestone.

Seqvex's intended workflow is:

**Planning → Implementation → Validation → Integrated**

The final **Validation → Integrated** transition remains a deliberate
integration step and should not be inferred merely because implementation or
validation work is complete.

### Classification rule

Use the dimensions together:

> **Issue Type = what kind of Issue**  
> **Area = where**  
> **Nature = what broad characteristic it has**  
> **Milestone = which development objective**  
> **Project status = workflow state**

This separation is important because Seqvex relies heavily on labels for issue
discovery, filtering, ownership, and contributor orientation.

## A Simple Rule

When in doubt:

> Understand the problem → build the smallest experiment → measure → then generalize.

Seqvex should grow from demonstrated requirements rather than from speculative complexity.
