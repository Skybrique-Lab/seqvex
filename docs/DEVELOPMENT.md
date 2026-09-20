# Seqvex Development Guide

> **This document is the development contract for Seqvex.**
>
> It applies to both human contributors and AI-assisted development.
> Before changing code, read this document together with the relevant
> architecture and failure/recovery documentation.

## 1. Purpose

`DEVELOPMENT.md` defines how Seqvex is developed, not merely how Rust
code is formatted.

The central development principle is:

> **Code should not outrun understanding.**

Seqvex is an early-stage project. Architecture, APIs, crate boundaries,
storage representations, and implementation strategies may change as
real problems are encountered and measured.

Do not freeze a design merely because it can be implemented.

## 2. Development Philosophy

Seqvex is developed experimentally and incrementally.

The preferred loop is:

```text
Learn
  ↓
Understand the problem
  ↓
Design the smallest experiment
  ↓
Implement
  ↓
Test
  ↓
Break it deliberately
  ↓
Understand failure
  ↓
Benchmark
  ↓
Profile
  ↓
Optimize
  ↓
Document
  ↓
Integrate
  ↺
Learn
```

### Core rules

1. Understand the problem before abstracting it.
2. Implement the smallest mechanism that can test the idea.
3. Test both expected and invalid behavior.
4. Deliberately exercise failure paths.
5. Measure before making performance claims.
6. Profile before optimizing.
7. Generalize only after recurring requirements justify generalization.
8. Document decisions and their evidence.
9. Keep unresolved decisions explicitly unresolved.
10. Prefer reversible decisions while the architecture is still being discovered.

> **Do not create an abstraction until you have experienced the problem it solves.**

## 3. Architectural Invariants

### 3.1 Streaming-first, not batchless

Seqvex is **streaming-first**, but it is not batchless.

Streaming is the semantic foundation. Single-observation execution is the
smallest execution unit. Bounded micro-batches and larger batches remain
valid where they improve computational or hardware efficiency without
violating ordering, temporal semantics, causality, state semantics, or
online-learning semantics.

A historical dataset may be replayed as a stream. Batching is an execution
strategy, not the definition of the computational model.

### 3.2 Execution semantics and compute placement are separate

Execution semantics describe **what a computation means**:

- single observation;
- streaming/online;
- bounded micro-batch;
- batch.

Compute placement describes **where and how it executes**:

- CPU;
- GPU/accelerator;
- heterogeneous CPU/accelerator execution.

Do not allow a hardware backend to silently change the semantic meaning of
an algorithm.

### 3.3 State is first-class

Many Seqvex workloads are stateful.

A conceptual state transition is:

```text
observation
    ↓
current valid state
    ↓
computation
    ↓
candidate next state
    ↓
validation
    ↓
commit
    ↓
valid next state
```

A failed update must not silently leave the model or runtime in a partially
modified or invalid state.

Detailed failure behavior is defined in `docs/FAILURE_AND_RECOVERY.md`.

> **Detect → stop unsafe transition → preserve valid state → classify →
> act → report → continue/recover/isolate/terminate.**

### 3.4 Heterogeneous memory and hardware locality

Do not assume all memory is equivalent.

The architecture must preserve the ability to reason about:

```text
CPU registers
L1/L2/L3/LLC cache
RAM
interconnect
GPU registers/cache
device memory / VRAM
NUMA
memory locality
alignment
SIMD access
memory bandwidth
data movement
synchronization
```

Potential future mechanisms include device-local storage, pinned memory,
unified/shared memory, zero-copy paths, NUMA-aware allocation, and explicit
transfers. These are optimization possibilities, not current API commitments.

> **Keep model state and frequently used data resident on one device when
> possible. Move data only when measured computational benefit exceeds
> transfer and synchronization cost.**

Do not equate heterogeneous execution with constant CPU↔GPU movement.

### 3.5 `std`, `no_std`, and constrained deployment

Seqvex is **std-first**.

The whole framework should not currently be forced into `#![no_std]`.
Foundational components should avoid unnecessary dependence on OS facilities
where practical so constrained or embedded deployment remains possible.

The project does **not** currently claim universal `no_std`, bare-metal,
hard real-time, safety certification, zero-allocation execution, or
cross-platform deterministic floating-point behavior.

## 4. Scope Boundary

Seqvex differentiates its scope by **process responsibility**, not by the
name of an individual function.

Seqvex owns the computational stages from:

- ML/RL preprocessing and representation;
- model execution;
- training;
- validation;
- inference;
- learning;
- numerical computation required by ML/RL;
- execution/runtime mechanisms.

General-purpose data ingestion, manipulation, cleaning, exploratory analysis,
visualization, databases, storage, ETL, and domain/business logic remain
outside the framework.

Operations such as one-hot encoding, normalization/standardization, PCA,
rolling statistics, and online statistics may be inside or outside Seqvex
depending on their role in the ML/RL computational pipeline.

## 5. Build From First Principles

Seqvex should be understandable from its foundations.

Existing libraries may be used when they provide clear value, but
dependencies should not determine the architecture unnecessarily.

Before adopting a dependency, consider its problem, ownership/licensing,
compile-time impact, runtime overhead, allocation behavior, portability,
`std`/`no_std` implications, hardware/backend constraints, and whether it
introduces an abstraction Seqvex would otherwise need to understand itself.

Do not reject dependencies merely because they are external. Do not add
dependencies merely because they are convenient.

Seqvex should build first-principles implementations where doing so
establishes Seqvex's own semantics and improves understanding. Mature
optimized infrastructure may be reused when it provides a clear
implementation benefit and does not become the semantic definition of the
framework.

## 6. Modularity

Seqvex should be modular at meaningful responsibility boundaries.

A functional module may contain focused sub-functional components:

```text
functional-module/
├── README.md
├── sub-function/
├── sub-function/
└── tests/
```

`README.md` belongs at the **functional-module level**, where shared context
is useful. Do not create a README for every tiny function or sub-function.

> **Document at the highest level where shared context is useful; keep
> detailed implementation knowledge close to the code and tests.**

A folder should represent a coherent responsibility, not simply a single
function, struct, or method.

## 7. Shared Infrastructure and Global Helpers

Functional locality is preferred over superficial deduplication.

Do not introduce global `common`, `utils`, or helper modules merely to remove
local duplication. The same applies to shared error types, state helpers,
numerical utilities, execution context, memory/storage abstractions, and
test support.

A shared abstraction should be introduced only when:

- the responsibility is genuinely cross-functional;
- the abstraction has a stable semantic contract;
- local ownership would create meaningful duplication of domain knowledge;
- or sharing is required for a concrete architectural property.

Similar implementation is not sufficient evidence. Prefer local ownership
first, then refactor upward when repeated responsibility and dependency
relationships demonstrate that the abstraction is genuinely shared.

In particular, do not create a single global error type merely because
multiple modules currently return errors. Domain-specific errors should
remain local until a genuinely cross-functional error contract emerges.

### Functional locality

A functional module should be substantially self-contained in its:

- implementation;
- internal helpers;
- state relevant to that responsibility;
- focused tests and test support.

Cross-functional composition and data flow are expected. Shared mutable state
and cross-functional coupling require stronger justification, especially as
concurrency and asynchronous execution are introduced.

> **DRY should remove duplicated knowledge, not erase useful ownership
> boundaries.**

## 8. Crate and Workspace Boundaries

Seqvex is currently a **single Cargo package**. A workspace and additional
crates may emerge later when demonstrated boundaries justify them.

A crate should not be created merely because a future architecture diagram
contains a possible component.

A new crate is justified when a meaningful boundary emerges through actual
requirements, such as dependency isolation, independent testing, backend
separation, API clarity, reuse, compile-time isolation, reduced coupling,
or materially different platform requirements.

Possible future areas include core types/errors, numerical computation,
streaming/online learning, sequential models, supervised/unsupervised
learning, reinforcement learning, validation, execution/runtime, and device
backends.

These are architectural directions, not a frozen crate list.

> **Crate boundaries should emerge from demonstrated responsibility
> boundaries.**

## 9. Rust Engineering Principles

These principles are design tools, not patterns to apply mechanically.

- **DRY:** avoid duplicated knowledge, invariants, and logic; do not abstract
  merely because code looks similar.
- **Composition over inheritance:** prefer focused structures and composable
  behavior.
- **Traits:** introduce them for real semantic contracts, multiple meaningful
  implementations, dependency inversion, backend abstraction, testability,
  or stable behavioral requirements.
- **Extension traits:** use for coherent domain behavior on types Seqvex does
  not own.
- **Builder pattern:** use for meaningful configuration, optional parameters,
  validation, or staged construction.
- **Typestate:** use when compile-time state distinctions provide real
  correctness; do not build a speculative typestate framework.
- **Newtype:** use when distinct semantic values should not be interchangeable.
- **Error enums:** prefer domain-specific errors and `Result<T,E>`; avoid
  stringly typed core errors and unnecessary `unwrap()`/`expect()`.
- **Iterators:** prefer them where they improve clarity without material
  performance cost; use explicit loops where clearer or measurably better.

## 10. Numerical Foundation

Seqvex requires numerical computation for ML/RL, but is not intended to
replace NumPy, SciPy, Pandas, Polars, a dataframe engine, or general
statistical software.

Numerical primitives should be introduced incrementally when they support
actual ML/RL requirements.

Possible foundations include sums, means, variance, covariance, dot products,
vector norms, online statistics, rolling statistics, and matrix/vector
operations required by implemented algorithms.

Do not build a complete numerical ecosystem before real Seqvex workloads
justify it.

## 11. Execution and State Semantics

Implementation should preserve the distinction between:

```text
single observation
streaming / online
micro-batch
batch
```

and:

```text
CPU
GPU / accelerator
heterogeneous
```

A backend optimization must not accidentally redefine the algorithm's
temporal or state semantics.

The precise state-transition implementation mechanism remains deliberately
open until the actual state model and failure characteristics are understood.

## 12. Failure, Recovery, and Determinism

Failure handling is part of the architecture.

Relevant failure classes include invalid input, numerical failure,
model/state invariant failure, memory/allocation failure, runtime failure,
and hardware/device failure.

For every stateful operation, consider what can fail, whether valid state
can be preserved, whether retry is safe, whether rollback is meaningful,
whether processing should continue/isolate/terminate, what must be reported,
and whether recovery preserves deterministic semantics.

Rollback is not universal. Do not add rollback machinery simply because
failure exists.

Detailed requirements belong in `docs/FAILURE_AND_RECOVERY.md`.

## 13. Testing

Testing should establish behavior and invariants, not merely increase
coverage numbers.

For foundational components, test valid behavior, invalid inputs, boundaries,
ordering, state transitions, state preservation after failure, repeated
streaming updates, deterministic behavior where promised, numerical edge
cases, single-observation semantics, and bounded micro-batch semantics where
implemented.

### Test ownership

The top-level `tests/` tree is intentionally **global**. It owns integration,
public API, cross-functional, and system-level behavior.

`tests/common/` is reserved for genuinely shared integration-test support.

Within implementation modules, focused tests should remain close to the
responsibility they verify. A focused sub-functional component should be
small enough to own focused tests.

> **Ask what invariant a test protects, not merely whether a line executes.**

## 14. Clippy and Formatting

Clippy is part of the development baseline.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Warnings should not be suppressed casually. Compilation success is not
sufficient evidence of correctness.

## 15. Performance Engineering

Performance must be measured.

```text
Implement
   ↓
Verify correctness
   ↓
Benchmark
   ↓
Profile
   ↓
Identify bottleneck
   ↓
Optimize
   ↓
Benchmark again
   ↓
Keep or revert based on evidence
```

Potential performance surfaces include allocations, cache locality/misses,
data layout, alignment, SIMD/vectorization, branch behavior, memory
bandwidth, synchronization, contention, copying, CPU/accelerator transfer,
accelerator kernel overhead, device utilization, and compiler/LLVM behavior.

> **Zero allocation is not the definition of performance.**

> **GPU execution is not automatically faster than CPU execution.**

Do not make performance claims without appropriate benchmarks or profiling.

## 16. Hardware-Aware Design and Optimization

Seqvex should remain capable of exploiting appropriate hardware without
turning every optimization mechanism into an architectural commitment.

The architecture should preserve the properties needed for efficient
implementations, including sensible data layout, ownership, locality,
vectorization, parallel execution, and accelerator placement.

Specific mechanisms such as explicit SIMD, Rayon, CUDA, ROCm, NUMA-aware
allocation, pinned memory, or zero-copy paths are implementation choices,
not requirements that every component must support.

Use this optimization ladder when relevant:

```text
correctness
    ↓
reference implementation
    ↓
benchmark
    ↓
profile
    ↓
data / layout optimization
    ↓
compiler / auto-vectorization
    ↓
explicit SIMD
    ↓
CPU parallelism
    ↓
accelerator / specialized backend
```

Not every workload should progress through every step. A small online update
may be best left as a simple CPU implementation; a large regular numerical
workload may justify further specialization.

External optimized infrastructure may be reused when it fits Seqvex's
semantic model. It must not silently redefine the meaning of the algorithm.

> **Hardware-aware design is an architectural constraint; hardware-specific
> optimization is an evidence-driven implementation decision.**

## 17. Predictability and Resource Behavior

Seqvex may eventually be used where latency, memory footprint, and failure
behavior matter.

Remain aware of bounded/unbounded work, allocation behavior, synchronization,
memory footprint, state growth, data movement, deterministic behavior, and
failure containment.

However, do not convert future goals into present guarantees.

Distinguish clearly between:

```text
architectural direction
implementation capability
measured property
formal guarantee
```

## 18. AI Development Contract

AI-assisted development is expected, but AI must operate within this
development contract.

Before modifying code, AI should read relevant documentation and existing
implementation, identify invariants, classify the change, distinguish
settled/tentative/deferred decisions, choose the smallest useful change,
and explain consequential assumptions.

AI must not invent architecture without approval, silently settle deferred
decisions, create speculative abstractions/future crates, add convenience
dependencies, generate opaque implementations the developer cannot
understand, replace foundational learning with generated code, claim
performance without evidence, or redesign unrelated components.

Prefer:

```text
small change → explain → test → measure → review → next change
```

The human developer retains responsibility for consequential design
decisions.

## 19. Change Classification

### Local implementation change

Bug fixes, focused improvements, tests, or implementation of an already
decided mechanism should proceed within existing architecture.

### Architectural change

Changes to state semantics, execution semantics, abstraction boundaries,
ownership/storage strategy, backend models, or crate responsibilities
require an explicit architectural decision before implementation.

### Experimental change

When architecture is uncertain:

> **Build an experiment instead of prematurely committing to an abstraction.**

Experiments should be small, isolated, measurable, and disposable.

## 20. Deferred Decisions

The following remain deliberately open until implementation experience
provides evidence:

- exact tensor/storage representation;
- memory ownership model;
- allocator architecture;
- device abstraction;
- backend abstraction;
- synchronization model;
- execution scheduler;
- graph/operator representation;
- exact crate boundaries;
- exact model trait hierarchy;
- serialization format;
- plugin architecture;
- GPU backend strategy;
- unified memory strategy;
- zero-copy strategy;
- sparse representation;
- distributed execution;
- multi-node execution;
- exact RL abstractions;
- exact deep-learning abstractions.

> **Deferred means intentionally deferred.**

A blank or undecided area is not an invitation to invent an answer.

## 21. Dependency and Abstraction Discipline

Every abstraction introduces conceptual complexity, maintenance cost, API
surface, compile-time consequences, potential performance implications,
and future compatibility constraints.

> **Prefer the smallest abstraction that accurately represents a demonstrated
> recurring requirement.**

Do not abstract for hypothetical reuse or optimize for theoretical elegance
at the expense of understanding.

## 22. Documentation Discipline

Documentation should record what was decided, why, supporting evidence,
uncertainty, material rejected alternatives, and what would cause a decision
to be revisited.

Use labels where appropriate:

- **Current**
- **Tentative**
- **Deferred**
- **Experimental**
- **Measured**
- **Not yet demonstrated**

Do not document speculation as fact.

## 23. Integration Discipline

Before integrating a meaningful change:

```text
Understand → Implement → Format → Test → Clippy
→ Benchmark/profile if relevant → Review architecture impact
→ Update documentation if the decision changed → Integrate
```

A change is not complete merely because it compiles.

## 24. Current Development Priority

Build Seqvex from its foundations outward:

1. Observation
2. Ordering / sequence
3. Streaming semantics
4. State
5. State transition
6. Failure atomicity
7. Minimal numerical primitives
8. Single-observation and bounded micro-batch semantics

Do not begin by implementing the full ML/RL framework.

## 25. What Success Looks Like

Early success is demonstrated by clear semantics, understandable Rust,
explicit invariants, strong tests, controlled failure behavior, measured
performance, justified abstractions, useful modular boundaries, preserved
hardware flexibility, and a credible path toward streaming and constrained
execution.

It is not measured by number of crates, algorithms, abstractions, generated
code, dependencies, premature GPU support, or theoretical performance
claims.

## 26. Guiding Rules

1. **Understand before abstracting.**
2. **Streaming-first does not mean batchless.**
3. **Separate computation semantics from hardware placement.**
4. **Treat state transitions as explicit operations with failure boundaries.**
5. **Preserve valid state when an update fails.**
6. **Keep heterogeneous memory and hardware locality possible.**
7. **Remain std-first while preserving a practical path toward constrained/no_std foundations.**
8. **Define scope by process responsibility, not function names.**
9. **Create modules and crates only when meaningful boundaries emerge.**
10. **Keep functional responsibilities locally owned until shared responsibility is demonstrated.**
11. **Document shared context at the functional-module level.**
12. **Use Rust patterns because they solve problems, not because they are fashionable.**
13. **Measure performance before optimizing.**
14. **Treat hardware-specific optimization as evidence-driven, not mandatory infrastructure.**
15. **Do not turn future goals into present guarantees.**
16. **Do not silently resolve deferred architectural decisions.**
17. **Keep AI-generated changes small enough to understand and review.**
18. **Prefer experiments when architecture is uncertain.**
19. **Let implementation evidence shape the architecture.**

## 27. Relationship to Other Documentation

`DEVELOPMENT.md` is the **canonical development contract**.

| Document | Purpose |
|---|---|
| `README.md` | Project identity, vision, scope, and public orientation |
| `ARCHITECTURE.md` | Architectural reasoning, system structure, and design decisions |
| `DEVELOPMENT.md` | Development rules for humans and AI |
| `ROADMAP.md` | Current development direction and phase planning |
| `FAILURE_AND_RECOVERY.md` | Failure paths, state integrity, and recovery principles |
| `CONTRIBUTING.md` | Contributor participation and contribution process |
| `KILOCODE_CONTEXT.md` | AI-oriented architectural comprehension before repository work |
| `KILO_SCAFFOLD.md` | Current KiloCode scaffolding constraints |

If another document conflicts with this development contract, determine
whether the conflict is an outdated document, a deliberate architectural
change, or a missing clarification. Do not silently choose one interpretation.

## Final Principle

> **Seqvex should be built by learning the problem deeply, implementing
> the smallest understandable mechanism, measuring its behavior, and
> allowing evidence—not speculation—to determine the architecture.**
