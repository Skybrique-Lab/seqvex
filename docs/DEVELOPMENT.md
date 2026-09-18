# Seqvex Development Guide

> **This document is the development contract for Seqvex.**
>
> It applies to both human contributors and AI-assisted development.
> Before changing code, read this document together with the relevant
> architecture and failure/recovery documentation.

------------------------------------------------------------------------

## 1. Purpose

`DEVELOPMENT.md` defines how Seqvex is developed, not merely how Rust
code is formatted.

It translates the project's architectural direction into practical rules
for:

- humans;
- coding agents;
- architecture changes;
- implementation;
- testing;
- benchmarking;
- profiling;
- documentation;
- integration.

The central development principle is:

> **Code should not outrun understanding.**

Seqvex is an early-stage project. Architecture, APIs, crate boundaries,
storage representations, and implementation strategies may change as
real problems are encountered and measured.

Do not freeze a design merely because it can be implemented.

------------------------------------------------------------------------

## 2. Development Philosophy

Seqvex is developed experimentally and incrementally.

The preferred loop is:

``` text
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
10. Prefer reversible decisions while the architecture is still being
    discovered.

> **Do not create an abstraction until you have experienced the problem
> it solves.**

------------------------------------------------------------------------

## 3. Architectural Invariants

The following principles should guide implementation even when the
concrete Rust design remains undecided.

### 3.1 Streaming-first, not batchless

Seqvex is **streaming-first**, but it is not batchless.

Streaming is the semantic foundation. Single-observation execution is
the smallest execution unit. Bounded micro-batches and larger batches
remain valid where they improve computational or hardware efficiency
without violating:

- ordering;
- temporal semantics;
- causality;
- state semantics;
- online-learning semantics.

A historical dataset may be replayed as a stream. A live deployment may
produce an effectively unbounded stream.

Batching is therefore an execution strategy, not the definition of the
computational model.

### 3.2 Execution semantics and compute placement are separate

Execution semantics describe **what a computation means**.

Possible semantics include:

- single observation;
- streaming/online;
- bounded micro-batch;
- batch.

Compute placement describes **where and how it executes**:

- CPU;
- GPU/accelerator;
- heterogeneous CPU/accelerator execution.

Do not allow a hardware backend to silently change the semantic meaning
of an algorithm.

### 3.3 State is first-class

Many Seqvex workloads are stateful.

A conceptual state transition is:

``` text
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

A failed update must not silently leave the model or runtime in a
partially modified or invalid state.

Detailed failure behavior is defined in:

`docs/FAILURE_AND_RECOVERY.md`

Core principle:

> **Detect → stop unsafe transition → preserve valid state → classify →
> act → report → continue/recover/isolate/terminate.**

### 3.4 Heterogeneous memory and hardware locality

Do not assume all memory is equivalent.

The architecture must preserve the ability to reason about:

``` text
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

Potential future mechanisms include:

- device-local storage;
- pinned memory;
- unified/shared memory;
- zero-copy paths;
- NUMA-aware allocation;
- explicit transfers.

These are optimization possibilities, not current API commitments.

> **Keep model state and frequently used data resident on one device
> when possible. Move data only when measured computational benefit
> exceeds transfer and synchronization cost.**

Do not equate heterogeneous execution with constant CPU↔GPU movement.

### 3.5 `std`, `no_std`, and constrained deployment

Seqvex is **std-first**.

The whole framework should not currently be forced into `#![no_std]`.
However, foundational components should avoid unnecessary dependence on
operating-system facilities where practical so that constrained or
embedded deployment remains possible.

Conceptually:

``` text
Seqvex
│
├── foundational components
│      └── potentially no_std-compatible
├── numerical / ML components
│      └── compatibility depends on requirements
├── runtime
│      └── may depend on platform facilities
└── integrations / tooling
       └── may depend on std / OS services
```

Bare-metal, embedded, low-power, and constrained deployment are future
architectural directions.

The project does **not** currently claim:

- universal `no_std` support;
- bare-metal support;
- hard real-time guarantees;
- safety certification;
- zero-allocation execution;
- cross-platform deterministic floating-point behavior.

These properties must be demonstrated before being claimed.

------------------------------------------------------------------------

## 4. Scope Boundary

Seqvex differentiates its scope by **process responsibility**, not by
the name of an individual function.

Seqvex owns the computational stages from:

- ML/RL preprocessing and representation;
- model execution;
- training;
- validation;
- inference;
- learning;
- numerical computation required by ML/RL;
- execution/runtime mechanisms.

General-purpose:

- data ingestion;
- data manipulation;
- data cleaning;
- exploratory data analysis;
- visualization;
- databases;
- storage;
- ETL;
- domain/business logic

remain outside the framework.

Therefore an operation is not classified solely by its name.

For example, one-hot encoding, normalization/standardization, PCA,
rolling statistics, and online statistics may be inside or outside
Seqvex depending on whether they are being implemented as part of
Seqvex's ML/RL computational responsibility.

The boundary is determined by **role and responsibility**.

Seqvex should not become a general-purpose dataframe, ETL, EDA, or
data-engineering framework.

------------------------------------------------------------------------

## 5. Build From First Principles

Seqvex should be understandable from its foundations.

Existing libraries may be used when they provide clear value, but
dependencies should not determine the architecture unnecessarily.

Before adopting a dependency, consider:

- what problem it solves;
- whether the problem is fundamental or incidental;
- ownership and licensing;
- compile-time impact;
- runtime overhead;
- allocation behavior;
- portability;
- `std`/`no_std` implications;
- hardware/backend constraints;
- whether the dependency introduces an abstraction Seqvex would
    otherwise need to understand itself.

Do not reject dependencies merely because they are external.

Do not add dependencies merely because they are convenient.

The objective is **deliberate dependency selection**.

------------------------------------------------------------------------

## 6. Modularity

Seqvex should be modular at meaningful responsibility boundaries.

A functional module may contain focused sub-functional components:

``` text
functional-module/
├── README.md
├── sub-function/
├── sub-function/
└── tests/
```

### Documentation rule

`README.md` belongs at the **functional-module level**, where shared
context is useful.

Do not create a README for every tiny function or sub-function.

Detailed implementation knowledge should remain close to:

- source code;
- tests;
- Rust documentation/comments;
- relevant architectural documentation.

> **Document at the highest level where shared context is useful; keep
> detailed implementation knowledge close to the code and tests.**

A folder should represent a coherent responsibility, not simply a single
function, struct, or method.

------------------------------------------------------------------------

## 7. Crate and Workspace Boundaries

Seqvex is a Cargo workspace, but a crate should not be created merely
because a future architecture diagram contains a possible component.

A new crate is justified when a meaningful boundary emerges through
actual requirements, such as:

- dependency isolation;
- independent testing;
- backend separation;
- API clarity;
- reuse;
- compile-time isolation;
- reduced coupling;
- materially different platform requirements.

Possible future areas include:

- core types and errors;
- numerical computation;
- streaming/online learning;
- sequential models;
- supervised learning;
- unsupervised learning;
- reinforcement learning;
- validation;
- execution/runtime;
- device backends.

These are architectural directions, not a frozen crate list.

> **Crate boundaries should emerge from demonstrated responsibility
> boundaries.**

------------------------------------------------------------------------

## 8. Rust Engineering Principles

These principles are important design tools. They are **not patterns to
apply mechanically**.

### 8.1 DRY

Avoid duplicated knowledge, invariants, and logic.

Do not abstract merely because two pieces of code currently look
similar.

Abstract repeated concepts when their shared behavior is real and likely
to remain shared.

### 8.2 Composition over inheritance

Rust does not use classical inheritance.

Prefer focused structures and composable behavior:

``` rust
struct Model<S, E> {
    state: S,
    executor: E,
}
```

Avoid deep inheritance-like hierarchies expressed through increasingly
complex traits.

### 8.3 Traits

Introduce a trait when it provides a real semantic contract, such as:

- multiple meaningful implementations;
- dependency inversion;
- backend abstraction;
- testability;
- stable behavioral requirements.

Do not create a trait for every struct.

### 8.4 Extension traits

Use extension traits for coherent domain-specific behavior on types
Seqvex does not own.

Do not use them merely to rename trivial methods or hide simple code.

### 8.5 Builder pattern

Use builders where construction involves meaningful configuration,
optional parameters, validation, or staged construction.

Do not introduce a builder for a trivial structure.

### 8.6 Typestate

Use typestate where compile-time state distinctions provide meaningful
correctness guarantees.

It may become useful for:

- lifecycle management;
- runtime state machines;
- initialization/ready states;
- asynchronous state transitions.

Do not build a speculative typestate framework before the problem
exists.

### 8.7 Newtype pattern

Use newtypes when distinct semantic values should not be
interchangeable:

``` rust
struct ObservationId(u64);
struct SequenceNumber(u64);
```

Do not wrap every primitive merely for abstraction's sake.

### 8.8 Error enums

Prefer explicit domain-specific error types:

``` rust
enum UpdateError {
    InvalidInput,
    NumericalFailure,
    InvariantViolation,
}
```

Prefer `Result<T, E>` and structured error propagation.

Avoid stringly typed core errors.

Avoid unnecessary `unwrap()` and `expect()` in production paths.

If an `unwrap()` or `expect()` is justified by an invariant, the
invariant should be clear.

### 8.9 Iterators and functional style

Prefer iterator composition when it improves clarity without causing a
material performance problem.

``` rust
let sum: f64 = values.iter().copied().sum();
```

Do not force functional style when an explicit loop is clearer or
measured to be more appropriate.

------------------------------------------------------------------------

## 9. Numerical Foundation

Seqvex requires numerical computation for ML/RL, but it is not intended
to replace general-purpose scientific or data-processing ecosystems.

Seqvex should not become:

- NumPy;
- SciPy;
- Pandas;
- Polars;
- a dataframe engine;
- general statistical software.

Numerical primitives should be introduced incrementally when they
support actual ML/RL requirements.

Possible foundations include:

- sums;
- means;
- variance;
- covariance;
- dot products;
- vector norms;
- online statistics;
- rolling statistics;
- matrix/vector operations required by implemented algorithms.

Do not build a complete numerical ecosystem before real Seqvex workloads
justify it.

------------------------------------------------------------------------

## 10. Execution and State Semantics

Implementation should preserve the distinction between:

``` text
single observation
streaming / online
micro-batch
batch
```

and:

``` text
CPU
GPU / accelerator
heterogeneous
```

A backend optimization must not accidentally redefine the algorithm's
temporal or state semantics.

### State update contract

A stateful update should conceptually behave as:

``` text
current valid state
       │
       ▼
compute candidate
       │
       ▼
validate candidate
       │
   ┌───┴───┐
   │       │
 valid   invalid
   │       │
   ▼       ▼
commit   preserve
state    prior state
```

The precise implementation mechanism is deliberately not fixed.

Possible mechanisms include:

- transactional construction;
- checkpoint/restore;
- copy-on-write;
- double buffering;
- rollback information;
- algorithm-specific approaches.

Choose the mechanism only after the actual state model and failure
characteristics are understood.

------------------------------------------------------------------------

## 11. Failure, Recovery, and Determinism

Failure handling is part of the architecture, not an afterthought.

Relevant failure classes include:

- invalid input;
- numerical failure;
- model/state invariant failure;
- memory/allocation failure;
- runtime failure;
- hardware/device failure.

For every stateful operation, consider:

1. What can fail?
2. Can the current state remain valid?
3. Can the operation be retried safely?
4. Is rollback meaningful?
5. Should processing continue?
6. Should the component be isolated?
7. Should execution terminate?
8. What must be reported?
9. Does recovery preserve deterministic semantics?

Rollback is not universal.

Do not add rollback machinery simply because failure exists.

Detailed requirements belong in:

`docs/FAILURE_AND_RECOVERY.md`

------------------------------------------------------------------------

## 12. Testing

Testing should establish behavior and invariants, not merely increase
coverage numbers.

For foundational components, test:

- valid behavior;
- invalid inputs;
- boundary conditions;
- ordering;
- state transitions;
- state preservation after failure;
- repeated streaming updates;
- deterministic behavior where promised;
- numerical edge cases;
- single-observation semantics;
- bounded micro-batch semantics where implemented.

Where meaningful, test properties rather than only examples.

### Test ownership

A focused sub-functional component should be small enough to own focused
tests.

Tests should help answer:

> **What invariant does this component guarantee?**

rather than merely:

> **Does this line execute?**

------------------------------------------------------------------------

## 13. Clippy and Formatting

Clippy is part of the development baseline.

Use:

``` bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Warnings should not be suppressed casually.

When an exception is genuinely justified:

- keep the scope narrow;
- document why it is justified;
- avoid turning a local exception into a global policy.

Compilation success is not sufficient evidence of correctness.

------------------------------------------------------------------------

## 14. Performance Engineering

Performance must be measured.

The preferred loop is:

``` text
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

Potential performance surfaces include:

- allocations;
- cache locality;
- cache misses;
- data layout;
- alignment;
- SIMD/vectorization;
- branch behavior;
- memory bandwidth;
- synchronization;
- contention;
- copying;
- CPU/accelerator transfer;
- accelerator kernel overhead;
- device utilization;
- compiler/LLVM behavior.

> **Zero allocation is not the definition of performance.**

Likewise:

> **GPU execution is not automatically faster than CPU execution.**

Small stateful online updates may favor CPU execution. Large parallel
numerical workloads may benefit from accelerators. The correct choice
depends on measured workload characteristics.

Do not make performance claims without benchmarks or profiling evidence
appropriate to the claim.

------------------------------------------------------------------------

## 15. Predictability and Resource Behavior

Seqvex may eventually be used in environments where latency, memory
footprint, and failure behavior matter.

Development should therefore remain aware of:

- bounded versus unbounded work;
- allocation behavior;
- synchronization;
- memory footprint;
- state growth;
- data movement;
- deterministic behavior;
- failure containment.

However, do not convert future goals into present guarantees.

The project should distinguish clearly between:

``` text
architectural direction
implementation capability
measured property
formal guarantee
```

These are not interchangeable.

------------------------------------------------------------------------

## 16. AI Development Contract

AI-assisted development is expected, but AI must operate within the
development contract.

### 16.1 Before modifying code, AI should

1. Read the relevant documentation.
2. Inspect the existing implementation.
3. Identify existing invariants.
4. Identify whether the requested change is architectural or local.
5. Distinguish settled decisions from tentative and deferred decisions.
6. Determine the smallest change that can test or implement the
    requirement.
7. Explain consequential assumptions before encoding them.

### 16.2 AI should preserve

- existing architectural boundaries;
- streaming/state semantics;
- failure atomicity;
- type safety;
- testability;
- deterministic behavior where promised;
- hardware-neutral high-level concepts;
- deferred decisions.

### 16.3 AI must not

- invent architecture without approval;
- silently settle deferred decisions;
- create speculative abstractions;
- create future crates prematurely;
- introduce dependencies merely for convenience;
- generate large opaque implementations that the developer cannot
    understand;
- replace foundational learning with generated code;
- claim performance without evidence;
- treat compilation as proof of correctness;
- redesign unrelated components while implementing a local
    requirement;
- turn every possible future requirement into current infrastructure.

### 16.4 AI should prefer

``` text
small change
   ↓
explain
   ↓
test
   ↓
measure
   ↓
review
   ↓
next change
```

over:

``` text
large generated implementation
   ↓
hope it matches the architecture
```

AI is particularly useful as:

- teacher;
- Socratic reviewer;
- research assistant;
- architecture reviewer;
- code reviewer;
- debugging assistant;
- test designer;
- documentation assistant.

The human developer retains responsibility for understanding and
accepting consequential design decisions.

------------------------------------------------------------------------

## 17. Change Classification

Before making a significant change, classify it.

### Local implementation change

Examples:

- fixing a bug;
- improving a focused function;
- adding a test;
- implementing an already-decided mechanism.

Proceed within existing architecture.

### Architectural change

Examples:

- changing state semantics;
- changing execution semantics;
- introducing a new abstraction boundary;
- changing ownership or storage strategy;
- introducing a new backend model;
- changing crate responsibilities.

Stop and document the decision before implementation.

### Experimental change

If the correct architecture is uncertain:

> **Build an experiment instead of prematurely committing to an
> abstraction.**

Experiments should be:

- small;
- isolated;
- measurable;
- disposable.

A successful experiment provides evidence. It does not automatically
become production architecture.

------------------------------------------------------------------------

## 18. Deferred Decisions

The following should remain deliberately open until implementation
experience provides evidence:

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
- exact reinforcement-learning abstractions;
- exact deep-learning abstractions.

A blank or undecided area is not an invitation to invent an answer.

> **Deferred means intentionally deferred.**

When evidence changes the decision, update the relevant architecture
documentation and development guidance.

------------------------------------------------------------------------

## 19. Dependency and Abstraction Discipline

Every abstraction introduces:

- conceptual complexity;
- maintenance cost;
- API surface;
- compile-time consequences;
- potential performance implications;
- future compatibility constraints.

Therefore:

> **Prefer the smallest abstraction that accurately represents a
> demonstrated recurring requirement.**

Do not abstract for hypothetical reuse.

Do not optimize for theoretical elegance at the expense of
understanding.

Do not allow the desire for a sophisticated architecture to outrun the
amount of implementation evidence available.

------------------------------------------------------------------------

## 20. Documentation Discipline

Documentation should record:

- what was decided;
- why it was decided;
- what evidence supports it;
- what remains uncertain;
- what alternatives were rejected when the rejection matters;
- what would cause the decision to be revisited.

Do not document speculation as fact.

Use clear labels where appropriate:

- **Current**
- **Tentative**
- **Deferred**
- **Experimental**
- **Measured**
- **Not yet demonstrated**

Architecture documentation should explain durable reasoning.

Code comments should explain local reasoning that is not obvious from
the code.

Avoid comments that merely restate syntax.

------------------------------------------------------------------------

## 21. Integration Discipline

Before integrating a meaningful change:

``` text
Understand
  ↓
Implement
  ↓
Format
  ↓
Test
  ↓
Clippy
  ↓
Benchmark/profile if relevant
  ↓
Review architecture impact
  ↓
Update documentation if the decision changed
  ↓
Integrate
```

A change is not complete merely because it compiles.

For foundational changes, ask:

- Does the state model remain valid?
- Does failure remain atomic?
- Does streaming semantics remain intact?
- Are ordering and temporal assumptions preserved?
- Did a local implementation accidentally freeze a deferred
    architectural decision?
- Did the change introduce unnecessary dependencies?
- Did it create an abstraction whose value has not been demonstrated?

------------------------------------------------------------------------

## 22. Current Development Priority

Seqvex should be built from its foundations outward.

The initial conceptual foundation is:

1. Observation
2. Ordering / sequence
3. Streaming semantics
4. State
5. State transition
6. Failure atomicity
7. Minimal numerical primitives
8. Single-observation and bounded micro-batch semantics

Do not begin by implementing the full ML/RL framework.

The objective of the foundation is to establish the semantics on which
later algorithms can depend.

------------------------------------------------------------------------

## 23. What Success Looks Like

Early Seqvex development is successful when the project increasingly
demonstrates:

- clear semantics;
- understandable Rust;
- explicit invariants;
- strong tests;
- controlled failure behavior;
- measured performance;
- justified abstractions;
- useful modular boundaries;
- preserved hardware flexibility;
- a credible path toward streaming and constrained execution.

It is **not** measured by:

- number of crates;
- number of algorithms implemented;
- number of abstractions;
- amount of generated code;
- number of dependencies;
- premature GPU support;
- theoretical performance claims.

------------------------------------------------------------------------

## 24. Guiding Rules

When uncertain, return to these rules:

1. **Understand before abstracting.**
2. **Streaming-first does not mean batchless.**
3. **Separate computation semantics from hardware placement.**
4. **Treat state transitions as explicit operations with failure
    boundaries.**
5. **Preserve valid state when an update fails.**
6. **Keep heterogeneous memory and hardware locality possible.**
7. **Remain std-first while preserving a practical path toward
    constrained/no_std foundations.**
8. **Define scope by process responsibility, not function names.**
9. **Create modules and crates only when meaningful boundaries
    emerge.**
10. **Document shared context at the functional-module level.**
11. **Use Rust patterns because they solve problems, not because they
    are fashionable.**
12. **Measure performance before optimizing.**
13. **Do not turn future goals into present guarantees.**
14. **Do not silently resolve deferred architectural decisions.**
15. **Keep AI-generated changes small enough to understand and review.**
16. **Prefer experiments when architecture is uncertain.**
17. **Let implementation evidence shape the architecture.**

------------------------------------------------------------------------

## 25. Relationship to Other Documentation

`DEVELOPMENT.md` is the **canonical development contract**.

Other documents serve narrower purposes:

-----------------------------------------------------------------------
  Document                            Purpose
----------------------------------- -----------------------------------
  `README.md`                         Project identity, vision, scope,
                                      and public orientation

  `ARCHITECTURE.md`                   Architectural reasoning, system
                                      structure, and design decisions

  `DEVELOPMENT.md`                    Development rules for humans and AI

  `FAILURE_AND_RECOVERY.md`           Failure paths, state integrity, and
                                      recovery principles

  `CONTRIBUTING.md`                   Contributor participation and
                                      contribution process

  `KILOCODE_CONTEXT.md`               AI-oriented architectural
                                      comprehension before repository
                                      work

  `KILO_SCAFFOLD.md`                  Current KiloCode scaffolding
                                      constraints
  -----------------------------------------------------------------------

If another document conflicts with this development contract, determine
whether the conflict represents:

1. an outdated document;
2. a deliberate architectural change;
3. a missing clarification.

Do not silently choose one interpretation.

Update the relevant documentation when a durable decision changes.

------------------------------------------------------------------------

## Final Principle

> **Seqvex should be built by learning the problem deeply, implementing
> the smallest understandable mechanism, measuring its behavior, and
> allowing evidence---not speculation---to determine the architecture.**
