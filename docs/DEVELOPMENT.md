# Seqvex Development Contract

> **Status:** Canonical development contract  
> **Audience:** Human contributors and AI-assisted development agents  
> **Project:** Seqvex

This document defines how Seqvex should be developed.

It is intentionally stricter than a conventional contribution guide because the project is still establishing its architecture. The goal is to prevent premature abstraction, accidental architectural commitments, and optimization decisions that later constrain hardware-aware execution.

---

# 1. Core Development Philosophy

Seqvex should be developed through a closed learning loop:

```text
Learn
  ↓
Implement the smallest useful experiment
  ↓
Test
  ↓
Break / find edge cases
  ↓
Understand
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

The loop is not a mandatory checklist for every tiny change. Documentation edits and straightforward bug fixes may require only part of it.

The principle is:

> **Implementation produces evidence; evidence shapes architecture.**

---

# 2. Architecture Before Implementation

Before implementing a non-trivial change:

1. Read the relevant documentation.
2. Inspect the existing implementation.
3. Identify the invariants being protected.
4. Classify the proposed change.
5. Determine whether the architecture is settled, tentative, experimental, or deferred.
6. Choose the smallest change that tests the requirement.
7. Identify any architecture-level consequences before coding.

Do not silently settle a deferred decision merely because an implementation needs an answer.

When architecture is uncertain:

> **Build an experiment instead of prematurely committing to an abstraction.**

---

# 3. Critical Architecture Review Guard

## 3.1 Mandatory rule

**Any implementation, optimization, or API change that could materially constrain, complicate, or prematurely freeze:**

- hardware-aware execution;
- heterogeneous memory/device placement;
- accelerator offload;
- CPU/GPU/device residency;
- per-stream versus model-owned state or scratch;
- synchronization strategy;
- Rust-native low-level optimization;
- low-level allocation/layout control;

**must undergo CRITICAL ARCHITECTURE REVIEW before implementation.**

This is a permanent development rule.

It applies even when the proposed change appears locally small.

## 3.2 Why this guard exists

Seqvex is deliberately leaving several high-reversal-cost decisions open.

A locally efficient implementation can accidentally make a later design much harder if it assumes:

```text
model owns all mutable execution resources
```

when the eventual architecture may require:

```text
model parameters
    +
per-stream state
    +
per-stream workspace
    +
device-specific execution
```

or another topology.

The purpose of the review is not to prevent optimization.

It is to prevent an optimization from silently becoming the architecture.

## 3.3 Examples that require review

Examples include:

- changing a public API from `&T` to `&mut T` because an implementation needs mutable scratch;
- making a model own a reusable workspace;
- introducing a generic allocator or memory pool;
- introducing a device abstraction;
- introducing a scheduler;
- making a tensor representation globally canonical;
- adding a synchronization model;
- adding GPU/accelerator-specific ownership assumptions;
- introducing unsafe low-level primitives that constrain later backends;
- changing state ownership to make a benchmark faster.

The GRU workspace and current `StreamingExecutor` `&mut M` API are active examples.

## 3.4 Review output

A CRITICAL ARCHITECTURE REVIEW should state:

- the concrete requirement;
- current evidence;
- affected invariants;
- alternatives considered;
- reversal cost;
- what is being intentionally left undecided;
- whether the implementation should proceed;
- what measurements or workloads are still required.

A review does not automatically authorize implementation.

---

# 4. Seqvex Semantic Hierarchy

Implementation must preserve the distinction between:

```text
single observation
        ↓
streaming / online
        ↓
bounded micro-batch
        ↓
larger batch
```

The intended hierarchy is:

- **Single observation** — fundamental semantic unit.
- **Streaming / online** — primary computational context.
- **Bounded micro-batching** — optimization/capability.
- **Larger batch execution** — secondary capability where useful.

This does not mean Seqvex cannot process large datasets or high-dimensional observations.

It means sequential/stateful semantics, temporal ordering, and evolving computation—not dataset size—are the organizing principles.

---

# 5. Execution Semantics vs Hardware Placement

Keep these dimensions separate.

Execution semantics describe:

```text
single observation
streaming / online
micro-batch
batch
```

Placement describes:

```text
CPU
GPU
accelerator
heterogeneous resources
```

A hardware optimization must not silently redefine the temporal or state semantics of an algorithm.

---

# 6. State and Failure Semantics

For stateful operations, reason explicitly about:

```text
current valid state
        ↓
candidate state
        ↓
validation
        ↓
commit
```

A failure must not silently commit an invalid or partial state.

Before implementing a stateful operation, determine:

- what can fail;
- whether valid state can be preserved;
- whether retry is safe;
- whether rollback is meaningful;
- whether processing should continue;
- what must be reported;
- whether recovery preserves required determinism.

Rollback is not universal. Do not add rollback machinery merely because failure exists.

See `docs/FAILURE_AND_RECOVERY.md` for detailed failure policy.

---

# 7. Scope Boundary

Seqvex owns computational responsibility from:

- ML/RL preprocessing and representation;
- model computation;
- training and learning;
- validation;
- inference;
- numerical computation required by ML/RL;
- execution/runtime mechanisms.

General-purpose ingestion, storage, databases, ETL, generic cleaning/manipulation, exploratory analysis, visualization, and domain/business logic remain outside the framework.

The boundary is based on the role of a computation in the ML/RL pipeline.

---

# 8. Build From First Principles

Seqvex should be understandable from its foundations.

Existing libraries may be used when they provide clear value, but dependencies should not determine Seqvex's architecture unnecessarily.

Before adopting a dependency, consider:

- problem solved;
- ownership/licensing;
- compile-time impact;
- runtime overhead;
- allocation behavior;
- portability;
- `std`/`no_std` implications;
- hardware/backend constraints;
- whether it introduces an abstraction Seqvex would otherwise need to understand itself.

Do not reject dependencies merely because they are external. Do not add dependencies merely because they are convenient.

---

# 9. Modularity

A module should represent a coherent responsibility.

Do not create a README, crate, abstraction, or helper solely because a folder or function exists.

A shared abstraction should emerge when:

- responsibility is genuinely cross-functional;
- a stable semantic contract exists;
- local ownership creates meaningful duplication of domain knowledge;
- or sharing is required for a concrete architectural property.

> **DRY should remove duplicated knowledge, not erase useful ownership boundaries.**

---

# 10. Shared Infrastructure and Global Helpers

Do not introduce global `common`, `utils`, or helper modules merely to remove local duplication.

The same applies to:

- shared error types;
- state helpers;
- numerical utilities;
- execution contexts;
- memory/storage abstractions;
- test support.

Similar implementation is not sufficient evidence for a shared abstraction.

Prefer local ownership first, then refactor upward when repeated responsibility and dependency relationships demonstrate that the abstraction is genuinely shared.

---

# 11. Crate and Workspace Boundaries

Seqvex is currently a **single Cargo package**.

A workspace and additional crates may emerge later when demonstrated responsibility boundaries justify them.

Possible future areas include:

- core types/errors;
- numerical computation;
- streaming/online learning;
- sequential models;
- supervised/unsupervised learning;
- reinforcement learning;
- validation;
- execution/runtime;
- device backends.

These are directions, not a frozen crate list.

---

# 12. Rust Engineering Principles

These principles are design tools, not patterns to apply mechanically.

- **DRY:** avoid duplicated knowledge and invariants.
- **Composition over inheritance:** prefer focused structures.
- **Traits:** introduce for real semantic contracts or meaningful implementations.
- **Extension traits:** use for coherent behavior on types Seqvex does not own.
- **Builder pattern:** use for meaningful configuration and validation.
- **Typestate:** use only when compile-time distinctions provide real correctness.
- **Newtype:** use for distinct semantic values.
- **Error enums:** prefer domain-specific errors and `Result<T,E>`.
- **Iterators:** use where they improve clarity without material performance cost; explicit loops are appropriate when clearer or measurably better.

## Unsafe Rust

`unsafe` may be justified for:

- SIMD;
- specialized memory access;
- FFI;
- accelerator interfaces;
- custom allocators;
- low-level kernels.

Unsafe boundaries require documented invariants and appropriate tests.

An unsafe optimization that could materially constrain later hardware-aware architecture requires CRITICAL ARCHITECTURE REVIEW first.

---

# 13. Numerical Foundation

Seqvex requires numerical computation for ML/RL but is not intended to replace general dataframe, scientific-computing, or statistical ecosystems.

Introduce numerical primitives incrementally when they support actual workloads.

Do not build a complete numerical ecosystem before real Seqvex workloads justify it.

---

# 14. Execution and State Semantics

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

A backend optimization must not accidentally redefine the algorithm's temporal or state semantics.

The precise state-transition implementation mechanism remains deliberately open until actual state models and failure characteristics justify a stable abstraction.

---

# 15. Automatic Execution Selection

Seqvex may eventually support both:

```text
Explicit
    user selects execution policy

Auto
    user permits Seqvex to select among available strategies
```

Automatic selection is intended to assist execution choice, not remove user control.

A user must be able to override Auto explicitly.

Potential signals include:

- workload dimensions;
- observation frequency;
- statefulness;
- latency/throughput requirements;
- computational intensity;
- memory behavior;
- available hardware;
- transfer/synchronization costs;
- historical replay versus live operation;
- online learning requirements;
- whether micro-batching preserves semantics.

This is a **future architecture/design direction**, not a current implementation.

Do not create a generic scheduler or automatic execution framework until multiple real execution strategies and workloads demonstrate a stable requirement.

---

# 16. Failure, Recovery, and Determinism

Failure classes may include:

- invalid input;
- numerical failure;
- state/model invariant failure;
- memory/allocation failure;
- runtime failure;
- hardware/device failure.

For each stateful operation, consider:

- what can fail;
- state preservation;
- retry safety;
- rollback semantics;
- continuation policy;
- error reporting;
- deterministic recovery where promised.

Determinism is a capability, not a universal requirement.

Do not promise deterministic floating-point behavior across arbitrary hardware unless measured and intentionally specified.

---

# 17. Testing

Testing should establish behavior and invariants, not merely increase coverage numbers.

For foundational components, test:

- valid behavior;
- invalid inputs;
- boundaries;
- ordering;
- state transitions;
- state preservation after failure;
- repeated streaming updates;
- deterministic behavior where promised;
- numerical edge cases;
- single-observation semantics;
- bounded micro-batch semantics where implemented.

Ask:

> **What invariant does this test protect?**

---

# 18. Formatting and Clippy

The development baseline is:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Warnings should not be suppressed casually.

Compilation success is not sufficient evidence of correctness.

---

# 19. Performance Engineering

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

Potential surfaces include:

- allocations;
- cache locality/misses;
- data layout;
- alignment;
- SIMD/vectorization;
- branch behavior;
- memory bandwidth;
- synchronization;
- contention;
- copying;
- CPU/accelerator transfer;
- kernel launch overhead;
- device utilization;
- compiler/LLVM behavior.

> **Zero allocation is not the definition of performance.**

> **GPU execution is not automatically faster than CPU execution.**

Do not make performance claims without appropriate benchmarks or profiling.

---

# 20. Hardware-Aware Design and Optimization

Seqvex should remain capable of exploiting appropriate hardware without turning every optimization mechanism into an architectural commitment.

Preserve the properties needed for efficient implementations:

- sensible data layout;
- ownership;
- locality;
- vectorization;
- parallel execution;
- accelerator placement;
- controlled data movement.

Specific mechanisms such as explicit SIMD, Rayon, CUDA, ROCm, NUMA-aware allocation, pinned memory, or zero-copy paths are implementation choices, not requirements for every component.

Use this optimization ladder where relevant:

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

Not every workload should progress through every step.

> **Hardware-aware design is an architectural constraint; hardware-specific optimization is an evidence-driven implementation decision.**

---

# 21. Predictability and Resource Behavior

Remain aware of:

- bounded/unbounded work;
- allocation behavior;
- synchronization;
- memory footprint;
- state growth;
- data movement;
- deterministic behavior;
- failure containment.

Distinguish:

```text
architectural direction
implementation capability
measured property
formal guarantee
```

Do not convert future goals into present guarantees.

---

# 22. AI Development Contract

AI-assisted development is expected, but AI must operate within this contract.

Before modifying code, AI should:

1. read relevant documentation;
2. inspect the existing implementation;
3. identify invariants;
4. classify the change;
5. distinguish settled/tentative/deferred decisions;
6. choose the smallest useful change;
7. explain consequential assumptions;
8. run appropriate validation;
9. document evidence for material decisions.

AI must not:

- invent architecture without approval;
- silently settle deferred decisions;
- create speculative abstractions;
- add convenience dependencies without justification;
- generate opaque implementations the developer cannot understand;
- claim performance without evidence;
- redesign unrelated components;
- bypass CRITICAL ARCHITECTURE REVIEW.

Prefer:

```text
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

The human developer retains responsibility for consequential design decisions.

---

# 23. Change Classification

## Local implementation change

Bug fixes, focused improvements, tests, or implementation of an already decided mechanism may proceed within existing architecture.

## Architectural change

Changes to:

- state semantics;
- execution semantics;
- abstraction boundaries;
- ownership/storage strategy;
- backend models;
- crate responsibilities;
- public API topology that affects hardware-aware execution;

require an explicit architectural decision before implementation.

If the change falls under the CRITICAL ARCHITECTURE REVIEW guard, review must occur **before** implementation.

## Experimental change

When architecture is uncertain:

> **Build an experiment instead of prematurely committing to an abstraction.**

Experiments should be:

- small;
- isolated;
- measurable;
- disposable;
- clearly separated from production code.

---

# 24. Issue and Planning Discipline

GitHub issues are part of the project's architectural paper trail.

Before creating, reopening, updating, or closing an issue:

1. inspect the relevant repository evidence;
2. confirm the issue is actually required;
3. consult the owner before making consequential issue-state changes;
4. use the repository's existing labels, issue types, and milestones;
5. do not create duplicate labels, milestones, or issues;
6. preserve the issue's relationship to the relevant architecture and evidence.

For KiloCode or AI reviews, record the review/audit outcome in the relevant GitHub issue comment even when the issue remains open or is later closed.

Implementation issues should not be created merely because an idea exists.

Prefer:

```text
evidence
   ↓
architecture/review
   ↓
decision
   ↓
issue
   ↓
implementation
```

when the work is architecture-sensitive.

---

# 25. Deferred Decisions

The following remain deliberately open until implementation experience provides evidence:

- exact tensor/storage representation;
- memory ownership model;
- allocator architecture;
- device abstraction;
- backend abstraction;
- synchronization model;
- execution scheduler;
- automatic execution-selection mechanism;
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

---

# 26. Dependency and Abstraction Discipline

Every abstraction introduces:

- conceptual complexity;
- maintenance cost;
- API surface;
- compile-time consequences;
- performance implications;
- future compatibility constraints.

> **Prefer the smallest abstraction that accurately represents a demonstrated recurring requirement.**

Do not abstract for hypothetical reuse.

Do not optimize for theoretical elegance at the expense of understanding.

---

# 27. Documentation Discipline

Documentation should record:

- what was decided;
- why;
- supporting evidence;
- uncertainty;
- material rejected alternatives;
- what would cause the decision to be revisited.

Use labels where appropriate:

- **Current**
- **Tentative**
- **Deferred**
- **Experimental**
- **Measured**
- **Not yet demonstrated**

Do not document speculation as fact.

When a public API or implementation topology is provisional, say so explicitly.

---

# 28. Integration Discipline

Before integrating a meaningful change:

```text
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

For architecture-sensitive changes, the integration record should identify:

- the evidence;
- the decision;
- the affected issue;
- the remaining uncertainty.

---

# 29. Current Development Priority

Build Seqvex from its foundations outward:

1. Observation
2. Ordering / sequence
3. Streaming semantics
4. State
5. State transition
6. Failure atomicity
7. Minimal numerical primitives
8. Single-observation semantics
9. Bounded micro-batch semantics where demonstrated useful

Do not begin by implementing the full ML/RL framework.

---

# 30. What Success Looks Like

Early success is demonstrated by:

- clear semantics;
- understandable Rust;
- explicit invariants;
- strong tests;
- controlled failure behavior;
- measured performance;
- justified abstractions;
- useful modular boundaries;
- preserved hardware flexibility;
- credible streaming execution;
- a path toward constrained execution.

It is not measured by:

- number of crates;
- number of algorithms;
- number of abstractions;
- generated code volume;
- dependency count;
- premature GPU support;
- theoretical performance claims.

---

# 31. Guiding Rules

1. **Understand before abstracting.**
2. **Streaming-first does not mean batchless.**
3. **Single observation is the fundamental semantic unit.**
4. **Micro-batching is an optimization, not the semantic foundation.**
5. **Separate computation semantics from hardware placement.**
6. **Treat state transitions as explicit operations with failure boundaries.**
7. **Preserve valid state when an update fails.**
8. **Keep heterogeneous memory and hardware locality possible.**
9. **Remain std-first while preserving a practical path toward constrained/no_std foundations.**
10. **Define scope by process responsibility, not function names.**
11. **Create modules and crates only when meaningful boundaries emerge.**
12. **Keep functional responsibilities locally owned until shared responsibility is demonstrated.**
13. **Use Rust patterns because they solve problems, not because they are fashionable.**
14. **Measure performance before optimizing.**
15. **Treat hardware-specific optimization as evidence-driven.**
16. **Do not turn future goals into present guarantees.**
17. **Do not silently resolve deferred architectural decisions.**
18. **Keep AI-generated changes small enough to understand and review.**
19. **Use CRITICAL ARCHITECTURE REVIEW before changes that could constrain hardware-aware execution, heterogeneous memory/device placement, accelerator offload, or Rust-native low-level optimization.**
20. **Prefer experiments when architecture is uncertain.**
21. **Let implementation evidence shape the architecture.**

---

# 32. Relationship to Other Documentation

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

If another document conflicts with this development contract, determine whether the conflict is an outdated document, a deliberate architectural change, or a missing clarification.

Do not silently choose one interpretation.

---

# Final Principle

> **Seqvex should be built by learning the problem deeply, implementing the smallest understandable mechanism, measuring its behavior, and allowing evidence—not speculation—to determine the architecture.**
