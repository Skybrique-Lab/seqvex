# Seqvex Development Guide

> **Status:** Early-stage development guide / living document

## 1. Purpose

This document describes how Seqvex should be developed.

Seqvex is intentionally built incrementally. The objective is not to design the complete framework first, but to gain implementation experience, test real behavior, measure actual constraints, and allow durable abstractions to emerge from those observations.

The core development loop is:

```text
Understand
    ↓
Define expected behavior
    ↓
Write test cases
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
    ↓
Automate with CI
```

The guiding rule is:

> **Do not create an abstraction until you have experienced the problem it solves.**

---

## 2. What We Develop

Seqvex owns the computational stages of the ML/RL pipeline rather than trying to own every function associated with data.

> **Seqvex differentiates its scope by process responsibility rather than by individual function. It owns the computational stages from ML/RL preprocessing and representation through model execution, training, validation, inference, and learning. General-purpose data ingestion, manipulation, cleaning, exploratory analysis, visualization, and storage remain outside the framework.**

This means that operations such as one-hot encoding, normalization, standardization, PCA, rolling statistics, or online statistics may be implemented when they serve the ML/RL pipeline.

The same operation may remain outside Seqvex when it is being used as general-purpose data analysis or manipulation.

The boundary is therefore based on **responsibility and role**, not a fixed list of functions.

---

## 3. Development Progression

Development should generally progress from small, understandable components toward integrated framework behavior.

```text
Rust foundations
      ↓
Numerical primitives
      ↓
Statistics
      ↓
Classical ML
      ↓
Online / streaming ML
      ↓
Temporal validation
      ↓
Performance fundamentals
      ↓
Execution/runtime experiments
      ↓
Model and state integration
      ↓
Hardware/device experiments
      ↓
Proven abstractions
      ↓
Framework integration
```

The sequence is a direction rather than a rigid schedule.

If implementation experience reveals that a different order is required, the order should change.

---

## 4. Start With the Smallest Useful Problem

Before creating a new abstraction, answer:

1. What concrete problem are we solving?
2. What behavior should the implementation provide?
3. What are the normal and failure cases?
4. What invariants must remain true?
5. Can the problem be demonstrated with a small experiment?

Prefer:

```text
small problem
    ↓
small implementation
    ↓
understand behavior
    ↓
test
```

over:

```text
large requirement
    ↓
large abstraction
    ↓
many traits
    ↓
large implementation
```

A capability should earn additional complexity through actual requirements.

---

## 5. Test Cases as Development Specifications

Test cases are a central development tool for Seqvex.

They serve three purposes:

1. Define expected behavior.
2. Provide a way to personally verify implementations.
3. Become automated regression protection through CI.

The preferred approach is to write understandable tests that the developer can explain and maintain.

A basic example:

```text
Given a valid model state
When a valid observation is processed
Then the expected state transition occurs.
```

Failure behavior should also be tested:

```text
Given a valid model state
When an update fails
Then the previous valid state remains active
And the failure is reported.
```

Tests should cover:

- normal behavior;
- edge cases;
- invalid input;
- state transitions;
- failure paths;
- numerical invariants;
- integration behavior.

---

## 6. TDD: Use It Where It Helps

Seqvex does not require strict project-wide Test-Driven Development.

Use a test-first approach when:

- expected behavior is already understood;
- the behavior can be stated clearly;
- the test is easier to write than the implementation;
- the test defines an important invariant or contract.

A useful TDD loop is:

```text
RED
  ↓
write failing test
  ↓
GREEN
  ↓
smallest implementation
  ↓
REFACTOR
  ↓
run tests again
```

Do not force TDD onto exploratory work where the purpose is to discover the correct algorithm, representation, or abstraction.

For exploratory low-level work, it is acceptable to:

```text
experiment
   ↓
understand
   ↓
define expected behavior
   ↓
write durable tests
   ↓
integrate
```

---

## 7. Unit Tests

Use unit tests for small, isolated behavior.

Typical examples:

```text
statistics
numerical operations
state transitions
model updates
validation
error handling
```

A unit test should answer a specific question.

Examples:

```text
Does online variance produce the expected result?

Does an update preserve the model invariant?

Does invalid input return the expected failure?

Does a rejected update leave committed state unchanged?
```

Keep tests readable enough that their purpose is immediately apparent.

---

## 8. Integration Tests

Use integration tests when multiple components must work together.

Examples:

```text
input
  ↓
ML/RL preprocessing
  ↓
model
  ↓
state update
  ↓
prediction / learning
```

Another important path is:

```text
input
  ↓
model
  ↓
failed update
  ↓
state preserved
  ↓
failure reported
  ↓
next observation
```

Integration tests should be introduced when separate components have become meaningful enough to exercise together.

Do not build a large integration-test framework before there is a real integration to test.

---

## 9. Property and Invariant Tests

Use property or invariant testing where behavior is better expressed as a rule than as one expected output.

Examples:

```text
variance >= 0

compatible dimensions remain compatible

valid state transitions preserve required invariants

a rejected update does not modify committed state

failed updates cannot expose partially committed state
```

Property-based testing may be introduced when it provides meaningful coverage of large input spaces.

The testing mechanism should remain proportional to the problem.

---

## 10. BDD and Behavioral Specifications

BDD is optional.

Use behavior-oriented specifications when describing externally observable behavior or system contracts.

For example:

```text
Given a valid model state
And a valid incoming observation
When the model performs an online update
Then the update becomes the new committed state.
```

BDD-style specifications are useful for framework-level behavior.

They are not necessary for every low-level numerical operation.

The practical hierarchy is:

```text
Unit tests
    ↓
component correctness

Property / invariant tests
    ↓
mathematical and state guarantees

Integration tests
    ↓
component interaction

Behavioral specifications
    ↓
observable framework behavior
```

---

## 11. DRY: Do Not Repeat Knowledge

Seqvex should follow the DRY principle:

> **Do not repeat knowledge that must remain consistent.**

However, DRY does **not** mean aggressively removing every repeated line of code.

There is an important distinction between:

```text
duplicated implementation
```

and:

```text
duplicated knowledge
```

Two pieces of code may look similar today without actually sharing the same abstraction.

Prefer:

```text
first implementation
    ↓
second implementation exposes real repetition
    ↓
confirm same responsibility / invariant
    ↓
extract shared abstraction
```

Avoid:

```text
similar-looking code
    ↓
immediate generic abstraction
    ↓
premature trait / helper / framework
```

The project rule remains:

> **Do not abstract similarity until the underlying reason for the similarity is understood.**

This is particularly important for traits, generic numerical abstractions, device interfaces, and runtime APIs.

---

## 12. When to Create a Trait

Create a trait when there is a real shared contract.

Good reasons include:

- multiple implementations must satisfy the same behavior;
- a backend boundary requires a stable interface;
- generic algorithms genuinely operate over different implementations;
- a capability must be substituted or tested independently.

Do not create a trait merely because two structs currently contain similar methods.

Ask:

```text
Do these implementations share a stable responsibility?
        │
   ┌────┴────┐
   │         │
  yes        no
   │         │
   ▼         ▼
consider   keep local
trait      implementation
```

---

## 13. When to Create a Crate

Seqvex uses a Cargo workspace, but exact crate boundaries are deliberately not frozen.

Create a crate when a meaningful architectural boundary has emerged.

Useful reasons include:

- dependency isolation;
- independent testing;
- backend separation;
- API clarity;
- reuse;
- reduced coupling;
- optional functionality;
- compilation behavior.

Do not create a new crate simply because a folder has become large.

The question is:

> **Does this represent a real dependency or responsibility boundary?**

---

## 14. When to Create an Abstraction

Before creating a permanent abstraction:

```text
1. Identify the concrete problem.
2. Implement enough real behavior to experience it.
3. Observe what is actually common.
4. Measure important consequences.
5. Determine whether the problem will recur.
6. Consider the reversal cost.
7. Create the smallest abstraction that solves the demonstrated problem.
```

High-reversal-cost decisions require stronger evidence.

Examples include:

- storage/ownership model;
- device abstraction;
- synchronization model;
- allocation model;
- serialization/ABI;
- foundational execution semantics.

Prefer a temporary local solution when the long-term shape is not yet known.

---

## 15. Choosing the Development Tool

Use the simplest tool that answers the current question.

### Rust compiler / Cargo

Use Cargo and the compiler for:

- compilation;
- dependency resolution;
- unit/integration tests;
- formatting;
- package validation;
- workspace management.

Typical commands:

```bash
cargo check
cargo test
cargo fmt
cargo clippy
cargo build
```

Use `cargo check` frequently during implementation because it provides fast feedback without requiring a full executable build.

Use `cargo test` whenever behavior has changed.

Use `cargo fmt` to keep formatting consistent.

Use `cargo clippy` when the code is sufficiently developed for lint feedback to be useful.

### Benchmarking

Use benchmarks when there is a concrete performance question.

Examples:

```text
Is this update faster?

Did an allocation disappear?

What is per-observation latency?

Does micro-batching improve throughput?

Does a data-layout change improve cache behavior?
```

Do not optimize based only on intuition.

### Profiling

Use profiling after a measurable performance problem exists.

The sequence should be:

```text
observe slow behavior
       ↓
benchmark
       ↓
profile
       ↓
identify bottleneck
       ↓
change implementation
       ↓
benchmark again
```

Do not introduce SIMD, custom allocation, GPU execution, complex memory layouts, or other low-level optimization merely because they might be faster.

---

## 16. Numerical and Algorithmic References

Reference implementations and established libraries may be used to check correctness.

Use them to answer questions such as:

```text
Is the numerical result correct?

Does the algorithm converge as expected?

Does Seqvex produce equivalent predictions?

What performance baseline should we compare against?
```

A reference implementation is evidence for behavior, not automatically an architectural template.

Seqvex should remain responsible for understanding the implementation rather than blindly reproducing another framework.

---

## 17. AI and Development Assistance

AI tools may assist development, but the developer should understand the code being incorporated.

AI can be used for:

- explaining Rust concepts;
- explaining algorithms;
- generating small examples;
- reviewing code;
- identifying edge cases;
- suggesting tests;
- researching implementation alternatives;
- reviewing architecture;
- debugging a specific problem.

Avoid using AI to generate large framework sections that the developer cannot personally explain and maintain.

Preferred pattern:

```text
developer identifies problem
        ↓
AI assists with reasoning / research
        ↓
developer implements
        ↓
tests verify behavior
        ↓
AI reviews if useful
        ↓
developer understands and accepts result
```

The objective is not maximum generated code.

The objective is a framework whose implementation is understood by its developers.

---

## 18. Git and Integration

Changes should remain small enough to understand and review.

A useful sequence is:

```text
small change
   ↓
check
   ↓
test
   ↓
review diff
   ↓
commit
   ↓
integrate
```

Avoid combining unrelated architectural changes into one change where possible.

When an architectural decision is still uncertain, prefer an experiment or isolated implementation over prematurely committing to a framework-wide abstraction.

---

## 19. GitHub Actions and CI

CI should automate tests that have already become meaningful locally.

The initial CI pipeline should remain simple:

```text
Pull Request / Push
        │
        ▼
 GitHub Actions
        │
   ┌────┼────┐
   ▼    ▼    ▼
 check test fmt
   │    │    │
   └────┼────┘
        ▼
     clippy
        │
        ▼
   CI result
```

The exact workflow can evolve as the project grows.

Eventually CI may include:

- compilation checks;
- unit tests;
- integration tests;
- formatting;
- Clippy;
- documentation checks;
- benchmarks or performance regression checks where justified;
- multiple supported Rust/toolchain/platform configurations.

The important principle is:

> **CI should automate established project requirements, not become a source of arbitrary process overhead.**

---

## 20. From Test Case to CI

The intended progression is:

```text
Developer defines behavior
        ↓
Developer writes test case
        ↓
Developer implements code
        ↓
Test passes locally
        ↓
Component becomes integrated
        ↓
Integration tests are added
        ↓
Tests become stable
        ↓
GitHub Actions runs them automatically
```

This allows the same test suite to serve as:

```text
learning tool
      +
development specification
      +
regression protection
      +
CI verification
```

---

## 21. Definition of Done

A small Seqvex capability should generally move through:

```text
Problem understood
      ↓
Expected behavior defined
      ↓
Implementation written
      ↓
Tests pass
      ↓
Failure / edge cases considered
      ↓
Integrated where appropriate
      ↓
Benchmark if performance matters
      ↓
Documented
      ↓
CI coverage added when stable
```

Not every experiment needs to reach the final stage.

Exploratory work may remain experimental until enough evidence exists to justify integration.

---

## 22. Development Principles

The following principles guide Seqvex development:

> **Understand before abstracting.**

> **Write code that you can personally explain and maintain.**

> **Use tests to define and protect behavior.**

> **Use TDD where it helps; do not apply it mechanically.**

> **Use DRY to remove duplicated knowledge, not merely duplicated syntax.**

> **Do not create traits, crates, or framework-wide abstractions without a demonstrated reason.**

> **Benchmark before optimizing.**

> **Profile before making performance assumptions.**

> **Keep experiments small and reversible where architecture is uncertain.**

> **Automate established requirements through CI.**

> **Let observed constraints, rather than imagined requirements, drive permanent architecture.**
