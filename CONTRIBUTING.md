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
- `area::execution` — `src/execution`
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
- **Sequential ML Slice — Recurrent Neural Network**
- **Sequential ML Slice — Classic ML**
- **Sequential ML Slice — Online ML**

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

## Issue Architecture and Scope

Create an Issue when work represents a distinct engineering objective that
can be described with its own scope and acceptance criteria.

A parent Issue should normally represent a meaningful feature, function,
defect, or engineering objective. When that objective contains distinct
categories of work with their own scope or evidence, use GitHub sub-issues to
organize those workstreams.

```text
                    PARENT ISSUE
              Feature / Function / Objective
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   Implementation      Audit        Benchmark
     sub-issue       sub-issue       sub-issue
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
              Documentation / Integration
                   when justified
```

The categories are workstreams, not individual actions.

Rerunning a test remains under the same Test/Audit sub-issue. Rerunning a
benchmark remains under the same Benchmark sub-issue. Do not create a new
Issue or sub-issue merely because the same category of work must be rerun,
adjusted, or repeated to satisfy the parent objective.

Use a new sub-issue when the work becomes a materially different category or
an independently meaningful objective.

### Audit and Benchmark Separation

Correctness/audit work and performance/benchmark work are separate categories
even when they concern the same implementation.

```text
Parent Feature
├── Implementation
├── Correctness / Audit
├── Benchmark / Performance
└── Documentation / Integration, when justified
```

An audit establishes whether behavior and invariants are correct. A benchmark
establishes measured performance/resource behavior. Neither substitutes for the
other.

### Issue References and Development Traceability

Every meaningful development action must reference the Issue that owns the
work.

Use the GitHub Issue number as the authoritative reference. Do not create a
second internal numbering system for comments, test runs, benchmark runs, or
development updates.

When a sub-issue exists, reference the sub-issue number for work performed
against that category. Reference the parent Issue when discussing the overall
feature or objective.

### Parent, Related, and Dependency Relationships

These relationships must not be conflated.

**Parent / sub-issue** — the child is a category required to complete the
parent objective.

**Dependency** — one Issue or sub-issue must be completed before another can
proceed, even if it is not conceptually a child.

```text
#28 Shared Benchmark Harness
          │
          └──── dependency ────► #33 GRU Benchmark
```

A shared infrastructure Issue may therefore be a dependency of multiple
feature or benchmark Issues without becoming a sub-issue of each one.

**Related Issue** — two Issues concern the same component or evidence area, but
neither is a parent/child relationship nor a prerequisite.

### Issue Scope and Reuse

Do not create a new Issue merely because an existing Issue requires an
implementation modification, additional test, benchmark adjustment, rerun,
audit repeat, or other work necessary to satisfy the same objective.

Same objective and same work category should remain under the existing
Issue/sub-issue.

Do not create a new sub-issue merely because:

- a test must be rerun;
- a benchmark must be rerun;
- a benchmark configuration changes within the same objective;
- an audit must repeat after a correction;
- additional evidence is required for the same acceptance criteria;
- implementation needs another iteration within the approved scope.

Create a new Issue/sub-issue when work introduces a materially different
objective, category, acceptance criteria, API contract, architectural change,
or independently meaningful optimization.

If a closed Issue's original objective remains incomplete, reopen the existing
Issue rather than creating a duplicate.

If a completed Issue is followed by a genuinely new regression or distinct
failure, create a new Bug Issue and reference the related Issue.

### Optimization and Modification Rule

An optimization is not automatically a new Issue. Determine whether it is
optimizing the same identified bottleneck or engineering objective.

A modification remains part of an existing Issue when it is needed to satisfy
that Issue's existing objective and acceptance criteria. A modification that
introduces a materially different API contract, architectural change, or
independently meaningful objective should be tracked separately.

### Seqvex Issue Decision Tree

```text
                  Does an owning Issue already exist?
                              │
                         ┌────┴────┐
                         │         │
                        Yes        No
                         │         │
                         ▼         ▼
                  Same objective?  Is this a
                         │          meaningful
                    ┌────┴────┐     objective?
                   Yes        No        │
                    │          │       Yes
                    ▼          ▼        ▼
              Same Issue /   New      CREATE
              Sub-Issue      Issue
                    │
                    ▼
              Is this a new
              work category?
                    │
               ┌────┴────┐
              No         Yes
               │           │
               ▼           ▼
           Continue     Create a
           existing     sub-issue
           category
                    │
                    ▼
             Is the work closed?
                    │
               ┌────┴────┐
              Yes        No
               │          │
               ▼          ▼
            REOPEN      CONTINUE
```

The key rule is:

> **Same objective + same category → same Issue/sub-issue.**  
> **Same objective + new meaningful category → new sub-issue.**  
> **New objective → new Issue.**  
> **Closed but same objective remains incomplete → reopen.**

### Parent Feature Branch and Sub-Issue Commit Traceability

When a parent Issue contains multiple native sub-issues, the parent objective should normally be implemented through **one feature branch for the parent objective**.

```text
Parent Issue
    │
    ├── Native Sub-Issue A
    ├── Native Sub-Issue B
    └── Native Sub-Issue C
             │
             ▼
      Parent feature branch
             │
             ├── commit → #A
             ├── commit → #A
             ├── commit → #B
             └── commit → #C
```

The branch represents the overall parent objective. The commit reference identifies the specific sub-issue whose work the commit primarily addresses.

For example:

```bash
git commit -m "perf: implement reusable GRU workspace — #33"
git commit -m "test: validate workspace allocation behavior — #33"
git commit -m "feat: establish bounded GRU micro-batch execution — #34"
```

A commit may reference multiple Issues/sub-issues when the change genuinely spans them. Do not split a logically single change merely to manufacture separate references.

The branch name should normally identify the parent objective:

```text
issue-17-gru-production-inference
```

This differs from the commit's Issue reference. The branch provides parent-level implementation context; the commit provides workstream-level traceability.

### Native Sub-Issue Rule

A sub-issue must be a genuine child/workstream of its parent, not merely an Issue that mentions the parent in its description.

Before converting or attaching an existing Issue as a sub-issue:

1. inspect the proposed parent;
2. verify that the candidate represents a meaningful category/workstream required by the parent objective;
3. confirm that it is not an independent Seqvex capability or objective;
4. only then establish the native GitHub parent/sub-issue relationship.

An existing standalone Issue may be attached to the parent through GitHub's native **Sub-issues → Add existing issue** mechanism. A textual `Parent #XX` reference in the Issue body is not itself a native parent/sub-issue relationship.

### Duplicate Replacement Rule

When an existing standalone Issue is determined to be a child by nature:

1. verify the supposed parent Issue first;
2. establish the replacement as a **native sub-issue** under that parent;
3. transfer the necessary scope, acceptance criteria, dependencies, evidence, and relevant history;
4. explicitly state in the replacement sub-issue that it replaces the original standalone Issue;
5. return to the original Issue;
6. add the `duplicate` label;
7. add a closing comment identifying the replacement;
8. close the original Issue as a duplicate.

Do not close the original duplicate before the replacement native sub-issue exists.

If an existing Issue was already created as an ordinary Issue and can be attached to the parent through **Add existing issue**, use that existing Issue rather than creating another replacement Issue. This preserves its history and avoids unnecessary duplicates.

### Parent Completion Rule

Closing all sub-issues does not by itself authorize closing the parent Issue.

After the sub-issues are complete, verify the parent's own objective and acceptance criteria. Close the parent only when the parent objective is actually complete and the applicable integration/validation gates have passed.

### Issue-Specific Git Workflow

Once an Issue or sub-issue is approved for implementation, use:

```text
GitHub Issue / Sub-Issue
  ↓
Maintainer confirms scope
  ↓
Dedicated issue branch
  ↓
Design / Planning review as required
  ↓
Implementation
  ↓
Validation / Audit / Benchmark as applicable
  ↓
One logical issue update
  ↓
Commit with issue reference (#XX)
  ↓
Push issue branch
  ↓
Pull Request
  ↓
Review + validation
  ↓
Merge into main
  ↓
Issue / sub-issue closes when appropriate
  ↓
User / maintainer deletes feature branch
```

Issue-specific development must not be performed directly on `main`.

Preferred branch naming is `issue-XX-short-description`. The branch should
correspond to the Issue or sub-issue whose work is being implemented.

The user/maintainer deletes the feature branch after merge; development
agents must not do so without explicit authorization.

### One Issue Update = One Logical Commit = One Push

For Seqvex issue work, treat each authorized issue update as one coherent
delivery unit:

```text
Issue / Sub-Issue update
        ↓
one logical commit
        ↓
one push
```

Use:

```bash
git commit -m "<type>: <concise description> — #XX"
git push origin issue-XX-short-description
```

The commit must contain only work belonging to that Issue/Sub-Issue update.
If a proposed update spans multiple independent Issues or categories, stop and
resolve the scope before committing.

### Separate Authorization Gates

The following permissions are independent:

- issue/sub-issue creation/reopening/update;
- branch creation;
- implementation;
- commit;
- push;
- Pull Request creation;
- merge;
- issue/sub-issue closure or other issue-state mutation;
- branch deletion.

Implementation authorization does not automatically authorize commit, push, PR
creation, merge, issue mutation, or branch deletion.

### Working-Tree and Commit Discipline

Before issue-specific work and before committing:

```bash
git status
git diff
```

Preserve unrelated pre-existing changes. Do not reset, stash, clean, overwrite, or discard them merely to make issue work easier.

Before committing, inspect the working tree and diff, stage only files belonging to the authorized Issue/Sub-Issue, and inspect the staged diff. Avoid `git add .` when unrelated changes may be present.

Do not force-push or rewrite published history without explicit authorization.

### Commit Convention

Use:

```text
<type>: <concise description> — #XX
```

Recommended types include `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, and `chore`.

Use `perf` only when supported by measurement. Use `Closes #XX`, `Fixes #XX`, or `Resolves #XX` only when the Issue/Sub-Issue objective is genuinely complete.

A Pull Request should state the Issue/Sub-Issue, parent Issue when applicable, scope, implementation, validation, architectural implications, and known limitations. Merge should occur only after the relevant review, validation, architecture, and authorization gates have passed.

## A Simple Rule

When in doubt:

> Understand the problem → build the smallest experiment → measure → then generalize.

Seqvex should grow from demonstrated requirements rather than from speculative complexity.

## Algorithm Development and Audit Lifecycle

For substantive ML/RL algorithms, Seqvex uses:

```text
Research / mathematical specification
  ↓
Reference implementation
  ↓
Independent correctness audit
  ↓
Sequential / non-IID audit
  ↓
Numerical / statistical audit
  ↓
Performance measurement
  ↓
Local optimization when justified
  ↓
Documentation / integration
```

After the fifth algorithm completes its lifecycle, conduct a **post-five Pre-Hardware Optimization Audit** across the algorithms and shared execution infrastructure. This is distinct from any earlier or partial audit.

The audit should consider correctness/reference behavior, validation, state/transition behavior, failure atomicity, reset/isolation, ordering/causality where applicable, numerical stability, statistical assumptions, long-horizon behavior, performance baselines, allocation/resource behavior, repeated implementation patterns, evidence for shared abstractions, and hardware-aware execution implications.

Seqvex may eventually formalize this as a **Common Audit Core + Algorithm-Specific Audit** model. A common category does not mean identical tests across algorithms, and a generic audit framework should not be implemented until repeated evidence justifies it.

### Optimization Gate

Optimization requires correctness evidence, a measured bottleneck, a local understandable change, preserved mathematical/sequential semantics, no silent settlement of a deferred architecture decision, and a follow-up benchmark or measurement.

Allocation reduction alone does not establish lower latency or higher throughput.

### Evidence Language

Distinguish:

- **FACT**
- **MEASURED EVIDENCE**
- **INFERENCE**
- **ASSUMPTION**
- **PROVISIONAL DECISION**
- **ARCHITECTURAL DECISION**
- **OPEN QUESTION**

Do not present an inference as measured evidence or a provisional decision as a settled architecture.
