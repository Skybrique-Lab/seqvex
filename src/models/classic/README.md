# Classic ML

## Purpose

Holds the classic (non-recurrent) ML vertical slices. The first is linear
regression prediction; the second is decision tree prediction.

## Responsibility

- `linear_regression` — the immutable model `ŷ = w · x + b`, its reference
  prediction, and a bounded micro-batch prediction over independent
  observations.
- `decision_tree` — read-only traversal of a caller-supplied trained regression
  tree, its reference prediction, and a bounded micro-batch prediction over
  independent observations.

## Relationship to Seqvex

This module is the architectural **control case** for the current ownership
review. Unlike the GRU slice, linear-regression prediction:

- needs no mutable model state;
- needs no reusable scratch/workspace;
- is therefore shareable behind `&self`, with no `&mut M` requirement.

Its characteristics are evidence for the hypothesis that immutable, shareable
model data and per-stream state/scratch should be separate concerns. It must not
be made to conform to the GRU's model-owned-workspace topology.

One observable consequence: the current generic `StreamingExecutor` demands
`&mut M`, yet linear regression needs no mutable model state at all. That is
direct evidence that the `&mut M` requirement is broader than any individual
model's needs and belongs to the executor/workspace review, not to this slice.
This slice does not change that signature.

## Inside

- `LinearRegression` and `RegressionError`.
- Reference prediction (`predict`).
- Streaming prediction via the foundation `StateModel` contract (the carried
  state is the most recent prediction).
- Bounded micro-batch prediction (`predict_batch`), valid because prediction is
  order-independent across observations.
- `DecisionTree`, `DecisionTreeNode`, and `DecisionTreeError` — read-only
  traversal of a model-local `Vec` of nodes (provisional layout), immutable
  `&self`, no scratch, with the same `State = f32` "most recent prediction"
  convention and model-local micro-batch.

## Decision Tree (read-only inference)

Inference traverses a trained tree supplied by the caller; training/fitting is
deliberately out of scope. The first task variant is **regression** (finite `f32`
leaves); classification is deferred. A split routes **left** when
`feature <= threshold` and right otherwise. Construction validates the graph
(empty/zero-dimension, feature index, child range, finite threshold and leaf,
cycle, shared child, unreachable node), and traversal is iterative with a step
bound. Because a prediction returns a validated leaf value with no arithmetic,
its output is always finite. `predict_batch` preserves input order; the first
element failure makes the entire call return `Err` and no partial output vector
is returned.

## Outside

- Online parameter updating, training, optimizers, and autodiff. Prediction is
  deliberately separate from learning; the online update is a later, ordered
  concern (RLS, Issue #26).
- Decision Tree training/fitting and classification; the slice is read-only
  regression traversal until those are explicitly approved.
- A generic linear-model trait, a generalized `predict` trait, a tree/arena/
  storage abstraction, or a micro-batch executor.
- Storage/tensor abstractions.

## Current tests / specification

- `tests/linear_regression.rs`
- `tests/decision_tree.rs`

## Micro-batch policy

Prediction may be micro-batched because each output depends only on its own
observation and the immutable weights. The batch must preserve input order and
must produce bitwise-identical results to repeated single-observation
prediction. No micro-batch path may silently change the reduction order of the
per-observation dot product.

## Numerical envelope

Prediction runs on the framework's `f32` substrate. Construction rejects
non-finite weights and bias (`NonFiniteParameter`), and prediction rejects a
dimension mismatch and non-finite features (`NonFiniteInput`). Prediction does
**not** validate its output.

Because the operands are `f32`, finite operands can still overflow during
multiplication or summation and produce `±inf`; a subsequent `inf + (-inf)` can
produce `NaN`. Finite parameters and finite inputs therefore do **not** guarantee
a finite prediction. Callers are responsible for keeping the products and sums
representable, for example by scaling features and coefficients. No numeric
threshold is imposed or implied: this is expected IEEE-754 `f32` behaviour, not a
defect, and Seqvex does not invent a magnitude cutoff.

The analogous pure-prediction path `Rls::predict` computes `wᵀx` with the same
no-output-validation behaviour; each model documents its own envelope rather than
sharing a rule or abstraction.

## Measured evidence (#23)

Release-mode benchmark (`cargo bench --bench linear_regression`), median of 20
runs, ns per observation:

| features | micro-batch | reference | streaming | micro-batch |
|---|---:|---:|---:|---:|
| 8 | 32 | 7.6 | 6.6 | 6.9 |
| 32 | 32 | 20.9 | 15.0 | 18.5 |
| 128 | 16 | 87.1 | 88.0 | 92.8 |

Reference and streaming allocate nothing; micro-batch allocates one output
`Vec` per batch (amortized `1/batch_size` per observation). Micro-batch is not
faster than single-observation prediction and is materially slower at
`features = 128` (effect ≈ 16× the single-observation IQR). No optimization was
implemented, per the slice sequence (no profile-identified bottleneck beyond the
inherent output allocation).

## Measured evidence (#24)

Release mode (`cargo bench --bench decision_tree`), median of 20 runs. Balanced
trees: depth 6 (127 nodes) at 8 features, depth 8 (511 nodes) at 32 features,
depth 10 (2047 nodes) at 128 and 256 features.

| features | depth | nodes | reference | streaming | micro-batch |
|---:|---:|---:|---:|---:|---:|
| 8 | 6 | 127 | 15.3 | 15.9 | 20.5 |
| 32 | 8 | 511 | 31.9 | 32.3 | 35.7 |
| 128 | 10 | 2047 | 79.5 | 78.1 | 83.7 |
| 256 | 10 | 2047 | 108.4 | 106.7 | 114.5 |

ns per observation. Reference and streaming allocate `0.000` allocations and `0`
bytes per observation; the bounded micro-batch allocates exactly one output
`Vec` per batch (amortized `1/batch_size` per observation). Streaming's measured
median is within 1.7 ns/obs of the direct reference (see table); the benchmark
applies no materiality test, so this is a reported measured difference, not a
statistical-equivalence claim. Micro-batch is not faster than single-observation
prediction. Construction builds the growing node `Vec` (11–16 heap allocations at
these tree sizes), separate from the allocation-free per-observation inference
and the per-batch output allocation. No optimization was implemented: there is no
profile-identified bottleneck.

## Major deferred decisions

- The numerical policy for classic ML (`f32` substrate vs the `f64` statistics
  layer). This slice uses the existing `f32` substrate because observations are
  already `Observation<Vector>`; a switch needs a measured requirement.
- Whether online/adaptive parameter updates belong in this module or the online
  ML namespace.
