# Classic ML

## Purpose

Holds the classic (non-recurrent) ML vertical slices. The first is linear
regression prediction.

## Responsibility

- `linear_regression` — the immutable model `ŷ = w · x + b`, its reference
  prediction, and a bounded micro-batch prediction over independent
  observations.

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

## Outside

- Online parameter updating, training, optimizers, and autodiff. Prediction is
  deliberately separate from learning; the online update is a later, ordered
  concern (RLS, Issue #26).
- A generic linear-model trait, a generalized `predict` trait, or a micro-batch
  executor.
- Storage/tensor abstractions.

## Current tests / specification

- `tests/linear_regression.rs`

## Micro-batch policy

Prediction may be micro-batched because each output depends only on its own
observation and the immutable weights. The batch must preserve input order and
must produce bitwise-identical results to repeated single-observation
prediction. No micro-batch path may silently change the reduction order of the
per-observation dot product.

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

## Major deferred decisions

- The numerical policy for classic ML (`f32` substrate vs the `f64` statistics
  layer). This slice uses the existing `f32` substrate because observations are
  already `Observation<Vector>`; a switch needs a measured requirement.
- Whether online/adaptive parameter updates belong in this module or the online
  ML namespace.
