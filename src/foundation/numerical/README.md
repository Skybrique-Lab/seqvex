# Numerical

## Purpose

Owns the minimal numerical computation required by Seqvex's own ML/RL methods,
not a general-purpose numerical or scientific-computing ecosystem.

## Responsibility

- Specify the contract for the smallest numerical primitives the foundation
  needs: `sum`, `mean`, `variance`, `dot`, `norm`.
- Specify the contract for online statistics: `online_mean`, `online_variance`.

## Relationship to Seqvex

Seqvex requires numerical computation for ML/RL but is not NumPy, SciPy,
Pandas, Polars, or a dataframe engine. Numerical capability is introduced only
when an actual ML/RL requirement justifies it. See `docs/DEVELOPMENT.md` §9.

## Inside

- `statistics` — contract signatures for the primitives above.

## Outside

- General linear algebra, matrix decompositions, dataframes, ETL, and EDA.
- Optimized, vectorized, or hardware-accelerated implementations.

## Current tests / specification

`tests/numerical.rs` — every test is `#[ignore = "awaiting implementation"]`.
The scaffold specifies behavior; it does not implement the algorithms.

## Major deferred decisions

- Rolling-window semantics (window boundary, update order) are not yet clear
  enough to fix a contract.
- Whether the primitives remain free functions, gain a trait, or delegate to a
  mature numerical crate.
- First-order numerical accuracy and stability requirements.
