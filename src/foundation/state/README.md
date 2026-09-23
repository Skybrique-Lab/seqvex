# State

## Purpose

Defines the smallest explicit semantics for persistent state, state
transitions, and failure atomicity.

## Responsibility

- `StateModel` — the transition contract: current valid state + observation →
  candidate next state, or a classified failure.
- Single-observation execution (`process_one`).
- Reference ordered-fold execution (`process_batch`): unbounded and
  stop-on-first-failure. This is **not** a bounded micro-batch strategy; a
  bounded micro-batch is introduced per algorithm (`docs/ML_VERTICAL_SLICES.md`).
- Streaming execution with failure preservation (`process_stream`).

## Relationship to Seqvex

State is first-class. A failed update must never silently commit invalid or
partial state, and the last committed valid state must remain usable for
subsequent observations. See `docs/DEVELOPMENT.md` §3.3/§10 and
`docs/FAILURE_AND_RECOVERY.md`.

## Inside

- `transition` — the `StateModel` contract, single-observation execution, and
  the reference ordered fold.
- `atomicity` — streaming replay that preserves the last valid state and
  reports failures.

## Outside

- Recovery mechanisms (rollback, checkpointing, retry policy): deferred.
- Scheduling, parallelism, and device placement.
- Any concrete model, optimizer, or ML algorithm.

## Current tests / specification

- `tests/streaming.rs`
- `tests/state.rs`
- `tests/failure.rs`
- `tests/determinism.rs`

## Provisional mechanism (explicitly not frozen)

The scaffold represents a candidate next state as a returned value and does not
mutate the current state. This makes failure atomicity structural, but it is a
reference mechanism only. `docs/FAILURE_AND_RECOVERY.md` §16 keeps
"transactional versus in-place updates" deferred; a future mechanism
(copy-on-write, double buffering, rollback information, or in-place mutation
with rollback) must satisfy the same tests.

## Major deferred decisions

- Exact state-update mechanism, ownership, and storage.
- Retry/recovery policy and failure-policy API.
- Determinism guarantees across platforms.
