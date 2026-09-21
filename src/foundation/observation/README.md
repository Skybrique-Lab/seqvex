# Observation

## Purpose

Represents a single unit of input to a stateful Seqvex computation.

## Responsibility

- Carry the input value required by a computation.
- Optionally carry sequence context. An observation does not require a
  timestamp, index, or any sequence metadata.
- Keep input representation distinct from persistent model state.

## Relationship to Seqvex

Observation is the smallest semantic unit of the streaming-first execution
model: a stream is an ordered sequence of observations, and single-observation
execution is the smallest execution unit. See `ARCHITECTURE.md`.

## Inside

- `Observation<T>` — an input value with optional sequence context.
- `SequenceNumber` — an explicit position within an ordered sequence.

## Outside

- Timestamps and wall-clock semantics.
- Storage or collection of observations. Streams are supplied as ordered
  iterators; their container is not part of this module.
- Any tensor, batch, or hardware representation.

## Current tests / specification

- `tests/observation.rs`
- `tests/ordering.rs`

## Major deferred decisions

- Whether observations need richer sequence metadata (timestamps, gaps,
  out-of-order handling).
- The relationship between observation representation and the deferred
  storage/tensor design.
