# Online ML

## Purpose

Holds the online (ordered, supervised) ML vertical slices. The first is
recursive least squares (RLS) with a forgetting factor.

## Responsibility

- `rls` — the immutable configuration `Rls`, its adaptive state `RlsState`, the
  supervised observation payload `RlsSample`, the reference prediction and
  update, and its explicit failure classification.

## Relationship to Seqvex

RLS is the first slice whose model **state changes with every observation**, so
it is the strongest evidence so far about where adaptive state belongs:

- `Rls` holds only immutable configuration (`D`, `λ`, `δ`) and implements
  `StateModel` with a `&self` contract. It is shareable.
- The adaptive `(w, P)` lives entirely in `StateModel::State`, owned by the
  execution session (`StreamingExecutor`), not by the model.
- No model-owned workspace, cache, or scratch is used. The value-returning
  contract computes a candidate `(w', P')` and returns it.

This confirms, on a stateful model, the same finding linear regression made on a
stateless one: streaming does not require `&mut M`, and adaptive state is a
per-stream property.

One consequence is recorded rather than solved: the generic
`StreamingExecutor::new(&mut M)` cannot share one immutable `Rls` across two
executors, even though RLS needs only `&self`. That `&mut M` limitation is a
pre-existing CRITICAL ARCHITECTURE REVIEW item; it is not changed here. One
shared `&Rls` with independent states is expressible today through the
foundation `process_one`/`process_stream` functions, which take `&M`, and the
tests exercise exactly that.

## Equations

For feature `x_t`, target `y_t`, parameters `w_t`, covariance `P_t`, and
forgetting factor `λ`:

```text
ŷ_t = w_{t-1}ᵀ x_t
e_t = y_t − ŷ_t
v_t = P_{t-1} x_t
d_t = λ + x_tᵀ v_t
k_t = v_t / d_t
w_t = w_{t-1} + k_t e_t
P_t = (P_{t-1} − v_t v_tᵀ / d_t) / λ
```

The covariance step uses the transparent rank-one form: since `P` is symmetric,
`k_t x_tᵀ P_{t-1} = v_t v_tᵀ / d_t`. Initialization is `w_0 = 0`,
`P_0 = δI` (symmetric positive-definite by construction).

## Failure and atomicity

Validation order is deterministic: dimensions (`x`, then `w`, then `P`),
observation finiteness, the denominator `d` (finite and `> 0`), and finally
candidate finiteness. Failures are reported as explicit `RlsError` variants; the
reference never panics on trust-boundary input.

Atomicity is structural. `w'` and `P'` are computed as locals and returned as
one `RlsState` only after both validate, so a failed update leaves the committed
`(w, P)` bitwise unchanged and there is no path that commits one component
without the other. The tests assert bitwise state preservation after every
failure class, directly and through the executor.

The incoming `P` is expected symmetric PSD; the reference does not run a general
positive-definiteness check (it would add a primitive and a hidden `O(D³)` cost).
The denominator guard catches corrupted/indefinite `P` and overflow.

## Micro-batch

RLS updates are ordered and state-dependent, so a bounded micro-batch must be an
ordered sequential fold — independent or parallel updates are invalid. The
foundation `process_batch` already expresses exactly this (ordered,
stop-on-first-failure, returns the last committed valid state). No generic
micro-batch API or executor is added for this slice.

## Measured evidence (#26)

`cargo bench --bench rls`, release, median of 20 runs. `λ = 1` is used for the
benchmark because it replays one fixed observation: a fixed input with `λ < 1`
is not persistently exciting, so `P` inflates in unexplored directions and
eventually trips the denominator guard. The update path and its `O(D²)` work are
unchanged; the correctness tests cover `λ < 1` over varying inputs.

| features | path | ns/obs | allocs/obs | bytes/obs |
|---:|---|---:|---:|---:|
| 8 | reference | 231.9 | 5.000 | 384 |
| 8 | streaming | 227.4 | 5.000 | 384 |
| 8 | bounded-fold | 240.3 | 6.000 | 416 |
| 32 | reference | 2255.8 | 5.000 | 4608 |
| 32 | streaming | 2268.4 | 5.000 | 4608 |
| 32 | bounded-fold | 2269.1 | 6.000 | 4736 |
| 128 | reference | 34952.7 | 5.000 | 67584 |
| 128 | streaming | 34622.8 | 5.000 | 67584 |
| 128 | bounded-fold | 34836.9 | 6.000 | 68096 |
| 256 | reference | 139636.4 | 5.000 | 266240 |
| 256 | streaming | 139734.8 | 5.000 | 266240 |
| 256 | bounded-fold | 141066.1 | 6.000 | 267264 |

Findings, stated without overclaiming:

- The reference is **not** allocation-free, unlike linear regression. Every
  observation returns a new `D×D` candidate `P'`, so the value-returning
  `StateModel` contract costs a steady **5 allocations/observation** dominated by
  `P'`; bytes/observation scale with `D²` (`384` at `D = 8`, `266240` at
  `D = 256`). Committed state is `D + D²` scalars (`288` bytes at `D = 8`,
  `263168` bytes at `D = 256`).
- Streaming is **not faster** than the direct reference (differences are within
  run noise); it is the same generic `update` path driven through the executor.
  Lower latency was never claimed for the streaming wrapper.
- The bounded foundation fold is **not faster** either, and costs one extra
  allocation/observation because it consumes owned observations (the batch is
  cloned to feed it). It is retained as the semantically-correct ordered
  micro-batch, not as a performance claim.
- Latency scales roughly with `D²` (`232 ns` at `D = 8` to `140 µs` at
  `D = 256`), consistent with the covariance update dominating.

The measured `O(D²)` candidate allocation is the RLS-specific evidence. A future
double-buffered `P` could remove it, but **where the second buffer lives**
(model vs stream vs external) is the unresolved ownership question; a workspace
is deliberately not introduced here. Any such optimization requires CRITICAL
ARCHITECTURE REVIEW.

The long-run test (`tests/rls.rs`) runs 10,000 deterministic observations at
`D = 4`, `λ = 0.999`, and asserts every state stays finite, matches the
independent scalar reference bitwise, and remains exactly symmetric. Symmetry is
preserved because `v_i v_j == v_j v_i` exactly in IEEE `f32`.

## Major deferred decisions

- Whether the generic executor should allow shared-model / multi-stream
  execution (the `&mut M` question) — escalated, not solved here.
- Where an optimized `P` double-buffer would live, if optimization is ever
  justified — requires a workspace decision.
- Whether `f32` suffices for RLS at larger `D` and long runs, or whether a
  numerical-representation decision is required. Numerical stability under weak
  excitation (variable/adaptive forgetting, ridge, `f64` covariance, UD or
  square-root forms) is deferred; the reference uses the existing `f32`
  substrate because observations are already `Observation<Vector>`.

## Current tests / specification

- `tests/rls.rs`
