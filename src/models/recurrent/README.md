# Gated Recurrent Unit (GRU)

The Gated Recurrent Unit (GRU) is a recurrent neural network (RNN) architecture designed for processing **sequential and temporal data**.

Unlike a feed-forward model, a GRU maintains an internal hidden state that is carried from one observation to the next.

In Seqvex, the GRU is implemented as a **stateful, single-observation CPU model with both a readable reference path and a provisional allocation-free production path**.

The reference path establishes mathematical correctness and semantic behavior. The production path demonstrates measured optimization without replacing the reference.

---

# 1. What is a GRU?

Suppose we observe:

$$
x_1, x_2, x_3, \ldots, x_t
$$

where `x_t` is one observation.

A feed-forward model could process each observation independently:

$$
y_t=f(x_t)
$$

A GRU instead maintains hidden state:

$$
h_t
$$

and performs:

$$
(x_t,h_{t-1})\rightarrow h_t
$$

This makes the GRU naturally compatible with Seqvex's streaming execution model.

---

# 2. GRU Mathematical Formulation

The Seqvex implementation uses:

## Update Gate

$$
z_t =
\sigma
\left(
W_zx_t + U_zh_{t-1} + b_z
\right)
$$

## Reset Gate

$$
r_t =
\sigma
\left(
W_rx_t + U_rh_{t-1} + b_r
\right)
$$

## Candidate Hidden State

$$
\tilde{h}_t =
\tanh
\left(
W_hx_t
+
U_h(r_t \odot h_{t-1})
+
b_h
\right)
$$

## New Hidden State

$$
h_t =
(1-z_t)\odot h_{t-1}
+
z_t\odot\tilde{h}_t
$$

where:

- `σ` is sigmoid;
- `tanh` is hyperbolic tangent;
- `⊙` is element-wise multiplication;
- `x_t` is the current observation;
- `h_{t-1}` is the previous hidden state;
- `z_t` is the update gate;
- `r_t` is the reset gate;
- `\tilde{h}_t` is the candidate state;
- `h_t` is the newly committed hidden state.

---

# 3. Understanding the Gates

## 3.1 Update gate

$$
z_t =
\sigma
\left(
W_zx_t + U_zh_{t-1} + b_z
\right)
$$

The sigmoid produces values between 0 and 1.

If:

$$
z_t\approx0
$$

then:

$$
h_t\approx h_{t-1}
$$

If:

$$
z_t\approx1
$$

then:

$$
h_t\approx\tilde{h}_t
$$

The update gate therefore controls the degree of state replacement.

## 3.2 Reset gate

$$
r_t =
\sigma
\left(
W_rx_t + U_rh_{t-1} + b_r
\right)
$$

The reset gate controls how much previous-state information participates in the candidate:

$$
r_t\odot h_{t-1}
$$

If:

$$
r_t\approx0
$$

the candidate largely ignores the previous hidden state.

If:

$$
r_t\approx1
$$

the previous hidden state contributes strongly.

---

# 4. Candidate Hidden State

The candidate is:

$$
\tilde{h}_t =
\tanh
\left(
W_hx_t
+
U_h(r_t\odot h_{t-1})
+
b_h
\right)
$$

The candidate is a **proposed state**, not automatically the committed state.

The transition can therefore be understood as:

```text
previous valid state
        ↓
candidate computation
        ↓
candidate validation
        ↓
new committed state
```

This matches Seqvex's state-transition and failure-atomicity semantics.

---

# 5. Dimensions and Shapes

Let:

- `d_x` = input dimension;
- `d_h` = hidden-state dimension.

Then:

$$
x_t\in\mathbb{R}^{d_x}
$$

and:

$$
h_t\in\mathbb{R}^{d_h}
$$

The input-weight matrices are:

$$
W_z,W_r,W_h\in\mathbb{R}^{d_h\times d_x}
$$

The recurrent-weight matrices are:

$$
U_z,U_r,U_h\in\mathbb{R}^{d_h\times d_h}
$$

The biases are:

$$
b_z,b_r,b_h\in\mathbb{R}^{d_h}
$$

| Quantity | Shape | Meaning |
|---|---|---|
| `x_t` | `d_x` | Current observation |
| `h_{t-1}` | `d_h` | Previous hidden state |
| `W_z` | `d_h × d_x` | Update-gate input weights |
| `U_z` | `d_h × d_h` | Update-gate recurrent weights |
| `b_z` | `d_h` | Update-gate bias |
| `W_r` | `d_h × d_x` | Reset-gate input weights |
| `U_r` | `d_h × d_h` | Reset-gate recurrent weights |
| `b_r` | `d_h` | Reset-gate bias |
| `W_h` | `d_h × d_x` | Candidate input weights |
| `U_h` | `d_h × d_h` | Candidate recurrent weights |
| `b_h` | `d_h` | Candidate bias |
| `z_t` | `d_h` | Update gate |
| `r_t` | `d_h` | Reset gate |
| `\tilde{h}_t` | `d_h` | Candidate state |
| `h_t` | `d_h` | New hidden state |

---

# 6. Why the Matrix Dimensions Work

For example:

$$
W_zx_t
$$

has:

$$
W_z\in\mathbb{R}^{d_h\times d_x}
$$

and:

$$
x_t\in\mathbb{R}^{d_x}
$$

so:

$$
W_zx_t\in\mathbb{R}^{d_h}
$$

Likewise:

$$
U_zh_{t-1}\in\mathbb{R}^{d_h}
$$

Therefore the two terms and the bias can be added element-wise.

The same dimensional reasoning applies to the reset and candidate equations.

---

# 7. Numerical Example

For a small illustrative example, let:

```text
d_x = 2
d_h = 2
```

and suppose:

$$
x_t=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

and:

$$
h_{t-1}=
\begin{bmatrix}
0.2\\
-0.1
\end{bmatrix}
$$

The GRU first computes the update and reset gates, then the candidate state, and finally blends the candidate with the previous hidden state.

The important structural lesson is that the same observation can produce a different result depending on the incoming hidden state.

That is the core recurrent property:

$$
f(x_t,h_{t-1})\neq f(x_t,h'_{t-1})
$$

in general.

---

# 8. GRU as a Seqvex Stateful Model

A GRU naturally fits the Seqvex transition:

$$
(\text{current state},\text{observation})
\rightarrow
\text{candidate next state}
$$

For the GRU:

```text
hidden state
    +
observation
    ↓
GRU computation
    ↓
candidate hidden state
    ↓
validation
    ↓
commit
```

If the transition fails, the previously committed hidden state remains valid.

Reset establishes a new execution boundary.

---

# 9. Reference and Production Implementations

Seqvex intentionally distinguishes two implementation roles.

## 9.1 Reference path

The reference path prioritizes:

- mathematical transparency;
- readable correspondence with the GRU equations;
- deterministic/reproducible behavior where useful;
- inspectability;
- independent correctness validation.

The current reference route is based on the `StateModel` contract and readable GRU computation.

It is the **semantic oracle** for optimized implementations.

## 9.2 Production path

The production path prioritizes measured execution properties.

The current GRU production path uses a private reusable workspace:

```text
Gru
├── parameters
├── hidden state
└── private workspace
    ├── acc
    ├── gate
    └── scratch
```

The workspace contains three hidden-dimension `Vec<f32>` buffers allocated once and reused across steps.

The production path therefore avoids the 20 steady-state allocations observed in the reference path during the Stage 2 allocation audit.

The production path validates the candidate before committing it.

## 9.3 Current architectural status

The production workspace is **provisional / experimental**.

It is a local GRU optimization, not a decision that all Seqvex models should own their execution scratch.

In particular, the current `StreamingExecutor` mutable model borrow exists so the executor can reach this model-owned workspace.

That creates an architectural question:

```text
immutable model parameters
        +
per-stream mutable state
        +
per-stream reusable workspace
```

versus:

```text
model-owned mutable workspace
```

The correct long-term topology has not been settled.

This is therefore subject to **CRITICAL ARCHITECTURE REVIEW**.

Do not generalize the GRU workspace into a framework-wide allocator/workspace abstraction until real classical and online ML workloads demonstrate that the requirement recurs.

---

# 10. Production Execution Semantics

The production path must preserve:

- the exact GRU equations;
- state ordering;
- reset semantics;
- failure atomicity;
- reference-path observable results;
- deterministic behavior where the API promises it.

The optimized implementation is not permitted to redefine the mathematical model merely to improve allocation behavior.

The intended relationship is:

```text
reference implementation
        ↓
independent validation
        ↓
production optimization
        ↓
equivalence tests
        ↓
benchmark / profile
```

---

# 11. Initialization

`GruParameters::deterministic` provides deterministic parameters for examples, tests, and benchmarks.

It should not be interpreted as a statistically justified production initialization scheme.

Production training/initialization policies remain future work.

---

# 12. Current Implementation Scope

The current GRU provides:

- configurable input and hidden dimensions;
- parameter validation;
- deterministic parameter generation for testing/examples;
- single-observation stepping;
- hidden-state reset;
- state-transition errors;
- reference computation;
- provisional allocation-free production computation;
- streaming integration.

The current implementation does **not** provide:

- training;
- loss functions;
- optimizers;
- autodiff;
- GPU execution;
- explicit SIMD;
- packed/fused gate kernels;
- generalized neural-network abstractions;
- distributed training.

---

# 13. Correctness and Tests

The implementation should be validated against:

- an independent scalar reference;
- dimension mismatch cases;
- non-finite parameters;
- non-finite inputs;
- invalid candidate behavior;
- reset behavior;
- repeated state transitions;
- long sequential folds;
- reference/production equivalence;
- failure atomicity.

The production path is only useful if it remains semantically equivalent to the reference path.

---

# 14. Performance

The Stage 2 allocation investigation established that the reference path performs:

```text
20 allocations / step
```

for the measured GRU configurations.

The production workspace path reduces this to:

```text
0 steady-state allocations / step
```

and uses:

```text
3 × hidden-dimension f32 buffers
```

for reusable scratch.

Measured streaming benefit was strongest for the small GRU configuration and became small or indistinguishable from benchmark noise at larger configurations.

Therefore:

> **Allocation elimination is demonstrated; universal latency improvement is not.**

The production path should remain evidence-driven and local until broader workloads justify generalization.

---

# 15. Real-Life Interpretation

A GRU can represent evolving context in systems such as:

- sensor streams;
- industrial telemetry;
- financial time series;
- user interaction sequences;
- robotics;
- network events;
- sequential decision systems.

For example, in a sensor stream:

```text
temperature_t
pressure_t
vibration_t
      ↓
    GRU
      ↓
hidden state
      ↓
next observation
```

The hidden state carries information from earlier observations into the next computation.

That is precisely the type of sequential dependency Seqvex is designed to represent.

---

# 16. Learning Checklist

A reader should be able to explain:

1. Why a GRU needs hidden state.
2. What the update gate controls.
3. What the reset gate controls.
4. Why the candidate is not automatically the final state.
5. The dimensions of every GRU matrix and vector.
6. Why the GRU is naturally sequential.
7. How GRU state maps onto Seqvex state semantics.
8. Why a reference implementation is useful.
9. Why the production path uses reusable workspace.
10. Why the current workspace ownership is still an architectural question.
11. Why allocation reduction does not automatically imply universal latency improvement.

---

# 17. References

The implementation should be understood against the standard GRU formulation and independently validated through Seqvex's own equations and tests.

For Seqvex-specific behavior, the authoritative sources are:

- `src/models/recurrent/gru.rs`
- `src/foundation/state/transition.rs`
- `src/execution/streaming.rs`
- `tests/gru.rs`
- `tests/streaming_execution.rs`

The mathematical reference and implementation contract should remain explicit rather than being inferred from an optimized kernel.
