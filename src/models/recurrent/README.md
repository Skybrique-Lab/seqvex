# Gated Recurrent Unit (GRU)

The Gated Recurrent Unit (GRU) is a recurrent neural network (RNN) architecture designed for processing **sequential and temporal data**.

Unlike a feed-forward model, a GRU maintains an internal hidden state that is carried from one observation to the next. This allows the model to represent information from previous observations when processing the current observation.

In Seqvex, the GRU is currently implemented as a **stateful, single-observation CPU reference implementation**. It is intended primarily to establish mathematical correctness, state semantics, and a clear baseline for future optimization.

---

# 1. What is a GRU?

Suppose we observe a sequence:

$$
x_1, x_2, x_3, \ldots, x_t
$$

where `x_t` is one observation at time `t`.

A conventional feed-forward neural network could process each observation independently:

$$
y_t = f(x_t)
$$

However, sequential problems often depend on what happened previously.

A GRU therefore maintains a hidden state:

$$
h_t
$$

The hidden state is updated every time a new observation arrives:

$$
h_{t-1} \rightarrow h_t
$$

The fundamental GRU transition is:

$$
(x_t, h_{t-1}) \rightarrow h_t
$$

This makes the GRU naturally compatible with Seqvex's streaming execution model.

---

# 2. GRU Mathematical Formulation

The Seqvex implementation uses the following GRU equations.

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

- `σ` is the sigmoid activation function.
- `tanh` is the hyperbolic tangent activation function.
- `⊙` denotes element-wise multiplication.
- `x_t` is the current observation.
- `h_{t-1}` is the previous hidden state.
- `z_t` is the update gate.
- `r_t` is the reset gate.
- `\tilde{h}_t` is the candidate hidden state.
- `h_t` is the newly committed hidden state.

The important point is that **the hidden state is not calculated from the current input alone**.

It depends on both the current observation and the previous state:

$$
x_t
\quad\text{and}\quad
h_{t-1}
$$

Therefore, sequence history is carried through the recurrent state.

---

# 3. Understanding the Gates

The two gates serve different purposes.

## 3.1 Update Gate

The update gate is:

$$
z_t =
\sigma
\left(
W_zx_t + U_zh_{t-1} + b_z
\right)
$$

The sigmoid produces values between 0 and 1:

$$
0 < z_t < 1
$$

The update gate determines how much of the candidate state should replace the previous state.

The final equation is:

$$
h_t =
(1-z_t)\odot h_{t-1}
+
z_t\odot\tilde{h}_t
$$

Consider a single hidden-state element.

If the update gate is close to zero:

$$
z_t \approx 0
$$

then:

$$
h_t \approx h_{t-1}
$$

The previous state is largely retained.

If the update gate is close to one:

$$
z_t \approx 1
$$

then:

$$
h_t \approx \tilde{h}_t
$$

The candidate state largely replaces the previous state.

Therefore, the update gate controls the **degree of state replacement**.

---

## 3.2 Reset Gate

The reset gate is:

$$
r_t =
\sigma
\left(
W_rx_t + U_rh_{t-1} + b_r
\right)
$$

It controls how much of the previous hidden state contributes to the candidate state.

The relevant term is:

$$
r_t\odot h_{t-1}
$$

If the reset gate is close to zero:

$$
r_t \approx 0
$$

then:

$$
r_t\odot h_{t-1}\approx0
$$

The candidate calculation largely ignores the previous hidden state.

If the reset gate is close to one:

$$
r_t \approx 1
$$

then:

$$
r_t\odot h_{t-1}\approx h_{t-1}
$$

The previous hidden state contributes strongly to the candidate.

Thus:

- **Update gate:** controls how much candidate information enters the state.
- **Reset gate:** controls how much previous-state information participates in creating the candidate.

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

The candidate should be understood as a **proposed new state**, not automatically as the final state.

The GRU first calculates the candidate:

$$
\tilde{h}_t
$$

and then uses the update gate to decide how much of that candidate becomes part of the actual state.

This distinction is important for understanding Seqvex's state semantics.

The transition can conceptually be viewed as:

$$
\text{previous state}
\rightarrow
\text{candidate state}
\rightarrow
\text{validated new state}
$$

---

# 5. Dimensions and Shapes

Let:

- `d_x` = input dimension
- `d_h` = hidden-state dimension

Then the observation has dimension:

$$
x_t \in \mathbb{R}^{d_x}
$$

and the hidden state has dimension:

$$
h_t \in \mathbb{R}^{d_h}
$$

For each GRU gate, the input-weight matrices have shape:

$$
W_z,\;W_r,\;W_h
\in
\mathbb{R}^{d_h\times d_x}
$$

The recurrent-weight matrices have shape:

$$
U_z,\;U_r,\;U_h
\in
\mathbb{R}^{d_h\times d_h}
$$

The biases have shape:

$$
b_z,\;b_r,\;b_h
\in
\mathbb{R}^{d_h}
$$

The main quantities are:

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

Consider the update-gate input calculation:

$$
W_zx_t
$$

If:

$$
W_z\in\mathbb{R}^{d_h\times d_x}
$$

and:

$$
x_t\in\mathbb{R}^{d_x}
$$

then:

$$
W_zx_t\in\mathbb{R}^{d_h}
$$

Similarly, the recurrent calculation is:

$$
U_zh_{t-1}
$$

where:

$$
U_z\in\mathbb{R}^{d_h\times d_h}
$$

and:

$$
h_{t-1}\in\mathbb{R}^{d_h}
$$

Therefore:

$$
U_zh_{t-1}\in\mathbb{R}^{d_h}
$$

All three terms can therefore be added:

$$
W_zx_t + U_zh_{t-1}+b_z
\in
\mathbb{R}^{d_h}
$$

The same dimensional reasoning applies to the reset gate and candidate calculation.

This is one of the fundamental correctness conditions of a neural-network implementation: **every matrix/vector operation must have compatible dimensions**.

---

# 7. Step-by-Step Computation

For each observation `x_t`, the GRU performs the following sequence.

## Step 1 — Read the Current Observation

$$
x_t
$$

## Step 2 — Read the Current Committed State

$$
h_{t-1}
$$

## Step 3 — Calculate the Update Gate

$$
z_t =
\sigma
\left(
W_zx_t+U_zh_{t-1}+b_z
\right)
$$

## Step 4 — Calculate the Reset Gate

$$
r_t =
\sigma
\left(
W_rx_t+U_rh_{t-1}+b_r
\right)
$$

## Step 5 — Apply the Reset Gate to the Previous State

$$
r_t\odot h_{t-1}
$$

## Step 6 — Calculate the Candidate State

$$
\tilde{h}_t =
\tanh
\left(
W_hx_t+
U_h(r_t\odot h_{t-1})+
b_h
\right)
$$

## Step 7 — Blend the Previous State and Candidate

$$
h_t =
(1-z_t)\odot h_{t-1}
+
z_t\odot\tilde{h}_t
$$

## Step 8 — Commit the New State

The resulting state becomes the state used for the next observation:

$$
h_t
\rightarrow
h_{t+1}
$$

---

# 8. A Small Numerical Example

Consider a single hidden-state element.

Suppose the GRU has already calculated:

$$
h_{t-1}=0.8
$$

and:

$$
z_t=0.25
$$

The candidate calculation produced:

$$
\tilde{h}_t=0.2
$$

The new state is:

$$
h_t =
(1-z_t)h_{t-1}
+
z_t\tilde{h}_t
$$

Substituting the values:

$$
h_t =
(1-0.25)(0.8)
+
(0.25)(0.2)
$$

Therefore:

$$
h_t =
0.75(0.8)+0.25(0.2)
$$

$$
h_t =
0.6+0.05
$$

Therefore:

$$
\boxed{h_t=0.65}
$$

The previous state was `0.8`, while the candidate was `0.2`.

Because the update gate was:

$$
z_t=0.25
$$

the GRU moved the state only part of the way toward the candidate.

This is the essential idea behind the update gate.

---

# 9. Real-World Example: Sensor Monitoring

Consider a temperature-monitoring system attached to an industrial machine.

Every second, the system receives a vector of sensor measurements:

$$
x_t =
\begin{bmatrix}
\text{temperature}\\
\text{vibration}\\
\text{pressure}
\end{bmatrix}
$$

Therefore:

$$
d_x=3
$$

Suppose the GRU uses a hidden dimension of:

$$
d_h=4
$$

Then:

$$
x_t\in\mathbb{R}^{3}
$$

and:

$$
h_t\in\mathbb{R}^{4}
$$

The input-weight matrices therefore have dimensions:

$$
W_z,\;W_r,\;W_h
\in
\mathbb{R}^{4\times3}
$$

while the recurrent matrices have dimensions:

$$
U_z,\;U_r,\;U_h
\in
\mathbb{R}^{4\times4}
$$

The hidden state does not represent one specific sensor reading. Instead, it is a learned numerical representation of information from the recent sequence.

For example, the system might observe:

$$
x_{t-2},\;x_{t-1},\;x_t
$$

where temperature has been gradually increasing while vibration has also increased.

The GRU processes them sequentially:

$$
h_{t-2}
\rightarrow
h_{t-1}
\rightarrow
h_t
$$

The current hidden state therefore depends on sequence history.

If the update gate produces values close to zero for some dimensions, those components can retain information from earlier observations.

If the update gate produces values closer to one, those components can incorporate more of the newly computed candidate state.

The GRU does not inherently know that increasing temperature means "machine failure." That interpretation must come from the task, training data, objective, and downstream model.

---

# 10. Another Sequential Example: Financial Time Series

A GRU can also be applied to financial time-series data.

For example, an observation could contain normalized features such as:

$$
x_t=
\begin{bmatrix}
\text{return}_t\\
\text{volume change}_t\\
\text{volatility}_t
\end{bmatrix}
$$

The GRU processes each observation sequentially:

$$
x_1,x_2,\ldots,x_t
$$

and maintains a corresponding sequence of hidden states:

$$
h_1,h_2,\ldots,h_t
$$

The hidden state can provide a learned representation of the recent sequence.

The recurrent relationship can be summarized as:

$$
h_t=f(x_t,h_{t-1})
$$

A separate downstream component could then use `h_t` for a task such as classification or forecasting.

**Important:** the current Seqvex GRU implementation does not implement a trading strategy, financial objective, prediction head, or training pipeline. The example only illustrates how a sequential model could consume time-series observations.

---

# 11. GRU and Long-Term Dependencies

A conventional recurrent neural network repeatedly applies transformations to its hidden state:

$$
h_t=f(h_{t-1},x_t)
$$

During training, repeated multiplication through many recurrent steps can contribute to vanishing or exploding gradients.

GRU architectures introduce gates that provide a mechanism for controlling information flow through the recurrent state.

This architecture was designed to make learning dependencies across time easier than with a basic RNN in many settings.

However, a GRU does **not** guarantee that information will be preserved indefinitely.

Actual memory retention depends on:

- learned parameters;
- input sequence;
- gate values;
- hidden-state dimensionality;
- numerical precision;
- training procedure;
- task characteristics.

Therefore, statements such as "a GRU remembers thousands of observations" should not be treated as an architectural guarantee.

---

# 12. GRU State Semantics in Seqvex

Seqvex treats state as a first-class part of sequential computation.

For the GRU, the state is the hidden vector:

$$
\text{state}=h_t
$$

The conceptual transition is:

$$
(x_t,h_{t-1})
\rightarrow
h_t
$$

The previous state must be valid before processing the observation.

The implementation follows the broader Seqvex state-transition principle:

$$
\text{observation}
\rightarrow
\text{current valid state}
\rightarrow
\text{compute}
\rightarrow
\text{candidate state}
\rightarrow
\text{validate}
\rightarrow
\text{commit}
$$

This is important because a recurrent model is not merely a mathematical function of the current input.

It is a **stateful computation over a sequence**.

---

# 13. Failure and State Atomicity

A recurrent model must be careful when updating state.

Suppose:

$$
h_{t-1}
$$

is the last known valid state.

The model attempts to calculate:

$$
h_t
$$

but the computation produces an invalid result, such as a non-finite value.

The system should not silently replace the valid state with an invalid state.

Instead:

$$
\boxed{
\text{invalid candidate}
\Rightarrow
\text{retain previous valid state}
}
$$

Conceptually:

$$
h_{t-1}
\rightarrow
\boxed{\text{candidate }h_t}
\rightarrow
\text{validation}
$$

Only after successful validation is the candidate committed.

This is particularly important for streaming systems because one corrupted state can otherwise propagate into every subsequent observation.

---

# 14. Initialization

The Seqvex GRU currently initializes its hidden state to zero:

$$
h_0=\mathbf{0}
$$

For example, if the hidden dimension is:

$$
d_h=4
$$

then:

$$
h_0=
\begin{bmatrix}
0\\
0\\
0\\
0
\end{bmatrix}
$$

The current parameter helper also provides deterministic parameters for tests, examples, and benchmarks.

These deterministic parameters are intended for **reproducibility**, not as a statistically justified training initialization strategy.

The current implementation does not perform parameter training.

---

# 15. Current Seqvex Scope

The current GRU implementation is deliberately small.

It provides:

- stateful single-observation inference;
- GRU update/reset/candidate equations;
- dimension validation;
- finite-value validation;
- deterministic parameter generation for reproducible tests and examples;
- state reset;
- atomic state commitment;
- CPU execution;
- a readable reference implementation.

It does **not** currently provide:

- backpropagation through time;
- automatic differentiation;
- loss functions;
- optimizers;
- parameter training;
- GPU execution;
- SIMD-specific kernels;
- batched training;
- packed/fused GRU kernels;
- model serialization;
- distributed execution.

These are separate engineering problems and should not be assumed to be part of the current GRU implementation.

---

# 16. Mapping the Mathematics to the Implementation

The implementation stores the three sets of GRU parameters:

```text
GruParameters
├── update gate
│   ├── W_z
│   ├── U_z
│   └── b_z
│
├── reset gate
│   ├── W_r
│   ├── U_r
│   └── b_r
│
└── candidate state
    ├── W_h
    ├── U_h
    └── b_h
```

The model itself maintains:

```text
Gru
├── input dimension
├── hidden dimension
├── parameters
└── hidden state
```

A call to:

```rust
gru.step(&observation)
```

represents one sequential transition:

$$
h_{t-1}
\xrightarrow{x_t}
h_t
$$

The next call uses the newly committed state.

```rust
gru.step(&observation_t);
gru.step(&observation_t1);
gru.step(&observation_t2);
```

Therefore, the model's behavior depends on the order of observations.

---

# 17. Why Observation Order Matters

Consider two observations:

$$
A,\;B
$$

A GRU processes:

$$
A\rightarrow B
$$

differently from:

$$
B\rightarrow A
$$

because the first observation determines the state used when processing the second observation.

For example:

$$
h_A=f(A,h_0)
$$

followed by:

$$
h_B=f(B,h_A)
$$

Changing the order changes the input state of the second transition.

Therefore, in general:

$$
f(B,f(A,h_0))
\neq
f(A,f(B,h_0))
$$

This is one of the defining characteristics of sequential models.

For Seqvex, this means observation ordering is not merely metadata. It is part of the computational semantics.

---

# 18. Reference Implementation

The current GRU should be understood as a **reference implementation**.

A reference implementation prioritizes:

1. mathematical correctness;
2. readability;
3. inspectability;
4. deterministic behavior;
5. testability.

It does not necessarily represent the final high-performance implementation.

This distinction is important for future optimization.

For example, a future implementation might:

- reduce temporary allocations;
- reuse scratch buffers;
- fuse operations;
- change memory layouts;
- introduce SIMD;
- use optimized matrix kernels;
- move computation to a GPU.

An optimized implementation must still produce results consistent with the reference semantics within an explicitly defined numerical tolerance.

The reference implementation therefore acts as a semantic baseline against which optimized implementations can be tested.

---

# 19. Testing Strategy

The GRU should be tested at several levels.

### Mathematical Correctness

Verify that the implementation produces the expected result for known inputs and parameters.

### Dimension Validation

Invalid combinations such as a weight matrix with shape `d_h × d_x` combined with an input of the wrong dimension must be rejected.

The mathematical requirement is:

$$
W\in\mathbb{R}^{d_h\times d_x}
$$

with:

$$
x\in\mathbb{R}^{d_x}
$$

producing:

$$
Wx\in\mathbb{R}^{d_h}
$$

### Numerical Validation

Non-finite parameters or inputs must not silently propagate through the model.

### State Progression

Repeated calls should produce the expected sequential state:

$$
h_0
\rightarrow
h_1
\rightarrow
h_2
\rightarrow
\cdots
$$

### Reset Behavior

Calling reset should return the model to its initial state:

$$
h_t
\rightarrow
h_0
$$

### Long Sequential Execution

The implementation should remain consistent with an independent reference calculation over many sequential steps.

The repository's GRU tests include an independent scalar reference implementation for this purpose.

---

# 20. Performance Considerations

The mathematical cost of a GRU comes primarily from its matrix-vector operations.

For each observation, the current formulation requires:

$$
W_zx_t,\quad
U_zh_{t-1}
$$

$$
W_rx_t,\quad
U_rh_{t-1}
$$

$$
W_hx_t,\quad
U_h(r_t\odot h_{t-1})
$$

There are therefore three input-side matrix-vector products and three recurrent matrix-vector products.

For input dimension `d_x` and hidden dimension `d_h`, the dominant arithmetic work scales approximately with:

$$
O(3d_hd_x+3d_h^2)
$$

per observation, ignoring lower-order vector operations.

For sufficiently large hidden dimensions, the recurrent matrix operations can become particularly significant because they scale with:

$$
d_h^2
$$

Performance optimization should therefore be driven by measurement rather than assumed bottlenecks.

Possible future optimization areas include:

- allocation reduction;
- memory layout;
- cache locality;
- SIMD;
- operation fusion;
- matrix-vector kernels;
- CPU-specific optimization;
- accelerator execution.

These are not automatically justified merely because they are possible.

---

# 21. Learning Checklist

After studying this README, you should be able to explain:

- What problem a GRU is designed to address.
- Why a GRU has a hidden state.
- What `z_t` represents.
- What `r_t` represents.
- What `\tilde{h}_t` represents.
- How the final hidden state `h_t` is calculated.
- Why `W_*` has shape `d_h × d_x`.
- Why `U_*` has shape `d_h × d_h`.
- Why the GRU is inherently sequential.
- Why observation order affects the result.
- How state is carried from one observation to the next.
- Why candidate state and committed state should be conceptually separated.
- Why invalid state must not silently replace valid state.
- What a reference implementation is.
- Which parts of the GRU are currently implemented in Seqvex.
- Which parts, such as training and GPU execution, are outside the current scope.

A useful final mental model is:

$$
\boxed{
(x_t,h_{t-1})
\rightarrow
\text{gates}
\rightarrow
\text{candidate}
\rightarrow
\text{state update}
\rightarrow
h_t
}
$$

The resulting state becomes the starting point for the next observation:

$$
\boxed{
h_t\rightarrow h_{t+1}
}
$$

That persistent state is what makes the GRU a recurrent model and makes it naturally suited to sequential computation.