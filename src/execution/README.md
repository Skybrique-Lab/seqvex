# Streaming Execution

The `execution` module defines **when and how sequential computation is executed** in Seqvex.

It sits between the caller and the underlying stateful model.

The execution layer does not define the mathematical behavior of an ML algorithm. Instead, it coordinates the application of a model's state-transition function over observations.

The fundamental execution relationship is:

$$
\text{observation}
\rightarrow
\text{model transition}
\rightarrow
\text{new committed state}
$$

For a sequential model such as a GRU:

$$
(x_t,h_{t-1})
\rightarrow
h_t
$$

The execution layer is responsible for repeatedly driving this transition while preserving the required ordering and state semantics.

---

# 1. Purpose

Seqvex is designed around sequential, temporal, non-IID, and streaming workloads.

In such workloads, observations arrive over time:

$$
x_1,x_2,x_3,\ldots,x_t
$$

The result of processing one observation may affect the processing of the next observation.

Therefore, execution cannot be treated simply as a collection of independent function calls.

The execution layer provides the mechanism for processing observations while maintaining the current committed state.

Its primary responsibilities are:

- driving sequential model execution;
- preserving observation order;
- maintaining the currently committed state;
- exposing single-observation execution;
- processing streams of observations;
- supporting state reset;
- preserving state atomicity when a transition fails.

---

# 2. Execution vs. Model Semantics

A central architectural distinction in Seqvex is:

> **The model defines what computation means. The execution layer defines when and how that computation is driven.**

For example, a GRU defines the mathematical transition:

$$
(x_t,h_{t-1})
\rightarrow
h_t
$$

The execution layer does not redefine the GRU equations.

Instead, it invokes the model transition for each observation.

Conceptually:

$$
x_1
\rightarrow
h_1
$$

then:

$$
x_2
\rightarrow
h_2
$$

using the state produced by the previous transition:

$$
h_1
\rightarrow
h_2
$$

and so on.

This separation allows different models to use the same execution semantics.

---

# 3. The `StateModel` Boundary

The execution layer operates against the `StateModel` abstraction defined in the foundation layer.

Conceptually, a stateful model provides a transition:

$$
(\text{state},\text{observation})
\rightarrow
\text{new state}
$$

The model is responsible for calculating the transition.

The execution layer is responsible for driving that transition.

This gives the architecture a separation similar to:

```text
┌─────────────────────────────┐
│        Execution Layer      │
│                             │
│  When/how observations run  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       StateModel             │
│                             │
│  What the transition means  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Model State            │
│                             │
│  Current committed state    │
└─────────────────────────────┘
```

The execution layer therefore does not need to know whether the underlying model is:

- a GRU;
- another recurrent model;
- an online statistical model;
- an anomaly detector;
- an RL state transition;
- or another future sequential model.

It only requires the model to satisfy the state-transition contract.

---

# 4. Streaming as the Primary Execution Context

Seqvex treats streaming as a first-class execution context.

A stream can be represented conceptually as:

$$
x_1,x_2,x_3,\ldots,x_n
$$

The execution layer processes observations in order:

$$
x_1
\rightarrow
x_2
\rightarrow
x_3
\rightarrow
\cdots
\rightarrow
x_n
$$

The corresponding state transitions are:

$$
h_0
\rightarrow
h_1
\rightarrow
h_2
\rightarrow
\cdots
\rightarrow
h_n
$$

The state produced by one successful transition becomes the state supplied to the next transition.

Therefore, execution order is semantically significant.

---

# 5. Single-Observation Execution

The fundamental operation is processing one observation.

Conceptually:

$$
(\text{current state},x_t)
\rightarrow
\text{next state}
$$

In the current implementation, `StreamingExecutor::process_one` performs this role.

A simplified execution sequence is:

```text
Current committed state
        │
        ▼
    observation
        │
        ▼
   StateModel::update
        │
        ▼
 candidate state
        │
        ▼
    validation
        │
        ▼
 committed state
```

The important property is that the executor maintains the state between calls.

For example:

```rust
executor.process_one(&observation_t)?;
executor.process_one(&observation_t1)?;
executor.process_one(&observation_t2)?;
```

The second call does not begin from the original state.

It begins from the state successfully produced by the first call.

---

# 6. Stream Execution

The execution layer also supports processing a sequence of observations.

Conceptually:

$$
\{x_1,x_2,\ldots,x_n\}
$$

is processed as:

$$
h_0
\xrightarrow{x_1}
h_1
\xrightarrow{x_2}
h_2
\xrightarrow{x_3}
\cdots
\xrightarrow{x_n}
h_n
$$

This is a sequential fold over state.

The important property is that the observations are **not assumed to be independent**.

For a stateful model:

$$
h_t=f(h_{t-1},x_t)
$$

Therefore:

$$
h_t
$$

depends on both the current observation and the state produced by previous observations.

---

# 7. Observation Ordering

Observation order is part of the execution semantics.

Consider two observations:

$$
A,\;B
$$

Processing them as:

$$
A\rightarrow B
$$

is generally different from:

$$
B\rightarrow A
$$

because the state entering the second transition is different.

For example:

$$
h_A=f(h_0,A)
$$

followed by:

$$
h_B=f(h_A,B)
$$

whereas reversing the observations produces:

$$
h'_B=f(h_0,B)
$$

followed by:

$$
h'_A=f(h'_B,A)
$$

In general:

$$
h_B\neq h'_A
$$

Therefore, the execution layer must preserve observation order unless an explicit execution mode defines another semantic.

---

# 8. State Ownership

The current `StreamingExecutor` owns the **committed execution state** associated with the execution session.

Conceptually:

```text
StreamingExecutor
├── model reference
└── committed state
```

The model defines how a transition is calculated.

The executor maintains the state that results from successfully executing those transitions.

This allows the same model definition to potentially participate in different execution sessions with independent state.

For example:

```text
Model
 ├── Executor A → state A
 └── Executor B → state B
```

The execution states are therefore associated with the execution context rather than being treated as one global stream.

---

# 9. State Transition and Atomicity

A central requirement of sequential execution is that an invalid transition must not silently corrupt the committed state.

The intended transition is:

$$
\text{previous valid state}
\rightarrow
\text{candidate state}
\rightarrow
\text{validation}
\rightarrow
\text{commit}
$$

If validation succeeds:

$$
\boxed{
\text{candidate state}
\rightarrow
\text{committed state}
}
$$

If validation fails:

$$
\boxed{
\text{previous valid state remains committed}
}
$$

This prevents an invalid state from becoming the input to subsequent observations.

The execution layer therefore relies on the state-transition semantics established by the foundation layer.

---

# 10. Failure Semantics

Consider a sequence:

$$
x_1,x_2,x_3,x_4
$$

Suppose:

$$
x_1
$$

and:

$$
x_2
$$

are processed successfully.

The committed state is then:

$$
h_2
$$

Suppose processing:

$$
x_3
$$

fails.

The desired state behavior is:

$$
h_2
\rightarrow
\boxed{\text{failed candidate}}
\rightarrow
h_2
$$

The invalid candidate must not become:

$$
h_3
$$

for the next successful transition.

If execution continues with:

$$
x_4
$$

then the model should continue from the last valid state:

$$
h_2
\xrightarrow{x_4}
h_4
$$

rather than from an invalid intermediate state.

This is particularly important in streaming systems because state corruption can propagate through all subsequent observations.

---

# 11. Reset Semantics

The execution layer supports resetting the execution state.

Conceptually:

$$
h_t
\rightarrow
h_0
$$

A reset establishes a new starting point for subsequent observations.

For the current GRU implementation, the initial hidden state is the zero vector:

$$
h_0=\mathbf{0}
$$

Therefore, after reset:

```text
previous execution history
        │
        ▼
      reset
        │
        ▼
initial state
        │
        ▼
new sequence
```

Reset is therefore a semantic operation, not merely a memory-management operation.

---

# 12. Relationship to the GRU

The GRU is an example of a stateful model that can be driven by the streaming execution layer.

The GRU itself defines:

$$
(x_t,h_{t-1})
\rightarrow
h_t
$$

The execution layer provides the repeated application:

$$
h_0
\xrightarrow{x_1}
h_1
\xrightarrow{x_2}
h_2
\xrightarrow{x_3}
\cdots
$$

Therefore:

```text
StreamingExecutor
        │
        │ process observation
        ▼
      GRU
        │
        │ calculate transition
        ▼
    new hidden state
        │
        ▼
StreamingExecutor
        │
        │ store committed state
        ▼
next observation
```

The executor does not need to understand GRU gates, matrix dimensions, activation functions, or recurrent equations.

Those belong to the model.

---

# 13. Execution Semantics vs. Compute Placement

Execution semantics and compute placement are intentionally separate concepts in Seqvex.

Execution semantics describe **how computation progresses**:

- single observation;
- streaming/online;
- bounded micro-batch;
- batch.

Compute placement describes **where computation runs**:

- CPU;
- GPU;
- accelerator;
- heterogeneous execution.

These dimensions should not be conflated.

For example:

```text
Execution:
streaming

Placement:
CPU
```

is valid.

So is:

```text
Execution:
streaming

Placement:
GPU
```

provided the relevant model and backend support it.

Similarly, a batch execution mode could eventually run on a CPU or GPU.

The execution layer therefore should not assume that streaming means CPU-only.

---

# 14. Why This Separation Matters

If execution and placement are tightly coupled, introducing a new hardware backend can require redesigning the execution semantics.

Seqvex instead aims to preserve the conceptual relationship:

```text
                 Execution semantics
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          single     streaming    batch
             │          │          │
             └──────────┼──────────┘
                        │
                        ▼
                 Compute placement
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
             CPU       GPU    accelerator
```

This separation is architectural intent.

It does **not** imply that every combination is currently implemented.

---

# 15. Streaming Is Not Automatically Batching

A streaming executor should not silently convert every sequence into a large batch.

The semantic unit of the current execution model is one observation:

$$
x_t
$$

followed by a state transition:

$$
h_{t-1}
\rightarrow
h_t
$$

Future micro-batching may be introduced as an optimization or additional execution capability.

However, batching must preserve the required semantics of the model and execution context.

Therefore:

> **Micro-batching is an optimization opportunity, not the semantic foundation of streaming execution.**

---

# 16. Current Implementation

The current execution module provides:

```text
src/execution/
├── mod.rs
└── streaming.rs
```

The main execution abstraction is:

```text
StreamingExecutor
```

It maintains:

```text
StreamingExecutor
├── model reference
└── committed state
```

The current interface supports the conceptual operations:

```text
process_one
process_stream
reset
```

### `process_one`

Processes one observation using the current committed state.

Conceptually:

$$
(h_{t-1},x_t)
\rightarrow
h_t
$$

### `process_stream`

Processes observations sequentially while carrying state from one successful transition to the next.

Conceptually:

$$
h_0
\xrightarrow{x_1}
h_1
\xrightarrow{x_2}
\cdots
\xrightarrow{x_n}
h_n
$$

### `reset`

Restores the executor to its initial execution state.

---

# 17. What the Execution Layer Does Not Own

The execution layer does **not** own:

- GRU equations;
- neural-network architecture;
- model parameters;
- loss functions;
- optimizers;
- training algorithms;
- feature engineering;
- data ingestion;
- database operations;
- visualization;
- business/domain logic;
- exchange or market-data protocols;
- GPU kernel implementation.

Those concerns belong elsewhere in the system.

The execution layer should remain focused on execution semantics.

---

# 18. Data Ingestion Is Separate

A stream of observations may originate from many sources:

```text
sensor
market data
file
network
simulation
database
application
```

The execution layer does not need to own those sources.

Conceptually:

```text
Data Source
    │
    ▼
Observation
    │
    ▼
Execution Layer
    │
    ▼
StateModel
    │
    ▼
New State
```

This separation allows Seqvex models to operate on observations produced by external systems without making the ML framework responsible for the external system itself.

---

# 19. Testing

Execution behavior should be tested independently from individual model mathematics.

Important execution tests include:

### Sequential state progression

Verify that state from one successful observation becomes the input state for the next.

### Observation ordering

Verify that observations are processed in the supplied order.

### Failure preservation

Verify that a failed transition does not replace the last valid committed state.

### Stream behavior

Verify that processing a stream produces the same sequential state progression as processing the observations individually.

### Reset behavior

Verify that reset removes the previous execution history and restores the expected initial state.

### Model integration

Verify that a real stateful model, such as the GRU, can execute correctly through the execution layer.

These tests establish the semantics of execution independently from performance optimization.

---

# 20. Performance Considerations

The execution layer is intentionally thin.

A streaming execution loop should avoid introducing unnecessary work around the model's actual computation.

For a sequence:

$$
x_1,x_2,\ldots,x_n
$$

the executor should ideally perform approximately:

```text
observation
    ↓
state transition
    ↓
commit
    ↓
next observation
```

without unnecessary intermediate abstraction layers in the hot path.

However, optimization should be measurement-driven.

Potential future areas include:

- reducing temporary allocations;
- avoiding unnecessary state copies;
- reducing synchronization;
- minimizing dispatch overhead;
- improving memory locality;
- supporting bounded micro-batching;
- specialized CPU execution;
- accelerator execution.

These should be introduced only when profiling demonstrates that they materially affect performance.

---

# 21. Determinism and Reproducibility

For a deterministic model and deterministic sequence of observations, execution should produce reproducible state progression under equivalent numerical conditions.

Conceptually:

$$
(x_1,x_2,\ldots,x_n)
\rightarrow
h_n
$$

should produce the same result when the same initial state, parameters, and numerical execution conditions are used.

This property is important for:

- testing;
- debugging;
- reference implementations;
- benchmark comparisons;
- validating optimized implementations.

Hardware-specific floating-point behavior may introduce small numerical differences when execution is moved to different backends.

Such differences should be measured and handled with explicitly defined numerical tolerances rather than assumed to be identical.

---

# 22. Current Scope

The current execution subsystem establishes the first streaming execution path for Seqvex.

Currently supported:

- stateful single-observation execution;
- sequential stream processing;
- explicit state ownership;
- reset;
- failure-aware state progression;
- integration with `StateModel`;
- CPU execution through the current model implementations.

The execution layer is intentionally small because its current purpose is to establish correct execution semantics before adding more advanced execution mechanisms.

---

# 23. Deferred Capabilities

The following are intentionally not assumed to be part of the current implementation:

- bounded micro-batching;
- general batch execution;
- asynchronous execution;
- task scheduling;
- thread pools;
- distributed execution;
- GPU scheduling;
- accelerator scheduling;
- graph execution;
- pipeline parallelism;
- automatic synchronization;
- zero-copy device transfer;
- heterogeneous execution policies.

These may become relevant as concrete requirements emerge.

They should not be added merely because they are common features in other ML frameworks.

---

# 24. Architectural Principle

The execution layer follows a simple principle:

> **Execution determines when and how a model transition is driven; the model determines what the transition means.**

For sequential execution, the essential state transition is:

$$
\boxed{
(\text{current state},\text{observation})
\rightarrow
\text{candidate state}
\rightarrow
\text{validated committed state}
}
$$

Repeated over a sequence:

$$
\boxed{
h_0
\xrightarrow{x_1}
h_1
\xrightarrow{x_2}
h_2
\xrightarrow{x_3}
\cdots
\xrightarrow{x_n}
h_n
}
$$

This is the foundation of Seqvex's streaming execution model.

The execution layer should remain simple enough that this semantic relationship is easy to understand, test, and eventually optimize.

---

# 25. Summary

The Seqvex execution layer provides the machinery required to run stateful models over sequential observations.

Its responsibilities are deliberately narrow:

1. accept observations;
2. preserve their order;
3. invoke the model transition;
4. maintain the committed state;
5. preserve state validity across failures;
6. support reset;
7. provide a foundation for future execution strategies.

The central abstraction is:

$$
(\text{state},\text{observation})
\rightarrow
\text{new state}
$$

The execution layer does not define the mathematics of the model.

Instead, it provides the controlled environment in which those mathematical transitions can be repeatedly executed over a sequence.

This separation is fundamental to Seqvex's architecture:

$$
\boxed{
\text{Execution semantics}
\neq
\text{Model semantics}
\neq
\text{Compute placement}
}
$$

Keeping these concerns separate allows Seqvex to evolve from its current CPU streaming reference implementation toward more advanced execution and hardware capabilities without prematurely coupling those concerns together.