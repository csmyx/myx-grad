# Dispatch Strategy for back_propagate

## Current approach: switch on `op_t` enum

```cpp
auto back_propagate() {
    switch (m_op) {
    case op_t::add:  { /* d(a+b)/da = 1, d(a+b)/db = 1 */ }
    case op_t::sub:  { /* d(a-b)/da = 1, d(a-b)/db = -1 */ }
    case op_t::mul:  { /* d(a*b)/da = b, d(a*b)/db = a */ }
    case op_t::tanh: { /* d(tanh(x))/dx = 1 - tanh(x)^2 */ }
    // ...
    }
}
```

### Pros

- **Cache-friendly**: all `Value` objects share the same memory layout. Traversing the
  topological order accesses contiguous or near-contiguous memory with no vtable
  indirection.
- **Simple**: adding a new op means appending an enum member + a case branch. No
  new subclass needed.
- **Zero virtual dispatch overhead**: no vtable pointer per object (saves 8 bytes
  per node), no indirect branch misprediction penalty.
- **Debuggable / serializable**: `op_t` is trivially printable (`to_string`),
  comparable, and storable — essential for DOT graph visualization.

### Cons

- All op logic lives in one function body; grows longer with each new op.
- Ops that need extra state (e.g. `ReLU` only needs a boolean, `Conv2D` needs
  stride/padding) must store that state as extra members on `Value`.

---

## Alternative designs

### B. Virtual functions / polymorphism

```cpp
struct Op {
    virtual void backprop(float grad, Value* left, Value* right) = 0;
    virtual ~Op() = default;
};
struct AddOp : Op { void backprop(...) override { ... } };
// Value holds a shared_ptr<Op>
```

| Pro                                                  | Con                                                                     |
| ---------------------------------------------------- | ----------------------------------------------------------------------- |
| Each op is an isolated class (Open/Closed Principle) | +8–16 bytes per object (vtable pointer)                                 |
| Adding an op never touches existing code             | Indirect call overhead on every backprop step                           |
| Complex ops can carry their own state naturally      | Objects scattered on heap → worse cache locality during graph traversal |

### C. `std::function` / function pointer

```cpp
std::function<void(float, Value*, Value*)> m_backprop;
```

| Pro                                                         | Con                                  |
| ----------------------------------------------------------- | ------------------------------------ |
| More flexible than virtual (can capture state via closures) | ~32 bytes per object (std::function) |
| Closures can be composed dynamically                        | Cannot be serialized / visualized    |
|                                                             | Terrible for cache locality          |

### D. Type-erased function object (dyno-style)

Rust trait object / C++ type erasure hybrids. Not used in production autodiff
frameworks — over-engineered for this domain.

---

## What production frameworks use

### PyTorch (ATen / C10)

**Hash-table dispatcher with codegen.** Each op registers at compile time:

```cpp
m.dispatcher().registerDef("aten::add(Tensor a, Tensor b) -> Tensor");
m.impl("aten::add", &add_kernel);
```

- Dispatch is a **string → function pointer** hash lookup, not per-tensor vtable.
- Thousands of kernels exist; per-tensor vtable would be prohibitive.
- Essentially the **upgraded version of switch dispatch**: the "switch" becomes a
  runtime hash table indexed by op name.

### llama.cpp / ggml

Uses **switch on op enum** — exactly like the current design:

```c
static void ggml_compute_forward(..., struct ggml_tensor *tensor) {
    switch (tensor->op) {
    case GGML_OP_ADD: ...
    case GGML_OP_MUL: ...
    case GGML_OP_MUL_MAT: ...
    }
}
```

Rationale:
- All ggml tensors are fixed-size structs with flat memory layout.
- During LLM inference, only a handful of ops execute at any time
  (ADD, MUL, MUL_MAT, ROPE, etc.) — branch predictor accuracy is near-perfect.
- Zero-abstraction, pure C, maximum performance.

### TensorFlow / XLA

**Compile-time IR.** The full compute graph (forward + backward) is lowered to
XLA HLO IR and JIT-compiled to machine code. Backward pass is not a runtime
switch at all — it is a new IR subgraph unrolled at compile time. Out of scope
for this project.

---

## Conclusion

**Switch on `op_t` is the correct choice for this engine's scale and goals.**

| Criteria         | Switch       | Virtual  | std::function |
| ---------------- | ------------ | -------- | ------------- |
| # ops < 30       | ✔            | —        | —             |
| Perf (cache)     | ✔            | ✘        | ✘             |
| Debuggable (DOT) | ✔            | ✘        | ✘             |
| Memory per node  | minimal      | +8–16 B  | +32 B         |
| Branch predict   | near-perfect | indirect | indirect      |

It is the same strategy used by ggml/llama.cpp to power production-grade LLM
inference.

### When to consider switching

- **Op count exceeds ~30** and ops have meaningfully different member state →
  consider a registration-based dispatcher (PyTorch-style string → kernel map).
- **Plugin system needed** for user-defined ops → string → factory function
  registry.
- **Never** move to virtual dispatch — in this domain, virtual functions are
  effectively an anti-pattern.
