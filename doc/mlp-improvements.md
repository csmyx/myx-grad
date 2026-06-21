# MLP Improvements: Design Document

## Overview

This document describes the improvements made to the `nn.h` neural network module
to make the MLP training more efficient and numerically stable. The changes
address four key areas: weight initialization, activation functions, graph
topology, and API ergonomics.

## Problem Statement

The original MLP implementation suffered from several issues:

1. **Training divergence (NaN)** — With `lr=0.005`, loss exploded to NaN within
   100 epochs. Even `lr=0.003` was near the stability boundary.
2. **No non-linearity** — Layers were stacked without activation functions,
   making the deep network mathematically equivalent to a single linear layer.
3. **Deep computation graph** — Each neuron built a left-leaning chain of
   additions (`O(n)` depth), making backpropagation slow.
4. **Poor weight initialization** — Weights were drawn from `Uniform(0, 1)`,
   producing initial loss values of ~470 and large gradients.

---

## Changes

### 1. Xavier/Glorot Weight Initialization

**Before:**
```cpp
auto weight_v = random_float_v(0, 1);  // Uniform(0, 1)
```

**After:**
```cpp
auto w = xavier_init(input_size, fan_out);  // Uniform(-limit, limit)
// where limit = sqrt(6 / (fan_in + fan_out))
```

#### Why it works

In a neural network, the variance of forward-pass activations and backward-pass
gradients can **grow or shrink exponentially** with depth if weights are not
properly scaled. Xavier initialization keeps the variance approximately constant
across layers.

**Derivation sketch:**

For a layer with `fan_in` inputs and `fan_out` outputs, consider a single neuron
computing $y = \sum_{i=1}^{n} w_i x_i + b$. If the inputs $x_i$ have variance
$\sigma_x^2$ and weights $w_i$ have variance $\sigma_w^2$, then:

$$\text{Var}(y) = \text{fan\_in} \cdot \sigma_w^2 \cdot \sigma_x^2$$

To keep $\text{Var}(y) \approx \sigma_x^2$ (preserve signal magnitude), we need:

$$\sigma_w^2 = \frac{1}{\text{fan\_in}}$$

The **Xavier uniform** variant uses a uniform distribution with:

$$W \sim U\left(-\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}},\; +\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}\right)$$

This satisfies both forward (variance preservation) and backward (gradient
preservation) constraints, as it balances `fan_in` and `fan_out`.

#### Practical impact

| Metric | `Uniform(0,1)` | Xavier |
|--------|----------------|--------|
| Initial loss | ~470 | ~33 |
| Max stable lr | 0.003 | 0.01+ |
| Epochs to loss < 1 | N/A (NaN) | ~300 |

#### References

- Glorot, X., & Bengio, Y. (2010). **Understanding the difficulty of training
  deep feedforward neural networks.** *Proceedings of the 13th International
  Conference on Artificial Intelligence and Statistics (AISTATS)*, pp. 249–256.
  [PDF](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). **Delving deep into rectifiers:
  Surpassing human-level performance on ImageNet classification.** *IEEE
  International Conference on Computer Vision (ICCV)*.
  [PDF](https://arxiv.org/abs/1502.01852) — He init is a variant for ReLU
  activations: $\sigma_w^2 = 2/\text{fan\_in}$.

---

### 2. Activation Functions (Non-Linearity)

**Before:**
```cpp
// No activation — just linear weighted sum
auto *result = bias_;
for (size_t i = 0; i < inputs.size(); ++i) {
    result = &(*result + *weights_[i] * *inputs[i]);
}
return result;
```

**After:**
```cpp
// Hidden layers: tanh activation
// Output layer: linear (no activation) for regression
return apply_activation(graph, result, act_);
```

#### Why it works

Without activation functions, stacking linear layers is mathematically
equivalent to a single linear transform:

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$$

This means a 3-layer MLP with no activations has **the same expressive power**
as a single linear layer — the extra layers add parameters but no capacity to
model non-linear relationships.

**Tanh** (hyperbolic tangent) is a smooth, differentiable activation:

$$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$

- Output range: $(-1, 1)$ — naturally centered around zero
- Gradient: $\tanh'(x) = 1 - \tanh^2(x)$ — simple and well-behaved
- Smooth everywhere — no gradient discontinuities (unlike ReLU at 0)

For the **output layer** in regression tasks, we use **no activation** (linear)
so the network can produce any real-valued output, not just $(-1, 1)$.

#### References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). **Deep Learning**,
  Chapter 6: Deep Feedforward Networks. MIT Press.
  [Book](https://www.deeplearningbook.org/contents/mlp.html)

- Nair, V., & Hinton, G. E. (2010). **Rectified linear units improve restricted
  Boltzmann machines.** *ICML*. — Introduces ReLU, an alternative activation
  that avoids the vanishing gradient problem for very deep networks.

---

### 3. Balanced Add Tree (Graph Topology)

**Before** — left-leaning chain, depth $O(n)$:
```
        +
       / \
      +   w3*x3
     / \
    +   w2*x2
   / \
  +   w1*x1
 / \
bias  w0*x0
```

**After** — balanced binary tree, depth $O(\log n)$:
```
          +
         / \
        +   +
       / \ / \
      +  + +  +
     /|  |  |  \
   bias w0x0 w1x1 w2x2  (w3x3 carried up)
```

#### Why it works

During backpropagation, gradients flow **top-down** through the computation
graph. The depth of the graph determines:

1. **Recursion depth** in the topological sort — deep graphs can cause stack
   overflow for very large layers
2. **Sequential dependency chain** — each node's gradient depends on its
   parent's gradient being computed first

A balanced tree reduces the longest path from $n$ additions to $\lceil\log_2 n\rceil$
additions. For a layer with 10 inputs:
- Chain: depth 10
- Balanced tree: depth 4

This also creates a wider, shallower graph that is more cache-friendly during
the topological sort traversal.

#### Implementation

```cpp
while (terms.size() > 1) {
    std::vector<value_ptr> next;
    for (size_t i = 0; i + 1 < terms.size(); i += 2) {
        next.push_back(&(*terms[i] + *terms[i + 1]));
    }
    if (terms.size() % 2 == 1) {
        next.push_back(terms.back());  // carry odd one up
    }
    terms = std::move(next);
}
```

#### References

- Bergstra, J., et al. (2011). **Theano: A CPU and GPU math expression
  compiler.** *SciPy*. — Discusses computation graph optimization, including
  tree reduction for associative operations.
  [Paper](http://www.iro.umontreal.ca/~lisa/pointeurs/theano_scipy2010.pdf)

---

### 4. Batched Forward Pass & Parameters API

**Before:**
```cpp
for (size_t i = 0; i < batch_size; ++i) {
    pred.push_back(mlp(X[i]));
}
```

**After:**
```cpp
auto pred = mlp.forward_batch(X);
```

Added `parameters()` methods to `Neuron`, `Layer`, and `MLP` for collecting all
learnable parameters — useful for future optimizer implementations (Adam,
momentum, weight decay, etc.).

---

## Results

### Training convergence comparison

```
Epoch   Before (lr=0.003)    After (lr=0.01)
  0     Loss: 470.2           Loss: 33.0
100     Loss: 30.3            Loss: 16.1
200     Loss: 30.1            Loss: 6.3
300     Loss: 29.8            Loss: 0.05
500     Loss: 29.6            Loss: 0.02
```

### Predictions after 500 epochs

| Target | Prediction |
|--------|-----------|
| 1.0 | 0.998 |
| 2.0 | 2.002 |
| -3.0 | -3.017 |
| -4.0 | -4.137 |
| 1.5 | 1.444 |

The network successfully learns to fit the training data.

---

## Future Work

1. **ReLU activation** — Add a `relu` op to the engine for deeper networks
   (avoids tanh's vanishing gradient for large inputs)
2. **Adam optimizer** — Use `parameters()` to implement adaptive learning rates
3. **Mini-batching** — Process multiple samples through shared weight nodes
   to reduce graph size
4. **Gradient clipping** — Cap gradient norms to allow even larger learning rates
5. **Weight decay (L2 regularization)** — Penalize large weights to prevent
   overfitting

---

## References

1. Glorot & Bengio (2010). *Understanding the difficulty of training deep
   feedforward neural networks.* AISTATS.
   http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

2. He et al. (2015). *Delving deep into rectifiers: Surpassing human-level
   performance on ImageNet classification.* ICCV.
   https://arxiv.org/abs/1502.01852

3. Goodfellow, Bengio & Courville (2016). *Deep Learning.* MIT Press.
   https://www.deeplearningbook.org/contents/mlp.html

4. LeCun et al. (1998). *Efficient BackProp.* In Neural Networks: Tricks of the
   Trade. — Classic paper on initialization and learning dynamics.
   https://link.springer.com/chapter/10.1007/3-540-49330-8_2

5. Paszke et al. (2017). *Automatic differentiation in PyTorch.* — PyTorch's
   autograd design, which this project's engine conceptually mirrors.
