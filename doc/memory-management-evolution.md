# 计算图内存管理演进记录

## 概述

本文档记录了 `myx-grad` 自动微分引擎在计算图节点内存管理方面的演进过程，包括每一版的设计选择、遇到的问题、解决方案，以及与业界最佳实践的对比。

---

## 问题背景

自动微分引擎的核心数据结构是 **计算图（DAG）**：每个 `Value` 节点通过 `m_left` / `m_right` 指针指向其输入节点，形成有向无环图。例如：

```
  c = a + b    →    c.m_left = &a, c.m_right = &b
```

这带来一个根本性的内存安全问题：**节点的生命周期必须长于所有引用它的子节点**。如果某个节点被销毁，所有指向它的指针立即悬空。

---

## 演进历程

### V1：栈分配 + 裸指针（原始版本）

```cpp
class Value {
  Value *m_left = nullptr;
  Value *m_right = nullptr;
};

auto a = Value(2.0);
auto b = Value(3.0);
auto c = a + b;  // c.m_left = &a, c.m_right = &b
```

**问题：**

| 问题                     | 描述                                                                       |
| ------------------------ | -------------------------------------------------------------------------- |
| 临时对象悬空             | `(a * b) + c` 中 `a*b` 的返回值是临时对象，语句结束后销毁，`c.m_left` 悬空 |
| vector 扩容              | `std::vector<Value<>>` 在 `push_back` 时可能 reallocation，使所有指针失效  |
| C 风格数组               | `T m_op_params[k_max_op_params]{}` 触发 `-W_decltype_cstyle_arrays` 警告   |
| `make_unique` 不支持填充 | `std::make_unique<T[]>(n, value)` 无此重载，无法初始化数组                 |

这是最直觉的写法，但存在严重的 UB（未定义行为）风险。

---

### V2：`std::array` + `std::vector` + `std::unique_ptr<T[]>`

```cpp
class Value {
  std::array<T, k_max_op_params> m_op_params{};  // 替代 C 数组
};

struct Layer {
  std::unique_ptr<Value<float_t>[]> m_weights;  // 替代 vector
};
```

**改进：**
- `std::array` 零开销、不会退化为指针、支持 STL 算法
- `std::vector` 支持 `reserve()` 预分配，避免 reallocation
- `std::make_unique<T[]>(n)` 确保堆分配

**遗留问题：**
- 二元运算的返回值仍在栈上，临时对象悬空问题未解决
- `unique_ptr<T[]>` 缺少 `.size()` 等安全接口

---

### V3：`Tensor` 包装类

```cpp
struct Tensor : private std::vector<std::unique_ptr<Value<float_t>>> {
  static auto zeros(size_t n) -> Tensor;
  static auto xavier_uniform(size_t n, ...) -> Tensor;
};
```

**改进：**
- 私有继承 + `using` 暴露，防止误用
- 工厂方法（`zeros`、`xavier_uniform`、`from_values`）保证所有元素都在堆上分配
- Xavier/Glorot 初始化直接内置于类型

**遗留问题：**
- 计算图节点的 `m_left`/`m_right` 仍为裸指针
- `operator()` 中 `new` 裸分配 `Value`，存在内存泄漏风险
- 临时 `Value` 对象（如 `w * input` 的结果）在栈上，指针悬空

---

### V4：`shared_ptr<Value>` + 自由运算符重载（当前版本）

```cpp
class Value : public Node<T>,
              public std::enable_shared_from_this<Value<T>> {
  std::shared_ptr<Value<T>> m_left;
  std::shared_ptr<Value<T>> m_right;
};

// 自由运算符 — 两侧对称
template <typename T>
auto operator+(const std::shared_ptr<Value<T>> &lhs,
               const std::shared_ptr<Value<T>> &rhs)
    -> std::shared_ptr<Value<T>>;

// 用户代码
auto c = a + b;   // a, b, c 都是 shared_ptr<Value<float_t>>
```

**彻底解决的问题：**

| 旧问题          | 解决方式                                                            |
| --------------- | ------------------------------------------------------------------- |
| 临时对象悬空    | 所有运算结果 `make_shared` 在堆上，引用计数保护                     |
| vector 扩容失效 | `shared_ptr` 拷贝只是增加引用计数，reallocation 不影响              |
| 裸指针风险      | `shared_ptr` 自动管理生命周期，无 `new`/`delete`                    |
| 运算符不对称    | 自由函数模板，两个参数都是 `shared_ptr`，`a + b` 自然对称           |
| DAG 共享子节点  | `shared_ptr` 引用计数天然支持一个节点被多个父节点引用（如 `a * a`） |

---

## 设计决策分析

### 为什么选择 `shared_ptr` 而非其他方案？

#### 方案对比

| 方案                        | 优点                                | 缺点                                                        | 业界使用               |
| --------------------------- | ----------------------------------- | ----------------------------------------------------------- | ---------------------- |
| **裸指针 + Arena 分配器**   | 零开销，缓存友好                    | 手动管理生命周期，arena 生命周期必须覆盖所有计算            | GCC/Clang AST、LLVM IR |
| **`unique_ptr` + 所有权树** | 明确所有权，无循环引用风险          | 无法表达 DAG 共享子节点（`a * a` 需要 `m_left == m_right`） | 不适用于计算图         |
| **`shared_ptr` + 引用计数** | 天然支持 DAG 共享，自动管理生命周期 | 循环引用风险、引用计数开销（原子操作）                      | PyTorch、MLIR 引用语义 |
| **侵入式引用计数**          | 比 `shared_ptr` 少一次间接寻址      | 需要侵入式基类，删除器不支持                                | CPython、V8 GC         |
| **GC 管理堆**               | 无手动管理                          | 不可预测的暂停，复杂度高                                    | Jax/JVM 生态           |

#### 我们的选择：`shared_ptr`

理由：
1. **DAG 共享是刚需**：`a * a` 等场景中同一节点作为多个输入，`unique_ptr` 无法表达
2. **计算图本身是无环的**：前向传播构建图、反向传播遍历图，不存在循环引用
3. **C++ 标准实践**：`enable_shared_from_this` 是标准库为"在成员函数中获取自身 `shared_ptr`"提供的正式机制，这正是我们的场景
4. **代码简洁性**：`shared_ptr` 是标准库中唯一同时支持共享所有权和自动销毁的智能指针

#### 循环引用分析

计算图中的边总是从 **新节点指向旧节点**（`c = a + b` 中 `c → a`, `c → b`），方向是单向的，不可能形成环。因此 `shared_ptr` 引用计数不会泄露内存。

```
     a     b        ← 叶子节点：引用计数 = 1（被一个子节点持有）
      \   /
       c            ← 中间节点：引用计数 = 1
        |
       d            ← 输出节点：引用计数 = 1（被用户持有）
```

当用户释放 `d` 时，引用计数链式递减，所有节点自动销毁。

---

### 为什么用自由运算符而非成员运算符？

**之前的成员运算符（不对称）：**
```cpp
// Value 类内
auto operator+(const shared_ptr<Value<T>> &other) -> shared_ptr<Value<T>>;

// 用户代码
auto c = *a + b;  // 左侧要解引用，右侧是 shared_ptr
```

**现在的自由运算符（对称）：**
```cpp
// namespace engine 内
template <typename T>
auto operator+(const shared_ptr<Value<T>> &lhs,
               const shared_ptr<Value<T>> &rhs) -> shared_ptr<Value<T>>;

// 用户代码
auto c = a + b;  // 两侧对称，无需解引用
```

**对称设计的优势：**

| 维度         | 成员运算符                     | 自由运算符                                           |
| ------------ | ------------------------------ | ---------------------------------------------------- |
| 调用语法     | `*a + b` 或 `a->operator+(b)`  | `a + b`                                              |
| 交换律       | 不对称，隐含 `this` 是特殊的   | 完全对称，左右地位相同                               |
| 模板推导     | 依赖 `enable_shared_from_this` | 两个参数类型相同，推导自然                           |
| 与标准库一致 | 否                             | 是（`std::complex`、`std::chrono` 等都用自由运算符） |

**C++ 核心指南相关条目：**

- [C.161](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c161-use-non-member-functions-for-symmetric-operators)：对称运算符应使用非成员函数
- [C.164](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c164-avoid-implicit-conversion-operators)：避免隐式转换运算符
- [R.24](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#r24-use-stdshared_ptr-or-stdunique_ptr-to-represent-ownership)：使用 `std::shared_ptr` 或 `std::unique_ptr` 表示所有权

---

### 为什么 `Tensor` 用私有继承？

```cpp
struct Tensor : private std::vector<std::shared_ptr<Value<float_t>>> {
private:
  using Base = std::vector<std::shared_ptr<Value<float_t>>>;
public:
  using Base::operator[];
  using Base::size;
  // ... 选择性暴露接口
};
```

**原因：**

1. **类型安全**：不允许将 `Tensor` 隐式转为 `vector`，防止误用如 `tensor.push_back(stack_value)`
2. **接口控制**：只暴露安全的操作（`[]`、`size`、`reserve`），隐藏危险的（直接 `emplace_back` 一个栈对象）
3. **零开销**：私有继承无虚表开销

**替代方案对比：**

| 方案                       | 优点                 | 缺点                               |
| -------------------------- | -------------------- | ---------------------------------- |
| `private` 继承 + `using`   | 零开销、精确接口控制 | 代码略冗长                         |
| `public` 继承              | 简单                 | 破坏封装，所有 `vector` 操作都暴露 |
| 组合（包含 `vector` 成员） | 完全封装             | 需要转发所有需要的接口             |
| **C++20 `using` 声明别名** | 最简洁               | C++20 才支持                       |

**业界实践**：私有继承 + `using` 暴露是 C++17 中"编译期多态 + 接口窄化"的标准模式。Boost.Asio 的 `io_context::executor_type`、Eigen 的表达式模板等均采用此模式。

---

## 初始化策略

### Xavier/Glorot 均匀初始化

```cpp
static auto xavier_uniform(size_t n, size_t fan_in, size_t fan_out,
                           std::mt19937 &gen) -> Tensor {
  float_t limit = std::sqrt(6.0 / static_cast<float_t>(fan_in + fan_out));
  std::uniform_real_distribution<float_t> dist(-limit, limit);
  // ...
}
```

**为什么选 Xavier 而非 He/Kaiming？**

| 初始化             | 公式                                                | 适用激活      | 说明                     |
| ------------------ | --------------------------------------------------- | ------------- | ------------------------ |
| Xavier/Glorot 均匀 | `U[-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out))]` | tanh, sigmoid | 保持前向/反向信号方差    |
| He/Kaiming 均匀    | `U[-√(6/fan_in), +√(6/fan_in)]`                     | ReLU 及变体   | 对于 tanh 会导致值域偏大 |
| LeCun 均匀         | `U[-√(3/fan_in), +√(3/fan_in)]`                     | SELU          | 自归一化网络专用         |

我们使用 Xavier/Glorot 因为本引擎使用 `tanh` 激活函数，与该初始化的理论假设一致。

**随机数引擎管理**：`std::mt19937` 通过引用传入而非内部创建，保证同一 `Layer` 内所有权重共享同一随机状态，同时允许外部控制种子实现可复现性。

---

## 与主流框架对比

| 特性         | myx-grad (当前)       | PyTorch                         | micrograd (Karpathy) |
| ------------ | --------------------- | ------------------------------- | -------------------- |
| 节点存储     | `shared_ptr<Value>`   | ATen Tensor + 引用计数          | Python 对象（GC）    |
| 生命周期管理 | 引用计数（自动）      | 引用计数 + autograd 作用域      | Python GC            |
| 运算符风格   | 自由函数模板          | 成员方法 + `__add__` 等魔法方法 | Python 运算符重载    |
| DAG 共享     | `shared_ptr` 天然支持 | `saved_tensors` 引用机制        | Python 引用语义      |
| 循环引用     | 无（计算图无环）      | 需 `torch.autograd` 显式管理    | Python GC 处理       |
| 内存开销     | 每节点 ~80B + 控制块  | 已优化 Tensor 存储              | Python 对象开销大    |

---

## 主流框架底层实现深度对比

### PyTorch（C++ 核心 + Python 绑定）

**架构层次：**

```
Python API (nn.Module, autograd.Function)
    ↕ pybind11 / torch.utils._pytree
C++ Dispatcher (ATen)
    ↕
TensorImpl (引用计数张量元数据)
    ↕
Storage (数据缓冲区，共享所有权)
```

**内存管理策略：**

| 维度       | PyTorch 实现                                              | myx-grad 实现                                |
| ---------- | --------------------------------------------------------- | -------------------------------------------- |
| 张量数据   | `Storage` + `RefCounted` 引用计数，多 Tensor 共享同一存储 | 每节点独立 `shared_ptr`，无存储共享          |
| 计算图节点 | `Node` (autograd) 在前向时按需创建，反向后自动释放        | `Value` 节点由 `shared_ptr` 生命周期管理     |
| 梯度累加   | `grad` 是独立 `Tensor`，按引用计数管理                    | `m_grad` 是成员变量，内联存储                |
| 图生命周期 | `torch.autograd` 提供 `retain_graph=True/False` 控制      | `shared_ptr` 自动释放，无显式 `retain_graph` |
| 内存池     | `CUDAAllocator` 分桶管理 + 缓存池，减少 `cudaMalloc` 开销 | 标准库分配器，无额外池化                     |

**关键差异：**

PyTorch 的 `TensorImpl` 使用侵入式引用计数（类似 Python 的 `PyObject`），比 `shared_ptr` 少一次间接寻址：

```cpp
// PyTorch (simplified)
class TensorImpl {
  mutable std::atomic<int64_t> refcount_;  // 侵入式引用计数
  Storage storage_;                          // 数据缓冲区
  // ... 少一次指针间接
};

// myx-grad
class Value : public std::enable_shared_from_this<Value<T>> {
  // shared_ptr 控制块在堆上单独分配（make_shared 可合并）
  std::shared_ptr<Value<T>> m_left;  // 一次指针间接
  std::shared_ptr<Value<T>> m_right;
};
```

**PyTorch 为什么要侵入式引用计数？**

1. **性能**：每次引用/解引用少一次缓存未命中（控制块和对象分离时两次缓存行访问）
2. **与 Python 集成**：Python 对象本身就是侵入式引用计数，PyTorch 的 C++ Tensor 需要与 Python 的 `tp_dealloc` 无缝对接
3. **多线程安全性**：`std::atomic<int64_t>` 的 CAS 操作比 `shared_ptr` 的 atomic 引用计数更可控

> **参考**：PyTorch `c10/util/intrusive_ptr.h`、`aten/src/ATen/core/TensorImpl.h`

---

### TensorFlow（C++ 核心 + XLA 编译后端）

**架构层次：**

```
Python API (tf.keras, tf.function)
    ↕ SWIG / pybind
C++ Session / Eager Runtime
    ↕
XLA Compiler (HLO IR → LLVM IR → 机器码)
    ↕
Device Runtime (GPU/TPU)
```

**内存管理策略：**

TensorFlow 2.x 默认 Eager 模式下的策略与 PyTorch 类似，但在计算图模式（`@tf.function`）下有本质不同：

| 维度       | TensorFlow 图模式                                  | TensorFlow Eager 模式                 | myx-grad                            |
| ---------- | -------------------------------------------------- | ------------------------------------- | ----------------------------------- |
| 计算图构建 | 先构建完整 `tf.Graph`，再执行                      | 逐操作即时执行                        | 逐操作即时执行                      |
| 内存管理   | `TensorBuffer` 引用计数 + `Allocator` 自定义分配器 | `TensorInterface` 引用计数            | `shared_ptr` 引用计数               |
| 梯度记录   | `tf.GradientTape` 持有所有前向张量的引用           | `tf.GradientTape` selective recording | 所有节点天然持有 `m_left`/`m_right` |
| 图优化     | XLA 融合、死代码消除、内存复用                     | 无                                    | 无                                  |
| 内存复用   | XLA `BufferAssignment` 将多个张量映射到同一内存区  | 无                                    | 无                                  |

**关键差异：**

TensorFlow 图模式用 `tf.Graph` 构建完整计算图后再执行，可以全局优化内存分配（如 XLA 的 `BufferAssignment` 可以让多个张量复用同一块内存）。myx-grad 和 PyTorch 的 Eager 模式都是即时执行，无法进行全局内存规划。

> **参考**：TensorFlow `tensorflow/core/framework/tensor.h`、`tensorflow/compiler/xla/service/buffer_assignment.h`

---

### JAX / XLA（函数式 + 追踪编译）

**架构层次：**

```
Python API (jax.numpy, jax.grad)
    ↕ jaxpr tracing
JAX 中间表示 (jaxpr)
    ↕
XLA Compiler → HLO → 机器码
```

**内存管理策略：**

| 维度       | JAX                                                | myx-grad                         |
| ---------- | -------------------------------------------------- | -------------------------------- |
| 计算图构建 | 函数追踪（trace），构建 `jaxpr` 后丢弃 Python 对象 | 运行时构建节点图                 |
| 内存管理   | XLA `BufferAssignment` 全局规划 + 引用计数         | `shared_ptr` 逐节点管理          |
| 梯度计算   | 反向模式 AD 直接编译为 XLA HLO                     | 遍历 `shared_ptr` 图反向传播     |
| JIT 编译   | `jax.jit` 将追踪的 `jaxpr` 编译为 XLA              | 无 JIT                           |
| 函数式保证 | 纯函数，无副作用                                   | 允许 `m_grad` 原地累加（副作用） |

**关键洞察：**

JAX 的设计哲学是 "追踪 → 编译 → 执行"：Python 函数只执行一次来追踪操作，生成的 `jaxpr` 完全脱离 Python 对象。这意味着 **Python 层不需要维护计算图的内存**。XLA 后端看到的只是一组 HLO 指令，可以自由地做死代码消除、内存复用、算子融合等优化。

myx-grad 的设计更接近 PyTorch Eager 模式——每个操作立即构建节点并持有引用。这种方式更直观，但牺牲了全局优化机会。

> **参考**：JAX `jax/_src/core.py` (Trace、Tracer)、`jax/_src/ad_util.py`

---

### ggml / llama.cpp（零分配推理引擎）

**架构层次：**

```
C API (ggml_init, ggml_mul, ...)
    ↕
ggml_context (Arena 分配器)
    ↕
Tensor 元数据 (ggml_tensor) + 数据缓冲区 (ggml_backend_buffer)
```

**内存管理策略：**

| 维度       | ggml / llama.cpp                            | myx-grad                          |
| ---------- | ------------------------------------------- | --------------------------------- |
| 分配策略   | Arena/Bump 分配器，一次性分配所有张量内存   | `shared_ptr` 逐节点 `make_shared` |
| 生命周期   | `ggml_free_ctx` 一次性释放所有              | `shared_ptr` 引用计数逐个释放     |
| 计算图     | `ggml_cgraph` 构建后执行，不持有前向引用    | `Value::m_left/m_right` 持续持有  |
| 内存碎片   | 零碎片（Arena 连续分配）                    | 可能碎片化                        |
| 缓存友好性 | 极好（连续内存布局）                        | 差（节点散布堆上）                |
| 反向传播   | `ggml_backprop` 遍历 `cgraph`，就地写入梯度 | 遍历 `shared_ptr` 图写入 `m_grad` |

**ggml 的 Arena 分配器核心：**

```cpp
// ggml/src/ggml.c (simplified)
struct ggml_context {
  size_t mem_size;
  void * mem_buffer;      // 一大块连续内存
  size_t mem_buffer_used;  // bump allocator 指针
  // ... 只进不退，零碎片
};

ggml_tensor * ggml_new_tensor(ctx, type, ndims, shape) {
  // 从 Arena 中 bump 分配，O(1)，无 malloc
  size_t size = compute_size(type, shape);
  auto * tensor = (ggml_tensor *)((char *)ctx->mem_buffer + ctx->mem_buffer_used);
  ctx->mem_buffer_used += size;
  return tensor;
}
```

**ggml 为什么不用 `shared_ptr`？**

1. **推理场景不需要自动释放**：推理时计算图结构固定，所有中间张量在推理结束前都活着，不需要引用计数
2. **极致性能**：Arena 分配是 O(1) 的 bump pointer，比 `shared_ptr` 的引用计数 + 堆分配快几个数量级
3. **缓存友好**：所有 `ggml_tensor` 元数据连续排列，CPU 缓存命中率极高
4. **嵌入式友好**：无异常、无 RTTI、无堆分配器依赖，可在嵌入式设备上运行

**myx-grad 与 ggml 的本质区别：**

myx-grad 是 **训练框架**（需要构建动态计算图、反向传播、梯度累加），而 ggml 是 **推理框架**（计算图静态已知、只执行前向传播）。这决定了内存管理策略的根本不同：

- 训练需要灵活的图构建 → `shared_ptr` 引用计数
- 推理需要极致性能 → Arena 分配 + 静态图

#### 为什么训练需要灵活的图构建？

训练时，**每次前向传播构建的计算图结构可能不同**。以下用 myx-grad 的代码举例说明：

**例 1：条件分支 — 图结构随数据变化**

```python
# 训练：对不同的输入走不同的路径
def forward(x, threshold=0.0):
    if x.value > threshold:     # ← 运行时才决定走哪条路
        return x.relu()         # 构建 relu 节点
    else:
        return x.tanh()         # 构建 tanh 节点
# 每次调用可能构建不同结构的计算图
# → 需要 shared_ptr 动态创建/销毁节点
```

推理时图结构固定：要么只要 relu 路径，要么只要 tanh 路径，编译器可以提前知道。

**例 2：动态形状 — 同一模型处理不同长度**

```cpp
// 训练：不同 batch 的输入大小不同
Layer l1(3, 4);   // 输入维度 3
auto out1 = l1({1.0, 2.0, 3.0});    // 构建 3 个乘法节点

Layer l2(5, 4);   // 输入维度 5
auto out2 = l2({1,2,3,4,5});          // 构建 5 个乘法节点
// 节点数量在运行时确定 → 无法提前分配 Arena
```

推理时输入形状固定（或变化模式已知），Arena 可以预先分配。

**例 3：反向传播需要持有整个前向图**

```
前向传播（训练时）：
  a ──┐
      ├─ c ──┐
  b ──┘      ├─ e (loss)
             │
  d ─────────┘

反向传播（必须从 e 回溯到 a、b、d）：
  ∂e/∂c → ∂c/∂a, ∂c/∂b    ← 需要 c.m_left 指向 a, c.m_right 指向 b
  ∂e/∂d                     ← 需要 e.m_right 指向 d

→ 整个前向图的所有中间节点必须存活到反向传播完成
→ shared_ptr 保证：只要 loss 节点还活着，整条链都不会被释放
```

推理只做前向传播，**不需要保存中间节点**：

```
推理：a → c → e (输出)，中间的 c 用完即可丢弃
→ Arena 分配器可以用 bump pointer 覆盖 c 的内存
```

**例 4：梯度累加 — 节点必须是可变的**

```cpp
// myx-grad 的反向传播
void back_propagate() {
  // 梯度累加到同一个节点上
  // a 可能同时是 c1 和 c2 的输入
  m_left->m_grad += 1.0 * m_grad;   // 第一次累加
  // ... 另一条路径也累加
  m_left->m_grad += other_grad;      // 第二次累加
  // → 节点必须是可变的（mutable），不能用只读的 Arena 缓冲区
}
```

推理的中间结果只需读取一次，不需要原地累加。

**例 5：共享子节点 — DAG 不是树**

```cpp
auto a = val(2.0, "a");
auto b = a * a;   // a 同时是 b.m_left 和 b.m_right
// → a 被两个指针引用，shared_ptr 引用计数 = 2
// → 反向传播时 a 的梯度需要从两条路径累加
```

如果用 Arena 分配，节点没有引用计数，无法知道何时安全释放。

#### 为什么推理不需要灵活的图构建？

推理的两个关键特性使 Arena 分配成为可能：

1. **图结构完全已知**：模型加载后，所有算子和连接关系确定，不会有 "if/else 走不同路径" 的需求
2. **只做前向传播**：不需要保存中间梯度，中间结果用完即可释放

```
推理时的内存布局（Arena/Bump 分配器）：

内存地址  0x1000       0x1100      0x1200      0x1300
         ┌────────────┬───────────┬───────────┬──────────┐
Arena:   │ tensor_0   │ tensor_1  │ tensor_2  │   ...    │
         └────────────┴───────────┴───────────┴──────────┘
         ↑ bump_ptr 从左到右分配，推理结束后一次性释放

对比 myx-grad 训练时的内存布局：
         0x5000    0x3200    0x7100    0x4900    0x6600
         ┌────┐    ┌────┐    ┌────┐    ┌────┐    ┌────┐
堆上:    │ a  │    │ b  │    │ c  │    │ d  │    │ e  │
         └────┘    └────┘    └────┘    └────┘    └────┘
         ↑ 散布在堆上，缓存不友好，但每个节点有独立生命周期
```

> **参考**：ggml `include/ggml.h`、`src/ggml.c`
````