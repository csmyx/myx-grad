#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace engine {

using float_t = double;

template <typename T> class Value;

using ValuePtr = std::shared_ptr<Value<float_t>>;

} // namespace engine

// std::hash specialization (must be in std namespace)
namespace std {
template <typename T> struct hash<engine::Value<T>> {
  auto operator()(const engine::Value<T> &s) const -> size_t {
    return std::hash<T>{}(s.value());
  }
};

template <typename T> struct hash<std::shared_ptr<engine::Value<T>>> {
  auto operator()(const std::shared_ptr<engine::Value<T>> &s) const -> size_t {
    return std::hash<T>{}(s->value());
  }
};
} // namespace std

namespace engine {

enum class op_t : std::uint8_t { nop, add, sub, mul, div, exp, tanh, pow };

inline auto to_string(op_t o) -> std::string {
  switch (o) {
  case op_t::add:
    return "+";
  case op_t::sub:
    return "-";
  case op_t::mul:
    return "*";
  case op_t::div:
    return "/";
  case op_t::exp:
    return "exp";
  case op_t::tanh:
    return "tanh";
  case op_t::nop:
    return "nop";
  case op_t::pow:
    return "pow";
  default:
    return "UNKNOWN";
  }
}

template <typename T> static auto format_v(T val) -> std::string {
  if constexpr (std::is_floating_point_v<T>) {
    return fmt::format("s[{:.6f}]", val);
  }
  return fmt::format("s[{}]", val);
}

template <typename T> struct Node {
  static inline int count = 0;
  explicit Node(T value, std::string id = "", std::string shape = "record")
      : m_value(std::move(value)),
        m_id(id.empty() ? "node_" + std::to_string(count) : id),
        m_shape(std::move(shape)) {
    ++count;
  }
  T m_value;
  std::string m_id;
  std::string m_shape;
};

template <typename T>
inline auto to_string(const engine::Value<T> &node) -> std::string {
  std::string label = fmt::format("{{id: {} | value: {} | grad: {} | op: {}",
                                  node.m_id, format_v(node.m_value),
                                  format_v(node.m_grad), to_string(node.m_op));
  if (node.m_op == op_t::pow) {
    label += fmt::format(" | exp: {}", format_v(node.m_op_params[0]));
  }
  if (!node.m_requires_grad) {
    label += " | frozen";
  }
  label += "}";
  return fmt::format("{} [label=\"{}\", shape={}]", node.m_id, label,
                     node.m_shape);
}

template <typename T> struct graph {
  explicit graph(std::shared_ptr<Value<T>> root) : m_root(std::move(root)) {}

  auto display() -> std::string {
    auto cur = m_root;
    std::string node_str;
    std::string edge_str;
    std::vector<std::shared_ptr<Value<T>>> stk;
    stk.push_back(cur);
    while (!stk.empty()) {
      cur = stk.back();
      stk.pop_back();
      node_str += to_string(*cur) + "\n";
      if (cur->m_left) {
        stk.push_back(cur->m_left);
        edge_str += cur->m_left->m_id + " -> " + cur->m_id + ";\n";
      }
      if (cur->m_right) {
        stk.push_back(cur->m_right);
        edge_str += cur->m_right->m_id + " -> " + cur->m_id + ";\n";
      }
    }
    return node_str + edge_str;
  }

  auto back_propagate() {
    std::vector<std::shared_ptr<Value<T>>> topo;
    std::unordered_set<std::shared_ptr<Value<T>>> visited;
    std::function<void(std::shared_ptr<Value<T>>)> build_topological;
    build_topological = [&](std::shared_ptr<Value<T>> s) -> void {
      if (!s || visited.contains(s) || !s->m_requires_grad) {
        return;
      }
      visited.insert(s);
      build_topological(s->m_left);
      build_topological(s->m_right);
      topo.push_back(s);
    };
    // set root gradient to 1.0
    m_root->m_grad = 1.0;
    build_topological(m_root);
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      (*it)->back_propagate();
    }
  }

  std::shared_ptr<Value<T>> m_root;
};

template <typename T>
class Value : public Node<T>, public std::enable_shared_from_this<Value<T>> {
public:
  static constexpr std::size_t k_max_op_params = 4;

  Value(std::string id, T val, float_t grad, op_t op,
        std::shared_ptr<Value<T>> left, std::shared_ptr<Value<T>> right)
      : Node<T>(val, id), m_value(val), m_grad(grad), m_op(op),
        m_left(std::move(left)), m_right(std::move(right)) {}

  explicit Value(T val) : Value("", val, 0.0, op_t::nop, nullptr, nullptr) {}

  Value(T val, std::string id)
      : Value(id, val, 0.0, op_t::nop, nullptr, nullptr) {}

  auto value() const -> T { return m_value; }
  auto with_id(std::string id) -> Value & {
    this->m_id = id;
    return *this;
  }

  auto set_requires_grad(bool v) -> Value & {
    m_requires_grad = v;
    return *this;
  }
  auto requires_grad() const -> bool { return m_requires_grad; }

  auto exp() -> std::shared_ptr<Value<T>> {
    auto value = std::exp(m_value);
    return std::make_shared<Value<T>>("", value, 0.0, op_t::exp,
                                      Value::shared_from_this(), nullptr);
  }

  auto tanh() -> std::shared_ptr<Value<T>> {
    auto exp2x = std::exp(2 * m_value);
    auto value = (exp2x - 1) / (exp2x + 1);
    return std::make_shared<Value<T>>("", value, 0.0, op_t::tanh,
                                      Value::shared_from_this(), nullptr);
  }

  // pow(base, exponent) — exponent stored in m_op_params[0]
  //  d(base^exp)/d(base) = exp * base^(exp-1)
  auto pow(T exponent) -> std::shared_ptr<Value<T>> {
    auto res = std::make_shared<Value<T>>("", std::pow(m_value, exponent), 0.0,
                                          op_t::pow, Value::shared_from_this(),
                                          nullptr);
    res->m_op_params[0] = exponent;
    return res;
  }

  auto operator==(const Value<T> &other) const -> bool {
    return this->m_id == other.m_id;
  }

  auto back_propagate() {
    switch (m_op) {
    case op_t::add: {
      if (m_left && m_left->m_requires_grad) {
        m_left->m_grad += 1.0 * m_grad;
      }
      if (m_right && m_right->m_requires_grad) {
        m_right->m_grad += 1.0 * m_grad;
      }
    }
      return;
    case op_t::sub: {
      if (m_left && m_left->m_requires_grad) {
        m_left->m_grad += 1.0 * m_grad;
      }
      if (m_right && m_right->m_requires_grad) {
        m_right->m_grad += -1.0 * m_grad;
      }
    }
      return;
    case op_t::mul: {
      if (m_left && m_left->m_requires_grad) {
        m_left->m_grad += m_right->m_value * m_grad;
      }
      if (m_right && m_right->m_requires_grad) {
        m_right->m_grad += m_left->m_value * m_grad;
      }
    }
      return;
    case op_t::div: {
      if (m_left && m_left->m_requires_grad) {
        m_left->m_grad += (1.0 / m_right->m_value) * m_grad;
      }
      if (m_right && m_right->m_requires_grad) {
        m_right->m_grad += (-m_value / m_right->m_value) * m_grad;
      }
    }
      return;
    case op_t::tanh: {
      if (m_left && m_left->m_requires_grad) {
        m_left->m_grad += (1 - m_value * m_value) * m_grad;
      }
    }
      return;
    case op_t::exp: {
      if (m_left && m_left->m_requires_grad) {
        m_left->m_grad += m_value * m_grad;
      }
    }
      return;
    case op_t::pow: {
      if (m_left && m_left->m_requires_grad) {
        T exponent = m_op_params[0];
        m_left->m_grad +=
            exponent * std::pow(m_left->m_value, exponent - 1) * m_grad;
      }
    }
      return;
    case op_t::nop: {
    }
      return;
    default:
      fmt::print("Unhandled operation\n");
      std::abort();
    }
    return;
  }

  T m_value{};
  float_t m_grad{}; // gradient
  op_t m_op{};
  std::shared_ptr<Value<T>> m_left;
  std::shared_ptr<Value<T>> m_right;
  bool m_requires_grad = true;
  std::array<T, k_max_op_params>
      m_op_params{}; // per-op parameters (pow: exponent,
                     // leaky_relu: slope, etc.)
};

// --- Free operator overloads for shared_ptr<Value<T>> ---
// Enables: auto c = a + b;  (where a, b are shared_ptr<Value<T>>)

template <typename T>
auto operator+(const std::shared_ptr<Value<T>> &lhs,
               const std::shared_ptr<Value<T>> &rhs)
    -> std::shared_ptr<Value<T>> {
  return std::make_shared<Value<T>>("", lhs->m_value + rhs->m_value, 0.0,
                                    op_t::add, lhs, rhs);
}

template <typename T>
auto operator-(const std::shared_ptr<Value<T>> &lhs,
               const std::shared_ptr<Value<T>> &rhs)
    -> std::shared_ptr<Value<T>> {
  return std::make_shared<Value<T>>("", lhs->m_value - rhs->m_value, 0.0,
                                    op_t::sub, lhs, rhs);
}

template <typename T>
auto operator*(const std::shared_ptr<Value<T>> &lhs,
               const std::shared_ptr<Value<T>> &rhs)
    -> std::shared_ptr<Value<T>> {
  return std::make_shared<Value<T>>("", lhs->m_value * rhs->m_value, 0.0,
                                    op_t::mul, lhs, rhs);
}

template <typename T>
auto operator/(const std::shared_ptr<Value<T>> &lhs,
               const std::shared_ptr<Value<T>> &rhs)
    -> std::shared_ptr<Value<T>> {
  return std::make_shared<Value<T>>("", lhs->m_value / rhs->m_value, 0.0,
                                    op_t::div, lhs, rhs);
}

struct Neuron {
  explicit Neuron(int input_sz) : m_input_sz(input_sz) {}
  int m_input_sz;
};

struct Tensor : private std::vector<std::shared_ptr<Value<float_t>>> {
private:
  using Base = std::vector<std::shared_ptr<Value<float_t>>>;

public:
  using Base::begin;
  using Base::const_iterator;
  using Base::end;
  using Base::iterator;
  using Base::operator[];
  using Base::emplace_back;
  using Base::push_back;
  using Base::reserve;
  using Base::size;

  Tensor() = default;

  /// Create N zero-initialized Values on the heap
  static auto zeros(size_t n) -> Tensor {
    Tensor t;
    t.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      t.emplace_back(std::make_shared<Value<float_t>>(0.0));
    }
    return t;
  }

  /// Create N Values with Xavier/Glorot uniform initialization
  static auto xavier_uniform(size_t n, size_t fan_in, size_t fan_out,
                             std::mt19937 &gen) -> Tensor {
    float_t limit = std::sqrt(6.0 / static_cast<float_t>(fan_in + fan_out));
    std::uniform_real_distribution<float_t> dist(-limit, limit);
    Tensor t;
    t.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      t.emplace_back(std::make_shared<Value<float_t>>(dist(gen)));
    }
    return t;
  }

  /// Create from raw float_t values (each allocated on the heap)
  static auto from_values(std::initializer_list<float_t> vals) -> Tensor {
    Tensor t;
    t.reserve(vals.size());
    for (auto v : vals) {
      t.emplace_back(std::make_shared<Value<float_t>>(v));
    }
    return t;
  }

  /// Create from a vector of float_t (each allocated on the heap)
  static auto from_values(const std::vector<float_t> &vals) -> Tensor {
    Tensor t;
    t.reserve(vals.size());
    for (auto v : vals) {
      t.emplace_back(std::make_shared<Value<float_t>>(v));
    }
    return t;
  }
};

struct Layer {
  Layer(size_t input_sz, size_t output_sz)
      : m_input_sz(input_sz), m_output_sz(output_sz) {
    std::mt19937 gen(std::random_device{}());
    m_weights =
        Tensor::xavier_uniform(input_sz * output_sz, input_sz, output_sz, gen);
    m_biases = Tensor::zeros(output_sz);
  }
  size_t m_input_sz = 0;
  size_t m_output_sz = 0;
  Tensor m_weights;
  Tensor m_biases;

  auto operator()(std::vector<float_t> inputs) -> Tensor {
    assert(inputs.size() == m_input_sz);
    Tensor outputs;
    outputs.reserve(m_output_sz);
    for (size_t i = 0; i < m_output_sz; ++i) {
      auto sum = m_biases[i];
      for (size_t j = 0; j < m_input_sz; ++j) {
        auto &w = m_weights[(i * m_input_sz) + j];
        auto in_val = std::make_shared<Value<float_t>>(inputs[j]);
        sum = w * in_val + sum;
      }
      outputs.push_back(sum);
    }
    return outputs;
  }

  auto backward() {}
};

struct MLP {
  MLP(size_t input_sz, std::vector<size_t> output_sz) {
    auto prev = input_sz;
    for (size_t output : output_sz) {
      m_layers.push_back(std::make_unique<Layer>(prev, output));
      prev = output;
    }
  }
  auto backward() {}
  std::vector<std::unique_ptr<Layer>> m_layers;
};

} // namespace engine
