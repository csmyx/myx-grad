#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fmt/format.h>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

namespace engine {

using float_t = double;

template <typename T> class Value;

} // namespace engine

// std::hash specialization (must be in std namespace)
namespace std {
template <typename T> struct hash<engine::Value<T>> {
  auto operator()(const engine::Value<T> &s) const -> size_t;
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
inline auto to_string(engine::Value<T> node) -> std::string {
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
  explicit graph(Value<T> *root) : m_root(root) {}

  auto display() -> std::string {
    Value<T> *cur = m_root;
    std::string node_str;
    std::string edge_str;
    std::vector<Value<T> *> stk;
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
    std::vector<engine::Value<T> *> topo;
    std::unordered_set<engine::Value<T> *> visited;
    std::function<void(engine::Value<T> *)> build_topological;
    build_topological = [&](engine::Value<T> *s) -> void {
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

  engine::Value<T> *m_root;
};

template <typename T> class Value : public Node<T> {
public:
  static constexpr std::size_t k_max_op_params = 4;

  Value(std::string id, T val, float_t grad, op_t op, Value<T> *left,
        Value<T> *right)
      : Node<T>(val, id), m_value(val), m_op(op), m_grad(grad), m_left(left),
        m_right(right) {}

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

  auto exp() -> Value {
    auto value = std::exp(m_value);
    auto res = Value("", value, 0.0, op_t::exp, this, nullptr);
    return res;
  }

  auto tanh() -> Value {
    auto exp2x = std::exp(2 * m_value);
    auto value = (exp2x - 1) / (exp2x + 1);
    auto res = Value("", value, 0.0, op_t::tanh, this, nullptr);
    return res;
  }

  auto operator+(this Value &self, Value &other) -> Value {
    auto res =
        Value("", self.m_value + other.m_value, 0.0, op_t::add, &self, &other);
    return res;
  }
  auto operator-(this Value &self, Value &other) -> Value {
    auto res =
        Value("", self.m_value - other.m_value, 0.0, op_t::sub, &self, &other);
    return res;
  }
  auto operator*(this Value &self, Value &other) -> Value {
    auto res =
        Value("", self.m_value * other.m_value, 0.0, op_t::mul, &self, &other);
    return res;
  }
  auto operator/(this Value &self, Value &other) -> Value {
    auto res =
        Value("", self.m_value / other.m_value, 0.0, op_t::div, &self, &other);
    return res;
  }
  // pow(base, exponent) — exponent stored in m_op_params[0]
  //  d(base^exp)/d(base) = exp * base^(exp-1)
  auto pow(T exponent) -> Value {
    auto res =
        Value("", std::pow(m_value, exponent), 0.0, op_t::pow, this, nullptr);
    res.m_op_params[0] = exponent;
    return res;
  }

  auto operator==(Value &other) const -> bool {
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
  Value<T> *m_left = nullptr;
  Value<T> *m_right = nullptr;
  bool m_requires_grad = true;
  T m_op_params[k_max_op_params]{}; // per-op parameters (pow: exponent,
                                    // leaky_relu: slope, etc.)
};

} // namespace engine

namespace std {
template <typename T>
auto hash<engine::Value<T>>::operator()(const engine::Value<T> &s) const
    -> size_t {
  return std::hash<T>{}(s.value());
}
} // namespace std
