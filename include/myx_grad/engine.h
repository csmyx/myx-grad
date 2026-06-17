#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fmt/format.h>
#include <functional>
#include <string>
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

enum class op_t : std::uint8_t { nop, add, sub, mul, div };

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
  case op_t::nop:
    return "<nop>";
  default:
    return "UNKNOWN";
  }
}

template <typename T> struct Node {
  static inline int count = 0;
  explicit Node(std::string label, std::string id = "",
                std::string shape = "box")
      : m_label(std::move(label)),
        m_id(id.empty() ? "node_" + std::to_string(count) : id),
        m_shape(std::move(shape)) {
    ++count;
  }
  std::string m_label;
  std::string m_id;
  std::string m_shape;
};

template <typename T>
inline auto to_string(engine::Value<T> node) -> std::string {
  return fmt::format(
      "{} [label=\"id: {} | value: {} | grad: {} | op: {} \", shape={}]",
      node.m_id, node.m_id, node.m_label, node.m_grad, to_string(node.m_op),
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
    std::vector<engine::Value<T> *> v;
    std::function<void(engine::Value<T> *)> build_tokological;
    build_tokological = [&](engine::Value<T> *s) -> void {
      if (!s) {
        return;
      }
      build_tokological(s->m_left);
      build_tokological(s->m_right);
      v.push_back(s);
    };
    m_root->m_grad = 1.0;
    build_tokological(m_root);
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
      (*it)->back_propagate();
    }
  }

  engine::Value<T> *m_root;
};

template <typename T> class Value : public Node<T> {
public:
  static auto label(T val) -> std::string {
    if constexpr (std::is_floating_point_v<T>) {
      return fmt::format("s[{:.3f}]", val);
    } else {
      return fmt::format("s[{}]", val);
    }
  }
  Value(std::string id, T val, float_t grad, op_t op, Value<T> *left,
        Value<T> *right)
      : Node<T>(label(val), id), m_value(val), m_op(op), m_grad(grad),
        m_left(left), m_right(right) {}

  explicit Value(T val) : Value("", val, 0.0, op_t::nop, nullptr, nullptr) {}

  Value(T val, std::string id)
      : Value(id, val, 0.0, op_t::nop, nullptr, nullptr) {}

  auto value() const -> T { return m_value; }
  auto with_id(std::string id) -> Value & {
    this->m_id = id;
    return *this;
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
    fmt::print("Division operation not implemented\n");
    auto res =
        Value("", self.m_value / other.m_value, 0.0, op_t::div, &self, &other);
    return res;
  }

  auto operator==(Value &other) const -> bool {
    return this->m_id == other.m_id;
  }

  auto back_propagate() {
    switch (m_op) {
    case op_t::add:
      if (m_left) {
        m_left->m_grad += 1.0 * m_grad;
      }
      if (m_right) {
        m_right->m_grad += 1.0 * m_grad;
      }
      return;
    case op_t::sub:
      if (m_left) {
        m_left->m_grad += -1.0 * m_grad;
      }
      if (m_right) {
        m_right->m_grad += -1.0 * m_grad;
      }
      return;
    case op_t::mul:
      if (m_left) {
        m_left->m_grad += m_right->m_value * m_grad;
      }
      if (m_right) {
        m_right->m_grad += m_left->m_value * m_grad;
      }
      return;
    case op_t::div:
      fmt::print("Division operation not implemented\n");
      return;
    case op_t::nop:
      return;
    default:
      fmt::print("Unhandled operation\n");
      std::abort();
      return;
    }
  }

  T m_value;
  float_t m_grad; // gradient
  op_t m_op;
  Value<T> *m_left;
  Value<T> *m_right;
};

} // namespace engine

namespace std {
template <typename T>
auto hash<engine::Value<T>>::operator()(const engine::Value<T> &s) const
    -> size_t {
  return std::hash<T>{}(s.value());
}
} // namespace std
