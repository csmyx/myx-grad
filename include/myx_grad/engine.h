#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace engine {

template <typename T> class scalar;

} // namespace engine

// std::hash specialization (must be in std namespace)
namespace std {
template <typename T> struct hash<engine::scalar<T>> {
  auto operator()(const engine::scalar<T> &s) const -> size_t;
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

template <typename T> struct node {
  static inline int count = 0;
  explicit node(std::string label, std::vector<const node<T> *> prev,
                op_t op = op_t::nop, std::string shape = "box")
      : label(std::move(label)), id("node_" + std::to_string(count)),
        shape(std::move(shape)), prev(std::move(prev)), m_op(op) {
    ++count;
  }

  void add_prev(const node<T> &prev_node) { prev.push_back(&prev_node); }

  std::string label;
  std::string id;
  std::string shape;
  std::vector<const node<T> *> prev;
  op_t m_op;
};

template <typename T> inline auto to_string(node<T> node) -> std::string {
  return fmt::format("{} [label=\"{} | {} | {}\", shape={}]", node.id, node.id,
                     node.label, to_string(node.m_op), node.shape);
}

template <typename T> struct graph {
  explicit graph(node<T> *root) : root(root) {}

  static auto display(graph g) -> std::string {
    const auto *cur = g.root;
    std::string node_str;
    std::string edge_str;
    std::vector<const node<T> *> stk;
    stk.push_back(cur);
    while (!stk.empty()) {
      cur = stk.back();
      stk.pop_back();
      node_str += to_string(*cur) + "\n";
      for (const node<T> *prev_node : cur->prev) {
        stk.push_back(prev_node);
        edge_str += prev_node->id + " -> " + cur->id + ";\n";
      }
    }
    return node_str + edge_str;
  }
  node<T> *root;
};

template <typename T> class scalar : public node<T> {
public:
  explicit scalar(T val)
      : node<T>("scalar[" + std::to_string(val) + "]", {}), m_value(val) {}
  scalar(T val, std::vector<const node<T> *> prev, op_t op)
      : node<T>("scalar[" + std::to_string(val) + "]", prev, op), m_value(val) {
  }

  auto value() const -> T { return m_value; }

  auto operator+(const scalar &other) const -> scalar {
    return scalar(m_value + other.m_value, {this, &other}, op_t::add);
  }
  auto operator-(const scalar &other) const -> scalar {
    return scalar(m_value - other.m_value, {this, &other}, op_t::sub);
  }
  auto operator*(const scalar &other) const -> scalar {
    return scalar(m_value * other.m_value, {this, &other}, op_t::mul);
  }
  auto operator/(const scalar &other) const -> scalar {
    return scalar(m_value / other.m_value, {this, &other}, op_t::div);
  }

  auto operator==(const scalar &other) const -> bool {
    return this->id == other.id;
  }

private:
  T m_value;
};

} // namespace engine

namespace std {
template <typename T>
auto hash<engine::scalar<T>>::operator()(const engine::scalar<T> &s) const
    -> size_t {
  return std::hash<T>{}(s.value());
}
} // namespace std
