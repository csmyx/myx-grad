#pragma once

#include <fmt/format.h>
#include <myx_grad/arena.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine {

using float_t = double;

template <typename T>
class Value;

}  // namespace engine

// std::hash specialization (must be in std namespace)
namespace std {
template <typename T>
struct hash<engine::Value<T>> {
    auto operator()(const engine::Value<T> &s) const -> size_t;
};
}  // namespace std

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

template <typename T>
static auto format_v(T val) -> std::string {
    if constexpr (std::is_floating_point_v<T>) {
        return fmt::format("s[{:.6f}]", val);
    }
    return fmt::format("s[{}]", val);
}

template <typename T>
struct Node {
    static inline int count = 0;
    explicit Node(const std::string &id = "", std::string shape = "record")
        : id_(id.empty() ? "node_" + std::to_string(count) : id), shape_(std::move((shape))) {
        ++count;
    }
    std::string id_;
    std::string shape_;
};

template <typename T>
inline auto to_string(const engine::Value<T> &node) -> std::string {
    std::string label = fmt::format("{{id: {} | value: {} | grad: {} | op: {}", node.id_, format_v(node.value_),
                                    format_v(node.grad_), to_string(node.op_));
    if (node.op_ == op_t::pow) {
        label += fmt::format(" | exp: {}", format_v(node.op_params_[0]));
    }
    if (!node.requires_grad_) {
        label += " | frozen";
    }
    label += "}";
    return fmt::format("{} [label=\"{}\", shape={}]", node.id_, label, node.shape_);
}

template <typename T>
class Graph {
public:
    Graph() = default;
    Graph(const Graph &) = delete;
    auto operator=(const Graph &) -> Graph & = delete;
    ~Graph() = default;

private:
    util::Arena arena_;
    std::vector<Value<T> *> nodes_;

    template <typename... Args>
    auto alloc_value(Args &&...args) -> Value<T> * {
        void *value_ptr = arena_.alloc(sizeof(Value<T>));
        auto *value = new (value_ptr) Value<T>(std::forward<Args>(args)...);
        nodes_.push_back(value);
        return value;
    }

public:
    auto leaf(T val, std::string id = "") -> Value<T> & {
        auto *node = alloc_value(val, id, this);
        return *node;
    }

    auto add(Value<T> &left, Value<T> &right) -> Value<T> & {
        assert(left.graph_ == this);
        assert(right.graph_ == this);
        auto *node = alloc_value("", left.value_ + right.value_, 0.0, op_t::add, &left, &right, this);
        return *node;
    }

    auto sub(Value<T> &left, Value<T> &right) -> Value<T> & {
        assert(left.graph_ == this);
        assert(right.graph_ == this);
        auto *node = alloc_value("", left.value_ - right.value_, 0.0, op_t::sub, &left, &right, this);
        return *node;
    }

    auto mul(Value<T> &left, Value<T> &right) -> Value<T> & {
        assert(left.graph_ == this);
        assert(right.graph_ == this);
        auto *node = alloc_value("", left.value_ * right.value_, 0.0, op_t::mul, &left, &right, this);
        return *node;
    }

    auto div(Value<T> &left, Value<T> &right) -> Value<T> & {
        assert(left.graph_ == this);
        assert(right.graph_ == this);
        auto *node = alloc_value("", left.value_ / right.value_, 0.0, op_t::div, &left, &right, this);
        return *node;
    }

    auto pow(Value<T> &base, T exponent) -> Value<T> & {
        assert(base.graph_ == this);
        auto *node = alloc_value("", std::pow(base.value_, exponent), 0.0, op_t::pow, &base, nullptr, this);
        node->op_params_[0] = exponent;
        return *node;
    }

    auto tanh(Value<T> &x) -> Value<T> & {
        assert(x.graph_ == this);
        auto exp2x = std::exp(2 * x.value_);
        auto val = (exp2x - 1) / (exp2x + 1);
        auto *node = alloc_value("", val, 0.0, op_t::tanh, &x, nullptr, this);
        return *node;
    }

    auto exp(Value<T> &x) -> Value<T> & {
        assert(x.graph_ == this);
        auto *node = alloc_value("", std::exp(x.value_), 0.0, op_t::exp, &x, nullptr, this);
        return *node;
    }

    // ---- display ----

    auto display(Value<T> &root) -> std::string {
        Value<T> *cur = &root;
        std::string node_str;
        std::string edge_str;
        std::vector<Value<T> *> stk;
        std::unordered_set<Value<T> *> seen;
        stk.push_back(cur);
        while (!stk.empty()) {
            cur = stk.back();
            stk.pop_back();
            if (seen.contains(cur)) {
                continue;
            }
            seen.insert(cur);
            node_str += to_string(*cur) + "\n";
            if (cur->left_) {
                stk.push_back(cur->left_);
                edge_str += cur->left_->id_ + " -> " + cur->id_ + ";\n";
            }
            if (cur->right_) {
                stk.push_back(cur->right_);
                edge_str += cur->right_->id_ + " -> " + cur->id_ + ";\n";
            }
        }
        return node_str + edge_str;
    }

    // ---- back_propagate ----

    auto back_propagate(Value<T> &root) {
        std::vector<Value<T> *> topo;
        std::unordered_set<Value<T> *> visited;
        std::function<void(Value<T> *)> build_topological;
        build_topological = [&](Value<T> *s) -> void {
            if (!s || visited.contains(s) || !s->requires_grad_) {
                return;
            }
            visited.insert(s);
            build_topological(s->left_);
            build_topological(s->right_);
            topo.push_back(s);
        };
        root.grad_ = 1.0;
        build_topological(&root);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->back_propagate();
        }
    }
};

template <typename T>
auto operator+(Value<T> &left, Value<T> &right) -> Value<T> & {
    assert(left.graph_ == right.graph_);
    return left.graph_->add(left, right);
}

template <typename T>
auto operator-(Value<T> &left, Value<T> &right) -> Value<T> & {
    assert(left.graph_ == right.graph_);
    return left.graph_->sub(left, right);
}

template <typename T>
auto operator*(Value<T> &left, Value<T> &right) -> Value<T> & {
    assert(left.graph_ == right.graph_);
    return left.graph_->mul(left, right);
}

template <typename T>
auto operator/(Value<T> &left, Value<T> &right) -> Value<T> & {
    assert(left.graph_ == right.graph_);
    return left.graph_->div(left, right);
}

template <typename T>
class Value : public Node<T> {
    friend class Graph<T>;

    Value(std::string id, T val, float_t grad, op_t op, Value<T> *left, Value<T> *right, Graph<T> *graph)
        : Node<T>(id), value_(val), grad_(grad), op_(op), left_(left), right_(right), graph_(graph) {}

    explicit Value(T val) : Value("", val, 0.0, op_t::nop, nullptr, nullptr, nullptr) {}

    Value(T val, std::string id, Graph<T> *graph) : Value(id, val, 0.0, op_t::nop, nullptr, nullptr, graph) {}

    Value(T val, std::string id) : Value(id, val, 0.0, op_t::nop, nullptr, nullptr, nullptr) {}

public:
    static constexpr std::size_t k_max_op_params = 4;

    // Non-copyable, non-movable: Value holds internal pointers (left_, right_)
    // that would dangle if the object were moved or copied. All Value objects
    // must be created through graph<T> factory methods, which own the arena.
    Value(const Value &) = delete;
    auto operator=(const Value &) -> Value & = delete;
    Value(Value &&) = delete;
    auto operator=(Value &&) -> Value & = delete;
    ~Value() = default;

    auto value() const -> T {
        return value_;
    }
    auto with_id(std::string id) -> Value & {
        this->id_ = id;
        return *this;
    }

    auto set_requires_grad(bool v) -> Value & {
        requires_grad_ = v;
        return *this;
    }
    auto requires_grad() const -> bool {
        return requires_grad_;
    }

    auto operator==(Value &other) const -> bool {
        return this->id_ == other.id_;
    }

    auto back_propagate() {
        switch (op_) {
            case op_t::add: {
                if (left_ && left_->requires_grad_) {
                    left_->grad_ += 1.0 * grad_;
                }
                if (right_ && right_->requires_grad_) {
                    right_->grad_ += 1.0 * grad_;
                }
            }
                return;
            case op_t::sub: {
                if (left_ && left_->requires_grad_) {
                    left_->grad_ += 1.0 * grad_;
                }
                if (right_ && right_->requires_grad_) {
                    right_->grad_ += -1.0 * grad_;
                }
            }
                return;
            case op_t::mul: {
                if (left_ && left_->requires_grad_) {
                    left_->grad_ += right_->value_ * grad_;
                }
                if (right_ && right_->requires_grad_) {
                    right_->grad_ += left_->value_ * grad_;
                }
            }
                return;
            case op_t::div: {
                if (left_ && left_->requires_grad_) {
                    left_->grad_ += (1.0 / right_->value_) * grad_;
                }
                if (right_ && right_->requires_grad_) {
                    right_->grad_ += (-value_ / right_->value_) * grad_;
                }
            }
                return;
            case op_t::tanh: {
                if (left_ && left_->requires_grad_) {
                    left_->grad_ += (1 - value_ * value_) * grad_;
                }
            }
                return;
            case op_t::exp: {
                if (left_ && left_->requires_grad_) {
                    left_->grad_ += value_ * grad_;
                }
            }
                return;
            case op_t::pow: {
                if (left_ && left_->requires_grad_) {
                    T exponent = op_params_[0];
                    left_->grad_ += exponent * std::pow(left_->value_, exponent - 1) * grad_;
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

    T value_{};
    float_t grad_{};  // gradient
    op_t op_{};
    Value<T> *left_ = nullptr;
    Value<T> *right_ = nullptr;
    bool requires_grad_ = true;
    T op_params_[k_max_op_params]{};  // per-op parameters (pow: exponent,
                                      // leaky_relu: slope, etc.)
    Graph<T> *graph_ = nullptr;
};


}  // namespace engine

namespace std {
template <typename T>
auto hash<engine::Value<T>>::operator()(const engine::Value<T> &s) const -> size_t {
    return std::hash<T>{}(s.value());
}
}  // namespace std
