#pragma once

#include <myx_grad/engine.h>

#include <random>
#include <vector>

namespace nn {

using dtype = double;
using value_t = engine::Value<dtype>;
using value_ptr = engine::Value<dtype> *;
// using Tensor = engine::Graph<dtype>::Tensor;
using Tensor = std::vector<engine::Value<dtype> *>;

inline auto to_string(const Tensor &tensor) -> std::string {
    std::string result = "[";
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += engine::format_v(tensor[i]->value());
    }
    result += "]";
    return result;
}

// class Tensor {
// private:
//     std::vector<value_ptr> data_;

// public:
//     Tensor() = default;
//     Tensor(std::initializer_list<value_ptr> init) : data_(init) {}
//     auto size() const -> size_t { return data_.size(); }
//     auto operator[](size_t i) -> value_ptr & { return data_[i]; }
//     auto begin() -> typename std::vector<value_ptr>::iterator { return data_.begin(); }
//     auto end() -> typename std::vector<value_ptr>::iterator { return data_.end(); }

//     auto add(const Tensor &other) -> Tensor {
//         Tensor result;
//         for (size_t i = 0; i < size(); ++i) {
//             result.data_.emplace_back(&(*data_[i] + *other.data_[i]));
//         }
//         return result;
//     }
// };

inline auto random_float_v(dtype min, dtype max) -> dtype {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<dtype> dist(min, max);
    return dist(gen);
}

struct Module {
    // auto zero_grad() {
    // }
    // static auto parameters() -> std::vector<value_ptr> {
    //     std::vector<value_ptr> params;
    //     return params;
    // }
};

class Neuron : public Module {
private:
    std::vector<value_ptr> weights_;
    value_ptr bias_;

public:
    Neuron(engine::Graph<dtype> &graph, size_t input_size, dtype bias_val = 0.0)
        : bias_{&graph.leaf(bias_val, "", /*is_parameter=*/true)} {
        weights_.reserve(input_size);
        for (size_t i = 0; i < input_size; ++i) {
            auto weight_v = random_float_v(0, 1);
            weights_.emplace_back(&graph.leaf(weight_v, "", /*is_parameter=*/true));
        }
    }
    auto operator()(std::vector<value_ptr> &inputs) -> value_ptr {
        auto *result = bias_;
        for (size_t i = 0; i < inputs.size(); ++i) {
            result = &(*result + *weights_[i] * *inputs[i]);
        }
        return result;
    }
};

class Layer : public Module {
private:
    std::vector<Neuron> neurons_;

public:
    Layer(engine::Graph<dtype> &graph, size_t input_size, size_t output_size) {
        neurons_.reserve(output_size);
        for (size_t i = 0; i < output_size; ++i) {
            neurons_.emplace_back(graph, input_size);
        }
    }
    auto operator()(std::vector<value_ptr> inputs) -> std::vector<value_ptr> {
        std::vector<value_ptr> outputs;
        outputs.reserve(neurons_.size());
        for (auto &neuron : neurons_) {
            outputs.emplace_back(neuron(inputs));
        }
        return outputs;
    }
};

class MLP : public Module {
private:
    std::vector<Layer> layers_;
    engine::Graph<dtype> *graph_ = nullptr;
    value_ptr root_ = nullptr;

public:
    explicit MLP(engine::Graph<dtype> &graph, size_t input_size, std::initializer_list<size_t> layer_sizes)
        : graph_(&graph) {
        auto prev_size = input_size;
        for (auto size : layer_sizes) {
            layers_.emplace_back(graph, prev_size, size);
            prev_size = size;
        }
    }
    auto operator()(std::vector<dtype> &inputs) -> value_ptr {
        auto outputs = graph_->create_tensor(inputs);
        for (auto &layer : layers_) {
            outputs = layer(outputs);
        }
        assert(outputs.size() == 1);
        root_ = outputs[0];
        return root_;
    }
};

}  // namespace nn