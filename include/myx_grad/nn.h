#pragma once

#include <myx_grad/engine.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace nn {

using dtype = double;
using value_t = engine::Value<dtype>;
using value_ptr = engine::Value<dtype> *;
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

// ---- random helpers ----

inline auto random_float_v(dtype min, dtype max) -> dtype {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<dtype> dist(min, max);
    return dist(gen);
}

// Xavier/Glorot uniform initialization: keeps variance roughly constant
// across layers. limit = sqrt(6 / (fan_in + fan_out))
inline auto xavier_init(size_t fan_in, size_t fan_out) -> dtype {
    dtype limit = std::sqrt(6.0 / static_cast<dtype>(fan_in + fan_out));
    return random_float_v(-limit, limit);
}

// ---- activation functions ----

enum class Activation { None, Tanh };

inline auto apply_activation(engine::Graph<dtype> &graph, value_ptr x, Activation act) -> value_ptr {
    switch (act) {
        case Activation::Tanh:
            return &graph.tanh(*x);
        case Activation::None:
        default:
            return x;
    }
}

// ---- Module base ----

struct Module {
    virtual ~Module() = default;
    virtual auto parameters() const -> std::vector<value_ptr> {
        return {};
    }
};

// ---- Neuron ----

class Neuron : public Module {
private:
    std::vector<value_ptr> weights_;
    value_ptr bias_;
    Activation act_;

public:
    Neuron(engine::Graph<dtype> &graph, size_t input_size, size_t fan_out, Activation act = Activation::Tanh,
           dtype bias_val = 0.0)
        : bias_{&graph.leaf(bias_val, "", /*requires_grad=*/true)}, act_{act} {
        weights_.reserve(input_size);
        for (size_t i = 0; i < input_size; ++i) {
            auto w = xavier_init(input_size, fan_out);
            weights_.emplace_back(&graph.leaf(w, "", /*requires_grad=*/true));
        }
    }

    // Balanced tree reduction: builds terms then sums pairwise.
    // Reduces graph depth from O(n) to O(log n), speeding up backprop.
    auto operator()(std::vector<value_ptr> &inputs) -> value_ptr {
        assert(inputs.size() == weights_.size());

        // Collect all terms: bias + w_i * x_i
        std::vector<value_ptr> terms;
        terms.reserve(inputs.size() + 1);
        terms.push_back(bias_);
        for (size_t i = 0; i < inputs.size(); ++i) {
            terms.push_back(&(*weights_[i] * *inputs[i]));
        }

        // Balanced pairwise reduction
        while (terms.size() > 1) {
            std::vector<value_ptr> next;
            next.reserve((terms.size() + 1) / 2);
            for (size_t i = 0; i + 1 < terms.size(); i += 2) {
                next.push_back(&(*terms[i] + *terms[i + 1]));
            }
            if (terms.size() % 2 == 1) {
                next.push_back(terms.back());
            }
            terms = std::move(next);
        }

        return apply_activation(*terms[0]->graph_, terms[0], act_);
    }

    auto parameters() const -> std::vector<value_ptr> override {
        auto params = weights_;
        params.push_back(bias_);
        return params;
    }
};

// ---- Layer ----

class Layer : public Module {
private:
    std::vector<Neuron> neurons_;
    Activation act_;

public:
    // input_size: number of inputs, output_size: number of neurons
    // act: activation for this layer (use None for output/regression layer)
    Layer(engine::Graph<dtype> &graph, size_t input_size, size_t output_size, Activation act = Activation::Tanh)
        : act_{act} {
        neurons_.reserve(output_size);
        for (size_t i = 0; i < output_size; ++i) {
            neurons_.emplace_back(graph, input_size, output_size, act);
        }
    }

    auto operator()(std::vector<value_ptr> &inputs) -> std::vector<value_ptr> {
        std::vector<value_ptr> outputs;
        outputs.reserve(neurons_.size());
        for (auto &neuron : neurons_) {
            outputs.emplace_back(neuron(inputs));
        }
        return outputs;
    }

    auto parameters() const -> std::vector<value_ptr> override {
        std::vector<value_ptr> params;
        for (const auto &neuron : neurons_) {
            auto np = neuron.parameters();
            params.insert(params.end(), np.begin(), np.end());
        }
        return params;
    }

    auto size() const -> size_t {
        return neurons_.size();
    }
};

// ---- MLP ----

class MLP : public Module {
private:
    std::vector<Layer> layers_;
    engine::Graph<dtype> *graph_ = nullptr;

public:
    // layer_sizes: e.g. {10, 5, 1} for 3-layer MLP
    // Hidden layers use Tanh activation; the last layer uses None (linear output)
    explicit MLP(engine::Graph<dtype> &graph, size_t input_size, std::initializer_list<size_t> layer_sizes)
        : graph_(&graph) {
        auto prev_size = input_size;
        size_t idx = 0;
        const size_t num_layers = layer_sizes.size();
        for (auto size : layer_sizes) {
            // Last layer is linear (no activation) for regression
            bool is_last = (idx == num_layers - 1);
            Activation act = is_last ? Activation::None : Activation::Tanh;
            layers_.emplace_back(graph, prev_size, size, act);
            prev_size = size;
            ++idx;
        }
    }

    // Forward pass for a single sample
    auto operator()(std::vector<dtype> &inputs) -> value_ptr {
        auto outputs = graph_->create_tensor(inputs);
        for (auto &layer : layers_) {
            outputs = layer(outputs);
        }
        assert(outputs.size() == 1);
        return outputs[0];
    }

    // Forward pass for a single sample (from pre-created input tensor)
    auto forward(std::vector<value_ptr> &inputs) -> std::vector<value_ptr> {
        auto outputs = inputs;
        for (auto &layer : layers_) {
            outputs = layer(outputs);
        }
        return outputs;
    }

    // Batched forward pass: processes all samples, returns one output per sample
    auto forward_batch(std::vector<std::vector<dtype>> &batch) -> Tensor {
        Tensor outputs;
        outputs.reserve(batch.size());
        for (auto &sample : batch) {
            outputs.push_back((*this)(sample));
        }
        return outputs;
    }

    auto parameters() const -> std::vector<value_ptr> override {
        std::vector<value_ptr> params;
        for (const auto &layer : layers_) {
            auto lp = layer.parameters();
            params.insert(params.end(), lp.begin(), lp.end());
        }
        return params;
    }
};

}  // namespace nn