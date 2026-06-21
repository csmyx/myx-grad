#include <fmt/core.h>
#include <myx_grad/nn.h>

#include <string>
#include <vector>

#include "fmt/base.h"
#include "myx_grad.h"

auto main() -> int {
    // myx_grad();

    // std::vector<std::string> vec;
    // vec.emplace_back("test_package");
    // myx_grad_print_vector(vec);


    // clang-format off
    std::vector<std::vector<nn::dtype>> X = {
        {0.4967, 0.1432, 0.8713, 0.2541},
        {0.6325, 0.9876, 0.1124, 0.7789},
        {0.3456, 0.3647, 0.9123, 0.4302},
        {0.2219, 0.6834, 0.7591, 0.0098},
        {0.1234, 0.5678, 0.9012, 0.3456},
    };
    // clang-format on
    std::vector<nn::dtype> Y = {1., 2., -3., -4., 1.5};
    engine::Graph<nn::dtype> graph;
    nn::MLP mlp(graph, 4, {10, 5, 1});

    auto loss = [&graph](nn::Tensor &pred, nn::Tensor &target) -> nn::value_ptr {
        auto *res = &graph.leaf(0);
        for (size_t i = 0; i < pred.size(); ++i) {
            auto &left = *pred[i];
            auto &right = *target[i];
            auto &diff = (left - right);
            res = &(*res + diff * diff);
        }
        return res;
    };

    auto batch_size = X.size();
    auto target = nn::Tensor{};
    for (size_t i = 0; i < batch_size; ++i) {
        target.push_back(graph.create_value(Y[i]));
    }
    fmt::println("target: {}", nn::to_string(target));
    size_t max_epoch = 2000;
    for (size_t epoch = 0; epoch < max_epoch; ++epoch) {
        auto pred = nn::Tensor{};
        for (size_t i = 0; i < batch_size; ++i) {
            pred.push_back(mlp(X[i]));
        }
        auto *root = loss(pred, target);
        graph.zero_grad();
        graph.back_propagate(*root);
        graph.learn(0.003);
        fmt::println("Epoch: {}, Loss: {}", epoch, root->value());
        if (epoch % 100 == 0) {
            fmt::println("epoch: {}, Loss: {}, pred: {}", epoch, root->value(), nn::to_string(pred));
        }
    }
}
