#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <fstream>
#include <myx_grad/engine.h>

TEST_CASE("test value", "[engine]") {
  {

    auto const val = engine::scalar<int>(5);
    REQUIRE(val.value() == 5);
    auto const val1 = engine::scalar<int>(3);
    REQUIRE(val1.value() == 3);
    REQUIRE(val.value() + val1.value() == 8);
    REQUIRE(val.value() - val1.value() == 2);
    REQUIRE(val.value() * val1.value() == 15);
    REQUIRE(val.value() / val1.value() == 1);
  }

  auto print_dot = [](engine::scalar<float> *root,
                      const std::string &file_name) {
    const std::string prefix_str = R"(digraph ComputeGraph {
rankdir=LR;
)";
    const std::string suffix_str = "}\n";
    engine::graph<float> g(root);
    g.back_propagate();
    auto dot_str =
        fmt::format("{}\n{}\n{}", prefix_str, g.display(), suffix_str);

    {
      std::ofstream dot_file(file_name);
      REQUIRE(dot_file.is_open());
      dot_file << dot_str;
      dot_file.close();
      fmt::println("Graph written to {}", file_name);
    }
  };

  {
    auto val1 = engine::scalar<float>(2.5f);
    auto val2 = engine::scalar<float>(3.5f);
    auto val3 = val1 + val2;

    print_dot(&val3, "graph1.dot");
  }

  {
    auto val1 = engine::scalar<float>(2.F);
    auto val2 = engine::scalar<float>(3.F);
    auto val3 = val1 + val2;
    auto val4 = engine::scalar<float>(4.F);
    auto val5 = val3 * val4;
    print_dot(&val5, "graph2.dot");
  }
}
