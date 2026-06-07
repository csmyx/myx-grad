#include <catch2/catch_test_macros.hpp>
#include <myx_grad/engine.h>

TEST_CASE("test value", "[engine]") {
  using namespace engine;
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
  {
    auto val1 = engine::scalar<float>(2.5f);
    auto val2 = engine::scalar<float>(3.5f);
    auto val3 = val1 + val2;

    graph<float> g(&val3);
    auto graph_str = graph<float>::display(g);

    const std::string prefix_str = R"(
digraph ComputeGraph {
    rankdir=LR;
)";
    const std::string suffix_str = "}\n";
    auto dot_str = fmt::format("{}\n{}\n{}", prefix_str, graph_str, suffix_str);

    {
      std::ofstream dot_file("graph.dot");
      REQUIRE(dot_file.is_open());
      dot_file << dot_str;
      fmt::println("Graph written to graph.dot");
    }
  }
}
