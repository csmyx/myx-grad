#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/core.h>
#include <fstream>
#include <myx_grad/engine.h>

TEST_CASE("test value", "[engine]") {
  {

    auto const val = engine::Value<int>(5);
    REQUIRE(val.value() == 5);
    auto const val1 = engine::Value<int>(3);
    REQUIRE(val1.value() == 3);
    REQUIRE(val.value() + val1.value() == 8);
    REQUIRE(val.value() - val1.value() == 2);
    REQUIRE(val.value() * val1.value() == 15);
    REQUIRE(val.value() / val1.value() == 1);
  }

  auto print_dot = [](engine::graph<float> graph,
                      const std::string &file_name) {
    const std::string prefix_str = R"(digraph ComputeGraph {
rankdir=TB;
)";
    const std::string suffix_str = "}\n";
    auto dot_str =
        fmt::format("{}\n{}\n{}", prefix_str, graph.display(), suffix_str);

    {
      std::ofstream dot_file(file_name);
      REQUIRE(dot_file.is_open());
      dot_file << dot_str;
      dot_file.close();
      fmt::println("Graph written to {}", file_name);
    }
  };

  {
    auto val1 = engine::Value<float>(2.5f);
    auto val2 = engine::Value<float>(3.5f);

    auto val3 = val1 + val2;

    engine::graph<float> g(&val3);
    g.back_propagate();

    print_dot(g, "graph1.dot");
  }

  // test: backpropagation
  {
    // v5 = (v1 + v2) * v4
    auto val1 = engine::Value<float>(2.F, "n1");
    auto val2 = engine::Value<float>(3.F, "n2");
    auto val3 = (val1 + val2).with_id("n3");
    auto val4 = engine::Value<float>(4.F, "n4");
    auto val5 = (val3 * val4).with_id("n5");
    engine::graph<float> g(&val5);
    g.back_propagate();
    print_dot(g, "graph2.dot");
    REQUIRE(val5.m_grad == 1.0F);
    REQUIRE(val4.m_grad == 5.0F);
    REQUIRE(val3.m_grad == 4.0F);
    REQUIRE(val2.m_grad == 4.0F);
    REQUIRE(val1.m_grad == 4.0F);
  }

  // test: use the same value as input multiple times
  {
    auto val1 = engine::Value<float>(3.F, "n1");
    auto val2 = (val1 + val1).with_id("n2");
    engine::graph<float> g(&val2);
    g.back_propagate();
    print_dot(g, "graph3.dot");
    REQUIRE(val1.m_grad == 2.0F);
  }

  // test: implement tanh
  {
    using namespace Catch::Matchers;
    auto x1 = engine::Value<float>(2.F, "x1");
    auto x2 = engine::Value<float>(0.F, "x2");
    auto w1 = engine::Value<float>(-3.F, "w1");
    auto w2 = engine::Value<float>(1.F, "w2");
    auto b = engine::Value<float>(6.8813735870195432F, "b");
    auto wx1 = (x1 * w1).with_id("wx1");
    auto wx2 = (x2 * w2).with_id("wx2");
    auto wx = (wx1 + wx2).with_id("wx");
    auto n = (wx + b).with_id("n");
    auto o = n.tanh().with_id("o");
    engine::graph<float> g(&o);
    g.back_propagate();
    print_dot(g, "graph_4.dot");
    REQUIRE_THAT(o.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(n.m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(wx1.m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(wx2.m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(w1.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(w2.m_grad, WithinAbs(0.0F, 1e-6));
    REQUIRE_THAT(x1.m_grad, WithinAbs(-1.5F, 1e-6));
    REQUIRE_THAT(x2.m_grad, WithinAbs(0.5F, 1e-6));
  }
}
