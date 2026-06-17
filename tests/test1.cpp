#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/core.h>
#include <fstream>
#include <myx_grad/engine.h>
using namespace Catch::Matchers;

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

  // test: subtraction backpropagation
  //  d(a-b)/da = 1, d(a-b)/db = -1
  {

    auto a = engine::Value<float>(5.F, "a");
    auto b = engine::Value<float>(3.F, "b");
    auto c = (a - b).with_id("c");
    engine::graph<float> g(&c);
    g.back_propagate();
    REQUIRE_THAT(c.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(b.m_grad, WithinAbs(-1.0F, 1e-6));
  }

  // test: deeper chain  v = (a*b + c) * d
  //  dv/da = d * b,  dv/db = d * a,  dv/dc = d * 1,  dv/dd = a*b + c
  {
    auto a = engine::Value<float>(2.F, "a");
    auto b = engine::Value<float>(3.F, "b");
    auto c = engine::Value<float>(1.F, "c");
    auto d = engine::Value<float>(4.F, "d");
    auto ab = (a * b).with_id("ab");    // 6
    auto abc = (ab + c).with_id("abc"); // 7
    auto v = (abc * d).with_id("v");    // 28
    engine::graph<float> g(&v);
    g.back_propagate();
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(abc.m_grad, WithinAbs(4.0F, 1e-6)); // d
    REQUIRE_THAT(d.m_grad, WithinAbs(7.0F, 1e-6));   // a*b+c
    REQUIRE_THAT(ab.m_grad, WithinAbs(4.0F, 1e-6));  // d * 1
    REQUIRE_THAT(c.m_grad, WithinAbs(4.0F, 1e-6));   // d * 1
    REQUIRE_THAT(a.m_grad, WithinAbs(12.0F, 1e-6));  // d * b
    REQUIRE_THAT(b.m_grad, WithinAbs(8.0F, 1e-6));   // d * a
  }

  // test: DAG with shared node — a used in two paths: a*b + a*c
  //  v = a*b + a*c
  //  dv/da = b + c,  dv/db = a,  dv/dc = a
  {

    auto a = engine::Value<float>(2.F, "a");
    auto b = engine::Value<float>(3.F, "b");
    auto c = engine::Value<float>(5.F, "c");
    auto ab = (a * b).with_id("ab"); // 6
    auto ac = (a * c).with_id("ac"); // 10
    auto v = (ab + ac).with_id("v"); // 16
    engine::graph<float> g(&v);
    g.back_propagate();
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(ab.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(ac.m_grad, WithinAbs(1.0F, 1e-6));
    // a accumulates gradient from both ab and ac paths: b + c = 3 + 5
    REQUIRE_THAT(a.m_grad, WithinAbs(8.0F, 1e-6));
    REQUIRE_THAT(b.m_grad, WithinAbs(2.0F, 1e-6)); // a
    REQUIRE_THAT(c.m_grad, WithinAbs(2.0F, 1e-6)); // a
  }

  // test: subtract same value  a - a = 0,  da = 1 + (-1) = 0
  // The gradient should cancel out since the node contributes to both left and
  // right. But in our engine, `a` is the same pointer for both sides, so m_grad
  // sums twice. Expected: m_left contributes +1, m_right contributes -1 → net 0
  // for a.
  {

    auto a = engine::Value<float>(7.F, "a");
    auto v = (a - a).with_id("v");
    engine::graph<float> g(&v);
    g.back_propagate();
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(0.0F, 1e-6));
  }

  // test: numerical gradient check (finite differences)
  //  f = a*b + tanh(a+b) + a - b
  //  Compare autograd against central finite differences for each leaf.
  {

    const float h = 1e-4F;

    // Define f(a,b) = a*b + tanh(a+b) + a - b
    auto compute = [](float av, float bv) -> float {
      return av * bv + std::tanh(av + bv) + av - bv;
    };

    // Analytical gradients:
    //  df/da = b + (1 - tanh(a+b)^2) + 1
    //  df/db = a + (1 - tanh(a+b)^2) - 1

    // Build the expression graph once with autograd
    auto a = engine::Value<float>(1.5F, "a");
    auto b = engine::Value<float>(0.8F, "b");
    auto prod = (a * b).with_id("prod");
    auto sum_ab = (a + b).with_id("sum");
    auto th = sum_ab.tanh().with_id("th");
    auto th_plus_a = (th + a).with_id("th_plus_a");
    auto v = (prod + th_plus_a).with_id("v");
    auto final_v = (v - b).with_id("final");
    engine::graph<float> g(&final_v);
    g.back_propagate();

    // Compute numerical gradients via central finite differences
    float f0 = compute(1.5F, 0.8F);
    float f_ap = compute(1.5F + h, 0.8F);
    float f_am = compute(1.5F - h, 0.8F);
    float f_bp = compute(1.5F, 0.8F + h);
    float f_bm = compute(1.5F, 0.8F - h);
    float num_grad_a = (f_ap - f_am) / (2.0F * h);
    float num_grad_b = (f_bp - f_bm) / (2.0F * h);

    REQUIRE_THAT(a.m_grad, WithinAbs(num_grad_a, 2e-3F));
    REQUIRE_THAT(b.m_grad, WithinAbs(num_grad_b, 2e-3F));
  }
}
