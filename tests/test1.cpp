#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/core.h>
#include <fstream>
#include <myx_grad/engine.h>
using namespace Catch::Matchers;

// NOLINTBEGIN(*-magic-numbers, *-identifier-length, *-avoid-do-while)

TEST_CASE("test value", "[engine]") {

  auto print_dot = [](engine::graph<float> &graph, engine::Value<float> &root,
                      const std::string &file_name) {
    const std::string prefix_str = R"(digraph ComputeGraph {
rankdir=TB;
)";
    const std::string suffix_str = "}\n";
    auto dot_str =
        fmt::format("{}\n{}\n{}", prefix_str, graph.display(root), suffix_str);

    {
      std::ofstream dot_file(file_name);
      REQUIRE(dot_file.is_open());
      dot_file << dot_str;
      dot_file.close();
      fmt::println("Graph written to {}", file_name);
    }
  };

  SECTION("basic Value creation and arithmetic") {
    auto const val = engine::Value<int>(5);
    REQUIRE(val.value() == 5);
    auto const val1 = engine::Value<int>(3);
    REQUIRE(val1.value() == 3);
    REQUIRE(val.value() + val1.value() == 8);
    REQUIRE(val.value() - val1.value() == 2);
    REQUIRE(val.value() * val1.value() == 15);
    REQUIRE(val.value() / val1.value() == 1);
  }

  SECTION("simple addition with graph output") {
    engine::graph<float> g;
    auto &v1 = g.leaf(2.5f);
    auto &v2 = g.leaf(3.5f);
    auto &v3 = g.add(v1, v2);
    g.back_propagate(v3);
    print_dot(g, v3, "graph1.dot");
  }

  SECTION("multiplication chain backpropagation") {
    // v5 = (v1 + v2) * v4
    engine::graph<float> g;
    auto &val1 = g.leaf(2.F, "n1");
    auto &val2 = g.leaf(3.F, "n2");
    auto &val3 = g.add(val1, val2);
    val3.with_id("n3");
    auto &val4 = g.leaf(4.F, "n4");
    auto &val5 = g.mul(val3, val4);
    val5.with_id("n5");
    g.back_propagate(val5);
    print_dot(g, val5, "graph2.dot");
    REQUIRE(val5.m_grad == 1.0F);
    REQUIRE(val4.m_grad == 5.0F);
    REQUIRE(val3.m_grad == 4.0F);
    REQUIRE(val2.m_grad == 4.0F);
    REQUIRE(val1.m_grad == 4.0F);
  }

  SECTION("shared input value") {
    engine::graph<float> g;
    auto &val1 = g.leaf(3.F, "n1");
    auto &val2 = g.add(val1, val1);
    val2.with_id("n2");
    g.back_propagate(val2);
    print_dot(g, val2, "graph3.dot");
    REQUIRE(val1.m_grad == 2.0F);
  }

  SECTION("tanh activation backpropagation") {
    engine::graph<float> g;
    auto &x1 = g.leaf(2.F, "x1");
    auto &x2 = g.leaf(0.F, "x2");
    auto &w1 = g.leaf(-3.F, "w1");
    auto &w2 = g.leaf(1.F, "w2");
    auto &b = g.leaf(6.8813735870195432F, "b");
    auto &wx1 = g.mul(x1, w1);
    wx1.with_id("wx1");
    auto &wx2 = g.mul(x2, w2);
    wx2.with_id("wx2");
    auto &wx = g.add(wx1, wx2);
    wx.with_id("wx");
    auto &n = g.add(wx, b);
    n.with_id("n");
    auto &o = g.tanh(n);
    o.with_id("o");
    g.back_propagate(o);
    print_dot(g, o, "graph_4.dot");
    REQUIRE_THAT(o.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(n.m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(wx1.m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(wx2.m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(w1.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(w2.m_grad, WithinAbs(0.0F, 1e-6));
    REQUIRE_THAT(x1.m_grad, WithinAbs(-1.5F, 1e-6));
    REQUIRE_THAT(x2.m_grad, WithinAbs(0.5F, 1e-6));
  }

  SECTION("subtraction backpropagation") {
    //  d(a-b)/da = 1, d(a-b)/db = -1
    engine::graph<float> g;
    auto &a = g.leaf(5.F, "a");
    auto &b = g.leaf(3.F, "b");
    auto &c = g.sub(a, b);
    c.with_id("c");
    g.back_propagate(c);
    REQUIRE_THAT(c.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(b.m_grad, WithinAbs(-1.0F, 1e-6));
  }

  SECTION("deeper chain backpropagation") {
    //  v = (a*b + c) * d
    engine::graph<float> g;
    auto &a = g.leaf(2.F, "a");
    auto &b = g.leaf(3.F, "b");
    auto &c = g.leaf(1.F, "c");
    auto &d = g.leaf(4.F, "d");
    auto &ab = g.mul(a, b);
    ab.with_id("ab"); // 6
    auto &abc = g.add(ab, c);
    abc.with_id("abc"); // 7
    auto &v = g.mul(abc, d);
    v.with_id("v"); // 28
    g.back_propagate(v);
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(abc.m_grad, WithinAbs(4.0F, 1e-6)); // d
    REQUIRE_THAT(d.m_grad, WithinAbs(7.0F, 1e-6));   // a*b+c
    REQUIRE_THAT(ab.m_grad, WithinAbs(4.0F, 1e-6));  // d * 1
    REQUIRE_THAT(c.m_grad, WithinAbs(4.0F, 1e-6));   // d * 1
    REQUIRE_THAT(a.m_grad, WithinAbs(12.0F, 1e-6));  // d * b
    REQUIRE_THAT(b.m_grad, WithinAbs(8.0F, 1e-6));   // d * a
  }

  SECTION("DAG with shared node") {
    //  v = a*b + a*c
    engine::graph<float> g;
    auto &a = g.leaf(2.F, "a");
    auto &b = g.leaf(3.F, "b");
    auto &c = g.leaf(5.F, "c");
    auto &ab = g.mul(a, b);
    ab.with_id("ab"); // 6
    auto &ac = g.mul(a, c);
    ac.with_id("ac"); // 10
    auto &v = g.add(ab, ac);
    v.with_id("v"); // 16
    g.back_propagate(v);
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(ab.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(ac.m_grad, WithinAbs(1.0F, 1e-6));
    // a accumulates gradient from both ab and ac paths: b + c = 3 + 5
    REQUIRE_THAT(a.m_grad, WithinAbs(8.0F, 1e-6));
    REQUIRE_THAT(b.m_grad, WithinAbs(2.0F, 1e-6)); // a
    REQUIRE_THAT(c.m_grad, WithinAbs(2.0F, 1e-6)); // a
  }

  SECTION("subtract same value") {
    engine::graph<float> g;
    auto &a = g.leaf(7.F, "a");
    auto &v = g.sub(a, a);
    v.with_id("v");
    g.back_propagate(v);
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(0.0F, 1e-6));
  }

  SECTION("numerical gradient check with finite differences") {
    const float h = 1e-4F;

    auto compute = [](float av, float bv) -> float {
      return (av * bv) + std::tanh(av + bv) + av - bv;
    };

    engine::graph<float> g;
    auto &a = g.leaf(1.5F, "a");
    auto &b = g.leaf(0.8F, "b");
    auto &prod = g.mul(a, b);
    prod.with_id("prod");
    auto &sum_ab = g.add(a, b);
    sum_ab.with_id("sum");
    auto &th = g.tanh(sum_ab);
    th.with_id("th");
    auto &th_plus_a = g.add(th, a);
    th_plus_a.with_id("th_plus_a");
    auto &v = g.add(prod, th_plus_a);
    v.with_id("v");
    auto &final_v = g.sub(v, b);
    final_v.with_id("final");
    g.back_propagate(final_v);

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

  SECTION("division backpropagation") {
    //  d(a/b)/da = 1/b,  d(a/b)/db = -a/b^2
    engine::graph<float> g;
    auto &a = g.leaf(8.F, "a");
    auto &b = g.leaf(2.F, "b");
    auto &v = g.div(a, b);
    v.with_id("v");
    g.back_propagate(v);
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(0.5F, 1e-6));  // 1/2
    REQUIRE_THAT(b.m_grad, WithinAbs(-2.0F, 1e-6)); // -8/4
  }

  SECTION("power backpropagation") {
    //  d(base^exp)/d(base) = exp * base^(exp-1)
    engine::graph<float> g;
    auto &a = g.leaf(3.F, "a");
    auto &v = g.pow(a, 2.0F);
    v.with_id("v"); // v = a^2 = 9
    g.back_propagate(v);
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(6.0F, 1e-6)); // 2*3
  }

  SECTION("pow in chain backpropagation") {
    //  v = (a * b)^3
    engine::graph<float> g;
    auto &a = g.leaf(2.F, "a");
    auto &b = g.leaf(3.F, "b");
    auto &prod = g.mul(a, b);
    prod.with_id("prod"); // 6
    auto &v = g.pow(prod, 3.0F);
    v.with_id("v"); // 216
    g.back_propagate(v);
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    // prod grad: 3 * 6^2 = 108
    REQUIRE_THAT(prod.m_grad, WithinAbs(108.0F, 1e-6));
    // a grad: 108 * b = 108 * 3 = 324
    REQUIRE_THAT(a.m_grad, WithinAbs(324.0F, 1e-6));
    // b grad: 108 * a = 108 * 2 = 216
    REQUIRE_THAT(b.m_grad, WithinAbs(216.0F, 1e-6));
  }

  SECTION("tanh shortcut vs manual exp-based implementation") {
    // Compare builtin tanh() against manual composition:
    //   tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)

    // --- builtin tanh ---
    engine::graph<float> g1;
    auto &x_builtin = g1.leaf(0.5F, "x");
    auto &builtin = g1.tanh(x_builtin);
    builtin.with_id("builtin");
    g1.back_propagate(builtin);

    // --- manual tanh via exp ---
    engine::graph<float> g2;
    auto &x_manual = g2.leaf(0.5F, "x");
    auto &two = g2.leaf(2.F);
    two.set_requires_grad(false);
    auto &one = g2.leaf(1.F);
    one.set_requires_grad(false);
    auto &two_x = g2.mul(x_manual, two);
    two_x.with_id("2x");
    auto &exp2x = g2.exp(two_x);
    exp2x.with_id("exp2x");
    auto &numerator = g2.sub(exp2x, one);
    numerator.with_id("num");
    auto &denominator = g2.add(exp2x, one);
    denominator.with_id("den");
    auto &manual = g2.div(numerator, denominator);
    manual.with_id("manual");
    g2.back_propagate(manual);

    // Forward values must match
    REQUIRE_THAT(builtin.m_value, WithinAbs(manual.m_value, 1e-6));

    // Gradient w.r.t x must match
    REQUIRE_THAT(x_builtin.m_grad, WithinAbs(x_manual.m_grad, 1e-6));
  }

  SECTION("requires_grad = false stops gradient") {
    engine::graph<float> g;
    auto &a = g.leaf(4.F, "a");
    auto &b = g.leaf(3.F, "b");
    b.set_requires_grad(false);
    REQUIRE_FALSE(b.requires_grad());
    REQUIRE(a.requires_grad());

    auto &c = g.mul(a, b);
    c.with_id("c"); // 12
    g.back_propagate(c);
    REQUIRE_THAT(a.m_grad, WithinAbs(3.0F, 1e-6)); // dc/da = b = 3
    REQUIRE(b.m_grad == 0.0F);                     // gradient blocked
  }

  SECTION("frozen constant in deep chain") {
    engine::graph<float> g;
    auto &a = g.leaf(2.F, "a");
    auto &frozen_b = g.leaf(5.F, "b");
    frozen_b.set_requires_grad(false);
    auto &c = g.leaf(1.F, "c");

    auto &prod = g.mul(a, frozen_b);
    prod.with_id("prod"); // 10
    auto &v = g.add(prod, c);
    v.with_id("v"); // 11
    g.back_propagate(v);

    REQUIRE_THAT(a.m_grad, WithinAbs(5.0F, 1e-6));    // dv/da = frozen_b = 5
    REQUIRE(frozen_b.m_grad == 0.0F);                 // blocked
    REQUIRE_THAT(c.m_grad, WithinAbs(1.0F, 1e-6));    // dv/dc = 1
    REQUIRE_THAT(prod.m_grad, WithinAbs(1.0F, 1e-6)); // dv/d(prod) = 1
  }

  SECTION("frozen intermediate node stops gradient to its children") {
    engine::graph<float> g;
    auto &a = g.leaf(2.F, "a");
    auto &b = g.leaf(3.F, "b");
    auto &c_var = g.leaf(1.F, "c");
    auto &d = g.leaf(4.F, "d");

    auto &ab = g.mul(a, b);
    ab.with_id("ab"); // 6
    auto &abc = g.add(ab, c_var);
    abc.with_id("abc"); // 7
    abc.set_requires_grad(false);
    auto &v = g.mul(abc, d);
    v.with_id("v"); // 28

    g.back_propagate(v);
    print_dot(g, v, "graph_requires_grad.dot");

    // d receives gradient: dv/dd = abc.m_value = 7
    REQUIRE_THAT(d.m_grad, WithinAbs(7.0F, 1e-6));

    // abc and everything below it gets nothing
    REQUIRE(abc.m_grad == 0.0F);
    REQUIRE(ab.m_grad == 0.0F);
    REQUIRE(a.m_grad == 0.0F);
    REQUIRE(b.m_grad == 0.0F);
    REQUIRE(c_var.m_grad == 0.0F);
  }
}

// NOLINTEND(*-magic-numbers, *-identifier-length, *-avoid-do-while)
