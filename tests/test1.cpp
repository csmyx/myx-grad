#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/core.h>
#include <fstream>
#include <myx_grad/engine.h>
using namespace Catch::Matchers;

// NOLINTBEGIN(*-magic-numbers, *-identifier-length, *-avoid-do-while)

TEST_CASE("test value", "[engine]") {

  auto print_dot = [](engine::graph<float> graph,
                      const std::string &file_name) {
    const std::string prefix_str = R"(digraph ComputeGraph {
rankdir=TB;
)";
    const std::string suffix_str = "}\n";
    auto dot_str =
        fmt::format("{}\n{}\n{}", prefix_str, graph.display(), suffix_str);

    {
      const std::string full_file_name = "doc/pictures/" + file_name;
      std::ofstream dot_file(full_file_name);
      REQUIRE(dot_file.is_open());
      dot_file << dot_str;
      dot_file.close();
      fmt::println("Graph written to {}", full_file_name);
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
    auto val1 = engine::Value<float>(2.5f);
    auto val2 = engine::Value<float>(3.5f);

    auto val3 = val1 + val2;

    engine::graph<float> g(&val3);
    g.back_propagate();

    print_dot(g, "graph1.dot");
  }

  SECTION("multiplication chain backpropagation") {
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

  SECTION("shared input value") {
    auto val1 = engine::Value<float>(3.F, "n1");
    auto val2 = (val1 + val1).with_id("n2");
    engine::graph<float> g(&val2);
    g.back_propagate();
    print_dot(g, "graph3.dot");
    REQUIRE(val1.m_grad == 2.0F);
  }

  SECTION("tanh activation backpropagation") {
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

  SECTION("subtraction backpropagation") {
    //  d(a-b)/da = 1, d(a-b)/db = -1
    auto a = engine::Value<float>(5.F, "a");
    auto b = engine::Value<float>(3.F, "b");
    auto c = (a - b).with_id("c");
    engine::graph<float> g(&c);
    g.back_propagate();
    REQUIRE_THAT(c.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(b.m_grad, WithinAbs(-1.0F, 1e-6));
  }

  SECTION("deeper chain backpropagation") {
    //  v = (a*b + c) * d
    //  dv/da = d * b,  dv/db = d * a,  dv/dc = d * 1,  dv/dd = a*b + c
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

  SECTION("DAG with shared node") {
    //  v = a*b + a*c
    //  dv/da = b + c,  dv/db = a,  dv/dc = a
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

  SECTION("subtract same value") {
    //  a - a = 0,  da = 1 + (-1) = 0
    // The gradient should cancel out since the node contributes to both left
    // and right. In our engine, `a` is the same pointer for both sides, so
    // m_grad sums twice. Expected: m_left contributes +1, m_right contributes
    // -1 → net 0 for a.
    auto a = engine::Value<float>(7.F, "a");
    auto v = (a - a).with_id("v");
    engine::graph<float> g(&v);
    g.back_propagate();
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(0.0F, 1e-6));
  }

  SECTION("numerical gradient check with finite differences") {
    //  f = a*b + tanh(a+b) + a - b
    //  Compare autograd against central finite differences for each leaf.
    const float h = 1e-4F;

    // Define f(a,b) = a*b + tanh(a+b) + a - b
    auto compute = [](float av, float bv) -> float {
      return (av * bv) + std::tanh(av + bv) + av - bv;
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

  SECTION("division backpropagation") {
    //  d(a/b)/da = 1/b,  d(a/b)/db = -a/b^2
    auto a = engine::Value<float>(8.F, "a");
    auto b = engine::Value<float>(2.F, "b");
    auto v = (a / b).with_id("v");
    engine::graph<float> g(&v);
    g.back_propagate();
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(0.5F, 1e-6));  // 1/2
    REQUIRE_THAT(b.m_grad, WithinAbs(-2.0F, 1e-6)); // -8/4
  }

  SECTION("power backpropagation") {
    //  d(base^exp)/d(base) = exp * base^(exp-1)
    //  pow(a, 2) = a^2, da = 2*a
    auto a = engine::Value<float>(3.F, "a");
    auto v = a.pow(2.0F).with_id("v"); // v = a^2 = 9
    engine::graph<float> g(&v);
    g.back_propagate();
    REQUIRE_THAT(v.m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a.m_grad, WithinAbs(6.0F, 1e-6)); // 2*3
  }

  SECTION("pow in chain backpropagation") {
    //  v = (a * b)^3
    //  dv/da = 3*(a*b)^2 * b,  dv/db = 3*(a*b)^2 * a
    auto a = engine::Value<float>(2.F, "a");
    auto b = engine::Value<float>(3.F, "b");
    auto prod = (a * b).with_id("prod");  // 6
    auto v = prod.pow(3.0F).with_id("v"); // 216
    engine::graph<float> g(&v);
    g.back_propagate();
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
    // Both forward value and backward gradient must match.

    // --- builtin tanh ---
    auto x_builtin = engine::Value<float>(0.5F, "x");
    auto builtin = x_builtin.tanh().with_id("builtin");
    engine::graph<float> g_builtin(&builtin);
    g_builtin.back_propagate();

    // --- manual tanh via exp ---
    auto x_manual = engine::Value<float>(0.5F, "x");
    auto two = engine::Value<float>(2.F);
    auto one = engine::Value<float>(1.F);
    auto two_x = (x_manual * two).with_id("2x");
    auto exp2x = two_x.exp().with_id("exp2x");
    auto numerator = (exp2x - one).with_id("num");
    auto denominator = (exp2x + one).with_id("den");
    auto manual = (numerator / denominator).with_id("manual");
    engine::graph<float> g_manual(&manual);
    g_manual.back_propagate();

    // Forward values must match
    REQUIRE_THAT(builtin.m_value, WithinAbs(manual.m_value, 1e-6));

    // Gradient w.r.t x must match
    // d(tanh(x))/dx = 1 - tanh(x)^2
    REQUIRE_THAT(x_builtin.m_grad, WithinAbs(x_manual.m_grad, 1e-6));
  }

  SECTION("requires_grad = false stops gradient") {
    //  c = a * b,  b set to requires_grad=false
    //  dc/da = b (grad flows),  dc/db = a (grad blocked)
    auto a = engine::Value<float>(4.F, "a");
    auto b = engine::Value<float>(3.F, "b");
    b.set_requires_grad(false);
    REQUIRE_FALSE(b.requires_grad());
    REQUIRE(a.requires_grad());

    auto c = (a * b).with_id("c"); // 12
    engine::graph<float> g(&c);
    g.back_propagate();
    REQUIRE_THAT(a.m_grad, WithinAbs(3.0F, 1e-6)); // dc/da = b = 3
    REQUIRE(b.m_grad == 0.0F);                     // gradient blocked
  }

  SECTION("frozen constant in deep chain") {
    //  v = (a * frozen_b) + c
    //  dv/da = frozen_b, dv/dc = 1
    //  frozen_b gets no gradient
    auto a = engine::Value<float>(2.F, "a");
    auto frozen_b = engine::Value<float>(5.F, "b");
    frozen_b.set_requires_grad(false);
    auto c = engine::Value<float>(1.F, "c");

    auto prod = (a * frozen_b).with_id("prod"); // 10
    auto v = (prod + c).with_id("v");           // 11
    engine::graph<float> g(&v);
    g.back_propagate();

    REQUIRE_THAT(a.m_grad, WithinAbs(5.0F, 1e-6));    // dv/da = frozen_b = 5
    REQUIRE(frozen_b.m_grad == 0.0F);                 // blocked
    REQUIRE_THAT(c.m_grad, WithinAbs(1.0F, 1e-6));    // dv/dc = 1
    REQUIRE_THAT(prod.m_grad, WithinAbs(1.0F, 1e-6)); // dv/d(prod) = 1
  }

  SECTION("frozen intermediate node stops gradient to its children") {
    //  v = (a*b + c) * d,  frozen the (a*b + c) sub-expression
    //  dv/d(abc) should be blocked → a,b,c,ab get no gradient
    //  dv/dd = abc.m_value should still flow normally
    auto a = engine::Value<float>(2.F, "a");
    auto b = engine::Value<float>(3.F, "b");
    auto c_var = engine::Value<float>(1.F, "c");
    auto d = engine::Value<float>(4.F, "d");

    auto ab = (a * b).with_id("ab");        // 6
    auto abc = (ab + c_var).with_id("abc"); // 7
    abc.set_requires_grad(false);
    auto v = (abc * d).with_id("v"); // 28

    engine::graph<float> g(&v);
    g.back_propagate();
    print_dot(g, "graph_requires_grad.dot");

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
