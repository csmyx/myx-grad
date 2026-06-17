#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/core.h>
#include <fstream>
#include <myx_grad/engine.h>
using namespace Catch::Matchers;

// Helper: create a shared_ptr<Value<float>> with a value and optional id
static auto val(double v, std::string id = "") -> engine::ValuePtr {
  return std::make_shared<engine::Value<engine::float_t>>(v, std::move(id));
}

// NOLINTBEGIN(*-magic-numbers, *-identifier-length, *-avoid-do-while)

TEST_CASE("test value", "[engine]") {

  auto print_dot = [](engine::graph<engine::float_t> graph,
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
    auto const v = engine::Value<int>(5);
    REQUIRE(v.value() == 5);
    auto const v1 = engine::Value<int>(3);
    REQUIRE(v1.value() == 3);
    REQUIRE(v.value() + v1.value() == 8);
    REQUIRE(v.value() - v1.value() == 2);
    REQUIRE(v.value() * v1.value() == 15);
    REQUIRE(v.value() / v1.value() == 1);
  }

  SECTION("simple addition with graph output") {
    auto v1 = val(2.5f);
    auto v2 = val(3.5f);
    auto v3 = v1 + v2;
    engine::graph<engine::float_t> g(v3);
    g.back_propagate();
    print_dot(g, "graph1.dot");
  }

  SECTION("multiplication chain backpropagation") {
    auto v1 = val(2.F, "n1");
    auto v2 = val(3.F, "n2");
    auto v3 = v1 + v2;
    v3->with_id("n3");
    auto v4 = val(4.F, "n4");
    auto v5 = v3 * v4;
    v5->with_id("n5");
    engine::graph<engine::float_t> g(v5);
    g.back_propagate();
    print_dot(g, "graph2.dot");
    REQUIRE(v5->m_grad == 1.0F);
    REQUIRE(v4->m_grad == 5.0F);
    REQUIRE(v3->m_grad == 4.0F);
    REQUIRE(v2->m_grad == 4.0F);
    REQUIRE(v1->m_grad == 4.0F);
  }

  SECTION("shared input value") {
    auto v1 = val(3.F, "n1");
    auto v2 = v1 + v1;
    v2->with_id("n2");
    engine::graph<engine::float_t> g(v2);
    g.back_propagate();
    print_dot(g, "graph3.dot");
    REQUIRE(v1->m_grad == 2.0F);
  }

  SECTION("tanh activation backpropagation") {
    auto x1 = val(2.F, "x1");
    auto x2 = val(0.F, "x2");
    auto w1 = val(-3.F, "w1");
    auto w2 = val(1.F, "w2");
    auto b = val(6.8813735870195432F, "b");
    auto wx1 = x1 * w1;
    wx1->with_id("wx1");
    auto wx2 = x2 * w2;
    wx2->with_id("wx2");
    auto wx = wx1 + wx2;
    wx->with_id("wx");
    auto n = wx + b;
    n->with_id("n");
    auto o = n->tanh();
    o->with_id("o");
    engine::graph<engine::float_t> g(o);
    g.back_propagate();
    print_dot(g, "graph_4.dot");
    REQUIRE_THAT(o->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(n->m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(wx1->m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(wx2->m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(w1->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(w2->m_grad, WithinAbs(0.0F, 1e-6));
    REQUIRE_THAT(x1->m_grad, WithinAbs(-1.5F, 1e-6));
    REQUIRE_THAT(x2->m_grad, WithinAbs(0.5F, 1e-6));
  }

  SECTION("subtraction backpropagation") {
    auto a = val(5.F, "a");
    auto b = val(3.F, "b");
    auto c = a - b;
    c->with_id("c");
    engine::graph<engine::float_t> g(c);
    g.back_propagate();
    REQUIRE_THAT(c->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(b->m_grad, WithinAbs(-1.0F, 1e-6));
  }

  SECTION("deeper chain backpropagation") {
    auto a = val(2.F, "a");
    auto b = val(3.F, "b");
    auto c = val(1.F, "c");
    auto d = val(4.F, "d");
    auto ab = a * b;
    ab->with_id("ab");
    auto abc = ab + c;
    abc->with_id("abc");
    auto v = abc * d;
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    REQUIRE_THAT(v->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(abc->m_grad, WithinAbs(4.0F, 1e-6));
    REQUIRE_THAT(d->m_grad, WithinAbs(7.0F, 1e-6));
    REQUIRE_THAT(ab->m_grad, WithinAbs(4.0F, 1e-6));
    REQUIRE_THAT(c->m_grad, WithinAbs(4.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(12.0F, 1e-6));
    REQUIRE_THAT(b->m_grad, WithinAbs(8.0F, 1e-6));
  }

  SECTION("DAG with shared node") {
    auto a = val(2.F, "a");
    auto b = val(3.F, "b");
    auto c = val(5.F, "c");
    auto ab = a * b;
    ab->with_id("ab");
    auto ac = a * c;
    ac->with_id("ac");
    auto v = ab + ac;
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    REQUIRE_THAT(v->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(ab->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(ac->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(8.0F, 1e-6));
    REQUIRE_THAT(b->m_grad, WithinAbs(2.0F, 1e-6));
    REQUIRE_THAT(c->m_grad, WithinAbs(2.0F, 1e-6));
  }

  SECTION("subtract same value") {
    auto a = val(7.F, "a");
    auto v = a - a;
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    REQUIRE_THAT(v->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(0.0F, 1e-6));
  }

  SECTION("numerical gradient check with finite differences") {
    const float h = 1e-4F;

    auto compute = [](float av, float bv) -> float {
      return (av * bv) + std::tanh(av + bv) + av - bv;
    };

    auto a = val(1.5F, "a");
    auto b = val(0.8F, "b");
    auto prod_val = a * b;
    prod_val->with_id("prod");
    auto sum_ab = a + b;
    sum_ab->with_id("sum");
    auto th = sum_ab->tanh();
    th->with_id("th");
    auto th_plus_a = th + a;
    th_plus_a->with_id("th_plus_a");
    auto v = prod_val + th_plus_a;
    v->with_id("v");
    auto final_v = v - b;
    final_v->with_id("final");
    engine::graph<engine::float_t> g(final_v);
    g.back_propagate();

    float f_ap = compute(1.5F + h, 0.8F);
    float f_am = compute(1.5F - h, 0.8F);
    float f_bp = compute(1.5F, 0.8F + h);
    float f_bm = compute(1.5F, 0.8F - h);
    float num_grad_a = (f_ap - f_am) / (2.0F * h);
    float num_grad_b = (f_bp - f_bm) / (2.0F * h);

    REQUIRE_THAT(a->m_grad, WithinAbs(num_grad_a, 2e-3F));
    REQUIRE_THAT(b->m_grad, WithinAbs(num_grad_b, 2e-3F));
  }

  SECTION("division backpropagation") {
    auto a = val(8.F, "a");
    auto b = val(2.F, "b");
    auto v = a / b;
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    REQUIRE_THAT(v->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(b->m_grad, WithinAbs(-2.0F, 1e-6));
  }

  SECTION("power backpropagation") {
    auto a = val(3.F, "a");
    auto v = a->pow(2.0F);
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    REQUIRE_THAT(v->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(6.0F, 1e-6));
  }

  SECTION("pow in chain backpropagation") {
    auto a = val(2.F, "a");
    auto b = val(3.F, "b");
    auto prod = a * b;
    prod->with_id("prod");
    auto v = prod->pow(3.0F);
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    REQUIRE_THAT(v->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(prod->m_grad, WithinAbs(108.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(324.0F, 1e-6));
    REQUIRE_THAT(b->m_grad, WithinAbs(216.0F, 1e-6));
  }

  SECTION("tanh shortcut vs manual exp-based implementation") {
    // --- builtin tanh ---
    auto x_builtin = val(0.5F, "x");
    auto builtin = x_builtin->tanh();
    builtin->with_id("builtin");
    engine::graph<engine::float_t> g_builtin(builtin);
    g_builtin.back_propagate();

    // --- manual tanh via exp ---
    auto x_manual = val(0.5F, "x");
    auto two = val(2.F);
    auto one = val(1.F);
    auto two_x = x_manual * two;
    two_x->with_id("2x");
    auto exp2x = two_x->exp();
    exp2x->with_id("exp2x");
    auto numerator = exp2x - one;
    numerator->with_id("num");
    auto denominator = exp2x + one;
    denominator->with_id("den");
    auto manual = numerator / denominator;
    manual->with_id("manual");
    engine::graph<engine::float_t> g_manual(manual);
    g_manual.back_propagate();

    REQUIRE_THAT(builtin->m_value, WithinAbs(manual->m_value, 1e-6));
    REQUIRE_THAT(x_builtin->m_grad, WithinAbs(x_manual->m_grad, 1e-6));
  }

  SECTION("requires_grad = false stops gradient") {
    auto a = val(4.F, "a");
    auto b = val(3.F, "b");
    b->set_requires_grad(false);
    REQUIRE_FALSE(b->requires_grad());
    REQUIRE(a->requires_grad());

    auto c = a * b;
    c->with_id("c");
    engine::graph<engine::float_t> g(c);
    g.back_propagate();
    REQUIRE_THAT(a->m_grad, WithinAbs(3.0F, 1e-6));
    REQUIRE(b->m_grad == 0.0F);
  }

  SECTION("frozen constant in deep chain") {
    auto a = val(2.F, "a");
    auto frozen_b = val(5.F, "b");
    frozen_b->set_requires_grad(false);
    auto c = val(1.F, "c");

    auto prod = a * frozen_b;
    prod->with_id("prod");
    auto v = prod + c;
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();

    REQUIRE_THAT(a->m_grad, WithinAbs(5.0F, 1e-6));
    REQUIRE(frozen_b->m_grad == 0.0F);
    REQUIRE_THAT(c->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(prod->m_grad, WithinAbs(1.0F, 1e-6));
  }

  SECTION("frozen intermediate node stops gradient to its children") {
    auto a = val(2.F, "a");
    auto b = val(3.F, "b");
    auto c_var = val(1.F, "c");
    auto d = val(4.F, "d");

    auto ab = a * b;
    ab->with_id("ab");
    auto abc = ab + c_var;
    abc->with_id("abc");
    abc->set_requires_grad(false);
    auto v = abc * d;
    v->with_id("v");

    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    print_dot(g, "graph_requires_grad.dot");

    REQUIRE_THAT(d->m_grad, WithinAbs(7.0F, 1e-6));
    REQUIRE(abc->m_grad == 0.0F);
    REQUIRE(ab->m_grad == 0.0F);
    REQUIRE(a->m_grad == 0.0F);
    REQUIRE(b->m_grad == 0.0F);
    REQUIRE(c_var->m_grad == 0.0F);
  }

  SECTION("division backpropagation") {
    auto a = val(8.F, "a");
    auto b = val(2.F, "b");
    auto v = a / b;
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    REQUIRE_THAT(v->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(0.5F, 1e-6));
    REQUIRE_THAT(b->m_grad, WithinAbs(-2.0F, 1e-6));
  }

  SECTION("power backpropagation") {
    auto a = val(3.F, "a");
    auto v = a->pow(2.0F);
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    REQUIRE_THAT(v->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(6.0F, 1e-6));
  }

  SECTION("pow in chain backpropagation") {
    auto a = val(2.F, "a");
    auto b = val(3.F, "b");
    auto prod = a * b;
    prod->with_id("prod");
    auto v = prod->pow(3.0F);
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    REQUIRE_THAT(v->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(prod->m_grad, WithinAbs(108.0F, 1e-6));
    REQUIRE_THAT(a->m_grad, WithinAbs(324.0F, 1e-6));
    REQUIRE_THAT(b->m_grad, WithinAbs(216.0F, 1e-6));
  }

  SECTION("tanh shortcut vs manual exp-based implementation") {
    auto x_builtin = val(0.5F, "x");
    auto builtin = x_builtin->tanh();
    builtin->with_id("builtin");
    engine::graph<engine::float_t> g_builtin(builtin);
    g_builtin.back_propagate();

    auto x_manual = val(0.5F, "x");
    auto two = val(2.F);
    auto one = val(1.F);
    auto two_x = x_manual * two;
    two_x->with_id("2x");
    auto exp2x = two_x->exp();
    exp2x->with_id("exp2x");
    auto numerator = exp2x - one;
    numerator->with_id("num");
    auto denominator = exp2x + one;
    denominator->with_id("den");
    auto manual = numerator / denominator;
    manual->with_id("manual");
    engine::graph<engine::float_t> g_manual(manual);
    g_manual.back_propagate();

    REQUIRE_THAT(builtin->m_value, WithinAbs(manual->m_value, 1e-6));
    REQUIRE_THAT(x_builtin->m_grad, WithinAbs(x_manual->m_grad, 1e-6));
  }

  SECTION("requires_grad = false stops gradient") {
    auto a = val(4.F, "a");
    auto b = val(3.F, "b");
    b->set_requires_grad(false);
    REQUIRE_FALSE(b->requires_grad());
    REQUIRE(a->requires_grad());

    auto c = a * b;
    c->with_id("c");
    engine::graph<engine::float_t> g(c);
    g.back_propagate();
    REQUIRE_THAT(a->m_grad, WithinAbs(3.0F, 1e-6));
    REQUIRE(b->m_grad == 0.0F);
  }

  SECTION("frozen constant in deep chain") {
    auto a = val(2.F, "a");
    auto frozen_b = val(5.F, "b");
    frozen_b->set_requires_grad(false);
    auto c = val(1.F, "c");

    auto prod = a * frozen_b;
    prod->with_id("prod");
    auto v = prod + c;
    v->with_id("v");
    engine::graph<engine::float_t> g(v);
    g.back_propagate();

    REQUIRE_THAT(a->m_grad, WithinAbs(5.0F, 1e-6));
    REQUIRE(frozen_b->m_grad == 0.0F);
    REQUIRE_THAT(c->m_grad, WithinAbs(1.0F, 1e-6));
    REQUIRE_THAT(prod->m_grad, WithinAbs(1.0F, 1e-6));
  }

  SECTION("frozen intermediate node stops gradient to its children") {
    auto a = val(2.F, "a");
    auto b = val(3.F, "b");
    auto c_var = val(1.F, "c");
    auto d = val(4.F, "d");

    auto ab = a * b;
    ab->with_id("ab");
    auto abc = ab + c_var;
    abc->with_id("abc");
    abc->set_requires_grad(false);
    auto v = abc * d;
    v->with_id("v");

    engine::graph<engine::float_t> g(v);
    g.back_propagate();
    print_dot(g, "graph_requires_grad.dot");

    REQUIRE_THAT(d->m_grad, WithinAbs(7.0F, 1e-6));
    REQUIRE(abc->m_grad == 0.0F);
    REQUIRE(ab->m_grad == 0.0F);
    REQUIRE(a->m_grad == 0.0F);
    REQUIRE(b->m_grad == 0.0F);
    REQUIRE(c_var->m_grad == 0.0F);
  }
}

// NOLINTEND(*-magic-numbers, *-identifier-length, *-avoid-do-while)
