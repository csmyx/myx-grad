#include <fmt/core.h>
#include <myx_grad/arena.h>
#include <myx_grad/engine.h>

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fstream>
using namespace Catch::Matchers;

// NOLINTBEGIN(*-magic-numbers, *-identifier-length, *-avoid-do-while)

TEST_CASE("test value", "[engine]") {
    auto print_dot = [](engine::Graph<float> &graph, engine::Value<float> &root, const std::string &file_name) {
        const std::string prefix_str = R"(digraph ComputeGraph {
rankdir=TB;
)";
        const std::string suffix_str = "}\n";
        auto dot_str = fmt::format("{}\n{}\n{}", prefix_str, graph.display(root), suffix_str);

        {
            std::ofstream dot_file(file_name);
            REQUIRE(dot_file.is_open());
            dot_file << dot_str;
            dot_file.close();
            fmt::println("Graph written to {}", file_name);
        }
    };

    SECTION("basic Value creation and arithmetic") {
        engine::Graph<int> graph;
        auto &val = graph.leaf(5);
        REQUIRE(val.value() == 5);
        auto &val1 = graph.leaf(3);
        REQUIRE(val1.value() == 3);
        REQUIRE(val.value() + val1.value() == 8);
        REQUIRE(val.value() - val1.value() == 2);
        REQUIRE(val.value() * val1.value() == 15);
        REQUIRE(val.value() / val1.value() == 1);
    }

    SECTION("simple addition with graph output") {
        engine::Graph<float> g;
        auto &v1 = g.leaf(2.5f);
        auto &v2 = g.leaf(3.5f);
        auto &v3 = v1 + v2;
        g.back_propagate(v3);
        print_dot(g, v3, "graph1.dot");
    }

    SECTION("multiplication chain backpropagation") {
        // v5 = (v1 + v2) * v4
        engine::Graph<float> g;
        auto &val1 = g.leaf(2.F, "n1");
        auto &val2 = g.leaf(3.F, "n2");
        auto &val3 = (val1 + val2).with_id("n3");
        auto &val4 = g.leaf(4.F, "n4");
        auto &val5 = (val3 * val4).with_id("n5");
        g.back_propagate(val5);
        print_dot(g, val5, "graph2.dot");
        REQUIRE(val5.grad_ == 1.0F);
        REQUIRE(val4.grad_ == 5.0F);
        REQUIRE(val3.grad_ == 4.0F);
        REQUIRE(val2.grad_ == 4.0F);
        REQUIRE(val1.grad_ == 4.0F);
    }

    SECTION("shared input value") {
        engine::Graph<float> g;
        auto &val1 = g.leaf(3.F, "n1");
        auto &val2 = (val1 + val1).with_id("n2");
        g.back_propagate(val2);
        print_dot(g, val2, "graph3.dot");
        REQUIRE(val1.grad_ == 2.0F);
    }

    SECTION("tanh activation backpropagation") {
        engine::Graph<float> g;
        auto &x1 = g.leaf(2.F, "x1");
        auto &x2 = g.leaf(0.F, "x2");
        auto &w1 = g.leaf(-3.F, "w1");
        auto &w2 = g.leaf(1.F, "w2");
        auto &b = g.leaf(6.8813735870195432F, "b");
        auto &wx1 = (x1 * w1).with_id("wx1");
        auto &wx2 = (x2 * w2).with_id("wx2");
        auto &wx = (wx1 + wx2).with_id("wx");
        auto &n = (wx + b).with_id("n");
        auto &o = g.tanh(n).with_id("o");
        g.back_propagate(o);
        print_dot(g, o, "graph_4.dot");
        REQUIRE_THAT(o.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(n.grad_, WithinAbs(0.5F, 1e-6));
        REQUIRE_THAT(wx1.grad_, WithinAbs(0.5F, 1e-6));
        REQUIRE_THAT(wx2.grad_, WithinAbs(0.5F, 1e-6));
        REQUIRE_THAT(w1.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(w2.grad_, WithinAbs(0.0F, 1e-6));
        REQUIRE_THAT(x1.grad_, WithinAbs(-1.5F, 1e-6));
        REQUIRE_THAT(x2.grad_, WithinAbs(0.5F, 1e-6));
    }

    SECTION("subtraction backpropagation") {
        //  d(a-b)/da = 1, d(a-b)/db = -1
        engine::Graph<float> g;
        auto &a = g.leaf(5.F, "a");
        auto &b = g.leaf(3.F, "b");
        auto &c = (a - b).with_id("c");
        g.back_propagate(c);
        REQUIRE_THAT(c.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(a.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(b.grad_, WithinAbs(-1.0F, 1e-6));
    }

    SECTION("deeper chain backpropagation") {
        //  v = (a*b + c) * d
        engine::Graph<float> g;
        auto &a = g.leaf(2.F, "a");
        auto &b = g.leaf(3.F, "b");
        auto &c = g.leaf(1.F, "c");
        auto &d = g.leaf(4.F, "d");
        auto &ab = (a * b).with_id("ab");     // 6
        auto &abc = (ab + c).with_id("abc");  // 7
        auto &v = (abc * d).with_id("v");     // 28
        g.back_propagate(v);
        REQUIRE_THAT(v.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(abc.grad_, WithinAbs(4.0F, 1e-6));  // d
        REQUIRE_THAT(d.grad_, WithinAbs(7.0F, 1e-6));    // a*b+c
        REQUIRE_THAT(ab.grad_, WithinAbs(4.0F, 1e-6));   // d * 1
        REQUIRE_THAT(c.grad_, WithinAbs(4.0F, 1e-6));    // d * 1
        REQUIRE_THAT(a.grad_, WithinAbs(12.0F, 1e-6));   // d * b
        REQUIRE_THAT(b.grad_, WithinAbs(8.0F, 1e-6));    // d * a
    }

    SECTION("DAG with shared node") {
        //  v = a*b + a*c
        engine::Graph<float> g;
        auto &a = g.leaf(2.F, "a");
        auto &b = g.leaf(3.F, "b");
        auto &c = g.leaf(5.F, "c");
        auto &ab = (a * b).with_id("ab");  // 6
        auto &ac = (a * c).with_id("ac");  // 10
        auto &v = (ab + ac).with_id("v");  // 16
        g.back_propagate(v);
        REQUIRE_THAT(v.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(ab.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(ac.grad_, WithinAbs(1.0F, 1e-6));
        // a accumulates gradient from both ab and ac paths: b + c = 3 + 5
        REQUIRE_THAT(a.grad_, WithinAbs(8.0F, 1e-6));
        REQUIRE_THAT(b.grad_, WithinAbs(2.0F, 1e-6));  // a
        REQUIRE_THAT(c.grad_, WithinAbs(2.0F, 1e-6));  // a
    }

    SECTION("subtract same value") {
        engine::Graph<float> g;
        auto &a = g.leaf(7.F, "a");
        auto &v = (a - a).with_id("v");
        g.back_propagate(v);
        REQUIRE_THAT(v.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(a.grad_, WithinAbs(0.0F, 1e-6));
    }

    SECTION("numerical gradient check with finite differences") {
        const float h = 1e-4F;

        auto compute = [](float av, float bv) -> float { return (av * bv) + std::tanh(av + bv) + av - bv; };

        engine::Graph<float> g;
        auto &a = g.leaf(1.5F, "a");
        auto &b = g.leaf(0.8F, "b");
        auto &prod = (a * b).with_id("prod");
        auto &sum_ab = (a + b).with_id("sum");
        auto &th = g.tanh(sum_ab).with_id("th");
        auto &th_plus_a = (th + a).with_id("th_plus_a");
        auto &v = (prod + th_plus_a).with_id("v");
        auto &final_v = (v - b).with_id("final");
        g.back_propagate(final_v);

        float f_ap = compute(1.5F + h, 0.8F);
        float f_am = compute(1.5F - h, 0.8F);
        float f_bp = compute(1.5F, 0.8F + h);
        float f_bm = compute(1.5F, 0.8F - h);
        float num_grad_a = (f_ap - f_am) / (2.0F * h);
        float num_grad_b = (f_bp - f_bm) / (2.0F * h);

        REQUIRE_THAT(a.grad_, WithinAbs(num_grad_a, 2e-3F));
        REQUIRE_THAT(b.grad_, WithinAbs(num_grad_b, 2e-3F));
    }

    SECTION("division backpropagation") {
        //  d(a/b)/da = 1/b,  d(a/b)/db = -a/b^2
        engine::Graph<float> g;
        auto &a = g.leaf(8.F, "a");
        auto &b = g.leaf(2.F, "b");
        auto &v = (a / b).with_id("v");
        g.back_propagate(v);
        REQUIRE_THAT(v.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(a.grad_, WithinAbs(0.5F, 1e-6));   // 1/2
        REQUIRE_THAT(b.grad_, WithinAbs(-2.0F, 1e-6));  // -8/4
    }

    SECTION("power backpropagation") {
        //  d(base^exp)/d(base) = exp * base^(exp-1)
        engine::Graph<float> g;
        auto &a = g.leaf(3.F, "a");
        auto &v = g.pow(a, 2.0F).with_id("v");  // v = a^2 = 9
        g.back_propagate(v);
        REQUIRE_THAT(v.grad_, WithinAbs(1.0F, 1e-6));
        REQUIRE_THAT(a.grad_, WithinAbs(6.0F, 1e-6));  // 2*3
    }

    SECTION("pow in chain backpropagation") {
        //  v = (a * b)^3
        engine::Graph<float> g;
        auto &a = g.leaf(2.F, "a");
        auto &b = g.leaf(3.F, "b");
        auto &prod = (a * b).with_id("prod");      // 6
        auto &v = g.pow(prod, 3.0F).with_id("v");  // 216
        g.back_propagate(v);
        REQUIRE_THAT(v.grad_, WithinAbs(1.0F, 1e-6));
        // prod grad: 3 * 6^2 = 108
        REQUIRE_THAT(prod.grad_, WithinAbs(108.0F, 1e-6));
        // a grad: 108 * b = 108 * 3 = 324
        REQUIRE_THAT(a.grad_, WithinAbs(324.0F, 1e-6));
        // b grad: 108 * a = 108 * 2 = 216
        REQUIRE_THAT(b.grad_, WithinAbs(216.0F, 1e-6));
    }

    SECTION("tanh shortcut vs manual exp-based implementation") {
        // Compare builtin tanh() against manual composition:
        //   tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)

        // --- builtin tanh ---
        engine::Graph<float> g1;
        auto &x_builtin = g1.leaf(0.5F, "x");
        auto &builtin = g1.tanh(x_builtin).with_id("builtin");
        g1.back_propagate(builtin);

        // --- manual tanh via exp ---
        engine::Graph<float> g2;
        auto &x_manual = g2.leaf(0.5F, "x");
        auto &two = g2.leaf(2.F).set_requires_grad(false);
        auto &one = g2.leaf(1.F).set_requires_grad(false);
        auto &two_x = (x_manual * two).with_id("2x");
        auto &exp2x = g2.exp(two_x).with_id("exp2x");
        auto &numerator = (exp2x - one).with_id("num");
        auto &denominator = (exp2x + one).with_id("den");
        auto &manual = (numerator / denominator).with_id("manual");
        g2.back_propagate(manual);

        // Forward values must match
        REQUIRE_THAT(builtin.value_, WithinAbs(manual.value_, 1e-6));

        // Gradient w.r.t x must match
        REQUIRE_THAT(x_builtin.grad_, WithinAbs(x_manual.grad_, 1e-6));
    }

    SECTION("requires_grad = false stops gradient") {
        engine::Graph<float> g;
        auto &a = g.leaf(4.F, "a");
        auto &b = g.leaf(3.F, "b").set_requires_grad(false);
        REQUIRE_FALSE(b.requires_grad());
        REQUIRE(a.requires_grad());

        auto &c = (a * b).with_id("c");  // 12
        g.back_propagate(c);
        REQUIRE_THAT(a.grad_, WithinAbs(3.0F, 1e-6));  // dc/da = b = 3
        REQUIRE(b.grad_ == 0.0F);                      // gradient blocked
    }

    SECTION("frozen constant in deep chain") {
        engine::Graph<float> g;
        auto &a = g.leaf(2.F, "a");
        auto &frozen_b = g.leaf(5.F, "b").set_requires_grad(false);
        auto &c = g.leaf(1.F, "c");

        auto &prod = (a * frozen_b).with_id("prod");  // 10
        auto &v = (prod + c).with_id("v");            // 11
        g.back_propagate(v);

        REQUIRE_THAT(a.grad_, WithinAbs(5.0F, 1e-6));     // dv/da = frozen_b = 5
        REQUIRE(frozen_b.grad_ == 0.0F);                  // blocked
        REQUIRE_THAT(c.grad_, WithinAbs(1.0F, 1e-6));     // dv/dc = 1
        REQUIRE_THAT(prod.grad_, WithinAbs(1.0F, 1e-6));  // dv/d(prod) = 1
    }

    SECTION("frozen intermediate node stops gradient to its children") {
        engine::Graph<float> g;
        auto &a = g.leaf(2.F, "a");
        auto &b = g.leaf(3.F, "b");
        auto &c_var = g.leaf(1.F, "c");
        auto &d = g.leaf(4.F, "d");

        auto &ab = (a * b).with_id("ab");                                  // 6
        auto &abc = (ab + c_var).with_id("abc").set_requires_grad(false);  // 7
        auto &v = (abc * d).with_id("v");                                  // 28

        g.back_propagate(v);
        print_dot(g, v, "graph_requires_grad.dot");

        // d receives gradient: dv/dd = abc.value_ = 7
        REQUIRE_THAT(d.grad_, WithinAbs(7.0F, 1e-6));

        // abc and everything below it gets nothing
        REQUIRE(abc.grad_ == 0.0F);
        REQUIRE(ab.grad_ == 0.0F);
        REQUIRE(a.grad_ == 0.0F);
        REQUIRE(b.grad_ == 0.0F);
        REQUIRE(c_var.grad_ == 0.0F);
    }
}

TEST_CASE("arena allocation", "[arena]") {
    SECTION("alloc returns non-null for small size") {
        util::Arena arena;
        void *ptr = arena.alloc(16, alignof(std::max_align_t));
        REQUIRE(ptr != nullptr);
        // Write to verify the memory is usable
        auto *iptr = static_cast<int *>(ptr);
        *iptr = 42;
        REQUIRE(*iptr == 42);
    }

    SECTION("alloc respects alignment") {
        util::Arena arena;
        constexpr std::size_t align = alignof(std::max_align_t);  // max guaranteed by new[]
        void *ptr = arena.alloc(8, align);
        REQUIRE(ptr != nullptr);
        auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        REQUIRE(addr % align == 0);
    }

    SECTION("alloc_unaligned returns non-null") {
        util::Arena arena;
        void *ptr = arena.alloc_unaligned(32);
        REQUIRE(ptr != nullptr);
        std::memset(ptr, 0xAB, 32);
        auto *cptr = static_cast<unsigned char *>(ptr);
        REQUIRE(cptr[0] == 0xAB);
        REQUIRE(cptr[31] == 0xAB);
    }

    SECTION("multiple small allocations are contiguous within a chunk") {
        util::Arena arena;
        void *p1 = arena.alloc_unaligned(8);
        void *p2 = arena.alloc_unaligned(8);
        REQUIRE(p1 != nullptr);
        REQUIRE(p2 != nullptr);
        REQUIRE(p2 == static_cast<char *>(p1) + 8);
    }

    SECTION("alloc handles big allocation (> chunk_size/4)") {
        util::Arena arena;
        constexpr std::size_t big = 2048;  // exceeds chunk_size/4 (1024)
        void *ptr = arena.alloc_unaligned(big);
        REQUIRE(ptr != nullptr);
        std::memset(ptr, 0xCD, big);
        auto *cptr = static_cast<unsigned char *>(ptr);
        REQUIRE(cptr[0] == 0xCD);
        REQUIRE(cptr[big - 1] == 0xCD);
    }

    SECTION("templated alloc<T> returns properly aligned storage") {
        util::Arena arena;
        void *ptr = arena.alloc<double>();
        REQUIRE(ptr != nullptr);
        auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        REQUIRE(addr % alignof(double) == 0);
        auto *dptr = static_cast<double *>(ptr);
        *dptr = 3.14;
        REQUIRE(*dptr == 3.14);
    }

    SECTION("many allocations exhaust and allocate new chunks") {
        util::Arena arena;
        // Allocate enough to exceed a single chunk (4096 bytes)
        std::vector<void *> ptrs;
        for (int i = 0; i < 512; ++i) {
            void *ptr = arena.alloc_unaligned(16);
            REQUIRE(ptr != nullptr);
            ptrs.push_back(ptr);
        }
        // All pointers should be unique
        std::unordered_set<void *> seen(ptrs.begin(), ptrs.end());
        REQUIRE(seen.size() == ptrs.size());
    }

    SECTION("arena destructor frees without crashing") {
        util::Arena *arena = new util::Arena();
        for (int i = 0; i < 100; ++i) {
            [[maybe_unused]] void *ptr = arena->alloc_unaligned(64);
        }
        delete arena;  // should not crash
    }

    SECTION("arena is movable") {
        util::Arena arena;
        void *ptr1 = arena.alloc_unaligned(16);
        util::Arena moved = std::move(arena);
        // The moved arena should still own the previously allocated chunk
        REQUIRE(ptr1 != nullptr);
        // New allocation from moved arena should work
        void *ptr2 = moved.alloc_unaligned(16);
        REQUIRE(ptr2 != nullptr);
    }
}


// NOLINTEND(*-magic-numbers, *-identifier-length, *-avoid-do-while)
// NOLINTBEGIN(*-magic-numbers, *-identifier-length)