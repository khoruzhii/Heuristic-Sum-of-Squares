// Test for newton::enumerate_lattice
// Compile: g++ -std=c++20 -O2 -I src -I third_party tests/test_newton.cpp -o tests/test_newton
// Run:     tests/test_newton

#include "newton.h"
#include "fmt.h"

#include <cassert>
#include <set>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <functional>

using Point = std::vector<int>;
using PointSet = std::set<Point>;

// Brute-force reference: enumerate all points in [lo,hi], filter by Ax+b<=0
PointSet brute_force(const newton::Polytope& poly,
                     const std::vector<int>& lo,
                     const std::vector<int>& hi) {
    int n = poly.n;
    PointSet result;
    std::vector<int> pt(n);

    std::function<void(int)> gen = [&](int d) {
        if (d == n) {
            bool ok = true;
            for (int i = 0; i < poly.m; ++i) {
                double val = poly.b[i];
                for (int j = 0; j < n; ++j)
                    val += poly.a(i, j) * pt[j];
                if (val > 1e-9) { ok = false; break; }
            }
            if (ok) result.insert(pt);
            return;
        }
        for (int x = lo[d]; x <= hi[d]; ++x) {
            pt[d] = x;
            gen(d + 1);
        }
    };
    gen(0);
    return result;
}

// --- Test cases ---

void test_unit_cube() {
    // 2D unit square: 0 <= x,y <= 2
    // Inequalities: -x <= 0, -y <= 0, x <= 2, y <= 2
    // i.e. A = [[-1,0],[0,-1],[1,0],[0,1]], b = [0,0,-2,-2]
    // Ax+b<=0: -x+0<=0, -y+0<=0, x-2<=0, y-2<=0
    newton::Polytope poly;
    poly.n = 2;
    poly.m = 4;
    poly.A = {-1, 0,  0, -1,  1, 0,  0, 1};
    poly.b = { 0, 0, -2, -2};

    auto pts = newton::enumerate_lattice(poly, {0, 0}, {2, 2});
    PointSet got(pts.begin(), pts.end());
    PointSet expected;
    for (int x = 0; x <= 2; ++x)
        for (int y = 0; y <= 2; ++y)
            expected.insert({x, y});

    assert(got == expected);
    fmt::print("test_unit_cube: OK ({} points)\n", got.size());
}

void test_triangle() {
    // 2D triangle: x >= 0, y >= 0, x + y <= 3
    // A = [[-1,0],[0,-1],[1,1]], b = [0,0,-3]
    newton::Polytope poly;
    poly.n = 2;
    poly.m = 3;
    poly.A = {-1, 0,  0, -1,  1, 1};
    poly.b = { 0, 0, -3};

    auto pts = newton::enumerate_lattice(poly, {0, 0}, {3, 3});
    PointSet got(pts.begin(), pts.end());

    PointSet expected;
    for (int x = 0; x <= 3; ++x)
        for (int y = 0; y <= 3 - x; ++y)
            expected.insert({x, y});

    assert(got == expected);
    fmt::print("test_triangle: OK ({} points)\n", got.size());
}

void test_simplex_3d() {
    // 3D simplex: x,y,z >= 0, x+y+z <= 5
    newton::Polytope poly;
    poly.n = 3;
    poly.m = 4;
    poly.A = {-1, 0, 0,  0, -1, 0,  0, 0, -1,  1, 1, 1};
    poly.b = { 0, 0, 0, -5};

    auto pts = newton::enumerate_lattice(poly, {0, 0, 0}, {5, 5, 5});
    PointSet got(pts.begin(), pts.end());
    auto ref = brute_force(poly, {0, 0, 0}, {5, 5, 5});

    assert(got == ref);
    fmt::print("test_simplex_3d: OK ({} points)\n", got.size());
}

void test_empty() {
    // Contradictory: x >= 1 and x <= 0
    newton::Polytope poly;
    poly.n = 1;
    poly.m = 2;
    poly.A = {-1, 1};
    poly.b = {1, 0};  // -x + 1 <= 0 => x >= 1;  x + 0 <= 0 => x <= 0

    auto pts = newton::enumerate_lattice(poly, {-5}, {5});
    assert(pts.empty());
    fmt::print("test_empty: OK\n");
}

void test_vs_brute_force_random(int n, int degree, int seed) {
    // Generate a random polytope from a "Newton-like" setup:
    // random support points in [0, 2*degree]^n, compute ConvexHull-like inequalities
    // We'll use a simplex-like structure for controlled testing:
    // x_i >= 0, sum(x_i) <= degree
    std::mt19937 rng(seed);

    newton::Polytope poly;
    poly.n = n;
    poly.m = n + 1;
    poly.A.resize(poly.m * n, 0.0);
    poly.b.resize(poly.m, 0.0);

    // x_i >= 0
    for (int i = 0; i < n; ++i)
        poly.A[i * n + i] = -1.0;

    // sum(x_i) <= degree
    for (int j = 0; j < n; ++j)
        poly.A[n * n + j] = 1.0;
    poly.b[n] = -degree;

    // Add a few random half-space constraints for variety
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    int extra = 3;
    poly.m += extra;
    for (int e = 0; e < extra; ++e) {
        std::vector<double> row(n);
        double rhs = 0;
        for (int j = 0; j < n; ++j) {
            row[j] = dist(rng);
            rhs += std::abs(row[j]) * degree / 2; // make it non-trivial
        }
        poly.A.insert(poly.A.end(), row.begin(), row.end());
        poly.b.push_back(-rhs);
    }

    std::vector<int> lo(n, 0), hi(n, degree);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto pts = newton::enumerate_lattice(poly, lo, hi);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dfs_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    PointSet got(pts.begin(), pts.end());

    t0 = std::chrono::high_resolution_clock::now();
    auto ref = brute_force(poly, lo, hi);
    t1 = std::chrono::high_resolution_clock::now();
    double bf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    assert(got == ref);
    fmt::print("test_random n={} d={}: OK ({} points, dfs={:.1f}ms, brute={:.1f}ms, speedup={:.0f}x)\n",
               n, degree, got.size(), dfs_ms, bf_ms,
               bf_ms > 0 ? bf_ms / dfs_ms : 0);
}

void test_high_dim_perf(int n, int degree) {
    // Simplex: x_i >= 0, sum <= degree. Only DFS, no brute-force reference.
    newton::Polytope poly;
    poly.n = n;
    poly.m = n + 1;
    poly.A.resize(poly.m * n, 0.0);
    poly.b.resize(poly.m, 0.0);

    for (int i = 0; i < n; ++i)
        poly.A[i * n + i] = -1.0;
    for (int j = 0; j < n; ++j)
        poly.A[n * n + j] = 1.0;
    poly.b[n] = -degree;

    std::vector<int> lo(n, 0), hi(n, degree);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto pts = newton::enumerate_lattice(poly, lo, hi);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Expected count: C(n+degree, n)
    long long expected = 1;
    for (int i = 1; i <= n; ++i)
        expected = expected * (degree + i) / i;

    assert((long long)pts.size() == expected);
    fmt::print("test_perf n={} d={}: {} points in {:.1f}ms (expected {})\n",
               n, degree, pts.size(), ms, expected);
}

int main() {
    fmt::print("=== Basic tests ===\n");
    test_unit_cube();
    test_triangle();
    test_simplex_3d();
    test_empty();

    fmt::print("\n=== Random correctness (DFS vs brute-force) ===\n");
    test_vs_brute_force_random(4, 6, 1);
    test_vs_brute_force_random(5, 5, 2);
    test_vs_brute_force_random(6, 4, 3);
    test_vs_brute_force_random(6, 6, 4);

    fmt::print("\n=== Performance (DFS only) ===\n");
    test_high_dim_perf(6, 6);    // C(12,6) = 924
    test_high_dim_perf(6, 10);   // C(16,6) = 8008
    test_high_dim_perf(8, 10);   // C(18,8) = 43758

    fmt::print("\nAll tests passed.\n");
    return 0;
}
