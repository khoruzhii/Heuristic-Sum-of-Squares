#pragma once
// Lattice point enumeration inside a convex polytope defined by Ax + b <= 0.
// Uses recursive DFS with coordinate-wise bound propagation (pruning).

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace newton {

struct Polytope {
    int n;                          // dimension
    int m;                          // number of half-space inequalities
    std::vector<double> A;          // row-major m x n
    std::vector<double> b;          // length m

    double a(int i, int j) const { return A[i * n + j]; }
};

// Enumerate all integer lattice points x satisfying A x + b <= 0.
//
// lo, hi: bounding box (per-coordinate).
// Returns list of points, each of length n.
inline std::vector<std::vector<int>> enumerate_lattice(
        const Polytope& poly,
        const std::vector<int>& lo_in,
        const std::vector<int>& hi_in)
{
    const int n = poly.n;
    const int m = poly.m;

    // --- Coordinate reordering: narrowest range first for better pruning ---
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return (hi_in[a] - lo_in[a]) < (hi_in[b] - lo_in[b]);
    });

    std::vector<int> lo(n), hi(n);
    std::vector<double> A(m * n);
    for (int j = 0; j < n; ++j) {
        lo[j] = lo_in[order[j]];
        hi[j] = hi_in[order[j]];
        for (int i = 0; i < m; ++i)
            A[i * n + j] = poly.A[i * n + order[j]];
    }

    // Access helper for permuted A
    auto a = [&](int i, int j) -> double { return A[i * n + j]; };

    // Precompute suffix bounds for rest-of-coordinates.
    // For inequality i at depth d (fixing coords 0..d-1, about to branch on d):
    //   rest_lo[i*n+d] = sum_{j>d} min(A[i][j]*lo[j], A[i][j]*hi[j])
    // This is the minimum possible contribution from coordinates after d.
    std::vector<double> rest_lo(m * n, 0.0);

    for (int i = 0; i < m; ++i) {
        double slo = 0.0;
        for (int d = n - 1; d >= 0; --d) {
            rest_lo[i * n + d] = slo;
            double v1 = a(i, d) * lo[d];
            double v2 = a(i, d) * hi[d];
            slo += std::min(v1, v2);
        }
    }

    // Per-row tolerance: eps * ||a_i||  (matches Python implementation)
    constexpr double eps = 1e-6;
    std::vector<double> row_tol(m);
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j)
            s += a(i, j) * a(i, j);
        row_tol[i] = eps * std::sqrt(s);
    }

    // Flat result storage (avoids per-point vector allocation)
    std::vector<int> flat_result;
    std::vector<int> point(n);
    // residual[i] = b[i] + sum_{j < depth} A[i][j] * point[j]
    std::vector<double> residual(poly.b);

    // Compute tight [clo, chi] for coordinate `depth` from current residual.
    auto compute_bounds = [&](int depth) -> std::pair<int, int> {
        int clo = lo[depth], chi = hi[depth];
        for (int i = 0; i < m; ++i) {
            double tol = row_tol[i];
            double a_id = a(i, depth);
            if (std::abs(a_id) < 1e-15) {
                if (residual[i] + rest_lo[i * n + depth] > tol) {
                    return {1, 0}; // infeasible
                }
                continue;
            }
            // Need: residual[i] + a_id * x + S <= tol for SOME S >= rest_lo.
            double rhs = -(residual[i] + rest_lo[i * n + depth] - tol);
            double bound = rhs / a_id;
            if (a_id > 0) {
                double v = std::floor(bound + 1e-9);
                if (v < lo[depth]) { return {1, 0}; }
                if (v < chi) chi = (int)v;
            } else {
                double v = std::ceil(bound - 1e-9);
                if (v > hi[depth]) { return {1, 0}; }
                if (v > clo) clo = (int)v;
            }
            if (clo > chi) return {clo, chi};
        }
        return {clo, chi};
    };

    auto recurse = [&](auto& self, int depth) -> void {
        auto [clo, chi] = compute_bounds(depth);

        bool is_leaf = (depth == n - 1);

        for (int x = clo; x <= chi; ++x) {
            point[depth] = x;

            if (is_leaf) {
                // Check all constraints at this complete point
                bool ok = true;
                for (int i = 0; i < m; ++i) {
                    if (residual[i] + a(i, depth) * x > row_tol[i]) {
                        ok = false;
                        break;
                    }
                }
                if (ok)
                    flat_result.insert(flat_result.end(), point.begin(), point.end());
            } else {
                // Update residual and recurse
                for (int i = 0; i < m; ++i)
                    residual[i] += a(i, depth) * x;

                self(self, depth + 1);

                for (int i = 0; i < m; ++i)
                    residual[i] -= a(i, depth) * x;
            }
        }
    };

    recurse(recurse, 0);

    // Unpermute coordinates back to original order
    size_t k = flat_result.size() / n;
    std::vector<std::vector<int>> result(k);
    for (size_t i = 0; i < k; ++i) {
        result[i].resize(n);
        for (int j = 0; j < n; ++j)
            result[i][order[j]] = flat_result[i * n + j];
    }

    return result;
}

} // namespace newton
