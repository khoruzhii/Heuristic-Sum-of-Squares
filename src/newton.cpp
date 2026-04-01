// Newton polytope lattice enumeration via DFS.
// Reads A.npy, b.npy, lo.npy, hi.npy from data/tmp/
// Writes result.npy (k x n int32) to data/tmp/
//
// Build: g++ -std=c++20 -O2 -I src -I third_party src/newton.cpp -o bin/newton_enum

#include "newton.h"
#include "cnpy.h"

#include <string>

static const std::string DIR = "data/tmp/";

int main() {
    auto A_arr  = cnpy::npy_load(DIR + "A.npy");
    auto b_arr  = cnpy::npy_load(DIR + "b.npy");
    auto lo_arr = cnpy::npy_load(DIR + "lo.npy");
    auto hi_arr = cnpy::npy_load(DIR + "hi.npy");

    int m = (int)A_arr.shape[0];
    int n = (int)A_arr.shape[1];

    newton::Polytope poly;
    poly.n = n;
    poly.m = m;
    poly.A = A_arr.as_vec<double>();
    poly.b = b_arr.as_vec<double>();

    auto lo_i32 = lo_arr.as_vec<int32_t>();
    auto hi_i32 = hi_arr.as_vec<int32_t>();
    std::vector<int> lo(lo_i32.begin(), lo_i32.end());
    std::vector<int> hi(hi_i32.begin(), hi_i32.end());

    auto points = newton::enumerate_lattice(poly, lo, hi);

    size_t k = points.size();
    std::vector<int32_t> flat(k * n);
    for (size_t i = 0; i < k; ++i)
        for (int j = 0; j < n; ++j)
            flat[i * n + j] = (int32_t)points[i][j];

    cnpy::npy_save(DIR + "result.npy", flat.data(),
                   {k, (size_t)n});

    return 0;
}
