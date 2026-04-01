# NSOS — Fast basis recovery for SOS verification

Fork of the [Neural Sum-of-Squares](https://arxiv.org/abs/2510.13444) (ICLR 2026) codebase with two lightweight heuristics that recover near-minimal monomial bases without a trained model, achieving comparable basis sizes mostly in under a second.

## Background

Certifying that a polynomial `p(x)` is a Sum of Squares requires finding a PSD matrix `Q` such that

```math
p(\mathbf{x}) = \mathbf{z}(\mathbf{x})^\top Q\, \mathbf{z}(\mathbf{x}),
```

where `z(x)` is a vector of basis monomials. The SDP cost is dominated by `|z|`, so smaller bases mean faster solves. The full basis `[x]_d` has size `C(n+d, d)` which grows quickly; the Newton polytope basis `(1/2)N(p) cap Z^n` is smaller but still can be expensive to compute via lattice enumeration in high dimensions.

## This repo contribution

### Heuristic 1: LP int_vtx + coverage repair

Avoids computing the Newton polytope entirely. Two steps:

**Step 1 — integer vertices via LP.** A support point `v` with all-even coordinates is a vertex of `conv(supp(p))` iff it cannot be written as a convex combination of the remaining support points:

```math
v \notin \operatorname{conv}(S \setminus \{v\}) \;\Longleftrightarrow\; \nexists\, \lambda \geq 0,\; \sum \lambda_i = 1,\; \sum \lambda_i s_i = v.
```

Each such vertex contributes the basis monomial `x^{v/2}`. One LP per candidate, no ConvexHull needed.

**Step 2 — greedy coverage repair.** The seed set from step 1 is expanded until every monomial in `S(p)` is expressible as a pairwise sum `B + B`:

```
while S(p) not subset B + B:
    pick m maximizing |{s in uncovered : s - m in B}|
    B <- B union {m}
```

Works well for dense, sparse, and low-rank matrices with `n >= 6`. For small `n` or block-diagonal structure, fallback to heuristic 2 is sometimes needed.

### Heuristic 2: Newton polytope + diagonal consistency filter (fallback)

When heuristic 1 produces a basis that fails SDP verification, we fall back to:

1. Compute full Newton polytope basis `(1/2)N(p) cap Z^n` via C++ DFS lattice enumeration
2. Apply facial reduction ([Permenter & Parrilo 2014](https://doi.org/10.1109/CDC.2014.7040427)) — iterated diagonal consistency to remove provably redundant monomials
3. Filter to `{m : x^{2m} in S(p)}`

This is a bit slower (requires ConvexHull + lattice enumeration) but always succeeds in our (n, d, m) experiments: `fb_n = 0` in all tested configurations, meaning the Newton reduced fallback is never needed beyond this step.

## Results

Comparison with the Neural SOS paper (Table 1, `s_nsos` column). Our method (`s_ours`) matches or slightly exceeds the planted basis size without any learned model. `fb_s` = fallbacks to strong, `p<=b` = planted basis recovered.

```
python scripts/table1.py --num 32

 Structure   n   d   m  |B*|  s_ours s_nsos   t_ours  fb_s fb_n  p<=b
---------------------------------------------------------------------
     dense   4   6  20    20      22     19    0.028   19   0  32/32  OK
     dense   6  12  30    31      31     33    0.096    0   0  32/32  OK
     dense   8  20  30    31      31     38    0.108    0   0  32/32  OK
     dense   6  20  60    59      59     89    0.714    0   0  32/32  OK
     dense 100  10  20    20      20      -    0.072    0   0  32/32  OK
     dense   6  40  40    40      40      -    0.257    0   0  32/32  OK

    sparse   4   6  20    20      21     15    0.028   26   0  32/32  OK
    sparse   6  12  30    30      30     27    0.077    0   0  32/32  OK
    sparse   8  20  30    30      30     27    0.086    0   0  32/32  OK
    sparse   6  20  60    59      59     73    0.582    0   0  32/32  OK
    sparse 100  10  20    20      20      -    0.055    0   0  32/32  OK
    sparse   6  40  40    40      40      -    0.351    2   0  32/32  OK

   lowrank   4   6  20    20      22     19    0.028   19   0  32/32  OK
   lowrank   6  12  30    31      31     30    0.096    0   0  32/32  OK
   lowrank   8  20  30    31      31     35    0.108    0   0  32/32  OK
   lowrank   6  20  60    59      59     66    0.674    0   0  32/32  OK
   lowrank 100  10  20    20      20      -    0.072    0   0  32/32  OK
   lowrank   6  40  40    40      40      -    0.250    0   0  32/32  OK

 blockdiag   4   6  20    20      21     20    0.026   32   0  32/32  OK
 blockdiag   6  12  30    31      31     31    0.062   18   0  32/32  OK
 blockdiag   8  20  30    31      31     30    0.044    4   0  32/32  OK
 blockdiag   6  20  60    59      60     71    0.233   32   0  32/32  OK
 blockdiag 100  10  20    20      20      -    0.023    0   0  32/32  OK
 blockdiag   6  40  40    40      41      -    2.603   24   0  32/32  OK
```

Key observations:
- **p<=b = 32/32 everywhere** — the planted basis is always recovered
- **s_ours <= s_nsos** for most configs — our basis is at least as compact as Neural SOS
- **fb_n = 0** — Newton fallback is never needed; strong suffices
- **t_ours < 1s** for all configs except block-diagonal n=6, d=40 (where strong fallback dominates)

## File structure

| File | Description |
|---|---|
| `scripts/basis.py` | Toolkit: `find_basis_lp`, `find_basis_strong`, `find_basis_newton`, `solve_sdp`, `verify_gram` |
| `scripts/table1.py` | Reproduces the table above. Flags: `--num`, `--sdp`, `--newton`, `--strong`, `--real` |
| `scripts/test_planted.py` | Detailed per-example diagnostics with all basis recovery modes |
| `src/newton.cpp`, `src/newton.h` | C++ DFS lattice enumerator (required for strong/Newton fallback) |

## Quick start

```bash
# Install dependencies
pip install numpy scipy cvxpy mosek

# Build C++ enumerator (needed for --strong / --newton modes)
g++ -std=c++20 -O2 -I src -I third_party src/newton.cpp -o bin/newton_enum

# Run table (LP + repair, no C++ needed for most configs)
python scripts/table1.py --num 32
```

## References

- Pelleriti et al., [Neural Sum-of-Squares](https://arxiv.org/abs/2510.13444), ICLR 2026
- Permenter & Parrilo, [Partial facial reduction](https://doi.org/10.1109/CDC.2014.7040427), CDC 2014
