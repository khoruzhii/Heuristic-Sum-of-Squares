"""
Test properties of planted basis vs half-Newton polytope:
  check_0: planted basis subset of Newton basis (lattice points in 1/2 N(p))  [--le]
  check_5: planted basis subset of reduced/strong basis  [--le]
  check_1: integer vertices of 1/2 N(p) subset of planted basis

Usage: python scripts/test_planted.py [--n 6] [--d 12] [--m 30] [--num 16] [--matrix dense] [--le] [--sdp] [--strong]
  e.g. python scripts/test_planted.py --n 8 --d 20 --m 30 --num 10 --matrix dense --lp
"""
import sys, types, argparse
sos_transformer = types.ModuleType('sos_transformer')
sos_transformer.__path__ = ['sos/src']
sys.modules['sos_transformer'] = sos_transformer

import os, time, subprocess
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from data_generation.monomials.monomial_sparseuniform import SparseUniformBasisSampler
from data_generation.matrix.matrix_simple import SimpleRandomPSDSampler
from data_generation.matrix.matrix_sparse import SparsePSDSampler
from data_generation.matrix.matrix_lowrank import LowRankPSDSampler
from data_generation.matrix.matrix_blockdiag import BlockDiagonalPSDSampler
from data_generation.polynomials.polynomial_sos import SOSPolynomialSampler
from data_generation.monomials.monomials import Monomial
from sdp_solver.cvxpy_solver import CVXPYSOSSolver

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWTON_BIN = os.path.join(REPO_ROOT, "bin", "newton_enum.exe")
TMP_DIR = os.path.join(REPO_ROOT, "data", "tmp")

def _half_hull_inequalities(pts):
    """Return (A, b) for 1/2 N(p): A x + b <= 0, handling degenerate cases via SVD."""
    try:
        hull = ConvexHull(pts)
        eq = hull.equations
        _, idx = np.unique(np.round(eq, 10), axis=0, return_index=True)
        eq = eq[idx]
        A = eq[:, :-1]
        b = eq[:, -1] / 2.0
        return A, b
    except Exception:
        pass

    if len(pts) == 1 or np.allclose(pts, pts[0]):
        n = pts.shape[1]
        A = np.vstack([np.eye(n), -np.eye(n)])
        p_half = pts[0] / 2.0
        b = np.concatenate([-p_half, p_half])
        return A, b

    centered = pts - pts.mean(axis=0)
    rank = np.linalg.matrix_rank(centered, tol=1e-10)
    mean = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=True)
    dirs = vh[:rank]

    projs = (pts - mean) @ dirs.T
    hull = ConvexHull(projs)
    A_proj = hull.equations[:, :-1]
    b_proj = hull.equations[:, -1]

    A_full = A_proj @ dirs
    b_full = b_proj - A_full @ mean

    null_dirs = vh[rank:]
    if len(null_dirs) > 0:
        A_null = np.vstack([null_dirs, -null_dirs])
        b_null = np.concatenate([-null_dirs @ mean, null_dirs @ mean])
        A_full = np.vstack([A_full, A_null])
        b_full = np.concatenate([b_full, b_null])

    b_full = b_full / 2.0
    return A_full, b_full


def newton_basis_cpp(exponent_vectors):
    """Compute lattice points in 1/2 N(p) using C++ DFS enumeration."""
    pts = np.asarray(exponent_vectors, dtype=float)
    if pts.size == 0:
        return None, {}

    t0 = time.perf_counter()
    try:
        A, b = _half_hull_inequalities(pts)
    except Exception:
        return None, {}
    t_hull = time.perf_counter() - t0

    margin = 1e-7
    lo = np.ceil(np.min(pts, axis=0) / 2 - margin).astype(np.int32)
    hi = np.floor(np.max(pts, axis=0) / 2 + margin).astype(np.int32)

    t0 = time.perf_counter()
    np.save(os.path.join(TMP_DIR, "A.npy"), A)
    np.save(os.path.join(TMP_DIR, "b.npy"), b)
    np.save(os.path.join(TMP_DIR, "lo.npy"), lo)
    np.save(os.path.join(TMP_DIR, "hi.npy"), hi)
    t_io_write = time.perf_counter() - t0

    t0 = time.perf_counter()
    subprocess.run([NEWTON_BIN], check=True, capture_output=True, cwd=REPO_ROOT)
    t_le = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = np.load(os.path.join(TMP_DIR, "result.npy"))
    t_io_read = time.perf_counter() - t0

    timing = {"hull": t_hull, "le": t_le, "io": t_io_write + t_io_read}
    if result.size == 0:
        return None, timing
    return result, timing

def coverage_repair(basis_set, support_tuples):
    """Greedy coverage repair: expand basis until S(p) is covered by B+B."""
    B = set(basis_set)
    def pairwise(B):
        sums = set()
        bl = list(B)
        for a in bl:
            for b in bl:
                sums.add(tuple(x + y for x, y in zip(a, b)))
        return sums

    covered = pairwise(B)
    uncovered = support_tuples - covered
    while uncovered:
        counts = {}
        for s in uncovered:
            for b in B:
                c = tuple(x - y for x, y in zip(s, b))
                if any(v < 0 for v in c):
                    continue
                if c in B:
                    continue
                if c not in counts:
                    counts[c] = 0
                counts[c] += 1
        if not counts:
            break
        best = max(counts, key=counts.get)
        B.add(best)
        for b in list(B):
            covered.add(tuple(x + y for x, y in zip(best, b)))
            covered.add(tuple(x + y for x, y in zip(b, best)))
        uncovered = support_tuples - covered
    return B

def coverage_repair_strong(basis_set, support_tuples):
    """Greedy coverage repair with candidates restricted to {m : 2m in support}."""
    B = set(basis_set)
    def pairwise(B):
        sums = set()
        bl = list(B)
        for a in bl:
            for b in bl:
                sums.add(tuple(x + y for x, y in zip(a, b)))
        return sums

    covered = pairwise(B)
    uncovered = support_tuples - covered
    while uncovered:
        counts = {}
        for s in uncovered:
            for b in B:
                c = tuple(x - y for x, y in zip(s, b))
                if any(v < 0 for v in c):
                    continue
                if c in B:
                    continue
                # strong filter: 2c must be in support
                c2 = tuple(2 * x for x in c)
                if c2 not in support_tuples:
                    continue
                if c not in counts:
                    counts[c] = 0
                counts[c] += 1
        if not counts:
            break
        best = max(counts, key=counts.get)
        B.add(best)
        for b in list(B):
            covered.add(tuple(x + y for x, y in zip(best, b)))
            covered.add(tuple(x + y for x, y in zip(b, best)))
        uncovered = support_tuples - covered
    return B

def coverage_repair_newton(basis_set, support_tuples, newton_set):
    """Greedy coverage repair with candidates restricted to newton_set."""
    B = set(basis_set)
    def pairwise(B):
        sums = set()
        bl = list(B)
        for a in bl:
            for b in bl:
                sums.add(tuple(x + y for x, y in zip(a, b)))
        return sums

    covered = pairwise(B)
    uncovered = support_tuples - covered
    while uncovered:
        best_m = None
        best_score = 0
        for m in newton_set - B:
            score = 0
            for b in B:
                s = tuple(x + y for x, y in zip(m, b))
                if s in uncovered:
                    score += 1
            s = tuple(x + x for x in m)
            if s in uncovered:
                score += 1
            if score > best_score:
                best_score = score
                best_m = m
        if best_m is None:
            break
        B.add(best_m)
        for b in list(B):
            covered.add(tuple(x + y for x, y in zip(best_m, b)))
            covered.add(tuple(x + y for x, y in zip(b, best_m)))
        uncovered = support_tuples - covered
    return B

def facial_reduce(poly, basis_tuples):
    """Iterated diagonal consistency (Permenter & Parrilo 2014).
    Removes exposed monomials with zero diagonal coefficient."""
    M = set(basis_tuples)
    poly_coeffs = {m.exponents: c for m, c in poly.terms.items()}

    changed = True
    while changed:
        changed = False
        M_list = list(M)
        midpoints = set()
        for i, a in enumerate(M_list):
            for j in range(i + 1, len(M_list)):
                b = M_list[j]
                s = tuple(x + y for x, y in zip(a, b))
                if all(v % 2 == 0 for v in s):
                    midpoints.add(tuple(v // 2 for v in s))
        M_plus = M - midpoints

        to_remove = set()
        for alpha in M_plus:
            two_alpha = tuple(2 * x for x in alpha)
            coeff = poly_coeffs.get(two_alpha, 0.0)
            if abs(coeff) < 1e-10:
                to_remove.add(alpha)

        if to_remove:
            M -= to_remove
            changed = True

    return M

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=6, help="num variables")
parser.add_argument("--d", type=int, default=12, help="polynomial degree (must be even; basis degree = d/2)")
parser.add_argument("--m", type=int, default=30, help="num monomials in planted basis")
parser.add_argument("--num", type=int, default=16, help="num examples")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--matrix", type=str, default="dense",
                    choices=["dense", "sparse", "lowrank", "blockdiag"],
                    help="matrix structure for Q")
parser.add_argument("--le", action="store_true",
                    help="enable lattice enumeration (Newton basis, facial reduction)")
parser.add_argument("--sdp", action="store_true",
                    help="verify basis with SDP feasibility check (MOSEK)")
parser.add_argument("--strong", action="store_true",
                    help="strong filter: keep only m with 2m in support (implies --le, skips repair)")
parser.add_argument("--strong-repair", action="store_true",
                    help="repair with candidates restricted to {m : 2m in support}")
parser.add_argument("--lp", action="store_true",
                    help="use LP vertex test instead of ConvexHull for int_vtx (faster in high dim)")
args = parser.parse_args()

use_le = args.le or args.strong
use_sdp = args.sdp

if use_sdp:
    solver = CVXPYSOSSolver(solver='MOSEK', verbose=False)
    sdp_options = {
        'mosek_params': {
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-3,
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-3,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-3,
            'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-3,
            'MSK_DPAR_INTPNT_CO_TOL_INFEAS': 1e-3,
        }
    }

basis_sampler = SparseUniformBasisSampler(
    num_monomials=args.m, num_vars=args.n, max_degree=args.d // 2, min_degree=1)

if args.matrix == "dense":
    matrix_sampler = SimpleRandomPSDSampler(random_state=args.seed)
elif args.matrix == "sparse":
    matrix_sampler = SparsePSDSampler(random_state=args.seed)
elif args.matrix == "lowrank":
    matrix_sampler = LowRankPSDSampler(random_state=args.seed)
elif args.matrix == "blockdiag":
    matrix_sampler = BlockDiagonalPSDSampler(random_state=args.seed)

poly_sampler = SOSPolynomialSampler(
    basis_sampler=basis_sampler, matrix_sampler=matrix_sampler, rational=False)

check0_fails = 0
check1_fails = 0
check4_fails = 0
check5_fails = 0
basis_eq_planted = 0
planted_sizes = []
newton_sizes = []
reduced_sizes = []
strong_sizes = []
int_vert_sizes = []
repair_sizes = []
hull_times = []
le_times = []
io_times = []
intvtx_times = []
reduce_times = []
sdp_times = []

flags = []
if use_le: flags.append("le")
if use_sdp: flags.append("sdp")
if args.strong: flags.append("strong")
if args.strong_repair: flags.append("strong-repair")
if args.lp: flags.append("lp")
flag_str = f", flags={'+'.join(flags)}" if flags else ""
print(f"config: n={args.n}, d={args.d}, m={args.m}, matrix={args.matrix}, examples={args.num}{flag_str}\n")

hdr = f"{'i':>3}  {'planted':>7}"
if use_le: hdr += f"  {'newton':>6}  {'reduced':>7}"
if args.strong: hdr += f"  {'strong':>6}"
hdr += f"  {'int_vtx':>7}"
if not args.strong: hdr += f"  {'repair':>6}"
hdr += f"  {'gap':>3}"
if use_le: hdr += f"  {'chk0':>4}  {'chk5':>4}"
hdr += f"  {'chk1':>4}"
if use_sdp: hdr += f"  {'sdp':>4}"
print(hdr)

for i in range(args.num):
    np.random.seed(args.seed + i)
    poly, true_basis, Q = poly_sampler.sample(num_vars=args.n, max_degree=args.d // 2)

    exponents = np.array([m.exponents for m in poly.terms.keys()])
    true_set = set(true_basis)

    # Newton basis via C++ DFS
    if use_le:
        newton_points, nt = newton_basis_cpp(exponents)
        if newton_points is None:
            continue
        newton_set = set(Monomial(tuple(map(int, p))) for p in newton_points)
        c0 = true_set.issubset(newton_set)
        if not c0:
            check0_fails += 1
        hull_times.append(nt["hull"])
        le_times.append(nt["le"])
        io_times.append(nt["io"])

        # Facial reduction
        t0 = time.perf_counter()
        newton_exps_for_reduce = set(tuple(map(int, p)) for p in newton_points)
        reduced_basis = facial_reduce(poly, newton_exps_for_reduce)
        reduce_times.append(time.perf_counter() - t0)

    # Integer vertices of 1/2 N(p)
    t0 = time.perf_counter()
    unique_exp = np.unique(exponents, axis=0)
    exp_set = set(map(tuple, unique_exp))

    if args.lp:
        # LP vertex test: only check candidates with all-even coordinates
        even_candidates = [v for v in unique_exp if np.all(v % 2 == 0)]
        int_verts = set()
        for v in even_candidates:
            tv = tuple(v)
            # midpoint check
            is_midpoint = False
            for a in unique_exp:
                ta = tuple(a)
                if ta == tv:
                    continue
                b = tuple(2 * v - a)
                if any(x < 0 for x in b):
                    continue
                if b != tv and b in exp_set:
                    is_midpoint = True
                    break
            if is_midpoint:
                continue
            # LP test: v is a vertex of conv(unique_exp) iff v not in conv(others)
            mask = ~np.all(unique_exp == v, axis=1)
            others = unique_exp[mask]
            if len(others) == 0:
                int_verts.add(Monomial(tuple(int(x) for x in v // 2)))
                continue
            m_lp, n_lp = others.shape
            c_lp = np.zeros(m_lp)
            A_eq = np.vstack([others.T, np.ones((1, m_lp))])
            b_eq = np.append(v.astype(float), 1.0)
            res = linprog(c_lp, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * m_lp, method='highs')
            if not res.success:  # infeasible => vertex
                int_verts.add(Monomial(tuple(int(x) for x in v // 2)))
    else:
        try:
            hull = ConvexHull(unique_exp)
            np_verts = unique_exp[hull.vertices]
        except Exception:
            np_verts = unique_exp

        int_verts = set()
        for v in np_verts:
            if not np.all(v % 2 == 0):
                continue
            tv = tuple(v)
            is_midpoint = False
            for a in unique_exp:
                ta = tuple(a)
                if ta == tv:
                    continue
                b = tuple(2 * v - a)
                if any(x < 0 for x in b):
                    continue
                if b != tv and b in exp_set:
                    is_midpoint = True
                    break
            if not is_midpoint:
                int_verts.add(Monomial(tuple(int(x) for x in v // 2)))

    t_intvtx = time.perf_counter() - t0
    intvtx_times.append(t_intvtx)

    # check_1: integer vertices subset of planted basis
    c1 = int_verts.issubset(true_set)
    if not c1:
        check1_fails += 1

    int_vtx_exps = set(m.exponents for m in int_verts)
    supp_set = set(map(tuple, exponents))
    planted_exps = set(m.exponents for m in true_set)

    # Strong filter: keep only m with 2m in support
    strong_basis = None
    if args.strong:
        strong_basis = set(m for m in reduced_basis if tuple(2*x for x in m) in supp_set)

    # check_5: planted subset of reduced/strong basis
    if use_le:
        check_basis = strong_basis if args.strong else reduced_basis
        c5 = planted_exps.issubset(check_basis)
        if not c5:
            check5_fails += 1

    # Build basis: strong filter or greedy repair
    if args.strong:
        repaired = set(strong_basis)
    elif args.strong_repair:
        repaired = coverage_repair_strong(set(int_vtx_exps), supp_set)
    elif use_le:
        repaired = coverage_repair_newton(set(int_vtx_exps), supp_set, reduced_basis)
    else:
        repaired = coverage_repair(set(int_vtx_exps), supp_set)
    repaired_monomials = set(Monomial(e) for e in repaired)

    # gap: planted monomials not in basis
    n_gap = len(true_set - repaired_monomials)
    if repaired_monomials == true_set:
        basis_eq_planted += 1

    # SDP feasibility
    c4 = None
    if use_sdp:
        t0 = time.perf_counter()
        is_sos, _ = solver.solve_sos_feasibility(poly, basis=list(repaired_monomials), solver_options=sdp_options)
        sdp_times.append(time.perf_counter() - t0)
        c4 = is_sos
        if not c4:
            check4_fails += 1

    planted_sizes.append(len(true_basis))
    int_vert_sizes.append(len(int_verts))
    repair_sizes.append(len(repaired))

    row = f"{i:>3}  {len(true_basis):>7}"
    if use_le:
        newton_sizes.append(len(newton_points))
        reduced_sizes.append(len(reduced_basis))
        row += f"  {len(newton_points):>6}  {len(reduced_basis):>7}"
    if args.strong:
        strong_sizes.append(len(strong_basis))
        row += f"  {len(strong_basis):>6}"
    row += f"  {len(int_verts):>7}"
    if not args.strong: row += f"  {len(repaired):>6}"
    row += f"  {n_gap:>3}"
    if use_le: row += f"  {'OK' if c0 else 'FAIL':>4}  {'OK' if c5 else 'FAIL':>4}"
    row += f"  {'OK' if c1 else 'FAIL':>4}"
    if use_sdp: row += f"  {'OK' if c4 else 'FAIL':>4}"
    print(row)

n_examples = len(planted_sizes)
planted_sizes = np.array(planted_sizes)
newton_sizes = np.array(newton_sizes)
reduced_sizes = np.array(reduced_sizes)
strong_sizes = np.array(strong_sizes)
int_vert_sizes = np.array(int_vert_sizes)
repair_sizes = np.array(repair_sizes)
hull_times = np.array(hull_times)
le_times = np.array(le_times)
io_times = np.array(io_times)
intvtx_times = np.array(intvtx_times)
reduce_times = np.array(reduce_times)
sdp_times = np.array(sdp_times)

print()
print("check:")
checks = [
    (1, "int_vtx <= planted",   check1_fails),
]
if use_le:
    checks.insert(0, (5, "planted <= reduced", check5_fails))
    checks.insert(0, (0, "planted <= newton", check0_fails))
if use_sdp:
    checks.append((4, "sdp feasibility",     check4_fails))
for num, desc, fails in checks:
    status = "ALL OK" if fails == 0 else f"{fails} VIOLATIONS"
    print(f"  {num}) {desc:<22s} {status}")

print(f"\naverages ({n_examples} examples):")
print(f"  planted basis:       {planted_sizes.mean():.1f}")
if use_le:
    print(f"  newton basis:        {newton_sizes.mean():.1f}")
    print(f"  reduced (facial):    {reduced_sizes.mean():.1f}")
    print(f"  reduce time:         {reduce_times.mean():.4f}s")
if args.strong:
    print(f"  strong (2m filter):  {strong_sizes.mean():.1f}")
print(f"  integer vertices:    {int_vert_sizes.mean():.1f}")
if not args.strong:
    print(f"  repaired basis:      {repair_sizes.mean():.1f}")
print(f"  basis == planted:    {basis_eq_planted}/{n_examples}")
if use_le:
    print(f"  newton time:         {(hull_times + le_times + io_times).mean():.3f}s  (hull={hull_times.mean():.3f}  le={le_times.mean():.3f}  io={io_times.mean():.3f})")
print(f"  int_vtx time:        {intvtx_times.mean():.4f}s")
if use_sdp:
    print(f"  sdp time:            {sdp_times.mean():.3f}s")
