"""SOS basis computation utilities.

Functions for computing compact monomial bases for SOS verification:
  - int_vtx via LP or ConvexHull
  - coverage repair (greedy, strong, newton-restricted)
  - Newton polytope basis via C++ lattice enumeration
  - facial reduction (diagonal consistency)
  - SDP feasibility check
"""

import sys, types, os, time, subprocess
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from math import comb

# Module setup for sos/src imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sos_mod = types.ModuleType('sos_transformer')
_sos_mod.__path__ = [os.path.join(_REPO_ROOT, 'sos', 'src')]
sys.modules.setdefault('sos_transformer', _sos_mod)

from data_generation.monomials.monomial_sparseuniform import SparseUniformBasisSampler
from data_generation.matrix.matrix_simple import SimpleRandomPSDSampler
from data_generation.matrix.matrix_sparse import SparsePSDSampler
from data_generation.matrix.matrix_lowrank import LowRankPSDSampler
from data_generation.matrix.matrix_blockdiag import BlockDiagonalPSDSampler
from data_generation.polynomials.polynomial_sos import SOSPolynomialSampler
from data_generation.monomials.monomials import Monomial
from sdp_solver.cvxpy_solver import CVXPYSOSSolver

REPO_ROOT = _REPO_ROOT
NEWTON_BIN = os.path.join(REPO_ROOT, "bin", "newton_enum.exe")
TMP_DIR = os.path.join(REPO_ROOT, "data", "tmp")

_MATRIX_SAMPLERS = {
    'dense': SimpleRandomPSDSampler,
    'sparse': SparsePSDSampler,
    'lowrank': LowRankPSDSampler,
    'blockdiag': BlockDiagonalPSDSampler,
}

# ---------------------------------------------------------------------------
# Polynomial generation
# ---------------------------------------------------------------------------

def generate_polynomial(n, d, m, matrix_type='dense', seed=42):
    """Generate SOS polynomial with known planted basis.

    Args:
        n: number of variables
        d: polynomial degree (even)
        m: number of monomials in planted basis
        matrix_type: 'dense' | 'sparse' | 'lowrank' | 'blockdiag'
        seed: random seed

    Returns:
        (poly, planted_set): polynomial object and set of Monomial objects
    """
    basis_degree = d // 2
    np.random.seed(seed)
    basis_sampler = SparseUniformBasisSampler(
        num_monomials=m, num_vars=n, max_degree=basis_degree, min_degree=1)
    matrix_sampler = _MATRIX_SAMPLERS[matrix_type](random_state=seed)
    poly_sampler = SOSPolynomialSampler(
        basis_sampler=basis_sampler, matrix_sampler=matrix_sampler, rational=False)
    poly, true_basis, Q = poly_sampler.sample(num_vars=n, max_degree=basis_degree)
    return poly, set(true_basis)


# ---------------------------------------------------------------------------
# Half-hull inequalities (H-representation of 1/2 N(p))
# ---------------------------------------------------------------------------

def _half_hull_inequalities(pts):
    """Return (A, b) for 1/2 N(p): A x + b <= 0, handling degenerate cases."""
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


# ---------------------------------------------------------------------------
# Newton basis via C++ DFS
# ---------------------------------------------------------------------------

def newton_basis_cpp(exponent_vectors):
    """Compute lattice points in 1/2 N(p) using C++ DFS enumeration.

    Returns:
        (result_array_or_None, timing_dict)
    """
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


# ---------------------------------------------------------------------------
# Integer vertices
# ---------------------------------------------------------------------------

def compute_int_vtx_lp(exponents):
    """Compute integer vertices of conv(support) with even coordinates via LP.

    Returns:
        (set of Monomial, elapsed_seconds)
    """
    unique_exp = np.unique(np.asarray(exponents), axis=0)
    exp_set = set(map(tuple, unique_exp))

    t0 = time.perf_counter()
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
        # LP test: v is vertex iff v not in conv(others)
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
        if not res.success:
            int_verts.add(Monomial(tuple(int(x) for x in v // 2)))
    elapsed = time.perf_counter() - t0
    return int_verts, elapsed


# ---------------------------------------------------------------------------
# Coverage repair
# ---------------------------------------------------------------------------

def _pairwise_sums(B):
    sums = set()
    bl = list(B)
    for a in bl:
        for b in bl:
            sums.add(tuple(x + y for x, y in zip(a, b)))
    return sums


def coverage_repair(basis_set, support_set):
    """Greedy coverage repair: expand basis until S(p) is covered by B+B."""
    B = set(basis_set)
    covered = _pairwise_sums(B)
    uncovered = support_set - covered
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
        uncovered = support_set - covered
    return B


def coverage_repair_strong(basis_set, support_set):
    """Greedy repair with candidates restricted to {m : 2m in support}."""
    B = set(basis_set)
    covered = _pairwise_sums(B)
    uncovered = support_set - covered
    while uncovered:
        counts = {}
        for s in uncovered:
            for b in B:
                c = tuple(x - y for x, y in zip(s, b))
                if any(v < 0 for v in c):
                    continue
                if c in B:
                    continue
                if tuple(2 * x for x in c) not in support_set:
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
        uncovered = support_set - covered
    return B


def coverage_repair_newton(basis_set, support_set, newton_set):
    """Greedy repair with candidates restricted to newton_set."""
    B = set(basis_set)
    covered = _pairwise_sums(B)
    uncovered = support_set - covered
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
        uncovered = support_set - covered
    return B


# ---------------------------------------------------------------------------
# Facial reduction
# ---------------------------------------------------------------------------

def facial_reduce(poly, basis_tuples):
    """Iterated diagonal consistency (Permenter & Parrilo 2014)."""
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


# ---------------------------------------------------------------------------
# High-level basis pipelines
# ---------------------------------------------------------------------------

def find_basis_lp(poly, strong_repair=False):
    """int_vtx (LP) + coverage repair.

    Returns: (basis_set_of_tuples, elapsed_seconds)
    """
    exponents = np.array([m.exponents for m in poly.terms.keys()])
    supp_set = set(map(tuple, exponents))

    t0 = time.perf_counter()
    int_vtx, _ = compute_int_vtx_lp(exponents)
    int_vtx_exps = set(m.exponents for m in int_vtx)

    if strong_repair:
        basis = coverage_repair_strong(int_vtx_exps, supp_set)
    else:
        basis = coverage_repair(int_vtx_exps, supp_set)
    elapsed = time.perf_counter() - t0
    return basis, elapsed


def find_basis_strong(poly):
    """Newton LE + facial reduce + 2m filter.

    Returns: (basis_set_of_tuples, elapsed_seconds)
    """
    exponents = np.array([m.exponents for m in poly.terms.keys()])
    supp_set = set(map(tuple, exponents))

    t0 = time.perf_counter()
    newton_pts, _ = newton_basis_cpp(exponents)
    if newton_pts is None:
        return set(), time.perf_counter() - t0
    newton_exps = set(tuple(map(int, p)) for p in newton_pts)
    reduced = facial_reduce(poly, newton_exps)
    strong = set(m for m in reduced if tuple(2 * x for x in m) in supp_set)
    elapsed = time.perf_counter() - t0
    return strong, elapsed


def find_basis_newton(poly):
    """Raw Newton polytope basis (lattice points in 1/2 N(p)).

    Returns: (basis_set_of_tuples, elapsed_seconds)
    """
    exponents = np.array([m.exponents for m in poly.terms.keys()])

    t0 = time.perf_counter()
    newton_pts, _ = newton_basis_cpp(exponents)
    elapsed = time.perf_counter() - t0

    if newton_pts is None:
        return set(), elapsed
    return set(tuple(map(int, p)) for p in newton_pts), elapsed


# ---------------------------------------------------------------------------
# SDP feasibility
# ---------------------------------------------------------------------------

def verify_gram(poly, basis_list, Q, tol=1e-6):
    """Verify SOS certificate: Q >= 0 and z^T Q z == p.

    Returns: (ok: bool, min_eig: float, max_coeff_err: float)
    """
    # Check Q is PSD
    eigvals = np.linalg.eigvalsh(Q)
    min_eig = float(eigvals[0])

    # Reconstruct polynomial from z^T Q z and compare coefficients
    poly_coeffs = {}
    for m, c in poly.terms.items():
        poly_coeffs[m.exponents] = c

    reconstructed = {}
    for i, mi in enumerate(basis_list):
        for j, mj in enumerate(basis_list):
            s = tuple(a + b for a, b in zip(mi, mj))
            if s not in reconstructed:
                reconstructed[s] = 0.0
            reconstructed[s] += Q[i, j]

    # Compare
    all_monos = set(poly_coeffs.keys()) | set(reconstructed.keys())
    max_err = 0.0
    for m in all_monos:
        err = abs(poly_coeffs.get(m, 0.0) - reconstructed.get(m, 0.0))
        max_err = max(max_err, err)

    ok = (min_eig > -tol) and (max_err < tol)
    return ok, min_eig, max_err


def solve_sdp(poly, basis_tuples, tol=1e-3, verify=True):
    """Check SDP feasibility with MOSEK, optionally verify certificate.

    Returns: (is_sos: bool, elapsed: float)
    """
    solver = CVXPYSOSSolver(solver='MOSEK', verbose=False)
    options = {
        'mosek_params': {
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
            'MSK_DPAR_INTPNT_CO_TOL_MU_RED': tol,
            'MSK_DPAR_INTPNT_CO_TOL_INFEAS': tol,
        }
    }
    basis_list = sorted(basis_tuples)
    basis_monomials = [Monomial(e) for e in basis_list]

    t0 = time.perf_counter()
    is_sos, Q = solver.solve_sos_feasibility(poly, basis=basis_monomials, solver_options=options)
    elapsed = time.perf_counter() - t0

    if is_sos and verify and Q is not None:
        ok, min_eig, max_err = verify_gram(poly, basis_list, Q, tol=tol * 10)
        if not ok:
            is_sos = False

    return is_sos, elapsed
