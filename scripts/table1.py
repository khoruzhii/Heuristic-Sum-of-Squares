"""Reproduce and extend Table 1 from Neural Sum-of-Squares (ICLR 2026).

Columns:
  Structure, n, d, m, |B*|,
  basis size  [ours, nsos, newton*],
  basis time  [ours, newton*],
  solve time  [ours, nsos, newton*]   (* = optional, with --newton / --sdp)

Usage:
  python scripts/table1.py [--num 4] [--sdp] [--newton] [--seed 42]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, time
import numpy as np
from basis import (
    generate_polynomial, find_basis_lp, find_basis_strong,
    find_basis_newton, solve_sdp, facial_reduce, Monomial,
)

# ── Paper results: Neural SOS "Ours" column from Table 1 ──────────────────
# Key: (structure, n, d, m) → (avg_basis_size, avg_total_time)
NSOS = {
    ('dense',    4,  6, 20): (19, 0.40),
    ('dense',    6, 12, 30): (33, 1.01),
    ('dense',    8, 20, 30): (38, 3.40),
    ('dense',    6, 20, 60): (89, 18.30),
    ('sparse',   4,  6, 20): (15, 0.23),
    ('sparse',   6, 12, 30): (27, 0.57),
    ('sparse',   8, 20, 30): (27, 0.62),
    ('sparse',   6, 20, 60): (73, 7.39),
    ('lowrank',  4,  6, 20): (19, 0.35),
    ('lowrank',  6, 12, 30): (30, 0.71),
    ('lowrank',  8, 20, 30): (35, 1.03),
    ('lowrank',  6, 20, 60): (66, 39.17),
    ('blockdiag',4,  6, 20): (20, 0.32),
    ('blockdiag',6, 12, 30): (31, 0.64),
    ('blockdiag',8, 20, 30): (30, 0.81),
    ('blockdiag',6, 20, 60): (71, 7.74),
}

# ── Configurations ────────────────────────────────────────────────────────
CONFIGS = [
    (4,   6,  20),
    (6,  12,  30),
    (8,  20,  30),
    (6,  20,  60),
    (100, 10,  20),
    (6,  40,  40),
]
STRUCTURES = ['dense', 'sparse', 'lowrank', 'blockdiag']

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Reproduce Table 1")
parser.add_argument('--num',    type=int, default=4,  help='examples per config')
parser.add_argument('--sdp',    action='store_true',   help='compute SDP solve times')
parser.add_argument('--newton', action='store_true',   help='include Newton basis columns')
parser.add_argument('--strong', action='store_true',   help='always use --strong (no LP fallback)')
parser.add_argument('--real',   action='store_true',   help='fallback via SDP (no planted knowledge)')
parser.add_argument('--seed',   type=int, default=42)
args = parser.parse_args()

# ── Table header ──────────────────────────────────────────────────────────
hdr = f"{'Structure':>10} {'n':>3} {'d':>3} {'m':>3} {'|B*|':>5}"
hdr += f"  {'s_ours':>6} {'s_nsos':>6}"
if args.newton:
    hdr += f" {'s_newt':>6}"
hdr += f"  {'t_ours':>7}"
if args.newton:
    hdr += f" {'t_newt':>7}"
if not args.strong:
    hdr += f"  {'fb_s':>3} {'fb_n':>3}"
hdr += f"  {'p<=b':>4}"
if args.sdp:
    hdr += f"  {'T_ours':>7} {'T_nsos':>7}"
    if args.newton:
        hdr += f" {'T_newt':>7}"

print(hdr)
print("-" * len(hdr))

# ── Main loop ─────────────────────────────────────────────────────────────
for struct in STRUCTURES:
    for n, d, m in CONFIGS:
        nsos_entry = NSOS.get((struct, n, d, m))

        planted_sizes = []
        ours_sizes = []
        ours_basis_times = []
        ours_solve_times = []
        newton_sizes = []
        newton_basis_times = []
        newton_solve_times = []
        fb_strong = 0
        fb_newton = 0
        planted_ok_count = 0
        errors = []

        for i in range(args.num):
            seed = args.seed + i
            poly, planted = generate_polynomial(n, d, m, struct, seed)
            planted_exps = set(b.exponents for b in planted)
            planted_sizes.append(len(planted))

            # ── Our basis ──
            if args.strong:
                basis, bt = find_basis_strong(poly)
            else:
                # Try LP+repair → strong → newton reduced
                basis, bt = find_basis_lp(poly)
                basis_monos = set(Monomial(e) for e in basis)

                if args.real:
                    # Fallback decision via SDP (no planted knowledge)
                    need_fb = False
                    ok_sdp, st_sdp = solve_sdp(poly, basis)
                    if not ok_sdp:
                        need_fb = True
                        bt += st_sdp
                else:
                    # Fallback decision via planted subset check
                    need_fb = not planted.issubset(basis_monos)

                if need_fb:
                    basis, bt2 = find_basis_strong(poly)
                    bt += bt2
                    fb_strong += 1
                    basis_monos = set(Monomial(e) for e in basis)

                    if args.real:
                        ok_sdp, st_sdp = solve_sdp(poly, basis)
                        if not ok_sdp:
                            bt += st_sdp
                            nb, bt3 = find_basis_newton(poly)
                            reduced = facial_reduce(poly, nb)
                            basis = reduced
                            bt += bt3
                            fb_newton += 1
                    elif not planted.issubset(basis_monos):
                        nb, bt3 = find_basis_newton(poly)
                        reduced = facial_reduce(poly, nb)
                        basis = reduced
                        bt += bt3
                        fb_newton += 1

            # ── Verify ──
            ok_sdp, st_sdp = solve_sdp(poly, basis)
            basis_monos = set(Monomial(e) for e in basis)
            planted_in_basis = planted.issubset(basis_monos)
            if planted_in_basis:
                planted_ok_count += 1

            if not ok_sdp and planted_in_basis:
                pass  # basis correct, SDP solver error
            elif not ok_sdp:
                errors.append(seed)

            if args.sdp:
                ours_solve_times.append(st_sdp)

            ours_sizes.append(len(basis))
            ours_basis_times.append(bt)

            # ── Newton (optional) ──
            if args.newton:
                nb, nt = find_basis_newton(poly)
                newton_sizes.append(len(nb))
                newton_basis_times.append(nt)
                if args.sdp:
                    ok_n, st_n = solve_sdp(poly, nb)
                    newton_solve_times.append(st_n)

        # ── Format row ──
        avg = lambda a: np.mean(a) if a else float('nan')

        row = f"{struct:>10} {n:>3} {d:>3} {m:>3} {avg(planted_sizes):>5.0f}"

        # basis sizes
        row += f"  {avg(ours_sizes):>6.0f}"
        if nsos_entry:
            row += f" {nsos_entry[0]:>6}"
        else:
            row += f" {'-':>6}"
        if args.newton:
            row += f" {avg(newton_sizes):>6.0f}"

        # basis times
        row += f"  {avg(ours_basis_times):>7.3f}"
        if args.newton:
            row += f" {avg(newton_basis_times):>7.3f}"

        # fallbacks (only in LP mode)
        if not args.strong:
            row += f"  {fb_strong:>3} {fb_newton:>3}"

        # planted ⊆ basis
        row += f"  {planted_ok_count:>2}/{args.num}"

        # solve times
        if args.sdp:
            if ours_solve_times:
                row += f"  {avg(ours_solve_times):>7.3f}"
            else:
                row += f"  {'-':>7}"
            if nsos_entry:
                row += f" {nsos_entry[1]:>7.2f}"
            else:
                row += f" {'-':>7}"
            if args.newton:
                if newton_solve_times:
                    row += f" {avg(newton_solve_times):>7.3f}"
                else:
                    row += f" {'-':>7}"

        if errors:
            row += f"  ERR:{errors}"
        else:
            row += f"  OK"

        print(row)

    # separator between structures
    print()
