#!/usr/bin/env python3
"""
Table 3 of the paper (the three reference readings), the closed-form
ranking-inversion condition, and the reconciliation of the five digitisation
biases.

Why this file exists
--------------------
The "Against biased ref." column of the published table was mislabelled: A and B
divided by the CONSISTENT reference (744.2255 cm) and only C divided by the
BIASED one (778.1718 cm). Here all three readings are computed in full, each
with its stated reference, and the ranking inversion is verified to survive.

Usage
-----
    python3 inversion.py            # table 3, inversion condition, reconciliation
    python3 inversion.py rsweep     # also sweeps R, regenerating the leaf (~90 s)

`general_condition` is the one that answers the reviewer: the published condition
needed bbar > 0 and did not state it, and there is a general form that needs no
premise at all. See its docstring.
"""

from __future__ import annotations

import sys

import numpy as np

# --------------------------------------------------------------------------
# Measured figures, seed 7, reference mesh (20,917 triangles).
# Regenerable with:  python3 demo.py  /  oldgt.py  /  leafperim.py
# --------------------------------------------------------------------------
P_A = 677.9005717976462      # level A, 4-direction Crofton on the mask
P_B = 713.5357392812726      # level B, flat mesh contour (simplify_px 0.30)
P_C = 778.0640251793623      # level C, the same contour lifted to the cylinder

P_MS_MASK = 715.0401512639276   # raw marching squares on the same mask
P_SK_MASK = 714.3330444827410   # skimage.measure.perimeter on the same mask
P_DSS_MASK = 685.1915           # DSS on the same mask

# BIASED reference: binary marching squares on the GT indicator
P_BIAS_INT = 778.171809
P_BIAS_PRJ = 715.689263
# CONSISTENT reference: zero level contour + order-1 Richardson
P_TRUE_INT = 744.2254917143233
P_TRUE_PRJ = 686.9815602950247
# zero level WITHOUT extrapolating, on the h = 0.100 cm grid of the refinement study
P_LVL_H010 = 743.1175
P_BIN_H010 = 778.2080

LEVELS = (("A - raster (Crofton 4 dir)", P_A),
          ("B - flat mesh", P_B),
          ("C - lifted mesh", P_C))


def pct(x, ref):
    return 100.0 * (x / ref - 1.0)


def table3():
    print("=" * 78)
    print("TABLE 3 - perimeter error per level, seed 7")
    print("=" * 78)
    print("references:")
    print(f"  biased     : {P_BIAS_INT:10.4f} cm intrinsic   "
          f"{P_BIAS_PRJ:10.4f} cm projected")
    print(f"  consistent : {P_TRUE_INT:10.4f} cm intrinsic   "
          f"{P_TRUE_PRJ:10.4f} cm projected")
    print(f"  pure projection factor p = {pct(P_TRUE_PRJ, P_TRUE_INT):+.4f} %")
    print()
    hdr = f"{'level':<28} {'(a) biased':>14} {'(b) intrinsic':>14} {'(c) reachable':>14}"
    print(hdr)
    print("-" * len(hdr))
    for i, (name, p) in enumerate(LEVELS):
        a = pct(p, P_BIAS_INT)                                  # a single GT
        b = pct(p, P_TRUE_INT)                                  # intrinsic truth
        c = pct(p, P_TRUE_PRJ if i < 2 else P_TRUE_INT)         # reachable
        print(f"{name:<28} {a:13.4f}% {b:13.4f}% {c:13.4f}%")
    print()
    for lab, key in (("(a) biased", "a"), ("(b) intrinsic", "b"),
                     ("(c) reachable", "c")):
        errs = []
        for i, (_, p) in enumerate(LEVELS):
            errs.append(pct(p, P_BIAS_INT) if key == "a" else
                        pct(p, P_TRUE_INT) if key == "b" else
                        pct(p, P_TRUE_PRJ if i < 2 else P_TRUE_INT))
        order = np.argsort(np.abs(errs))
        print(f"  order in |error| under {lab:<15} : "
              f"{'  <  '.join('ABC'[i] for i in order)}")
    eA_a, eB_a = pct(P_A, P_BIAS_INT), pct(P_B, P_BIAS_INT)
    eA_c, eB_c = pct(P_A, P_TRUE_PRJ), pct(P_B, P_TRUE_PRJ)
    print()
    print(f"  A/B INVERSION: under (a) B looks {abs(eA_a) / abs(eB_a):.2f} x better "
          f"than A; under (c) A is {abs(eB_c) / abs(eA_c):.2f} x better than B  "
          f"(total swing {abs(eA_a) / abs(eB_a) * abs(eB_c) / abs(eA_c):.2f} x)")


def inversion_condition():
    print()
    print("=" * 78)
    print("CLOSED-FORM INVERSION CONDITION")
    print("=" * 78)
    p = P_TRUE_PRJ / P_TRUE_INT - 1.0
    bA = P_A / P_TRUE_PRJ - 1.0
    bB = P_B / P_TRUE_PRJ - 1.0
    EA = P_A / P_TRUE_INT - 1.0
    EB = P_B / P_TRUE_INT - 1.0
    print("  E_X = (1+b_X)(1+p) - 1")
    print("  |E_A| > |E_B|  <=>  (E_A-E_B)(E_A+E_B) > 0")
    print("                 <=>  (1+p)(b_A-b_B) * [ (1+p)(b_A+b_B) + 2p ] > 0")
    print("  with b_A < b_B: (1+p)(b_A+b_B) + 2p < 0")
    print("                 <=>  p < p* = -bbar/(1+bbar),   bbar = (b_A+b_B)/2")
    print("                 <=>  (1+p)(1+bbar) < 1")
    print()
    print(f"  p   = {100 * p:+.4f} %   b_A = {100 * bA:+.4f} %   b_B = {100 * bB:+.4f} %")
    print(f"  exact composition: (1+b_A)(1+p)-1 = {100 * ((1 + bA) * (1 + p) - 1):+.6f} % "
          f"vs measured {100 * EA:+.6f} % (resid {(1 + bA) * (1 + p) - 1 - EA:.1e})")
    print(f"  exact composition: (1+b_B)(1+p)-1 = {100 * ((1 + bB) * (1 + p) - 1):+.6f} % "
          f"vs measured {100 * EB:+.6f} % (resid {(1 + bB) * (1 + p) - 1 - EB:.1e})")
    bbar = 0.5 * (bA + bB)
    pstar = -bbar / (1.0 + bbar)
    print(f"  bbar = {100 * bbar:+.6f} %    p* = {100 * pstar:+.6f} %")
    print(f"  p < p* ?  {100 * p:+.4f} < {100 * pstar:+.4f}  ->  {p < pstar}   "
          f"(margin {100 * (pstar - p):.3f} pp)")
    s = 1.0 + pstar
    print(f"  at p = p*: E_A = {100 * (bA * s + pstar):+.4f} %, "
          f"E_B = {100 * (bB * s + pstar):+.4f} %  (exact tie in |.|: "
          f"{abs(bA * s + pstar) - abs(bB * s + pstar):.1e})")
    print()
    print("  sweep of p with b fixed:")
    print(f"   {'p %':>8} {'E_A %':>10} {'E_B %':>10}   smaller |error|")
    for pv in (-0.16, -0.12, -0.10, p, -0.05, -0.03, -0.02, pstar, -0.01, 0.0, 0.02):
        ea, eb = bA * (1 + pv) + pv, bB * (1 + pv) + pv
        who = "A" if abs(ea) < abs(eb) else ("B" if abs(eb) < abs(ea) else "tie")
        print(f"   {100 * pv:+8.4f} {100 * ea:+10.4f} {100 * eb:+10.4f}   {who}")


def general_condition(n=2_000_000, seed=7):
    """The hypothesis Eq. (2) was missing, and the general form that needs none.

    A reviewer pointed it out and was right. The published condition

        inversion occurs  <=>  (1+p)(1+bbar) < 1

    characterises |E_A| > |E_B|, i.e. the FINAL ORDER, not the INVERSION of the
    order. They coincide only if, with no shared distortion, the order was
    |b_A| < |b_B|, and that, given b_A < b_B, is equivalent to bbar > 0.

    Subtracting squares gives everything:

        E_A^2 - E_B^2 = 2(1+p)(b_A-b_B) [ (1+p)(1+bbar) - 1 ]
        b_A^2 - b_B^2 = 2(b_A-b_B) bbar

    The factor (b_A-b_B) is COMMON, so for p > -1 it cancels and what remains is

        there is an inversion  <=>  sign[(1+p)(1+bbar) - 1]  !=  sign(bbar)

    which asks for no premise about which bias is larger. The leaf satisfies
    bbar > 0, so the published figure (p* = -1.26 %) still holds.
    """
    print()
    print("=" * 78)
    print("THE GENERAL FORM OF THE CONDITION, AND THE MISSING HYPOTHESIS")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    bA = rng.uniform(-0.6, 0.6, n)
    bB = rng.uniform(-0.6, 0.6, n)
    p = rng.uniform(-0.6, 0.6, n)
    bbar = 0.5 * (bA + bB)
    keep = (np.abs(bA - bB) > 1e-9) & (np.abs(bbar) > 1e-9)   # drop the ties
    bA, bB, p, bbar = bA[keep], bB[keep], p[keep], bbar[keep]

    EA, EB = (1 + bA) * (1 + p) - 1, (1 + bB) * (1 + p) - 1
    # INVERSION = the order in |E| differs from the order in |b| (no distortion)
    inv = np.sign(np.abs(EA) - np.abs(EB)) != np.sign(np.abs(bA) - np.abs(bB))

    pub = (1 + p) * (1 + bbar) < 1                               # published Eq. (2)
    gen = np.sign((1 + p) * (1 + bbar) - 1) != np.sign(bbar)     # general form
    sub = bA < bB                                                # stated premise

    print(f"  {len(bA)} triples (b_A, b_B, p) uniform in [-0.6, 0.6], no ties")
    print("  truth criterion: sign(|E_A|-|E_B|) != sign(|b_A|-|b_B|)")
    print()
    print(f"  {'predictor':<46} {'accuracy':>8}")
    print("  " + "-" * 55)
    print(f"  {'published Eq. (2): (1+p)(1+bbar) < 1':<46} "
          f"{100 * np.mean(pub == inv):7.4f} %")
    print(f"  {'  + its stated premise b_A < b_B':<46} "
          f"{100 * np.mean(pub[sub] == inv[sub]):7.4f} %")
    print(f"  {'  + the missing hypothesis, bbar > 0':<46} "
          f"{100 * np.mean(pub[sub & (bbar > 0)] == inv[sub & (bbar > 0)]):7.4f} %")
    print(f"  {'general form (sign disagreement)':<46} "
          f"{100 * np.mean(gen == inv):7.4f} %")
    print()
    bad = sub & pub & ~inv
    print("  counterexamples with b_A < b_B and Eq. (2) TRUE but no inversion:")
    print(f"    {bad.sum()} cases, i.e. {100 * bad.sum() / (sub & pub).sum():.1f} % "
          f"of those satisfying the premise and the inequality")
    print(f"    is bbar < 0 in all of them ?  {bool(np.all(bbar[bad] < 0))}   "
          f"(largest bbar {bbar[bad].max():+.6f})")
    print()
    print("  exact counterexample:  b_A = -50 %, b_B = +10 %, p = -10 %")
    a, b, q = -0.50, 0.10, -0.10
    bb = 0.5 * (a + b)
    print(f"    bbar = {100 * bb:+.1f} %,  (1+p)(1+bbar) = {(1 + q) * (1 + bb):.3f} < 1  "
          f"-> Eq. (2) says IT INVERTS")
    print(f"    |b_A| = {abs(a):.2f} > |b_B| = {abs(b):.2f}   -> with no distortion B "
          f"was already the better one")
    print(f"    |E_A| = {abs((1 + a) * (1 + q) - 1):.4f} > "
          f"|E_B| = {abs((1 + b) * (1 + q) - 1):.4f}   -> B is still the better one: "
          f"there is NO inversion")
    print()
    pl = P_TRUE_PRJ / P_TRUE_INT - 1.0
    bAl, bBl = P_A / P_TRUE_PRJ - 1.0, P_B / P_TRUE_PRJ - 1.0
    bbl = 0.5 * (bAl + bBl)
    print(f"  the leaf:  b_A = {100 * bAl:+.4f} %, b_B = {100 * bBl:+.4f} %, "
          f"p = {100 * pl:+.4f} %")
    print(f"    bbar = {100 * bbl:+.4f} % > 0  ->  the hypothesis DOES hold and "
          f"p* = {100 * (-bbl / (1 + bbl)):+.4f} % stands")
    print(f"    |b_A| {abs(bAl):.5f} < |b_B| {abs(bBl):.5f}   and   "
          f"|E_A| {abs((1 + bAl) * (1 + pl) - 1):.5f} > "
          f"|E_B| {abs((1 + bBl) * (1 + pl) - 1):.5f}   ->  it inverts")


def reconcile():
    print()
    print("=" * 78)
    print("RECONCILIATION OF THE FIVE DIGITISATION BIASES")
    print("=" * 78)
    rows = [
        ("+3.87 %", "level B: MS + Douglas-Peucker 0.30 px + CDT", P_B, P_TRUE_PRJ),
        ("+4.08 %", "raw binary MS on the same mask", P_MS_MASK, P_TRUE_PRJ),
        ("+4.55 %", "level C: B's contour lifted to the cylinder", P_C, P_TRUE_INT),
        ("+4.56 %", "the biased REFERENCE against the consistent one",
         P_BIAS_INT, P_TRUE_INT),
        ("+4.72 %", "MS against zero level on the SAME grid (plateau)",
         P_BIN_H010, P_LVL_H010),
    ]
    print(f"{'cited':<9} {'what is measured':<48} {'measured':>10} "
          f"{'reference':>10} {'exact':>9}")
    print("-" * 92)
    for lab, what, num, den in rows:
        print(f"{lab:<9} {what:<48} {num:10.4f} {den:10.4f} {pct(num, den):+8.4f}%")
    print()
    print("  differences:")
    print(f"   +4.08 -> +3.87 : {pct(P_B, P_TRUE_PRJ) - pct(P_MS_MASK, P_TRUE_PRJ):+.4f} pp, "
          f"the 0.30 px Douglas-Peucker simplification")
    print(f"   +3.87 -> +4.55 : {pct(P_C, P_TRUE_INT) - pct(P_B, P_TRUE_PRJ):+.4f} pp, "
          f"the lift (it stretches the digitised contour by x{P_C / P_B:.6f} against "
          f"x{P_TRUE_INT / P_TRUE_PRJ:.6f} for the truth)")
    print(f"   +4.55 -> +4.56 : {pct(P_BIAS_INT, P_TRUE_INT) - pct(P_C, P_TRUE_INT):+.4f} pp, "
          f"and they are different objects: one is the error of the METHOD, the "
          f"other that of the REFERENCE")
    print(f"   +4.56 -> +4.72 : "
          f"{pct(P_BIN_H010, P_LVL_H010) - pct(P_BIAS_INT, P_TRUE_INT):+.4f} pp, "
          f"same estimator, denominator without extrapolating "
          f"(743.1175 sits {pct(P_LVL_H010, P_TRUE_INT):+.4f} % below the limit)")


def rsweep():
    """Regenerate the leaf at several R: p, b_A, b_B and the real order."""
    import os
    import warnings
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, os.pardir, os.pardir))
    if os.path.isdir(os.path.join(root, "foliar")):
        sys.path.insert(0, root)
    from foliar import pipeline, synth
    print()
    print("=" * 78)
    print("R SWEEP (the leaf is regenerated in full at each R, seed 7)")
    print("=" * 78)
    print(f"{'R (cm)':>8} {'p %':>9} {'b_A %':>9} {'b_B %':>9} {'E_A %':>10} "
          f"{'E_B %':>10} {'p* %':>9} inversion")
    for R in (38.0, 50.0, 60.0, 70.0, 80.0, 82.0, 84.0, 85.0, 86.0, 88.0,
              90.0, 100.0, 120.0, 150.0, 200.0, 400.0):
        cfg = synth.LeafConfig(seed=7, curl_radius_cm=R)
        s = synth.generate(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = pipeline.run(s, curl_radius_cm=R)
        gt = s.truth
        pA, pB = r.level_A["perimeter_cm"], r.level_B["perimeter_cm"]
        p = gt.projected_perimeter_cm / gt.perimeter_cm - 1
        bA = pA / gt.projected_perimeter_cm - 1
        bB = pB / gt.projected_perimeter_cm - 1
        EA, EB = pA / gt.perimeter_cm - 1, pB / gt.perimeter_cm - 1
        pstar = -(bA + bB) / (2 + bA + bB)
        print(f"{R:8.1f} {100 * p:+9.4f} {100 * bA:+9.4f} {100 * bB:+9.4f} "
              f"{100 * EA:+10.4f} {100 * EB:+10.4f} {100 * pstar:+9.4f} "
              f"{'YES' if abs(EA) > abs(EB) else 'no'}", flush=True)


if __name__ == "__main__":
    table3()
    inversion_condition()
    general_condition()
    reconcile()
    if len(sys.argv) > 1 and sys.argv[1] == "rsweep":
        rsweep()
