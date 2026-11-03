#!/usr/bin/env python3
"""
The two- (three-) grid convergence test the paper promises and does not show.

Answers reviewer question 7: "Could you share the exact two-grid convergence
test you propose (grid spacings, expected orders, tolerance thresholds) as a
minimal checklist that libraries or papers can adopt?"

The core is three lines
-----------------------
    L = [est(shape.sample(n)) for n in (400, 800, 1600)]
    e = [abs(v / shape.exact - 1) for v in L]
    p = math.log2(e[0] / e[-1]) / 2        # PASSES if p >= 0.5 or e[-1] <= 1e-3

Everything else in this file is the justification of the constants and the
demonstration that the test separates the estimators that need separating.

Usage
-----
    python3 twogrid.py            # full checklist (~2 s)
    python3 twogrid.py quick      # canonical shape only (< 1 s)
    python3 twogrid.py shapes     # why the prescribed shape is not negotiable
    python3 twogrid.py self       # why self-convergence is NO use

What is measured and with what thresholds
-----------------------------------------
See the constants GRIDS, ORDER_MIN, TOL_FLOOR and PLATEAU_BAND below; each
carries its derivation and the measured number it comes from. The three lines
above are the core; PLATEAU_BAND adds a third verdict, INCONCLUSIVE, which
exists because the binary one gives a MEASURED false rejection (see that
constant).
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
from skimage import measure

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir))


def _load_dss():
    """dss.py sits next to this file; do not depend on the cwd."""
    spec = importlib.util.spec_from_file_location("dss", os.path.join(_HERE, "dss.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dss = _load_dss()


# ==========================================================================
# 1 - THE THREE PARAMETERS OF THE TEST
# ==========================================================================

GRIDS = (400, 800, 1600)
"""
Spacings: three grids in ratio 2, i.e. a total range of 4x in h.

Why THREE and not two, even though the test is called "two-grid": with two
grids the observed order is estimated over a 2x range and the estimator's noise
enters whole into the only available ratio. Measured on the canonical square,
the pair (400, 800) gives DSS order 4.48 and the pair (800, 1600) gives -1.65:
neither is informative on its own, while the average over the 4x range gives
1.42, which is. Two grids suffice to DETECT a plateau (the symptom is that the
error does not move); three are needed so as not to reject, out of noise, an
estimator that does converge.

Why 400 -> 1600 and not 100 -> 400: the plateau of the binary estimator is
already established at n = 400 (error +7.86 % against the limit +8.24 %), so the
coarsest grid has to sit to the right of the transient. Why not go to 3200: it
costs 4x more and changes no verdict in the table.

In physical units, over the domain [-1.6, 1.6]^2 of the square of side 1.6:
h = 8.0e-3, 4.0e-3, 2.0e-3, i.e. between 200 and 800 pixels across the object.
That is the regime real work happens in; it is not an unreachable asymptotic
limit.
"""

ORDER_MIN = 0.5
"""
Minimum observed order to pass through the order branch.

It comes from half the weakest theoretical order among the estimators one would
want to pass. The zero level set estimator is order 2 on smooth boundary and
order 1 at corners, and the corners dominate, so its useful order is 1. DSS is
multigrid-convergent with a proof but with no sharp rate; measured here it gives
1.42 on the canonical square, 1.64 on the circle, 3.26 on the leaf, and -0.40 on
a square at 35 degrees. Setting the threshold at 1 would reject both out of
noise; setting it at 0.5 leaves 50 % of slack over order 1 and still rejects a
plateau, whose order is 0 by construction.

Measured margin on the canonical shape: zero level set 1.15, DSS 1.42 against
the 0.5 threshold; marching squares -0.03 and Crofton-4 -0.09. The separation is
not marginal.

MEASURED WARNING. This branch alone gives a FALSE NEGATIVE when the convergent
estimator has already reached its accuracy floor: on the repository's leaf the
zero level set gives order 0.36 with an error of +0.0043 %, because there is
nothing left to converge. Hence the verdict is resolved together with TOL_FLOOR
and not by the order alone.
"""

TOL_FLOOR = 1.0e-3
"""
Relative error on the finest grid below which the estimator PASSES without
looking at the order (0.1 %).

It is placed between two measured things, not by taste:
  - above the worst error of a convergent estimator on the fine grid of the
    prescribed shapes: zero level set 0.040 %, DSS 0.030 % on the canonical
    square; 0.0043 % and 0.0003 % on the leaf. Margin 2.5x.
  - below the smallest plateau of a NON-convergent estimator on the prescribed
    shape: Crofton-4 +2.58 %, marching squares +8.20 %. Margin 26x.
"""

PLATEAU_BAND = 0.25
"""
PLATEAU signature: the final error stayed within +-25 % of the initial error
after refining 4x, i.e. e_3 / e_1 in [0.8, 1.25].

It exists because the binary PASS/FAIL verdict gives a MEASURED false negative.
DSS's error is not monotone in h, and on the square at 35 degrees the 4x window
gives it order -0.40 with a final error of 0.120 %: a multigrid-convergent
estimator WITH A PROOF would be rejected. But its ratio e_3/e_1 is 1.74, i.e.
the error MOVED by 74 %; that is noise, not a plateau. An estimator with a bias
floor gives ratios from 0.97 to 1.14 in every case measured here.

And the INCONCLUSIVE instruction closes the case: running the same DSS at 35
degrees on n = 1600/3200/6400 gives order 4.27 and PASSES, while marching
squares on that same window gives order -0.01, ratio 1.02, and stays at FAIL.

Where the 25 % comes from: an order-1 estimator loses 75 % of its error when
refining 4x, and ORDER_MIN = 0.5 already demands losing 50 %. Staying within a
quarter in either direction is not slow convergence, it is a floor.

Measured e_3/e_1 ratios, canonical square: marching squares 1.04,
skimage.perimeter 1.08, Crofton-4 1.14 (all three PLATEAU); zero level set 0.20,
DSS 0.14 (both converge). On the repository's leaf: 1.00, 1.04, 1.01 against
0.61 and 0.01.
"""

# --------------------------------------------------------------------------
# Prescribed shapes.  The angle is NOT free: see `shapes_demo`.
# --------------------------------------------------------------------------
CANON_ANGLE = 22.5
"""
The square is rotated 22.5 degrees because there the Proffitt & Rosen (1979)
bound

    L_est / L_true = cos(theta) + (sqrt(2) - 1) sin(theta)

reaches its maximum: d/dtheta = 0 <=> tan(theta) = sqrt(2) - 1 <=> theta = 22.5.
It is the angle at which the naive estimator is as far from the truth as it can
get, and therefore where the test has the best signal-to-noise ratio. The angles
0 and 45 are the ZEROS of that same formula and must be avoided: see
`shapes_demo`, where marching squares passes the test with order 1.00 on an
axis-aligned square.
"""


# ==========================================================================
# 2 - FORMAS Y MUESTREO
# ==========================================================================

@dataclass(frozen=True)
class Sample:
    """What an estimator sees on one grid. `phi` is None if there is no field."""
    mask: np.ndarray          # indicadora booleana (lo unico que trae una red)
    g: np.ndarray             # 1-D coordinate of the NODES (x axis); square grid
    h: float                  # spacing, isotropic
    phi: np.ndarray | None    # signed field, if the shape has one


@dataclass
class Shape:
    name: str
    exact: float                       # exact perimeter, closed form
    exact_src: str                     # where that number comes from
    sampler: Callable[[int], Sample]

    def sample(self, n: int) -> Sample:
        return self.sampler(n)


_HALF = 1.6


def _grid_shape(phi_fn):
    def sampler(n: int) -> Sample:
        g = np.linspace(-_HALF, _HALF, n)          # NODOS, con extremos
        XX, YY = np.meshgrid(g, g)
        phi = phi_fn(XX, YY)
        return Sample(phi > 0, g, float(g[1] - g[0]), phi)
    return sampler


def _phi_square(ang_deg: float, side: float = 1.6):
    a = 0.5 * side

    def phi(XX, YY):
        t = math.radians(ang_deg)
        c, s = math.cos(t), math.sin(t)
        xr, yr = c * XX + s * YY, -s * XX + c * YY
        return np.minimum(np.minimum(a - xr, a + xr), np.minimum(a - yr, a + yr))
    return phi


def _phi_circle(XX, YY):
    return 1.0 - np.hypot(XX, YY)


def square(ang_deg: float) -> Shape:
    return Shape("square %g deg" % ang_deg, 6.4,
                 "side 1.6 x 4",
                 _grid_shape(_phi_square(ang_deg)))


CIRCLE = Shape("circle r=1", 2.0 * math.pi, "2*pi", _grid_shape(_phi_circle))
CANON = square(CANON_ANGLE)


# --------------------------------------------------------------------------
# Repository-native shape: the tear-free blade, whose perimeter has a closed
# quadrature (the same one validate.py uses, section 9, check 4).
# --------------------------------------------------------------------------
def leaf_shape() -> Shape | None:
    if not os.path.isdir(os.path.join(_ROOT, "foliar")):
        return None
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    try:
        from scipy import integrate
        from foliar import synth
    except Exception:
        return None

    cfg = synth.LeafConfig(n_tears=0, n_lesions=0)
    leaf = synth.SyntheticLeaf(cfg)
    L, H, A = cfg.length_cm, cfg.halfwidth_cm, 0.42   # 0.42 = exponente de halfwidth()

    def dW(u, side):
        s = np.clip(u / L, 1e-300, 1.0 - 1e-300)
        k = H / (0.5 ** (2 * A)) * (cfg.asymmetry if side < 0 else 1.0)
        return k * (A / L) * s ** (A - 1) * (1 - s) ** (A - 1) * (1 - 2 * s)

    exact = 0.0
    for side in (+1, -1):
        f = lambda u, side=side: np.sqrt(1.0 + dW(u, side) ** 2)
        exact += integrate.quad(lambda t: f(0.5 * L * t * t) * (L * t), 0, 1,
                                limit=400, epsabs=1e-10, epsrel=1e-10)[0]
        exact += integrate.quad(lambda t: f(L - 0.5 * L * t * t) * (L * t), 0, 1,
                                limit=400, epsabs=1e-10, epsrel=1e-10)[0]

    # n is reinterpreted as "nodes along the leaf"; the grid is forced
    # isotropic (hu == hv exactly) so DSS receives square pixels.
    def sampler(n: int) -> Sample:
        ug = np.linspace(-0.03 * L, 1.03 * L, n)
        h = float(ug[1] - ug[0])
        Hv = H * 1.03
        nv = int(math.ceil(2 * Hv / h)) + 1
        half_v = 0.5 * (nv - 1) * h
        vg = np.linspace(-half_v, half_v, nv)
        UG, VG = np.meshgrid(ug, vg)
        phi = leaf.blade_signed(UG, VG)
        return Sample(phi > 0, ug, h, phi)

    return Shape("leaf, no tears (repo)", exact,
                 "arc-length quadrature of v = +-W(u)", sampler)


LEAF_GRIDS = (584, 1167, 2333)   # h = 0.400, 0.200, 0.100 cm


# ==========================================================================
# 3 - ESTIMADORES
# ==========================================================================

def est_marching_squares(s: Sample) -> float:
    """The de facto default: marching squares at level 0.5 on the indicator."""
    return _contour_len(s.mask.astype(float), s.g, s.h, 0.5)


def est_skimage_perimeter(s: Sample) -> float:
    return float(measure.perimeter(s.mask)) * s.h


def est_crofton4(s: Sample) -> float:
    return float(measure.perimeter_crofton(s.mask, directions=4)) * s.h


def est_levelset(s: Sample) -> float:
    """Zero level set of the signed field (needs the generator itself)."""
    if s.phi is None:
        raise ValueError("this shape exposes no signed field")
    return _contour_len(s.phi, s.g, s.h, 0.0)


def est_dss(s: Sample) -> float:
    """Digital straight segments on the SAME binary mask."""
    return dss.dss_length(s.mask, s.h)


def _contour_len(field, g, h, level) -> float:
    # The grid is isotropic, so the origin does not affect the length; both
    # coordinates are scaled by the SAME h and the chords are summed.
    tot = 0.0
    for c in measure.find_contours(field, level):
        pts = c * h
        tot += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return tot


ESTIMATORS = [
    ("marching squares", est_marching_squares, False),
    ("skimage.perimeter", est_skimage_perimeter, False),
    ("skimage.crofton(4)", est_crofton4, False),
    ("zero level set", est_levelset, True),
    ("DSS", est_dss, True),
]  # (nombre, fn, se_espera_que_pase)


# ==========================================================================
# 4 - THE TEST
# ==========================================================================

def two_grid_check(est, shape: Shape, grids=None) -> dict:
    """
    Measure `est` on `shape` over three grids and return the verdict.

    Core (the paper's three lines):
        L = [est(shape.sample(n)) for n in grids]
        e = [abs(v / shape.exact - 1) for v in L]
        order = log2(e[0] / e[-1]) / 2
    """
    grids = tuple(grids or GRIDS)
    assert len(grids) == 3, "the test uses three grids in ratio 2"

    L = [est(shape.sample(n)) for n in grids]
    e = [abs(v / shape.exact - 1) for v in L]

    eps = 1e-15
    order = math.log2(max(e[0], eps) / max(e[-1], eps)) / (len(grids) - 1)

    ratio = e[-1] / max(e[0], eps)       # plateau signature: ~1 => it did not move
    if order >= ORDER_MIN or e[-1] <= TOL_FLOOR:
        verdict = "PASS"
    elif abs(ratio - 1.0) <= PLATEAU_BAND:
        verdict = "FAIL"         # plateau: the error did not move under 4x refinement
    else:
        verdict = "INCONCL"      # serie no monotona; extender, ver PLATEAU_BAND
    passes = verdict == "PASS"

    # diagnostic, NOT criterion: the self-convergence order, which is what one
    # can measure without exact truth.  See `self_demo`.
    d = [abs(L[i + 1] - L[i]) for i in range(len(L) - 1)]
    order_self = (math.log2(max(d[0], eps) / max(d[1], eps))
                  if d[1] > 0 else float("inf"))

    why = "-"
    if verdict == "PASS":
        why = "+".join([w for w, c in (("order", order >= ORDER_MIN),
                                       ("floor", e[-1] <= TOL_FLOOR)) if c])
    elif verdict == "INCONCL":
        why = "extend"

    return dict(shape=shape.name, grids=grids, values=L,
                signed_err=[v / shape.exact - 1 for v in L], abs_err=e,
                order=order, order_self=order_self, passes=passes,
                verdict=verdict, why=why, ratio=ratio)


# ==========================================================================
# 5 - INFORMES
# ==========================================================================

def _row(name, r):
    return ("  %-19s %s  %7.2f %6.2f %8s  %-11s %s"
            % (name,
               " ".join("%+9.4f%%" % (100 * x) for x in r["signed_err"]),
               r["order"], r["ratio"],
               "%.2f" % r["order_self"] if math.isfinite(r["order_self"]) else "inf",
               r["why"] or "-",
               r["verdict"]))


def _header(shape, grids):
    print("  shape: %s   P_exact = %.8f" % (shape.name, shape.exact))
    print("  source of the exact value: %s" % shape.exact_src)
    print("  %-19s %-31s %7s %6s %8s  %-11s %s"
          % ("estimator", "relative error at n = %d/%d/%d" % grids,
             "order", "e3/e1", "p_self", "passes via", "verdict"))
    print("  " + "-" * 95)


# DOCUMENTED false PASSes: (shape, estimator) where the test approves a
# non-convergent estimator. These are failures of the shape, not of the
# criterion, and they are exactly why the checklist PRESCRIBES the shape.
KNOWN_FALSE_PASS = {
    ("circle r=1", "skimage.crofton(4)"),
    ("square 0 deg", "marching squares"),
    ("square 0 deg", "skimage.perimeter"),
}


def run_shape(shape: Shape, grids=None, ests=None) -> dict:
    grids = tuple(grids or GRIDS)
    ests = ests or ESTIMATORS
    _header(shape, grids)
    out = {}
    for name, fn, expect in ests:
        try:
            r = two_grid_check(fn, shape, grids)
        except ValueError as exc:
            print("  %-19s no aplicable (%s)" % (name, exc))
            continue
        known = (shape.name, name) in KNOWN_FALSE_PASS
        r["expected"] = expect or known
        r["known_false_pass"] = known
        out[name] = r
        print(_row(name, r))
        if known and r["passes"]:
            print("      (DOCUMENTED false PASS: the shape has no generic"
                  " orientation; see `twogrid.py shapes`)")
        elif r["passes"] != r["expected"]:
            print("      >>> DISCREPANCY with what was expected (%s)"
                  % ("should pass" if r["expected"] else "should fail"))
    print()
    return out


def checklist():
    print("=" * 92)
    print("TWO-GRID CONVERGENCE TEST - minimal checklist")
    print("=" * 92)
    print("umbrales:  ORDER_MIN = %.2f   TOL_FLOOR = %.1e   PLATEAU_BAND = %.2f"
          "   grillas = %s" % (ORDER_MIN, TOL_FLOOR, PLATEAU_BAND, GRIDS))
    print("verdict: PASS    order >= ORDER_MIN  OR  final |e| <= TOL_FLOOR")
    print("           FAIL    if not, and also |e3/e1 - 1| <= PLATEAU_BAND (plateau)")
    print("           INCONCL if not -> non-monotone series: extend, do not conclude")
    print("p_self = SELF-convergence order (no exact truth); it is"
          " diagnostico, no criterio.")
    print()

    print("--- (1) canonical shape: square at %g degrees ---" % CANON_ANGLE)
    r1 = run_shape(CANON)

    print("--- (2) second shape, smooth boundary: circle ---")
    r2 = run_shape(CIRCLE)

    print("--- (3) repository-native shape: the tear-free blade ---")
    lf = leaf_shape()
    if lf is None:
        print("  (foliar/ not importable from %s; skipped)\n" % _ROOT)
        r3 = {}
    else:
        r3 = run_shape(lf, LEAF_GRIDS)

    print("=" * 92)
    print("RESUMEN")
    print("=" * 92)
    ok = True
    for res in (r1, r2, r3):
        for r in res.values():
            if r["passes"] != r["expected"]:
                ok = False
    print("  (i)  the test FAILS on binary marching squares:            %s"
          % ("YES" if not r1["marching squares"]["passes"] else "NO"))
    print("  (ii) the test PASSES on the zero level set estimator:      %s"
          % ("YES" if r1["zero level set"]["passes"] else "NO"))
    print("       the test PASSES on DSS:                               %s"
          % ("YES" if r1["DSS"]["passes"] else "NO"))
    print("  (iii) it also fails on skimage.perimeter and Crofton-4:    %s"
          % ("YES" if (not r1["skimage.perimeter"]["passes"]
                      and not r1["skimage.crofton(4)"]["passes"]) else "NO"))
    print()
    print("  MEASURED NOTE, not decoration: on the CIRCLE, Crofton-4 passes through"
          " the floor branch")
    print("  (error %.4f %%). That is a failure of the shape chosen, not of the test"
          % (100 * r2["skimage.crofton(4)"]["abs_err"][-1]))
    print("  : see `python3 twogrid.py shapes`. That is why the checklist"
          " PRESCRIBES the angle.")
    print()
    print("  overall checklist verdict: %s"
          % ("consistent with expectations" if ok else "THERE ARE DISCREPANCIES"))
    return 0 if ok else 1


def shapes_demo():
    """Why the angle is prescribed: two documented false PASSes."""
    print("=" * 92)
    print("WHY THE SHAPE IS NOT FREE - two reproducible false PASSes")
    print("=" * 92)
    print()
    print("(a) square ALIGNED with the axes (theta = 0): the Proffitt & Rosen")
    print("    bound is cos0 + (sqrt2-1) sin0 - 1 = 0, i.e. the bias of the naive")
    print("    estimator VANISHES by construction. All that is left is")
    print("    resolution error, which is indeed order 1.")
    run_shape(square(0.0), ests=[e for e in ESTIMATORS if e[0] != "DSS"])
    print("(b) circle: Kulpa's constant averages the bias over directions")
    print("    and Crofton-4 ends up below the floor.")
    run_shape(CIRCLE, ests=[e for e in ESTIMATORS if "crofton" in e[0]])
    print("(c) the same square at 22.5 degrees, where the bound is MAXIMAL:")
    run_shape(CANON, ests=[e for e in ESTIMATORS if e[0] != "DSS"])
    print("RULE: run the test on at least one shape whose boundary is in")
    print("      GENERIC orientation. The square at 22.5 degrees is the")
    print("      canonical choice because it maximises the 1979 bound. A circle")
    print("      alone, or an aligned square, give the false PASSes measured above.")
    print()
    print("angle sweep, n = 1600, to locate the zeros of each bias:")
    print("  %5s %11s %11s %11s %11s" % ("theta", "MS", "crofton4", "zero level", "DSS"))
    for a in (0, 5, 10, 15, 20, 22.5, 25, 30, 35, 40, 45):
        sh = square(a)
        s = sh.sample(1600)
        row = [est_marching_squares(s), est_crofton4(s), est_levelset(s), est_dss(s)]
        print("  %5.1f " % a + " ".join("%+10.4f%%" % (100 * (v / 6.4 - 1)) for v in row))
    print()
    print("  crofton4 crosses zero near 10 and 35 degrees: at those angles")
    print("  it would pass through the floor. 22.5 is as far from both as possible.")
    print()
    print("verdict by angle (n = %d/%d/%d), and DSS's FALSE NEGATIVE:" % GRIDS)
    print("  %6s %-9s %-9s %-9s %-9s" % ("theta", "MS", "crofton4", "zerolvl", "DSS"))
    for a in (0, 5, 10, 15, 20, 22.5, 25, 30, 35, 40, 45):
        sh = square(a)
        cells = []
        for fn in (est_marching_squares, est_crofton4, est_levelset, est_dss):
            r = two_grid_check(fn, sh)
            cells.append("%-9s" % r["verdict"])
        print("  %6.1f " % a + " ".join(cells))
    print()
    print("  At 35 degrees DSS -- multigrid-convergent WITH A PROOF -- gives")
    print("  order -0.40 with a final error of 0.120 %: the 4x window lands on a")
    print("  non-monotone stretch of its series. With a binary verdict that would")
    print("  be a false REJECTION. The plateau signature saves it: its e3/e1 is")
    print("  1.74, i.e. the error DID MOVE, whereas a bias floor gives 0.97..1.14.")
    print("  It is a measured limitation of the test, not a presentation detail.")
    print()
    print("  And the INCONCLUSIVE instruction closes the case, measured: at")
    print("  n = 1600/3200/6400 the same DSS at 35 degrees gives")
    for fn, nm in ((est_dss, "DSS"), (est_marching_squares, "MS ")):
        r = two_grid_check(fn, square(35.0), (1600, 3200, 6400))
        print("    %s  %s  order %5.2f  e3/e1 %.2f  ->  %s"
              % (nm, " ".join("%+8.4f%%" % (100 * x) for x in r["signed_err"]),
                 r["order"], r["ratio"], r["verdict"]))
    print("  so the inconclusive band is not a permanent tie: it separates")
    print("  the two with one more grid.")


def self_demo():
    """
    The point that connects to the inverse crime: SELF-convergence is no use.

    A practitioner with no exact truth measures |L(h) - L(h/2)| and its order.
    That number does not distinguish "converges" from "converges to the wrong
    value".
    """
    print("=" * 92)
    print("SELF-CONVERGENCE VS CONVERGENCE - why exact truth is needed")
    print("=" * 92)
    print()
    print("  canonical shape, square at %g degrees, grids %s" % (CANON_ANGLE, GRIDS))
    print("  %-19s %10s %12s %14s" % ("estimator", "p_self", "real order", "final error"))
    print("  " + "-" * 60)
    for name, fn, _ in ESTIMATORS:
        r = two_grid_check(fn, CANON)
        print("  %-19s %10.2f %12.2f %13.4f%%"
              % (name, r["order_self"], r["order"], 100 * r["abs_err"][-1]))
    print()
    print("  Marching squares exhibits self-convergence of order ~2.8 while its")
    print("  true error stays pinned at +8.2 %. Refining inside the shared")
    print("  scheme exposes NOTHING: that is exactly why the test needs the exact")
    print("  perimeter in the denominator, and not a second run of the same")
    print("  estimator on a finer grid.")


CHECKLIST_TEXT = """
CHECKLIST (to copy into a repository or an appendix)

  1. Pick a shape with a CLOSED-FORM perimeter and a boundary in GENERIC
     orientation. Canonical: square of side a rotated 22.5 degrees, P = 4a.
     A circle alone is not enough, nor is an axis-aligned square.
  2. Sample on three node grids in ratio 2: n = 400, 800, 1600 across the
     domain.
  3. e_i = |L_i / P_exact - 1|.
  4. order = log2(e_1 / e_3) / 2.
  5. PASS if order >= 0.5 OR e_3 <= 1e-3.
     REJECT if it does not pass AND |e_3/e_1 - 1| <= 0.25: the error did not
     move when refining 4x, i.e. there is a bias floor and the estimator is no
     use as a REFERENCE.
     INCONCLUSIVE in any other case: the series is non-monotone, extend it to
     n = 3200 and 6400 before concluding. Do not read it as a pass.
  6. Do not replace P_exact with a finer run of the same estimator:
     self-convergence gives order ~2.8 on an estimator whose real error is
     +8.2 % and constant. It is the same reason the standard remedy for the
     inverse crime -- generating the data on a finer mesh -- exposes nothing
     here: the bias has a floor independent of h.
"""


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "quick":
        run_shape(CANON)
        return 0
    if cmd == "shapes":
        shapes_demo()
        return 0
    if cmd == "self":
        self_demo()
        return 0
    rc = checklist()
    print()
    self_demo()
    print(CHECKLIST_TEXT)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
