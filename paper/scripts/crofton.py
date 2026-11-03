#!/usr/bin/env python3
"""
Reviewer questions 1 and 3.

(A) `identity`  -- PROOF that scikit-image's 4-direction discrete Crofton is
    marching squares rescaled by Kulpa's constant on masks with no saddle
    configuration, and that saddles are the only correction. The proof is
    algebraic (see the `LEMMAS` docstring below) and is verified here
    exhaustively.

(B) `directions` -- discrete Crofton with 2, 4, 8, 16 and 32 directions,
    implemented by hand (skimage only goes up to 4). Measures bias on the
    circle, orientation spread on the square, and whether it converges or
    plateaus.

Usage:
    python3 crofton.py identity
    python3 crofton.py directions
    python3 crofton.py all
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from skimage import measure

# scripts/ lives two levels below the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir))

SQ2 = math.sqrt(2.0)
K_KULPA = 4.0 / math.pi * (2.0 * SQ2 - 2.0)          # 1.05478618...

LEMMAS = """
Notation.  Let B be a subset of Z^2, a binary mask (background 0 outside the
support).  On the grid of pixel CENTRES define the four transition counters

    N_h  = #{ (i,j) : B[i,j] != B[i,j+1] }        (crossings of horizontal lines)
    N_v  = #{ (i,j) : B[i,j] != B[i+1,j] }        (crossings of vertical lines)
    N_d1 = #{ (i,j) : B[i,j] != B[i+1,j+1] }      (diagonal +45)
    N_d2 = #{ (i,j) : B[i,j+1] != B[i+1,j] }      (diagonal -45)
    S    = # of 2x2 cells in a SADDLE configuration (the two pixels of one
           diagonal at 1 and those of the other at 0, or vice versa)

Lemma 1 (4-direction discrete Crofton).
    perimeter_crofton(B, directions=4) = (pi/8) [ N_h + N_v + (N_d1+N_d2)/sqrt2 ].
  Proof in two steps.
  (i) The Cauchy-Crofton formula gives L = (1/2) int_0^pi N(theta) d(theta).
      With four equispaced directions (d(theta) = pi/4) and perpendicular
      spacing between neighbouring lines of the family (p,q) equal to
      1/sqrt(p^2+q^2) -- 1 for the axes, 1/sqrt2 for the diagonals -- the
      quadrature is L = (1/2)(pi/4)[N_h + N_v + (N_d1+N_d2)/sqrt2].  That is the
      RIGHT-HAND SIDE.
  (ii) The left-hand side, as skimage implements it, is coefs . H with H[code]
      the histogram of the 16 2x2 configurations and coefs its LUT.  The
      right-hand side is also linear in H: each horizontal adjacency is the top
      row of one cell and the bottom row of another, each vertical one is the
      left column of one and the right column of another, and each diagonal
      adjacency lives in a single cell, so the right-hand side equals w . H with
          w[code] = (pi/16)(t + b + l + r) + (pi/(8 sqrt2))(d1 + d2),
      where t,b,l,r,d1,d2 are the six edge indicators of the cell.
      Now, coefs != w component by component; what holds is that the DIFFERENCE
      annihilates every realisable histogram.  For a mask with a zero frame
      exactly five conservation laws hold over the cells:
          sum t = sum b,   sum l = sum r,
          sum TL = sum TR = sum BL = sum BR   (each pixel is the TL corner of
          exactly one cell, the TR of exactly one, etc.)
      and their linear span has dimension 5, which is exactly the dimension of
      the annihilator of the cone of realisable histograms (rank 11 of 16,
      measured).  The difference is written in that basis with closed-form
      coefficients:
          coefs - w = pi[ (1/16)(t-b) - (1/16)(l-r)
                          + (1/8 - 1/(8 sqrt2))(TL-BR)
                          - (1/4 - 1/(8 sqrt2))(TR-BR)
                          - (1/(8 sqrt2))(BL-BR) ]
      (residual 5.6e-17, verified in `check_lut`).  Therefore coefs.H = w.H for
      EVERY mask, which is the statement.

Lemma 2 (marching squares at level 0.5 on the indicator).
    MS(B) = (sqrt2 - 1)(N_h + N_v) + ((2 - sqrt2)/2)(N_d1 + N_d2) + (2 - sqrt2) S.
  Proof: with values in {0,1} and level 0.5 every crossing falls at the MIDPOINT
  of the cell edge, so each 2x2 cell contributes a segment of fixed length
  according to its configuration.  Up to symmetry there are five cases, and for
  each we record (h_c, v_c, d_c) and the length L_c:
      uniform      (0,0,0)  L=0
      corner       (1,1,1)  L=sqrt2/2      (8 of the 16 configurations)
      straight     (0,2,2) or (2,0,2)  L=1 (4 configurations)
      saddle       (2,2,0)  L=sqrt2        (2 configurations)
  The system  alpha*(h_c+v_c) + beta*d_c = L_c  restricted to the "corner" and
  "straight" rows has the UNIQUE solution alpha = (sqrt2-1)/2,
  beta = (2-sqrt2)/2, and that solution also reproduces the uniform row.  The
  only row it does NOT satisfy is the saddle: it predicts 4*alpha = 2(sqrt2-1)
  and the value is sqrt2, a deficit of sqrt2 - 2(sqrt2-1) = 2 - sqrt2 per
  saddle.  Summing over cells and using  sum_c h_c = 2 N_h,  sum_c v_c = 2 N_v,
  sum_c d_c = N_d1+N_d2  (each diagonal adjacency lives in a single cell) gives
  the statement.

Theorem.  For every binary mask
    MS(B) = (4/pi)(2 sqrt2 - 2) * crofton_4(B) + (2 - sqrt2) * S(B),
  and in particular MS = 1.05478618 * crofton_4 exactly when S = 0.
  Proof: by Lemma 1, (4/pi)(2sqrt2-2)*crofton_4 =
  (2sqrt2-2)/2 * [N_h+N_v+(N_d1+N_d2)/sqrt2] = (sqrt2-1)(N_h+N_v) +
  ((2-sqrt2)/2)(N_d1+N_d2), which is exactly the saddle-free term of Lemma 2;
  the diagonal coefficient matches because (sqrt2-1)/sqrt2 = (2-sqrt2)/2.  QED.

Corollary (the one the paper uses).  On saddle-free masks the two estimators are
the SAME linear functional of (N_h, N_v, N_d1, N_d2) up to a multiplicative
constant that does not depend on the mask.  Therefore Crofton-4 cannot correct
the orientation dependence of marching squares: it only rescales it, and its
measured orientation spread in percentage points is that of marching squares
divided by 1.05478618.  Saddles are the only exception, and they are a subpixel
effect that becomes rare under refinement: they do not change the limit.
"""


# --------------------------------------------------------------------------
# contadores
# --------------------------------------------------------------------------
def counts(mask: np.ndarray, pad: int = 1):
    """(N_h, N_v, N_d1, N_d2, S) with a zero background all around."""
    m = np.pad(np.asarray(mask, dtype=bool), pad)
    n_h = int(np.count_nonzero(m[:, :-1] != m[:, 1:]))
    n_v = int(np.count_nonzero(m[:-1, :] != m[1:, :]))
    n_d1 = int(np.count_nonzero(m[:-1, :-1] != m[1:, 1:]))
    n_d2 = int(np.count_nonzero(m[:-1, 1:] != m[1:, :-1]))
    a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
    n_s = int(np.count_nonzero((a == d) & (b == c) & (a != b)))
    return n_h, n_v, n_d1, n_d2, n_s


def crofton4_predicted(mask) -> float:
    n_h, n_v, n_d1, n_d2, _ = counts(mask)
    return math.pi / 8.0 * (n_h + n_v + (n_d1 + n_d2) / SQ2)


def ms_predicted(mask) -> float:
    n_h, n_v, n_d1, n_d2, n_s = counts(mask)
    return ((SQ2 - 1.0) * (n_h + n_v)
            + (2.0 - SQ2) / 2.0 * (n_d1 + n_d2)
            + (2.0 - SQ2) * n_s)


def ms_measured(mask, pad: int = 2) -> float:
    """skimage marching squares at level 0.5 (padded so nothing is clipped)."""
    f = np.pad(np.asarray(mask, dtype=float), pad)
    tot = 0.0
    for c in measure.find_contours(f, 0.5):
        tot += float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
    return tot


def crofton4_measured(mask, pad: int = 2) -> float:
    return float(measure.perimeter_crofton(np.pad(np.asarray(mask, bool), pad), 4))


# --------------------------------------------------------------------------
# (A) verification of the proof
# --------------------------------------------------------------------------
def check_lut() -> int:
    """
    Lemma 1, step (ii): coefs - w is an exact combination of the FIVE
    conservation laws satisfied by every 2x2 histogram of a mask with a zero
    frame.  That proves coefs.H = w.H for every mask, not only for the ones
    enumerated afterwards.
    """
    from skimage.measure import _regionprops_utils as ru
    import inspect
    src = inspect.getsource(ru.perimeter_crofton)
    ok_k = "[[0, 0, 0], [0, 1, 4], [0, 2, 8]]" in src
    print(f"  [{'OK ' if ok_k else 'BAD'}] skimage's 2x2 kernel is still"
          f" [[0,0,0],[0,1,4],[0,2,8]]  (bit0=BR bit1=TR bit2=BL bit3=TL)")

    coefs = np.array([
        0.0,
        math.pi / 4 * (1 + 1 / SQ2), math.pi / (4 * SQ2), math.pi / (2 * SQ2),
        0.0,
        math.pi / 4 * (1 + 1 / SQ2), 0.0, math.pi / (4 * SQ2),
        math.pi / 4, math.pi / 2, math.pi / (4 * SQ2), math.pi / (4 * SQ2),
        math.pi / 4, math.pi / 2, 0.0, 0.0])

    c = np.arange(16)
    br = ((c >> 0) & 1).astype(float)
    tr = ((c >> 1) & 1).astype(float)
    bl = ((c >> 2) & 1).astype(float)
    tl = ((c >> 3) & 1).astype(float)
    t = (tl != tr).astype(float)
    b = (bl != br).astype(float)
    lf = (tl != bl).astype(float)
    rt = (tr != br).astype(float)
    d1 = (tl != br).astype(float)
    d2 = (tr != bl).astype(float)
    w = math.pi / 16 * (t + b + lf + rt) + math.pi / (8 * SQ2) * (d1 + d2)

    rel = np.column_stack([t - b, lf - rt, tl - br, tr - br, bl - br])
    lam = math.pi * np.array([1 / 16, -1 / 16,
                              1 / 8 - 1 / (8 * SQ2),
                              -(1 / 4 - 1 / (8 * SQ2)),
                              -1 / (8 * SQ2)])
    res = float(np.abs(rel @ lam - (coefs - w)).max())
    rk = int(np.linalg.matrix_rank(rel))
    ok = res < 1e-14 and rk == 5
    print(f"  [{'OK ' if ok else 'BAD'}] Lemma 1 (ii): coefs - w lies in the span of"
          f" the 5 conservation laws (rank {rk}), with closed-form coefficients;"
          f" residual {res:.2e}")

    # and the five laws really do hold on random masks
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(400):
        n_r = int(rng.integers(3, 12))
        n_c = int(rng.integers(3, 12))
        mm = rng.random((n_r, n_c)) < float(rng.random())
        hh = _hist2x2(mm)
        worst = max(worst, float(np.abs(rel.T @ hh).max()))
    ok2 = worst < 1e-12
    print(f"  [{'OK ' if ok2 else 'BAD'}] the 5 conservation laws hold"
          f" over 400 random masks  (maximum deviation {worst:.2e})")
    return int(not ok_k) + int(not ok) + int(not ok2)


def _hist2x2(mask) -> np.ndarray:
    """Histogram of the 16 2x2 configurations, with skimage's kernel."""
    from scipy import ndimage as ndi
    im = np.pad(np.asarray(mask, dtype=bool).astype(np.uint8), 1)
    xf = ndi.convolve(im, np.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]]),
                      mode="constant", cval=0)
    return np.bincount(xf.ravel(), minlength=16).astype(float)


def exhaustive_4x4() -> int:
    """All 2^16 configurations of a 4x4 window."""
    bad = 0
    n_tot = n_sad = 0
    w_lema1 = w_lema2 = w_teo_sin = w_teo_con = 0.0
    for bits in range(1 << 16):
        g = np.array([(bits >> i) & 1 for i in range(16)], bool).reshape(4, 4)
        if not g.any():
            continue
        n_tot += 1
        mm = np.zeros((8, 8), bool)
        mm[2:6, 2:6] = g
        _, _, _, _, n_s = counts(mm)
        cro = crofton4_measured(mm)
        ms = ms_measured(mm)
        w_lema1 = max(w_lema1, abs(cro - crofton4_predicted(mm)))
        w_lema2 = max(w_lema2, abs(ms - ms_predicted(mm)))
        res = abs(ms - (K_KULPA * cro + (2.0 - SQ2) * n_s))
        w_teo_con = max(w_teo_con, res)
        if n_s == 0:
            w_teo_sin = max(w_teo_sin, abs(ms - K_KULPA * cro))
        else:
            n_sad += 1

    tol = 1e-9
    for name, w in (("Lemma 1  crofton_4 = (pi/8)[N_h+N_v+(N_d1+N_d2)/sqrt2]", w_lema1),
                    ("Lemma 2  MS = (sqrt2-1)(N_h+N_v) + ((2-sqrt2)/2)(N_d1+N_d2)"
                     " + (2-sqrt2) S", w_lema2),
                    (f"Theorem MS = {K_KULPA:.8f} * crofton_4  (saddle-free only)",
                     w_teo_sin),
                    (f"Theorem MS = {K_KULPA:.8f} * crofton_4 + (2-sqrt2) S"
                     "  (ALL)", w_teo_con)):
        ok = w < tol
        bad += not ok
        print(f"  [{'OK ' if ok else 'BAD'}] {name}   max residual {w:.2e}")
    print(f"       4x4 window: {n_tot} non-empty masks, "
          f"{n_tot - n_sad} saddle-free, {n_sad} with at least one saddle")
    return bad


def leaf_mask():
    """The project's blade mask (seed 7, 4 px/cm).

    The root is derived from the location of THIS file, not from an absolute
    path: scripts/ sits two levels below the repository root. There used to be
    paths carrying the machine's username, which also do not run anywhere else.
    """
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from foliar import synth
    s = synth.generate(synth.LeafConfig())
    return s.mask_blade, 1.0 / s.px_per_cm


def check_leaf() -> int:
    bad = 0
    m, h = leaf_mask()
    n_h, n_v, n_d1, n_d2, n_s = counts(m)
    cro = crofton4_measured(m)
    ms = ms_measured(m)
    print(f"       leaf {m.shape[1]}x{m.shape[0]} px, h = {h:g} cm/px;  "
          f"N_h={n_h} N_v={n_v} N_d1={n_d1} N_d2={n_d2} saddles={n_s}")
    for name, lhs, rhs in (
            ("Lemma 1", cro, crofton4_predicted(m)),
            ("Lemma 2", ms, ms_predicted(m)),
            ("Theorem (with saddle term)", ms, K_KULPA * cro + (2 - SQ2) * n_s)):
        rel = abs(lhs - rhs) / max(1.0, abs(lhs))
        ok = rel < 1e-12
        bad += not ok
        print(f"  [{'OK ' if ok else 'BAD'}] {name} on the leaf: "
              f"{lhs * h:.6f} cm vs {rhs * h:.6f} cm  (relative residual {rel:.2e})")
    print(f"       measured ratio MS / crofton_4 = {ms / cro:.10f}   "
          f"Kulpa constant = {K_KULPA:.10f}")
    return bad


def cmd_identity() -> int:
    print(LEMMAS)
    print("=" * 78)
    print("VERIFICATION")
    print("=" * 78)
    bad = check_lut()
    bad += exhaustive_4x4()
    bad += check_leaf()
    print(f"\n  --> {'ALL GREEN' if bad == 0 else str(bad) + ' FAILURES'}")
    return bad


# --------------------------------------------------------------------------
# (B) discrete Crofton with K directions
# --------------------------------------------------------------------------
def direction_family(k: int):
    """
    The K directions are the PRIMITIVE vectors of the upper half-plane with
    |v|^2 <= R^2, sorted by angle.  The cut-offs giving exactly 2/4/8/16/32
    directions are R^2 = 1, 2, 5, 13, 29.

    Returns [(p, q, delta_theta, spacing)], with spacing = 1/|v| = perpendicular
    separation between neighbouring lines of the family (p,q) over Z^2, and
    delta_theta the Voronoi angular weight (theta_{k+1}-theta_{k-1})/2 on the
    projective circle of period pi.
    """
    cut = {2: 1, 4: 2, 8: 5, 16: 13, 32: 29}
    if k not in cut:
        raise ValueError(f"K must be one of {sorted(cut)}")
    r2 = cut[k]
    rmax = int(math.isqrt(r2)) + 1
    vs = []
    for p in range(-rmax, rmax + 1):
        for q in range(0, rmax + 1):
            if q == 0 and p <= 0:
                continue                       # half-plane: q>0, or q=0 and p>0
            if p * p + q * q > r2:
                continue
            if math.gcd(abs(p), q) != 1:
                continue                       # primitive vectors only
            vs.append((p, q))
    vs.sort(key=lambda v: math.atan2(v[1], v[0]) % math.pi)
    assert len(vs) == k, f"expected {k} directions, got {len(vs)}"
    th = [math.atan2(q, p) % math.pi for p, q in vs]
    out = []
    for i, (p, q) in enumerate(vs):
        nxt = th[(i + 1) % k] + (math.pi if i == k - 1 else 0.0)
        prv = th[i - 1] - (math.pi if i == 0 else 0.0)
        out.append((p, q, (nxt - prv) / 2.0, 1.0 / math.hypot(p, q)))
    return out


def crofton_k(mask: np.ndarray, k: int, h: float = 1.0,
              fam=None) -> float:
    """
    K-direction discrete Crofton:  L = (1/2) sum_k dtheta_k * Delta_k * N_k
    with N_k = number of pairs (x, x+v_k) where the mask changes value.
    """
    fam = fam if fam is not None else direction_family(k)
    pad = max(max(abs(p), abs(q)) for p, q, _, _ in fam)
    m = np.pad(np.asarray(mask, dtype=bool), pad)
    n_r, n_c = m.shape
    tot = 0.0
    for p, q, dth, sep in fam:
        # v = (p, q):  p = step in columns (x),  q = step in rows (y).
        # Common crop so B[x] can be compared with B[x+v] without leaving the array.
        r0, r1 = max(0, -q), n_r - max(0, q)
        c0, c1 = max(0, -p), n_c - max(0, p)
        lhs = m[r0:r1, c0:c1]
        rhs = m[r0 + q:r1 + q, c0 + p:c1 + p]
        tot += dth * sep * int(np.count_nonzero(lhs != rhs))
    return 0.5 * tot * h


# --------------------------------------------------------------------------
# test shapes (SAME convention as conv.py)
# --------------------------------------------------------------------------
HALF = 1.6
P_CIR = 2.0 * math.pi
P_SQ = 6.4                                    # square of side 1.6


def grid(n: int):
    g = np.linspace(-HALF, HALF, n)           # NODE grid, endpoints included
    return g, float(g[1] - g[0])


def mask_circle(n: int) -> np.ndarray:
    g, _ = grid(n)
    return (1.0 - np.hypot(g[None, :], g[:, None])) > 0.0


def mask_square(n: int, ang_deg: float) -> np.ndarray:
    g, _ = grid(n)
    t = math.radians(ang_deg)
    c, s = math.cos(t), math.sin(t)
    xx, yy = g[None, :], g[:, None]
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    phi = np.minimum(np.minimum(0.8 - xr, 0.8 + xr),
                     np.minimum(0.8 - yr, 0.8 + yr))
    return phi > 0.0


def asymptotic_ratio(k: int, phi_deg: float, fam=None) -> float:
    """
    Limit h -> 0 of the ratio (estimated / true) for a STRAIGHT BOUNDARY of
    direction phi.

    A straight line of direction phi and length L cuts the family (p,q) in
    N_k -> L |sin(theta_k - phi)| / Delta_k intersections, so
        L_est / L  ->  (1/2) sum_k dtheta_k |sin(theta_k - phi)|,
    which is the quadrature of (1/2) int_0^pi |sin(theta - phi)| d(theta) = 1 at
    the nodes theta_k.  Since |sin| has a KINK at theta = phi, the quadrature
    cannot be exact except by coincidence: that is the residual bias that
    survives refinement, and it explains why the estimator plateaus at a value
    != 0 for any FIXED K.
    """
    fam = fam if fam is not None else direction_family(k)
    phi = math.radians(phi_deg)
    return 0.5 * sum(dth * abs(math.sin(math.atan2(q, p) - phi))
                     for p, q, dth, _ in fam)


def asymptotic_square(k: int, ang_deg: float, fam=None) -> float:
    """Square: two pairs of sides, at phi and at phi + 90 degrees."""
    return 0.5 * (asymptotic_ratio(k, ang_deg, fam)
                  + asymptotic_ratio(k, ang_deg + 90.0, fam))


KS = (2, 4, 8, 16, 32)
NS = (200, 400, 800, 1600, 3200)
ANGS = tuple(float(a) for a in range(0, 50, 5))


def cmd_directions() -> int:
    fams = {k: direction_family(k) for k in KS}
    print("### Direction families\n")
    print("| K | cut-off |v|^2 | vectors (half-plane) |")
    print("|---:|---:|---|")
    cut = {2: 1, 4: 2, 8: 5, 16: 13, 32: 29}
    for k in KS:
        vv = " ".join(f"({p},{q})" for p, q, _, _ in fams[k])
        print(f"| {k} | {cut[k]} | {vv if k <= 16 else vv[:120] + ' ...'} |")

    # sanity: K=4 must reproduce skimage exactly
    m = mask_square(401, 30.0)
    a = crofton_k(m, 4)
    b = float(measure.perimeter_crofton(np.pad(m, 1), 4))
    print(f"\n  [{'OK ' if abs(a - b) < 1e-9 * b else 'BAD'}] control: our own"
          f" implementation with K=4 reproduces skimage.perimeter_crofton"
          f"  ({a:.9f} vs {b:.9f})")
    m = mask_square(401, 30.0)
    a2 = crofton_k(m, 2)
    b2 = float(measure.perimeter_crofton(np.pad(m, 1), 2))
    print(f"  [{'OK ' if abs(a2 - b2) < 1e-9 * b2 else 'BAD'}] control: K=2"
          f" too  ({a2:.9f} vs {b2:.9f})")

    # ---- circle --------------------------------------------------------
    print("\n### Circle r = 1 (exact perimeter 2*pi = %.6f), error in %%\n" % P_CIR)
    print("| n | " + " | ".join(f"K={k}" for k in KS) + " |")
    print("|---:|" + "---:|" * len(KS))
    last = {}
    for n in NS:
        mk = mask_circle(n)
        _, h = grid(n)
        row = []
        for k in KS:
            v = crofton_k(mk, k, h, fams[k])
            row.append(100.0 * (v / P_CIR - 1.0))
            last[k] = row[-1]
        print(f"| {n} | " + " | ".join(f"{e:+.4f}" for e in row) + " |")
        del mk
    print("\n  observed asymptotic bias (n = 3200): "
          + ",  ".join(f"K={k}: {last[k]:+.4f} %" for k in KS))

    # ---- orientation spread --------------------------------------------
    print("\n### Orientation spread, square of side 1.6 (P = 6.4)")
    print("    fixed definition: max(err) - min(err) in PERCENTAGE POINTS,")
    print("    angles 0..45 step 5, node grid, n = 3200\n")
    tab = {}
    for n in (800, 1600, 3200):
        errs = {k: [] for k in KS}
        for ang in ANGS:
            mk = mask_square(n, ang)
            _, h = grid(n)
            for k in KS:
                errs[k].append(100.0 * (crofton_k(mk, k, h, fams[k]) / P_SQ - 1.0))
            del mk
        tab[n] = {k: np.array(v) for k, v in errs.items()}
    print("| K | spread n=800 | spread n=1600 | spread n=3200 | err min (n=3200) | "
          "err max (n=3200) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for k in KS:
        d = [tab[n][k].max() - tab[n][k].min() for n in (800, 1600, 3200)]
        e = tab[3200][k]
        print(f"| {k} | {d[0]:.4f} | {d[1]:.4f} | {d[2]:.4f} | "
              f"{e.min():+.4f} | {e.max():+.4f} |")

    print("\n  full angular profile at n = 3200 (error in %)\n")
    print("| angle | " + " | ".join(f"K={k}" for k in KS) + " |")
    print("|---:|" + "---:|" * len(KS))
    for i, ang in enumerate(ANGS):
        print(f"| {ang:.0f} | " + " | ".join(f"{tab[3200][k][i]:+.4f}" for k in KS)
              + " |")

    # ---- plateau --------------------------------------------------------
    print("\n### Does it converge or plateau?")
    print("    'closed-form limit' = (1/2) sum_k dtheta_k |sin(theta_k - phi)|,")
    print("    averaged over the two pairs of sides. It is the h -> 0 limit")
    print("    PREDICTED analytically, not a fit.\n")
    for ang in (0.0, 30.0):
        print(f"  square at {ang:.0f} degrees, error in %\n")
        print("| n | " + " | ".join(f"K={k}" for k in KS) + " |")
        print("|---:|" + "---:|" * len(KS))
        for n in NS:
            mk = mask_square(n, ang)
            _, h = grid(n)
            row = [100.0 * (crofton_k(mk, k, h, fams[k]) / P_SQ - 1.0) for k in KS]
            print(f"| {n} | " + " | ".join(f"{e:+.5f}" for e in row) + " |")
            del mk
        print("| closed-form limit | "
              + " | ".join(f"{100 * (asymptotic_square(k, ang, fams[k]) - 1):+.5f}"
                           for k in KS) + " |")
        print()

    print("  Plateau summary (square at 0 degrees, which is EXACTLY")
    print("  digitisable: a multigrid-convergent estimator would give 0):\n")
    print("| K | err n=1600 | err n=3200 | change (pp) | closed-form limit |")
    print("|---:|---:|---:|---:|---:|")
    m16, m32 = mask_square(1600, 0.0), mask_square(3200, 0.0)
    _, h16 = grid(1600)
    _, h32 = grid(3200)
    for k in KS:
        e16 = 100.0 * (crofton_k(m16, k, h16, fams[k]) / P_SQ - 1.0)
        e32 = 100.0 * (crofton_k(m32, k, h32, fams[k]) / P_SQ - 1.0)
        lim = 100.0 * (asymptotic_square(k, 0.0, fams[k]) - 1.0)
        print(f"| {k} | {e16:+.5f} | {e32:+.5f} | {e32 - e16:+.5f} | {lim:+.5f} |")
    del m16, m32

    print("\n  Orientation spread in the closed-form LIMIT (rasterising"
          " nothing):\n")
    print("| K | limit spread (pp) | measured spread n=3200 (pp) |")
    print("|---:|---:|---:|")
    for k in KS:
        lims = np.array([100 * (asymptotic_square(k, a, fams[k]) - 1) for a in ANGS])
        print(f"| {k} | {lims.max() - lims.min():.4f} | "
              f"{tab[3200][k].max() - tab[3200][k].min():.4f} |")
    # how many directions would be needed? Power law on the closed-form limit.
    ks = np.array(KS, float)
    dl = np.array([max(100 * (asymptotic_square(k, a, fams[k]) - 1) for a in ANGS)
                   - min(100 * (asymptotic_square(k, a, fams[k]) - 1) for a in ANGS)
                   for k in KS])
    sl, ic = np.polyfit(np.log(ks), np.log(dl), 1)
    print(f"\n  Power law of the limit: spread ~ K^({sl:.3f})  "
          f"(fit over K = 2..32, R^2 not reported: there are 5 points)")
    for tgt, nm in ((0.060, "DSS"), (0.023, "zero level set")):
        k_need = math.exp((math.log(tgt) - ic) / sl)
        r2 = k_need / 1.10          # measured density: ~1.1 directions per unit of |v|^2
        print(f"    to match {nm} ({tgt:.3f} pp) would take K ~ {k_need:.0f}"
              f" directions, i.e. |v| up to ~{math.sqrt(r2):.0f} px"
              f" (window ~{2 * round(math.sqrt(r2)) + 1:.0f}x"
              f"{2 * round(math.sqrt(r2)) + 1:.0f})   [EXTRAPOLATION, not measured]")

    print("\n  Paper references for the SAME spread definition:")
    print("    binary marching squares 8.13 pp | skimage.perimeter 8.10 pp |")
    print("    Crofton-4 7.71 pp | DSS 0.060 pp | zero level set 0.023 pp")
    return 0


def cmd_leaf() -> int:
    """The project's real leaf, and the failure mode of large K."""
    fams = {k: direction_family(k) for k in KS}
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from foliar import synth

    print("### PROJECTED perimeter of the blade (seed 7), fixed reference\n")
    print("| px/cm | ref (cm) | " + " | ".join(f"K={k}" for k in KS) + " |")
    print("|---:|---:|" + "---:|" * len(KS))
    for ppc in (2.0, 4.0, 8.0, 16.0, 32.0):
        st = synth.generate(synth.LeafConfig(px_per_cm=ppc))
        h = 1.0 / ppc
        ref = st.truth.projected_perimeter_cm
        row = [100.0 * (crofton_k(st.mask_blade, k, h, fams[k]) / ref - 1.0)
               for k in KS]
        print(f"| {ppc:g} | {ref:.3f} | "
              + " | ".join(f"{e:+.3f} %" for e in row) + " |", flush=True)
        del st

    # honest failure mode: a direction (p,q) with large |v| SKIPS features finer
    # than its step, and the transition count loses them.
    print("\n### Failure mode of large K: features finer than the step\n")
    print("    bar 1 px wide and 400 px long (exact perimeter 802 px)\n")
    print("| width (px) | exact perim. | " + " | ".join(f"K={k}" for k in KS) + " |")
    print("|---:|---:|" + "---:|" * len(KS))
    for wpx in (1, 2, 3, 5, 9):
        m = np.zeros((wpx + 8, 408), bool)
        m[4:4 + wpx, 4:404] = True
        exact = 2.0 * (400 + wpx)
        row = [100.0 * (crofton_k(m, k, 1.0, fams[k]) / exact - 1.0) for k in KS]
        print(f"| {wpx} | {exact:.0f} | " + " | ".join(f"{e:+.2f} %" for e in row)
              + " |")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    rc = 0
    if cmd in ("identity", "all"):
        rc += cmd_identity()
    if cmd in ("leaf", "all"):
        print("\n" + "=" * 78 + "\n")
        rc += cmd_leaf()
    if cmd in ("directions", "all"):
        print("\n" + "=" * 78 + "\n")
        rc += cmd_directions()
    sys.exit(1 if rc else 0)
