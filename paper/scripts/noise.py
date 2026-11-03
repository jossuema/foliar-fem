#!/usr/bin/env python3
"""
Robustness of the ranking-inversion condition under boundary noise.
(Answer to reviewer question 2.)

Question
--------
Eq. (2) is currently exact algebra but with no empirical evidence of robustness.
Does it survive when the mask comes with segmentation error: contour jitter,
dilation/erosion, opening/closing, salt and pepper?

The condition, in the form CORRECTED after the paper review, is

    there is an inversion  <=>  sign[(1+p)(1+bbar) - 1]  !=  sign(bbar)

and it reduces to (1+p)(1+bbar) < 1 only when bbar > 0. The previous version of
this file counted "B wins in |E|" and called that an inversion; they are
different things and are now reported separately. See `inverted`.

How it is answered
------------------
The delicate point is that noise changes the TRUE perimeter of the object. That
is why the perturbations are NOT applied to the mask but to the CONTINUOUS
object, and the mask is obtained by rasterising the perturbed object. That gives
two well-defined references, and both are reported:

  reading (i)  "user"      : against the perimeter of the ORIGINAL leaf
                             (686.9816 cm projected / 744.2255 cm intrinsic).
                             The noise is segmentation error: the physical
                             object did not change, what the network saw did.
  reading (ii) "estimator" : against the perimeter of the PERTURBED object,
                             measured exactly on the continuous polygon. It
                             isolates the estimator's digitisation bias.

The continuous -> raster bridge is validated: the reference polygon, rasterised
on the same 4 px/cm grid, reproduces `sample.mask_blade` with ZERO pixels of
difference out of 253,120, and on it gives b_A = -1.3219 %, b_B = +3.8653 %,
i.e. exactly the published figures.

Shortcut (verified, not assumed)
--------------------------------
The level B perimeter is `fem.boundary_length` on the mesh, and the CDT places
the Steiner points ON the boundary without moving it, so it coincides with the
length of the simplified polygon:

    mask_to_polygon(mask, org, px, 0.30).length  ==  r.level_B["perimeter_cm"]

verified to 4.2e-12 cm on the reference mesh and re-verified on five perturbed
masks, RAW salt and pepper included (`python3 noise.py meshcheck`). That makes it
possible to run ~70 masks without meshing, and without exposing the run to the
Steiner cap.

Usage
-----
    python3 noise.py            # everything (~1.6 min)
    python3 noise.py cont       # the continuous family only
    python3 noise.py raster     # the raster family only (salt and pepper, etc.)
    python3 noise.py critical   # jitter amplitude at which the inversion is lost
    python3 noise.py seeds      # replication over five leaves
    python3 noise.py iid        # why iid jitter does not define an object
    python3 noise.py meshcheck  # validates the B == polygon length shortcut
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings

import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, orient
from skimage import measure, morphology

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir))
if os.path.isdir(os.path.join(_ROOT, "foliar")):
    sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from foliar import synth, meshing  # noqa: E402

# --------------------------------------------------------------------------
# Published references (seed 7, 4 px/cm). Regenerable with leafperim.py
# --------------------------------------------------------------------------
P_TRUE_PRJ = 686.9815602950247      # nivel cero + Richardson, proyectado
P_TRUE_INT = 744.2254917143233      # nivel cero + Richardson, intrinseco
P_REF = -0.076917                   # p = P_TRUE_PRJ / P_TRUE_INT - 1

H_REF = 0.025                       # grid step of the reference polygon
PPC = 4.0                           # px/cm de la imagen


# --------------------------------------------------------------------------
# 1 - continuous reference object
# --------------------------------------------------------------------------
def ref_polygon(leaf, cfg, h=H_REF):
    """Zero level contour of the signed field, in PROJECTED coordinates (cm).

    It is the same estimator `synth.perimeter_levelset` uses, but returning the
    polygon instead of its length, so that it can be perturbed.
    """
    R = cfg.curl_radius_cm
    L, Hv = cfg.length_cm, cfg.halfwidth_cm * 1.03
    ug = np.linspace(-0.03 * L, 1.03 * L, int(round(1.06 * L / h)) + 1)
    vg = np.linspace(-Hv, Hv, int(round(2 * Hv / h)) + 1)
    du, dv = ug[1] - ug[0], vg[1] - vg[0]
    UG, VG = np.meshgrid(ug, vg)
    fld = leaf.blade_signed(UG, VG)

    rings = []
    for c in measure.find_contours(fld, 0.0):
        if len(c) < 4:
            continue
        u = ug[0] + c[:, 1] * du
        v = vg[0] + c[:, 0] * dv
        p = Polygon(np.column_stack([u, R * np.sin(v / R)]))
        if not p.is_valid:
            p = p.buffer(0)
            if p.geom_type == "MultiPolygon":
                p = max(p.geoms, key=lambda g: g.area)
        if p.is_empty or p.geom_type != "Polygon":
            continue
        rings.append(p)
    if not rings:
        return Polygon()

    # same even-odd rule as meshing.mask_to_polygon (invariant 5)
    rings.sort(key=lambda p: p.area, reverse=True)
    probes = [Point(p.exterior.coords[0]) for p in rings]
    depth = [sum(1 for j, q in enumerate(rings) if j != i and q.contains(probes[i]))
             for i in range(len(rings))]
    pieces = []
    for i, p in enumerate(rings):
        if depth[i] % 2:
            continue
        holes = [rings[j].exterior.coords for j in range(len(rings))
                 if depth[j] == depth[i] + 1 and p.contains(probes[j])]
        q = Polygon(p.exterior.coords, holes)
        if not q.is_valid:
            q = q.buffer(0)
        if not q.is_empty:
            pieces.append(q)
    return unary_union(pieces) if pieces else Polygon()


def rings_of(g):
    out = []
    if g.is_empty:
        return out
    for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
        out.append(np.asarray(p.exterior.coords))
        for hl in p.interiors:
            out.append(np.asarray(hl.coords))
    return out


def intrinsic_length(g, R):
    """Length of the same contour UNPROJECTED onto the cylinder: v = R asin(y/R)."""
    tot = 0.0
    for r in rings_of(g):
        v = R * np.arcsin(np.clip(r[:, 1] / R, -1.0, 1.0))
        tot += float(np.sum(np.hypot(np.diff(r[:, 0]), np.diff(v))))
    return tot


def rasterize(g, origin_cm, ppc, shape):
    """Gauss digitisation: pixel on if its centre falls inside."""
    H, W = shape
    X = origin_cm[0] + np.arange(W) / ppc
    Y = origin_cm[1] + np.arange(H) / ppc
    XX, YY = np.meshgrid(X, Y)
    return shapely.contains_xy(g, XX.ravel(), YY.ravel()).reshape(H, W)


# --------------------------------------------------------------------------
# 2 - perturbaciones en el dominio CONTINUO
# --------------------------------------------------------------------------
def morpho(g, r_cm, kind):
    """Minkowski with a disc of radius r_cm on the continuous object."""
    qs = 32
    if kind == "dilate":
        return g.buffer(r_cm, quad_segs=qs)
    if kind == "erode":
        return g.buffer(-r_cm, quad_segs=qs)
    if kind == "open":
        return g.buffer(-r_cm, quad_segs=qs).buffer(r_cm, quad_segs=qs)
    if kind == "close":
        return g.buffer(r_cm, quad_segs=qs).buffer(-r_cm, quad_segs=qs)
    raise ValueError(kind)


def _band_limited(s, L, sigma, ell, rng, oversample=8):
    """Gaussian field periodic in arc length, with Gaussian autocorrelation.

    Power P_k ~ exp(-(2 pi k ell / L)^2 / 2), i.e. autocorrelation
    exp(-(ds/ell)^2/2). `ell` is the CORRELATION LENGTH: it controls how fast
    the noise oscillates along the boundary. It is the parameter per-vertex iid
    jitter does NOT have, and that is why iid jitter does not define a
    continuous object (see `iid_demo`).

    It is synthesised by FFT on a uniform arc-length grid and interpolated to
    the vertices: this avoids an (N x K) product and its floating-point noise.
    The amplitude is renormalised to an EXACT sample deviation = sigma, so the
    parameter is reproducible and not a realisation of a random variable.
    """
    M = 1
    while M < oversample * L / max(ell, 1e-9):
        M *= 2
    M = int(min(max(M, 1024), 1 << 20))
    k = np.fft.rfftfreq(M, d=1.0 / M)                 # 0..M/2
    P = np.exp(-0.5 * (2 * math.pi * k * ell / L) ** 2)
    P[0] = 0.0                                        # media nula
    c = (rng.normal(size=k.size) + 1j * rng.normal(size=k.size)) * np.sqrt(P)
    d_uni = np.fft.irfft(c, n=M)
    sd = float(d_uni.std())
    if sd <= 0:
        return np.zeros_like(s)
    d_uni *= sigma / sd
    su = np.arange(M) * (L / M)
    return np.interp(np.mod(s, L), su, d_uni, period=L)


def jitter(g, sigma_cm, ell_cm, seed):
    """Smooth normal displacement of the continuous boundary."""
    rng = np.random.default_rng(seed)
    out = []
    for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
        p = orient(p, 1.0)
        new_rings = []
        for r in [np.asarray(p.exterior.coords)] + \
                 [np.asarray(hl.coords) for hl in p.interiors]:
            xy = r[:-1]                                     # sin el punto repetido
            n = len(xy)
            seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
            s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
            L = float(seg.sum())
            d = _band_limited(s, L, sigma_cm, ell_cm, rng)
            # tangent by central difference on the closed ring
            t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
            tn = np.hypot(t[:, 0], t[:, 1])
            tn[tn == 0] = 1.0
            nx, ny = t[:, 1] / tn, -t[:, 0] / tn             # normal (dy, -dx)
            q = np.column_stack([xy[:, 0] + d * nx, xy[:, 1] + d * ny])
            new_rings.append(np.vstack([q, q[:1]]))
        q = Polygon(new_rings[0], new_rings[1:])
        if not q.is_valid:
            q = q.buffer(0)                                  # deshace los lazos
        out.append(q)
    res = unary_union(out)
    if res.geom_type not in ("Polygon", "MultiPolygon"):
        res = unary_union([x for x in getattr(res, "geoms", [])
                           if x.geom_type in ("Polygon", "MultiPolygon")])
    return res


# --------------------------------------------------------------------------
# 3 - perturbations in the RASTER domain (no continuous truth)
# --------------------------------------------------------------------------
def bin_morpho(mask, rad_px, kind):
    fp = morphology.disk(rad_px)
    if kind == "dilate":
        return morphology.binary_dilation(mask, fp)
    if kind == "erode":
        return morphology.binary_erosion(mask, fp)
    if kind == "open":
        return morphology.binary_opening(mask, fp)
    if kind == "close":
        return morphology.binary_closing(mask, fp)
    raise ValueError(kind)


def salt_pepper(mask, q, seed):
    rng = np.random.default_rng(seed)
    return mask ^ (rng.random(mask.shape) < q)


def clean(mask, min_px=16):
    m = morphology.binary_opening(mask, morphology.disk(1))
    m = morphology.remove_small_objects(m, min_size=min_px)
    m = morphology.remove_small_holes(m, area_threshold=min_px)
    return m


# --------------------------------------------------------------------------
# 4 - estimadores
# --------------------------------------------------------------------------
def est_A(mask, ppc):
    """Level A: 4-direction Crofton (baseline.py)."""
    return float(measure.perimeter_crofton(mask, directions=4)) / ppc


def est_B(mask, origin_cm, ppc):
    """Level B: flat mesh contour == length of the simplified polygon."""
    g = meshing.mask_to_polygon(mask, origin_cm, ppc, 0.30)
    return float(g.length)


def est_MS(mask, ppc):
    tot = 0.0
    for c in measure.find_contours(mask.astype(float), 0.5):
        tot += float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
    return tot / ppc


def est_DSS(mask, ppc):
    import dss as D
    return D.dss_length(mask, 1.0 / ppc)


# --------------------------------------------------------------------------
# 5 - the inversion condition
# --------------------------------------------------------------------------
def eq2(bA, bB, p):
    """Ranking-inversion condition.

    Eq. (2) EXACTLY AS PUBLISHED, which assumes b_A < b_B:

        |E_A| > |E_B|   <=>   (1+p)(1+bbar) < 1,      bbar = (b_A+b_B)/2

    The premise b_A < b_B is not cosmetic. From (E_A-E_B)(E_A+E_B) > 0 with
    E_A - E_B = (1+p)(b_A - b_B) and 1+p > 0 comes the GENERAL form

        |E_A| > |E_B|   <=>   sign(b_A - b_B) * [ (1+p)(1+bbar) - 1 ] > 0

    which coincides with Eq. (2) only while b_A < b_B. RAW salt and pepper
    inverts that premise (Crofton counts every isolated pixel; level B discards
    them via `min_area_px`), and there the published Eq. (2) fails while the
    general form still holds.

    Returns (bbar, p*, predicts_Eq2, predicts_general, premise_ok).
    """
    bbar = 0.5 * (bA + bB)
    pstar = -bbar / (1.0 + bbar)
    lhs = (1.0 + p) * (1.0 + bbar) - 1.0
    pred_eq2 = lhs < 0.0
    sgn = -1.0 if bA < bB else (1.0 if bA > bB else 0.0)
    pred_gen = sgn * lhs > 0.0
    return bbar, pstar, pred_eq2, pred_gen, (bA < bB)


def observed(pA, pB, L_int):
    """|E_A| > |E_B| with E_X measured against the matching INTRINSIC reference.

    MIND the name of the third output: it is "B wins", NOT "there is an
    inversion". They are the same thing only if A was winning without the shared
    distortion, i.e. if |b_A| < |b_B|. For the inversion proper, see `inverted`.
    """
    EA, EB = pA / L_int - 1.0, pB / L_int - 1.0
    return EA, EB, abs(EA) > abs(EB)


def inverted(bA, bB, b_wins_in_E):
    """Did the shared distortion INVERT the order, or only confirm it?

    A paper reviewer pointed this out and was right: the final order is not the
    inversion. The inversion is that the order in |E| DIFFERS from the order in
    |b|, which is what would be measured without the shared distortion:

        there is an inversion  <=>  (|E_A| > |E_B|)  !=  (|b_A| > |b_B|)

    Equivalently, and without touching the measurements,

        there is an inversion  <=>  sign[(1+p)(1+bbar) - 1]  !=  sign(bbar)

    The two perturbation families that do NOT produce an inversion fall for the
    same single reason: they leave |b_A| > |b_B|, i.e. A was not the better one
    even before projecting. That happens with dilation and closing (which push
    b_A more negative than b_B is positive, bbar < 0) and with raw salt and
    pepper (which sends b_A to +247 %, bbar >> 0). There is no order to invert.
    """
    return b_wins_in_E != (abs(bA) > abs(bB))


# --------------------------------------------------------------------------
# 6 - infraestructura
# --------------------------------------------------------------------------
class Bench:
    def __init__(self):
        self.cfg = synth.LeafConfig()                 # seed 7, 4 px/cm
        self.leaf = synth.SyntheticLeaf(self.cfg)
        self.R = self.cfg.curl_radius_cm
        t0 = time.time()
        self.sample = synth.generate(self.cfg)
        self.shape = self.sample.mask_blade.shape
        self.org = self.sample.origin_cm
        self.ppc = self.sample.px_per_cm
        self.g0 = ref_polygon(self.leaf, self.cfg)
        self.L0 = float(self.g0.length)
        self.L0i = intrinsic_length(self.g0, self.R)
        self.t_setup = time.time() - t0

    def header(self):
        m0 = rasterize(self.g0, self.org, self.ppc, self.shape)
        nd = int((m0 ^ self.sample.mask_blade).sum())
        print("=" * 100)
        print("CONTINUOUS -> RASTER BRIDGE  (assembly validation, seed 7, %g px/cm)"
              % self.ppc)
        print("=" * 100)
        print("  reference polygon: h = %.4f cm, %d vertices, %d rings"
              % (H_REF, sum(len(r) for r in rings_of(self.g0)), len(rings_of(self.g0))))
        print("  L0 projected   = %10.4f cm   (%+.4f %% against 686.9816 -> polygonisation offset)"
              % (self.L0, 100 * (self.L0 / P_TRUE_PRJ - 1)))
        print("  L0 intrinseco  = %10.4f cm   (%+.4f %% contra 744.2255)"
              % (self.L0i, 100 * (self.L0i / P_TRUE_INT - 1)))
        print("  p of the polygon = %+.4f %%   (against %+.4f %% analytic)"
              % (100 * (self.L0 / self.L0i - 1), 100 * P_REF))
        print("  rasterizado a %g px/cm contra sample.mask_blade: %d pixeles de %d distintos"
              % (self.ppc, nd, m0.size))
        pA, pB = est_A(m0, self.ppc), est_B(m0, self.org, self.ppc)
        print("  b_A = %+.4f %%   b_B = %+.4f %%   (publicados: -1.3219 / +3.8653)"
              % (100 * (pA / P_TRUE_PRJ - 1), 100 * (pB / P_TRUE_PRJ - 1)))
        print("  setup %.1f s" % self.t_setup)
        print()

    def measure(self, mask, with_dss=False):
        d = dict(A=est_A(mask, self.ppc), B=est_B(mask, self.org, self.ppc),
                 MS=est_MS(mask, self.ppc), npx=int(mask.sum()))
        if with_dss:
            d["DSS"] = est_DSS(mask, self.ppc)
        return d


ROWFMT = ("%-28s %8.2f | %8.3f %7.3f %7.3f %7.3f %-3s %-3s %-4s | "
          "%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %-3s %-3s %-4s")
HDR = ("%-28s %8s | %8s %7s %7s %7s %-3s %-3s %-4s | "
       "%7s %7s %7s %7s %7s %7s %-3s %-3s %-4s"
       % ("perturbacion", "L_prj",
          "b_A", "b_B", "p*", "marg", "prd", "obs", "ok",
          "b_A", "b_B", "b_MS", "b_DSS", "p*", "marg", "prd", "obs", "ok"))


def _ok(pred_eq2, pred_gen, obs, premise):
    """'ok' = the published Eq. (2) is right. 'PREM' = failed via the b_A<b_B premise.
    '**' = real failure (should never happen: the general form is an identity)."""
    if pred_eq2 == obs:
        return "ok"
    return "PREM" if (not premise and pred_gen == obs) else "**"


def emit(label, Lp, Li, pA, pB, pMS, pDSS, refP_fix, refI_fix, p_fix,
         refP_var, refI_var, p_var):
    bA1, bB1 = pA / refP_fix - 1, pB / refP_fix - 1
    _, ps1, pe1, pg1, prem1 = eq2(bA1, bB1, p_fix)
    _, _, obs1 = observed(pA, pB, refI_fix)
    bA2, bB2 = pA / refP_var - 1, pB / refP_var - 1
    bMS2 = pMS / refP_var - 1
    bD2 = pDSS / refP_var - 1
    _, ps2, pe2, pg2, prem2 = eq2(bA2, bB2, p_var)
    _, _, obs2 = observed(pA, pB, refI_var)
    print(ROWFMT % (label, Lp,
                    100 * bA1, 100 * bB1, 100 * ps1, 100 * (ps1 - p_fix),
                    "YES" if pe1 else "no", "YES" if obs1 else "no",
                    _ok(pe1, pg1, obs1, prem1),
                    100 * bA2, 100 * bB2, 100 * bMS2, 100 * bD2, 100 * ps2,
                    100 * (ps2 - p_var),
                    "YES" if pe2 else "no", "YES" if obs2 else "no",
                    _ok(pe2, pg2, obs2, prem2)),
          flush=True)
    return dict(label=label, Lp=Lp, Li=Li,
                bA1=bA1, bB1=bB1, ps1=ps1, p1=p_fix,
                pred1=pe1, gen1=pg1, obs1=obs1, prem1=prem1,
                inv1=inverted(bA1, bB1, obs1),
                bA2=bA2, bB2=bB2, bMS2=bMS2, bD2=bD2, ps2=ps2, p2=p_var,
                pred2=pe2, gen2=pg2, obs2=obs2, prem2=prem2,
                inv2=inverted(bA2, bB2, obs2))


# --------------------------------------------------------------------------
# 7 - familia continua
# --------------------------------------------------------------------------
def cont(bench, quick=False):
    print("=" * 100)
    print("CONTINUOUS FAMILY - the perturbation is applied to the OBJECT, the mask is rasterised")
    print("=" * 100)
    print("  reading (i)  left  : FIXED reference    = original leaf (686.9816 / 744.2255,"
          " p = -7.6917 %)")
    print("  reading (ii) right : VARIABLE reference = perturbed object"
          " (continuous polygon, p re-measured)")
    print("  prd = the published Eq. (2) predicts B wins;  obs = B wins in |E|.")
    print("  MIND: 'B wins' is NOT 'the order inverted'. The inversion also requires that")
    print("  without the shared distortion A was winning, i.e. |b_A| < |b_B|; counted in the summary.")
    print("  ok  = they agree;  marg = p* - p in pp (positive = there is an inversion, with margin).")
    print("  WARNING: 'ok' is NOT evidence. Given b_A, b_B and p measured coherently,")
    print("  E_X = (1+b_X)(1+p)-1 is an IDENTITY, so prd == obs always. The column")
    print("  is there to catch assembly errors. The empirical content is in how much")
    print("  b_A and b_B move, and in whether the margin p* - p survives.")
    print()
    print(HDR)
    print("-" * len(HDR))

    px = 1.0 / bench.ppc
    cases = [("sin perturbar", bench.g0)]
    for r in (1, 2):
        for k in ("dilate", "erode", "open", "close"):
            cases.append(("%s %d px (%.2f cm)" % (k, r, r * px),
                          morpho(bench.g0, r * px, k)))
    sigmas = (0.25, 0.5, 1.0, 2.0)
    ells = (0.5, 2.0) if quick else (0.5, 1.0, 2.0)
    seeds = (101,) if quick else (101, 202, 303)
    for ell in ells:
        for sg in sigmas:
            for sd in seeds:
                cases.append(("jitter s=%.2fpx l=%.1fcm #%d" % (sg, ell, sd),
                              jitter(bench.g0, sg * px, ell, sd)))

    rows = []
    for label, g in cases:
        if g.is_empty:
            print("%-26s  VACIO" % label)
            continue
        Lp = float(g.length)
        Li = intrinsic_length(g, bench.R)
        m = rasterize(g, bench.org, bench.ppc, bench.shape)
        d = bench.measure(m, with_dss=True)
        rows.append(emit(label, Lp, Li, d["A"], d["B"], d["MS"], d["DSS"],
                         P_TRUE_PRJ, P_TRUE_INT, P_REF,
                         Lp, Li, Lp / Li - 1.0))
    summarize(rows, "familia continua")
    return rows


# --------------------------------------------------------------------------
# 8 - familia raster
# --------------------------------------------------------------------------
def raster(bench, quick=False):
    print()
    print("=" * 100)
    print("RASTER FAMILY - the perturbation is applied to the MASK; there is NO continuous object")
    print("=" * 100)
    print("  Only reading (i) exists. The right-hand column uses the SAME fixed")
    print("  reference and is there only to keep the table readable; it is NOT reading (ii).")
    print()
    print(HDR)
    print("-" * len(HDR))
    m0 = bench.sample.mask_blade
    cases = [("original mask", m0)]
    for r in (1, 2):
        for k in ("dilate", "erode", "open", "close"):
            cases.append(("bin %s disk(%d)" % (k, r), bin_morpho(m0, r, k)))
    seeds = (101,) if quick else (101, 202, 303)
    for q in (0.001, 0.005, 0.01):
        for sd in seeds:
            sp = salt_pepper(m0, q, sd)
            cases.append(("s&p q=%.3f raw #%d" % (q, sd), sp))
            cases.append(("s&p q=%.3f limpio #%d" % (q, sd), clean(sp)))
    rows = []
    for label, m in cases:
        d = bench.measure(m, with_dss=True)
        rows.append(emit(label, float("nan"), float("nan"), d["A"], d["B"],
                         d["MS"], d["DSS"],
                         P_TRUE_PRJ, P_TRUE_INT, P_REF,
                         P_TRUE_PRJ, P_TRUE_INT, P_REF))
    summarize(rows, "familia raster (solo lectura (i))", only_fixed=True)
    return rows


KULPA = 4.0 / math.pi * (2.0 * math.sqrt(2.0) - 2.0)     # 1.05478618


def summarize(rows, name, only_fixed=False):
    print()
    print("  --- resumen: %s (n = %d) ---" % (name, len(rows)))
    for tag, ka, kb, kp, kq in (("i", "bA1", "bB1", "ps1", "p1"),
                                ("ii", "bA2", "bB2", "ps2", "p2")):
        if only_fixed and tag == "ii":
            continue
        j = "1" if tag == "i" else "2"
        a = np.array([r[ka] for r in rows]) * 100
        b = np.array([r[kb] for r in rows]) * 100
        s = np.array([r[kp] for r in rows]) * 100
        mar = np.array([r[kp] - r[kq] for r in rows]) * 100
        inv = sum(1 for r in rows if r["obs" + j])
        bad = sum(1 for r in rows if r["pred" + j] != r["obs" + j])
        badg = sum(1 for r in rows if r["gen" + j] != r["obs" + j])
        nprem = sum(1 for r in rows if not r["prem" + j])
        print("   lectura (%-2s)  b_A [%+7.3f, %+7.3f] %%   b_B [%+7.3f, %+7.3f] %%   "
              "b_B - b_A [%.3f, %.3f] pp" % (tag, a.min(), a.max(), b.min(), b.max(),
                                             (b - a).min(), (b - a).max()))
        print("                p* [%+7.3f, %+7.3f] %%   margen p*-p  min %+.3f  "
              "mediana %+.3f  max %+.3f pp"
              % (s.min(), s.max(), mar.min(), float(np.median(mar)), mar.max()))
        # equivalent and more readable form: with p fixed, the inversion holds iff
        # bbar < bbar* = -p/(1+p).  A single number as the threshold.
        bb = 0.5 * (a + b)
        pv = np.array([r[kq] for r in rows]) * 100
        bbstar = -pv / (1.0 + pv / 100.0)
        print("                bbar [%+7.3f, %+7.3f] %%   umbral bbar* = -p/(1+p) = "
              "[%+.3f, %+.3f] %%   holgura min %.3f pp"
              % (bb.min(), bb.max(), bbstar.min(), bbstar.max(),
                 float((bbstar - bb).min())))
        ginv = sum(1 for r in rows if r["inv" + j])
        print("                B gana en |E| en %d de %d;  "
              "published Eq. (2) fails in %d;  GENERAL form fails in %d (must be 0)"
              % (inv, len(rows), bad, badg))
        print("                GENUINE INVERSION (the order in |E| differs from the order in |b|)"
              " en %d de %d" % (ginv, len(rows)))
        if ginv < len(rows):
            faltan = [r["label"] for r in rows if not r["inv" + j]]
            print("                no inversion because |b_A| > |b_B| already before projecting:")
            for lab in faltan:
                print("                   %s" % lab)
        print("                premise b_A < b_B violated in %d of %d;  "
              "sign b_A < 0 < b_B preserved in %d of %d"
              % (nprem, len(rows),
                 sum(1 for x, y in zip(a, b) if x < 0 < y), len(rows)))
    if not only_fixed:
        ms = np.array([r["bMS2"] for r in rows]) * 100
        print("   isotropic asymptote (boundary of uniform orientation):"
              " MS -> %+.4f %%,  Crofton-4 -> %+.4f %%"
              % (100 * (KULPA - 1), 0.0))
        ds = np.array([r["bD2"] for r in rows]) * 100
        print("                b_MS  measured in [%+.3f, %+.3f] %% under reading (ii)"
              % (ms.min(), ms.max()))
        print("   DSS estimator (multigrid-convergent, the paper's proposal):")
        print("                b_DSS measured in [%+.3f, %+.3f] %%   |b_DSS| max %.3f %%"
              % (ds.min(), ds.max(), np.abs(ds).max()))
    print()


# --------------------------------------------------------------------------
# 8b - critical noise amplitude under reading (i)
# --------------------------------------------------------------------------
def critical(bench, seeds=(101, 202, 303, 404, 505)):
    """At what jitter amplitude does the inversion stop showing under reading (i)?

    With p FIXED the condition reduces to a single threshold:
    bbar < bbar* = -p/(1+p) = +8.333 %. sigma is swept finely and the crossing
    of bbar is reported.
    """
    print()
    print("=" * 100)
    print("AMPLITUD CRITICA DE JITTER  (lectura (i), p fijo = -7.6917 %,"
          " umbral bbar* = +8.3330 %)")
    print("=" * 100)
    print("  bbar = (b_A + b_B)/2 against the FIXED reference. While bbar < bbar*")
    print("  the inversion is observed; once it crosses, segmentation error dominates")
    print("  the digitisation bias and the A/B order stops being informative.")
    print("  %d seeds per point; mean +- sd reported." % len(seeds))
    print()
    px = 1.0 / bench.ppc
    bstar = 100 * (-P_REF / (1.0 + P_REF))
    sig = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5)
    for ell in (0.5, 1.0, 2.0):
        print("  longitud de correlacion l = %.1f cm  (%.0f px)" % (ell, ell * bench.ppc))
        print("   %8s %16s %16s %16s %8s"
              % ("sigma px", "b_A %", "b_B %", "bbar %", "inv?"))
        prev = None
        for sg in sig:
            bb = []
            for sd in seeds:
                g = jitter(bench.g0, sg * px, ell, sd)
                m = rasterize(g, bench.org, bench.ppc, bench.shape)
                d = bench.measure(m)
                bb.append((100 * (d["A"] / P_TRUE_PRJ - 1),
                           100 * (d["B"] / P_TRUE_PRJ - 1)))
            bb = np.array(bb)
            a, b = bb[:, 0], bb[:, 1]
            m_ = 0.5 * (a + b)
            inv = m_.mean() < bstar
            print("   %8.3f %8.3f+-%-6.3f %8.3f+-%-6.3f %8.3f+-%-6.3f %8s"
                  % (sg, a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1),
                     m_.mean(), m_.std(ddof=1), "YES" if inv else "no"), flush=True)
            if prev is not None and (prev[1] < bstar) != inv:
                s0, m0 = prev
                sc = s0 + (bstar - m0) * (sg - s0) / (m_.mean() - m0)
                print("   %8s  <-- cruce por bbar* en sigma ~ %.3f px (%.4f cm)"
                      % ("", sc, sc * px))
            prev = (sg, m_.mean())
        print()

    # does the threshold collapse onto the rms SLOPE of the boundary, sigma/ell?
    # For a Gaussian field with correlation length ell and deviation sigma, the
    # rms slope is sigma/ell. If the mechanism is "the noise randomises the
    # local orientation", the threshold should depend on sigma/ell and not on
    # sigma and ell separately.
    print("  COLLAPSE ONTO THE RMS SLOPE sigma/ell")
    print("   %6s %8s %8s %16s %6s" % ("l(cm)", "sig(px)", "sig/l", "bbar %", "inv?"))
    for ell, sg in ((0.5, 0.75), (1.0, 1.5), (2.0, 3.0), (4.0, 6.0),
                    (0.5, 0.875), (1.0, 1.75), (2.0, 3.5)):
        v = []
        for sd in seeds:
            m = rasterize(jitter(bench.g0, sg * px, ell, sd),
                          bench.org, bench.ppc, bench.shape)
            d = bench.measure(m)
            v.append(0.5 * (100 * (d["A"] / P_TRUE_PRJ - 1)
                            + 100 * (d["B"] / P_TRUE_PRJ - 1)))
        v = np.array(v)
        print("   %6.1f %8.2f %8.4f %8.3f+-%-6.3f %6s"
              % (ell, sg, sg * px / ell, v.mean(), v.std(ddof=1),
                 "YES" if v.mean() < bstar else "no"), flush=True)
    print("   The collapse is APPROXIMATE: at sigma/l = 0.375 bbar drops from 8.31 to 5.16 %")
    print("   as l goes from 0.5 to 4.0 cm, and the spread between seeds blows up")
    print("   (+-5.1 pp at l = 4 cm: only ~170 correlation lengths fit along the")
    print("   boundary, i.e. very few independent samples).")
    print()


# --------------------------------------------------------------------------
# 8c - is the robustness an accident of seed 7?
# --------------------------------------------------------------------------
def seeds_check(seed_list=(7, 8, 9, 10, 11)):
    """Repeat three perturbations over five different leaves, reading (ii)."""
    print()
    print("=" * 100)
    print("REPLICATION OVER FIVE LEAVES  (reading (ii), reference = perturbed object)")
    print("=" * 100)
    print("  Each leaf carries its own reference polygon and its own p.")
    print()
    print("%6s %-22s %9s %9s %9s %8s %8s %7s %6s"
          % ("hoja", "perturbacion", "b_A %", "b_B %", "bbar %", "p %", "bbar* %",
             "B gana", "inv?"))
    n_inv = n_tot = 0
    for sd in seed_list:
        cfg = synth.LeafConfig(seed=sd)
        leaf = synth.SyntheticLeaf(cfg)
        smp = synth.generate(cfg)
        g0 = ref_polygon(leaf, cfg)
        px = 1.0 / smp.px_per_cm
        for lab, g in (("sin perturbar", g0),
                       ("dilate 1 px", morpho(g0, px, "dilate")),
                       ("jitter s=.5px l=1cm", jitter(g0, 0.5 * px, 1.0, 101))):
            Lp = float(g.length)
            Li = intrinsic_length(g, cfg.curl_radius_cm)
            m = rasterize(g, smp.origin_cm, smp.px_per_cm, smp.mask_blade.shape)
            pA = est_A(m, smp.px_per_cm)
            pB = est_B(m, smp.origin_cm, smp.px_per_cm)
            bA, bB = pA / Lp - 1, pB / Lp - 1
            p = Lp / Li - 1
            bbar = 0.5 * (bA + bB)
            bstar = -p / (1 + p)
            _, _, obsv = observed(pA, pB, Li)
            ginv = inverted(bA, bB, obsv)
            n_tot += 1
            n_inv += ginv
            print("%6d %-22s %+9.3f %+9.3f %+9.3f %+8.3f %+8.3f %7s %6s"
                  % (sd, lab, 100 * bA, 100 * bB, 100 * bbar, 100 * p,
                     100 * bstar, "YES" if obsv else "no",
                     "YES" if ginv else "no"), flush=True)
        del smp
    print()
    print("  GENUINE INVERSION in %d of %d  (the remaining ones have |b_A| > |b_B|,"
          " i.e. A was not the better one even before projecting)" % (n_inv, n_tot))
    print()


# --------------------------------------------------------------------------
# 9 - why iid jitter does not define an object
# --------------------------------------------------------------------------
def iid_demo(bench):
    print()
    print("=" * 100)
    print("WHY PER-VERTEX IID JITTER DOES NOT DEFINE A CONTINUOUS OBJECT")
    print("=" * 100)
    print("  The SAME sigma is applied to the SAME object discretised at two vertex")
    print("  densities. If the noise were a perturbation of the object, the perturbed")
    print("  length would converge; with iid noise it diverges as 1/sqrt(spacing).")
    print()
    print("%-10s %10s %12s %12s %12s %12s"
          % ("h (cm)", "vertices", "L0 (cm)", "iid s=0.5px", "razon", "correl l=1cm"))
    px = 1.0 / bench.ppc
    for h in (0.100, 0.050, 0.025):
        g = ref_polygon(bench.leaf, bench.cfg, h)
        nv = sum(len(r) for r in rings_of(g))
        rng = np.random.default_rng(7)
        r = np.asarray((g.geoms[0] if g.geom_type == "MultiPolygon" else g)
                       .exterior.coords)[:-1]
        t = np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0)
        tn = np.hypot(t[:, 0], t[:, 1]); tn[tn == 0] = 1
        d = rng.normal(scale=0.5 * px, size=len(r))
        q = np.column_stack([r[:, 0] + d * t[:, 1] / tn, r[:, 1] - d * t[:, 0] / tn])
        Liid = float(np.sum(np.hypot(*np.diff(np.vstack([q, q[:1]]), axis=0).T)))
        Lc = float(jitter(g, 0.5 * px, 1.0, 7).length)
        print("%-10.4f %10d %12.4f %12.4f %12.4f %12.4f"
              % (h, nv, g.length, Liid, Liid / g.length, Lc), flush=True)
    print()
    print("  The iid column grows without bound under refinement: it is not an object,")
    print("  it is a property of the discretisation. That is why the jitter in this")
    print("  study is parametrised by (sigma, correlation length) and not by sigma alone.")
    print()


# --------------------------------------------------------------------------
# 10 - validation of the B == polygon length shortcut
# --------------------------------------------------------------------------
def meshcheck(bench):
    from foliar import fem
    print()
    print("=" * 100)
    print("SHORTCUT: level B perimeter == length of the simplified polygon")
    print("=" * 100)
    px = 1.0 / bench.ppc
    m0 = bench.sample.mask_blade
    tests = [("sin perturbar", bench.g0),
             ("dilate 1 px", morpho(bench.g0, px, "dilate")),
             ("jitter s=2px l=0.5cm", jitter(bench.g0, 2.0 * px, 0.5, 101)),
             ("jitter s=0.5px l=1cm", jitter(bench.g0, 0.5 * px, 1.0, 101)),
             # the two worst for the mesher: salt and pepper. They do not hang
             # because mask_to_polygon already discards specks under 2 px2.
             ("s&p q=0.005 limpio", clean(salt_pepper(m0, 0.005, 101))),
             ("s&p q=0.005 raw", salt_pepper(m0, 0.005, 101))]
    for label, g in tests:
        m = g if isinstance(g, np.ndarray) else \
            rasterize(g, bench.org, bench.ppc, bench.shape)
        blade = meshing.mask_to_polygon(m, bench.org, bench.ppc, 0.30)
        les = meshing.mask_to_polygon(bench.sample.mask_lesion & m, bench.org,
                                      bench.ppc, 0.18, min_area_px=2.0)
        mid = meshing.mask_to_polygon(bench.sample.mask_midrib & m, bench.org,
                                      bench.ppc, 0.30)
        t0 = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mesh = meshing.build_mesh(blade, les, mid, bench.ppc,
                                      target_edge_cm=2.0, lesion_edge_cm=0.45)
            wl = [str(x.message) for x in w]
        bl = fem.boundary_length(mesh.V, mesh.F)
        print("  %-24s poly %12.6f   mesh %12.6f   dif %9.2e cm   "
              "(%d tri, ang %.2f, cap %s, %.1fs)%s"
              % (label, blade.length, bl, abs(bl - blade.length), len(mesh.F),
                 mesh.min_angle_deg, mesh.steiner_capped, time.time() - t0,
                 ("  WARN: " + wl[0][:40]) if wl else ""), flush=True)
    print()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    quick = "quick" in sys.argv
    b = Bench()
    b.header()
    if cmd in ("all", "cont"):
        cont(b, quick)
    if cmd in ("all", "raster"):
        raster(b, quick)
    if cmd in ("all", "critical"):
        critical(b, (101, 202, 303) if quick else (101, 202, 303, 404, 505))
    if cmd in ("all", "seeds"):
        seeds_check()
    if cmd in ("all", "iid"):
        iid_demo(b)
    if cmd in ("all", "meshcheck"):
        meshcheck(b)
