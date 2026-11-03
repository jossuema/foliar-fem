#!/usr/bin/env python3
"""
Length estimator by DIGITAL STRAIGHT SEGMENTS (DSS) on a binary mask.

Motivation
----------
The zero-level-set perimeter estimator requires a signed field. A mask coming
out of a segmentation network does NOT carry one: it carries ones and zeros. The
question is whether, on that impoverished input, a multigrid-convergent
estimator exists. The classical answer from digital geometry is yes: the
decomposition of the digital contour into digital straight segments
(Kovalevsky & Fuchs 1992; survey in Coeurjolly, Lachaud & Roussillon 2012). This
file implements it and MEASURES it.

Full chain
----------
1. `crack_contours`  : mask -> interpixel contour ("crack boundary"), one or
                       several closed 4-connected curves on the grid of pixel
                       CORNERS. It is exact and unambiguous: there is no Moore
                       tracing and no neighbourhood convention to choose beyond
                       the tie-break at diagonal pinch points.
2. `_extend`         : incremental DSS recognition (Debled-Rennesson /
                       Reveilles) along that curve.
3. `dss_length`      : greedy forward and backward segmentation, sum of the
                       Euclidean chords, average of the two.

The bridge between 1 and 2: a 4-connected curve is not a NAIVE (8-connected)
line but a STANDARD line, whose arithmetic condition is

    mu <= a*x - b*y < mu + |a| + |b|

instead of `mu + max(|a|,|b|)`. Rather than writing a second recogniser, the
exact equivalence by shear is used. With (X, Y) = (x + y, y), which is
unimodular and therefore bijective on Z^2:

    a*x - b*y  =  a*(X - Y) - b*Y  =  a*X - (a + b)*Y

so the standard condition (a, b, mu) on (x, y) is EXACTLY the naive condition
(A, B, mu) = (a, a + b, mu) on (X, Y), because max(A, B) = a + b = B for
a, b >= 0. The shear maps the steps {E, N} to {(1,0), (1,1)}, i.e. to octant 1.
The other three quadrants reduce to the first by a rotation of the grid. The
identity verifies itself in `selftest`; it is not taken on trust.

Usage
-----
    python3 dss.py selftest     # correctness of the recogniser against brute force
    python3 dss.py bench        # convergence and spread tables
    python3 dss.py deep         # convergence up to n = 12800
    python3 dss.py leaf         # projected perimeter of the project's leaf
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

# scripts/ lives two levels below the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir))

# --------------------------------------------------------------------------
# 1 · Contorno interpixel (crack boundary)
# --------------------------------------------------------------------------
# Coordinate convention: pixel (row i, column j) occupies the square
# [j, j+1] x [i, i+1]. A vertex is an integer CORNER pair (x, y) = (column,
# row). Edges are oriented so the foreground always stays on the same side;
# the actual turning sense is irrelevant for the length.

_ROT_BACK = {(1, 0): (0, -1), (0, 1): (1, 0), (-1, 0): (0, 1), (0, -1): (-1, 0)}
#  ^ ROT^-1(d) = (dy, -dx). At a diagonal pinch (two foreground pixels
#    touching only at a corner) the vertex has out-degree 2. Preferring
#    ROT^-1 amounts to CROSSING the pinch, i.e. treating the foreground as
#    8-connected and the background as 4-connected, which is the usual
#    convention for the Gauss digitisation of a connected shape.
#    The total crack length is the same either way.


def crack_contours(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Interpixel contour of a binary mask.

    Returns a list of closed curves; each is a list of integer vertices with the
    first one REPEATED at the end. The steps are unit and 4-connected by
    construction.
    """
    mask = np.asarray(mask, dtype=bool)
    m = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    m[1:-1, 1:-1] = mask
    core = m[1:-1, 1:-1]

    # (mask of exposed sides, directed edge in corner offsets)
    sides = (
        (core & ~m[0:-2, 1:-1], (0, 0, 1, 0)),   # arriba : (j,i)     -> (j+1,i)
        (core & ~m[1:-1, 2:],   (1, 0, 1, 1)),   # derecha: (j+1,i)   -> (j+1,i+1)
        (core & ~m[2:, 1:-1],   (1, 1, 0, 1)),   # abajo  : (j+1,i+1) -> (j,i+1)
        (core & ~m[1:-1, 0:-2], (0, 1, 0, 0)),   # izq.   : (j,i+1)   -> (j,i)
    )

    succ: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for sel, (sx, sy, ex, ey) in sides:
        ii, jj = np.nonzero(sel)
        for i, j in zip(ii.tolist(), jj.tolist()):
            succ.setdefault((j + sx, i + sy), []).append((j + ex, i + ey))

    loops = []
    for s0 in list(succ.keys()):
        while succ.get(s0):
            loop = [s0]
            v, d_in = s0, None
            while True:
                outs = succ[v]
                if len(outs) == 1 or d_in is None:
                    w = outs.pop()
                else:
                    pref = (v[0] + _ROT_BACK[d_in][0], v[1] + _ROT_BACK[d_in][1])
                    k = outs.index(pref) if pref in outs else 0
                    w = outs.pop(k)
                if not succ[v]:
                    del succ[v]
                loop.append(w)
                if w == s0:
                    break
                d_in = (w[0] - v[0], w[1] - v[1])
                v = w
            loops.append(loop)
    return loops


# --------------------------------------------------------------------------
# 2 · Incremental recognition of a naive DSS in octant 1
# --------------------------------------------------------------------------
class _Oct1DSS:
    """
    Naive DSS with steps in {(1,0), (1,1)}, characteristics (a, b, mu):

        mu <= a*x - b*y <= mu + b - 1,      0 <= a <= b,   gcd(a, b) = 1

    Since r = a*x - b*y decreases with y, r = mu are the UPPER leaning points
    and r = mu + b - 1 the LOWER ones. The first and the last of each class are
    kept, which is all that is needed to update the slope in O(1)
    (Debled-Rennesson & Reveilles 1995).
    """

    __slots__ = ("a", "b", "mu", "ufx", "ufy", "ulx", "uly",
                 "lfx", "lfy", "llx", "lly")

    def __init__(self, x: int, y: int):
        self.a, self.b, self.mu = 0, 1, -y
        self.ufx = self.ulx = self.lfx = self.llx = x
        self.ufy = self.uly = self.lfy = self.lly = y

    def push(self, x: int, y: int) -> bool:
        a, b, mu = self.a, self.b, self.mu
        r = a * x - b * y
        if mu <= r <= mu + b - 1:                      # queda dentro de la banda
            if r == mu:
                self.ulx, self.uly = x, y
            if r == mu + b - 1:
                self.llx, self.lly = x, y
            return True
        if r == mu - 1:                                # se sale por arriba
            a2 = y - self.ufy
            b2 = x - self.ufx
            self.a, self.b = a2, b2
            self.mu = a2 * x - b2 * y
            self.lfx, self.lfy = self.llx, self.lly    # L_f' = L_l' = L_l viejo
            self.ulx, self.uly = x, y
            return True
        if r == mu + b:                                # se sale por abajo
            a2 = y - self.lfy
            b2 = x - self.lfx
            self.a, self.b = a2, b2
            self.mu = a2 * x - b2 * y - b2 + 1
            self.ufx, self.ufy = self.ulx, self.uly    # U_f' = U_l' = U_l viejo
            self.llx, self.lly = x, y
            return True
        return False


# quadrants of a 4-connected curve and the integer matrix T = shear or rotation
# that maps them to octant 1 (steps (1,0) and (1,1))
_QUAD = (
    (((1, 0), (0, 1)),   (1, 1, 0, 1)),      # E, N  -> (x+y, y)
    (((0, 1), (-1, 0)),  (-1, 1, -1, 0)),    # N, W  -> (y-x, -x)
    (((-1, 0), (0, -1)), (-1, -1, 0, -1)),   # W, S  -> (-x-y, -y)
    (((0, -1), (1, 0)),  (1, -1, 1, 0)),     # S, E  -> (x-y, x)
)
_QUAD_OF = {}
for _q, (_dirs, _M) in enumerate(_QUAD):
    _QUAD_OF[(_dirs[0], _dirs[1])] = _q
    _QUAD_OF[(_dirs[1], _dirs[0])] = _q


def _extend(path, i: int, n: int):
    """
    Largest j such that path[i..j] is a standard DSS. Returns
    (j, alpha, beta, mu), where (alpha, beta, mu) is the CERTIFICATE in the
    original coordinates:

        mu <= alpha*x + beta*y < mu + |alpha| + |beta|   for every point

    It is returned so that `selftest` can verify the output without going
    through the transformation again, i.e. without sharing code with the
    recogniser.
    """
    d1 = None
    quad = None
    M = None
    dss = None
    j = i
    while j + 1 < n:
        px, py = path[j]
        qx, qy = path[j + 1]
        step = (qx - px, qy - py)
        if d1 is None:
            d1 = step
        elif quad is None:
            if step != d1:
                if step == (-d1[0], -d1[1]):
                    break
                quad = _QUAD_OF[(d1, step)]
                M = _QUAD[quad][1]
                m00, m01, m10, m11 = M
                sx, sy = path[i]
                dss = _Oct1DSS(m00 * sx + m01 * sy, m10 * sx + m11 * sy)
                for t in range(i + 1, j + 1):
                    tx, ty = path[t]
                    dss.push(m00 * tx + m01 * ty, m10 * tx + m11 * ty)
        elif step != _QUAD[quad][0][0] and step != _QUAD[quad][0][1]:
            break
        if quad is not None:
            m00, m01, m10, m11 = M
            if not dss.push(m00 * qx + m01 * qy, m10 * qx + m11 * qy):
                break
        j += 1

    if quad is None:                       # tramo recto puro (0 o 1 direccion)
        if d1 is None:
            return j, 0, 0, 0
        dx, dy = d1
        alpha, beta = dy, -dx              # forma lineal nula sobre la direccion
        mu = alpha * path[i][0] + beta * path[i][1]
        return j, alpha, beta, mu

    m00, m01, m10, m11 = M
    A, B, MU = dss.a, dss.b, dss.mu
    # g(x,y) = A*X - B*Y with (X,Y) = M(x,y); linear with integer coefficients
    alpha = A * m00 - B * m10
    beta = A * m01 - B * m11
    return j, alpha, beta, MU


def greedy_breaks(path) -> list[int]:
    """Greedy segmentation: indices where each DSS ends. path[0] == path[-1]."""
    n = len(path)
    out, i = [0], 0
    while i < n - 1:
        j, *_ = _extend(path, i, n)
        if j <= i:                                     # no puede pasar; guarda
            j = i + 1
        out.append(j)
        i = j
    return out


def _poly_len(path, breaks) -> float:
    tot = 0.0
    for u, v in zip(breaks[:-1], breaks[1:]):
        dx = path[v][0] - path[u][0]
        dy = path[v][1] - path[u][1]
        tot += math.hypot(dx, dy)
    return tot


def dss_length(mask: np.ndarray, h: float = 1.0, mode: str = "fb") -> float:
    """
    Length of `mask`'s contour by DSS decomposition, in units of `h` (the pixel
    side).

    mode = "fb"   average of the greedy forward and backward segmentations
                  (default; cancels part of the starting-point bias)
    mode = "f"    forward only
    """
    tot = 0.0
    for loop in crack_contours(mask):
        fwd = _poly_len(loop, greedy_breaks(loop))
        if mode == "f":
            tot += fwd
        else:
            rev = loop[::-1]
            tot += 0.5 * (fwd + _poly_len(rev, greedy_breaks(rev)))
    return tot * h


def dss_nsegments(mask: np.ndarray) -> int:
    return sum(len(greedy_breaks(l)) - 1 for l in crack_contours(mask))


# --------------------------------------------------------------------------
# 3 · Brute-force oracles (for `selftest` only)
# --------------------------------------------------------------------------
def _fits_vertical(x, y) -> bool:
    """Is there a primitive (a, b) with |a| <= b and max(a*x - b*y) - min < b."""
    n = len(x)
    cands = {(0, 1), (1, 1), (-1, 1)}
    for i in range(n):
        for j in range(i + 1, n):
            dx = int(x[j] - x[i])
            dy = int(y[j] - y[i])
            if dx == 0:
                continue
            a, b = (dy, dx) if dx > 0 else (-dy, -dx)
            g = math.gcd(abs(a), b)
            if g:
                a //= g
                b //= g
            if abs(a) <= b:
                cands.add((a, b))
    for a, b in cands:
        r = a * x - b * y
        if int(r.max() - r.min()) < b:
            return True
    return False


def is_naive_dss_bf(pts) -> bool:
    """Exact brute force: the set admits a naive line containing it."""
    P = np.asarray(pts, dtype=np.int64)
    if len(P) <= 2:
        return True
    return (_fits_vertical(P[:, 0], P[:, 1]) or   # |a| <= |b|
            _fits_vertical(P[:, 1], P[:, 0]))     # |a| >  |b|, por simetria x<->y


def _shear(pts):
    P = np.asarray(pts, dtype=np.int64)
    return np.column_stack([P[:, 0] + P[:, 1], P[:, 1]])


# --------------------------------------------------------------------------
# 4 · Reference estimators
# --------------------------------------------------------------------------
def ms_binary_length(mask, h=1.0) -> float:
    """Marching squares at level 0.5 on the binary indicator."""
    from skimage import measure
    tot = 0.0
    for c in measure.find_contours(mask.astype(float), 0.5):
        tot += float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
    return tot * h


def sk_perimeter(mask, h=1.0) -> float:
    from skimage import measure
    return float(measure.perimeter(mask)) * h


def sk_crofton4(mask, h=1.0) -> float:
    from skimage import measure
    return float(measure.perimeter_crofton(mask, directions=4)) * h


def cv_arclength(mask, h=1.0):
    try:
        import cv2
    except ImportError:
        return None
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST,
                             cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(c, True) for c in cs)) * h


ESTIMATORS = [
    ("DSS", lambda m, h: dss_length(m, h)),
    ("MS bin 0.5", ms_binary_length),
    ("sk.perimeter", sk_perimeter),
    ("Crofton-4", sk_crofton4),
    ("cv2.arcLength", cv_arclength),
]


# --------------------------------------------------------------------------
# 5 · Formas de prueba
# --------------------------------------------------------------------------
HALF = 1.6


def grid(n):
    g = np.linspace(-HALF, HALF, n)
    h = float(g[1] - g[0])
    XX, YY = np.meshgrid(g, g)
    return XX, YY, h


def _blocked(n, fn):
    """Rasterise in row blocks: at n = 12800 a meshgrid would ask for 2.6 GB."""
    g = np.linspace(-HALF, HALF, n)
    h = float(g[1] - g[0])
    out = np.zeros((n, n), dtype=bool)
    step = max(1, 4_000_000 // n)
    for s in range(0, n, step):
        out[s:s + step] = fn(g[None, :], g[s:s + step][:, None])
    return out, h


def mask_circle(n, r=1.0):
    return _blocked(n, lambda X, Y: X * X + Y * Y <= r * r)


def mask_ellipse(n, a=1.3, b=0.7):
    return _blocked(n, lambda X, Y: (X / a) ** 2 + (Y / b) ** 2 <= 1.0)


def mask_square(n, side=1.6, deg=0.0):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    q = 0.5 * side

    def f(X, Y):
        return (np.abs(c * X + s * Y) <= q) & (np.abs(-s * X + c * Y) <= q)

    return _blocked(n, f)


# --------------------------------------------------------------------------
# 6 · Autotest
# --------------------------------------------------------------------------
def selftest(seed=0):
    rng = np.random.default_rng(seed)
    print("\n== selftest ==")
    bad = 0

    # (a) identidad de cizalla: recta estandar <-> recta naive
    n_ok = 0
    for _ in range(400):
        a = int(rng.integers(0, 12))
        b = int(rng.integers(1, 12))
        mu = int(rng.integers(-20, 20))
        xs = np.arange(0, 40, dtype=np.int64)
        # points of the standard line mu <= a*x - b*y < mu + a + b
        ys = np.floor((a * xs - mu) / b).astype(np.int64)
        r = a * xs - b * ys
        keep = (r >= mu) & (r < mu + a + b)
        P = np.column_stack([xs[keep], ys[keep]])
        Q = _shear(P)
        A, B = a, a + b
        rq = A * Q[:, 0] - B * Q[:, 1]
        if rq.min() >= mu and rq.max() < mu + max(A, B):
            n_ok += 1
    ok = n_ok == 400
    bad += not ok
    print(f"  [{'OK ' if ok else 'BAD'}] shear (x+y, y): standard(a,b) == naive(a,a+b)"
          f"   {n_ok}/400")

    # (b) reconocedor naive octante 1 contra fuerza bruta
    n_tested = n_dis = 0
    for _ in range(600):
        k = int(rng.integers(2, 26))
        steps = rng.integers(0, 2, size=k)
        pts = [(0, 0)]
        for s in steps:
            x, y = pts[-1]
            pts.append((x + 1, y + int(s)))
        d = _Oct1DSS(*pts[0])
        alive = True
        for t in range(1, len(pts)):
            got = d.push(*pts[t]) if alive else False
            if alive and not got:
                alive = False
            want = is_naive_dss_bf(pts[:t + 1])
            n_tested += 1
            if alive != want:
                n_dis += 1
            if not alive:
                break
    ok = n_dis == 0
    bad += not ok
    print(f"  [{'OK ' if ok else 'MAL'}] reconocedor naive == fuerza bruta"
          f"   {n_dis} discrepancias en {n_tested} prefijos")

    # (c) primitive gcd and certificate on real curves
    n_seg = n_cert = n_max = 0
    shapes = [mask_circle(97)[0], mask_ellipse(83)[0], mask_square(71, deg=27.0)[0]]
    blob = rng.random((60, 60)) < 0.55
    from scipy import ndimage
    blob = ndimage.binary_closing(ndimage.binary_opening(blob, np.ones((3, 3))),
                                  np.ones((3, 3)))
    shapes.append(blob)
    for msk in shapes:
        for loop in crack_contours(msk):
            n = len(loop)
            i = 0
            while i < n - 1:
                j, al, be, mu = _extend(loop, i, n)
                n_seg += 1
                # certificado en coordenadas ORIGINALES
                w = abs(al) + abs(be)
                seg = np.array(loop[i:j + 1], dtype=np.int64)
                r = al * seg[:, 0] + be * seg[:, 1]
                if (w > 0 and r.min() >= mu and r.max() < mu + w
                        and math.gcd(abs(al), abs(be)) == 1):
                    n_cert += 1
                # maximality: one more point is no longer a standard DSS
                if j + 1 < n:
                    ext = np.array(loop[i:j + 2], dtype=np.int64)
                    if not is_naive_dss_bf(_shear(ext)):
                        n_max += 1
                else:
                    n_max += 1
                i = j
    ok = (n_cert == n_seg) and (n_max == n_seg)
    bad += not ok
    print(f"  [{'OK ' if ok else 'BAD'}] every greedy segment is a standard DSS (certificate"
          f" in original coords)   {n_cert}/{n_seg}")
    print(f"  [{'OK ' if n_max == n_seg else 'BAD'}] every greedy segment is MAXIMAL"
          f" (brute force on the sheared version)      {n_max}/{n_seg}")

    # (d) the crack contour closes and is 4-connected
    n_loops = n_bad_step = 0
    for msk in shapes:
        for loop in crack_contours(msk):
            n_loops += 1
            if loop[0] != loop[-1]:
                n_bad_step += 1
            for u, v in zip(loop[:-1], loop[1:]):
                if abs(v[0] - u[0]) + abs(v[1] - u[1]) != 1:
                    n_bad_step += 1
    ok = n_bad_step == 0
    bad += not ok
    print(f"  [{'OK ' if ok else 'MAL'}] contorno crack cerrado y 4-conectado"
          f"   {n_loops} curvas, {n_bad_step} pasos invalidos")

    # (e) a straight staircase is ONE single DSS: the chord is exact
    #     (the property that separates DSS from marching squares, which would
    #      measure it as the sum of the steps)
    pts = [(0, 0)]
    for t in range(37):
        x, y = pts[-1]
        pts.append((x + 1, y) if t % 3 else (x, y + 1))
    pts.append(pts[0])
    j, al, be, mu = _extend(pts, 0, len(pts) - 1)
    ok = j == len(pts) - 2
    bad += not ok
    print(f"  [{'OK ' if ok else 'BAD'}] staircase of slope 1/3 (37 steps)"
          f" = ONE single DSS   j = {j} of {len(pts) - 2}")

    # (f) corner polynomial: on a rectangle the ABSOLUTE error is bounded
    #     (one constant per corner) and therefore the RELATIVE one is O(h).
    #     It is the convergence property, proved as such and not as an exact
    #     equality the estimator cannot satisfy at resolution 1.
    errs = []
    for k in (4, 8, 16, 32, 64):
        mm = np.zeros((4 + 3 * k, 4 + 5 * k), bool)
        mm[2:2 + 3 * k, 2:2 + 5 * k] = True
        errs.append(dss_length(mm) - 2 * (3 * k + 5 * k))
    errs = np.array(errs)
    ok = (np.abs(errs) < 3.5).all() and abs(errs[-1] - errs[-2]) < 0.02
    bad += not ok
    print(f"  [{'OK ' if ok else 'BAD'}] rectangle k*3 x k*5: absolute error bounded"
          f" -> {errs[-1]:+.4f} px (rel {100 * errs[-1] / (16 * 64):+.3f} %)")

    rel = np.abs(errs / np.array([16.0 * k for k in (4, 8, 16, 32, 64)]))
    ratios = rel[:-1] / rel[1:]
    ok = (ratios > 1.8).all()
    bad += not ok
    print(f"  [{'OK ' if ok else 'BAD'}] and the relative error halves when"
          f" k doubles (order 1)   ratios {np.array2string(ratios, precision=2)}")

    # (g) KULPA IDENTITY between the two reference estimators.
    #     skimage.perimeter_crofton(4) is NOT an estimator independent of
    #     marching squares: on any mask with no saddle configuration (a 2x2
    #     window with the two pixels on a diagonal) it holds, exactly, that
    #
    #         ms_binary_length = (4/pi)*(2*sqrt(2)-2) * sk_crofton4
    #
    #     The constant is Kulpa's factor 1.05478618. The consequence is that
    #     Crofton rescales the bias and does NOT touch the angle dependence:
    #     its orientation spread is that of marching squares divided by that
    #     same constant. Verified by exhaustive enumeration of ALL subsets of a
    #     4x4 window (2^16) that have no saddles.
    from skimage import measure as _meas
    K_KULPA = 4.0 / math.pi * (2.0 * math.sqrt(2.0) - 2.0)
    n_chk = n_dis = n_sad = 0
    for bits in range(1 << 16):
        g = np.array([(bits >> i) & 1 for i in range(16)], bool).reshape(4, 4)
        if not g.any():
            continue
        mm = np.zeros((8, 8), bool)
        mm[2:6, 2:6] = g
        a, b_, c_, d_ = mm[:-1, :-1], mm[:-1, 1:], mm[1:, :-1], mm[1:, 1:]
        if (((a & d_) & ~(b_ | c_)) | ((b_ & c_) & ~(a | d_))).any():
            n_sad += 1
            continue
        n_chk += 1
        lhs = ms_binary_length(mm)
        rhs = K_KULPA * sk_crofton4(mm)
        if abs(lhs - rhs) > 1e-9 * max(1.0, lhs):
            n_dis += 1
    ok = n_dis == 0
    bad += not ok
    print(f"  [{'OK ' if ok else 'BAD'}] marching squares == {K_KULPA:.8f} * Crofton-4"
          f" (saddle-free)   {n_chk - n_dis}/{n_chk} exact, {n_sad} with a saddle skipped")

    print(f"  --> {'TODO EN VERDE' if bad == 0 else str(bad) + ' FALLAS'}")
    return bad


# --------------------------------------------------------------------------
# 7 · Banco de convergencia
# --------------------------------------------------------------------------
NS = (200, 400, 800, 1600, 3200)
ANGLES = tuple(float(a) for a in range(0, 50, 5))
P_CIRCLE = 2.0 * math.pi
P_ELLIPSE = 6.4253707428
P_SQUARE = 6.4


def _measure(mask, h):
    out = {}
    for name, fn in ESTIMATORS:
        v = fn(mask, h)
        out[name] = v
    return out


def bench():
    names = [n for n, _ in ESTIMATORS]
    have_cv = cv_arclength(np.ones((3, 3), bool), 1.0) is not None
    if not have_cv:
        names = [n for n in names if n != "cv2.arcLength"]

    res = {}          # (shape, n) -> {est: err_rel_pct}
    sq = {}           # (n, ang)  -> {est: value}
    t0 = time.time()
    for n in NS:
        for tag, mk, ref in (("circle", mask_circle, P_CIRCLE),
                             ("ellipse", mask_ellipse, P_ELLIPSE)):
            msk, h = mk(n)
            vals = _measure(msk, h)
            res[(tag, n)] = {k: 100.0 * (v / ref - 1.0)
                             for k, v in vals.items() if v is not None}
            del msk
        for ang in ANGLES:
            msk, h = mask_square(n, deg=ang)
            vals = _measure(msk, h)
            sq[(n, ang)] = {k: v for k, v in vals.items() if v is not None}
            del msk
        print(f"  ... n = {n} listo ({time.time() - t0:.0f} s)", file=sys.stderr)

    # ---- table 1: circle and ellipse
    print("\n### Convergence: relative error (%) against the exact value\n")
    print("| forma | n | h | " + " | ".join(names) + " |")
    print("|---|---:|---:|" + "---:|" * len(names))
    for tag in ("circle", "ellipse"):
        for n in NS:
            h = 3.2 / (n - 1)
            row = res[(tag, n)]
            cells = " | ".join(f"{row[k]:+.4f}" for k in names)
            print(f"| {tag} | {n} | {h:.5f} | {cells} |")

    # ---- table 2: rotated square, worst case and spread
    #
    # SPREAD DEFINITION. It is max(error) - min(error), in PERCENTAGE POINTS,
    # over rotations from 0 to 45 degrees in steps of 5.
    #
    # The previous version used max(value)/min(value) - 1, which is wrong for
    # razones independientes:
    #
    #   (a) it makes no sense when the errors cross zero -- Crofton runs from
    #       -5.18 % to +2.52 %, which is 7.71 POINTS of travel, not "8.13 %" --;
    #
    #   (b) it is scale invariant, so TWO ESTIMATORS DIFFERING BY A CONSTANT
    #       FACTOR give exactly the same number. That is precisely what was
    #       happening here: on any mask with no saddle configuration (2x2 with
    #       the two pixels on a diagonal),
    #
    #           ms_binary_length == (4/pi)*(2*sqrt(2)-2) * sk_crofton4
    #
    #       exactly -- the constant being Kulpa's factor 1.0547862 -- so the
    #       Crofton column reproduced the marching squares one digit for digit.
    #       Verified by exhaustive enumeration of the 23,857 saddle-free subsets
    #       of a 4x4 window, with zero residual, and on the leaf mask (which has
    #       0 saddles over 251,937 windows).
    #       With the definition in percentage points the relation becomes
    #       visible: spread(Crofton) = spread(MS) / 1.0547862.
    print("\n### Square of side 1.6 rotated 0..45 deg (step 5): "
          "worst absolute error and spread (pp = max err - min err)\n")
    print("| n | " + " | ".join(f"{k} worst % / spread pp" for k in names) + " |")
    print("|---:|" + "---:|" * len(names))
    disp_tbl = {}
    for n in NS:
        cells = []
        for k in names:
            v = np.array([sq[(n, a)][k] for a in ANGLES])
            e = 100.0 * (v / P_SQUARE - 1.0)
            worst = float(np.abs(e).max())
            disp = float(e.max() - e.min())      # puntos porcentuales
            disp_tbl[(n, k)] = (worst, disp)
            cells.append(f"{worst:.4f} / {disp:.4f}")
        print(f"| {n} | " + " | ".join(cells) + " |")

    # ---- table 3: by angle at the finest resolution
    nfine = NS[-1]
    print(f"\n### Square by angle, n = {nfine}: relative error (%)\n")
    print("| angle | " + " | ".join(names) + " |")
    print("|---:|" + "---:|" * len(names))
    for a in ANGLES:
        row = sq[(nfine, a)]
        cells = " | ".join(f"{100.0 * (row[k] / P_SQUARE - 1.0):+.4f}" for k in names)
        print(f"| {a:.0f} | {cells} |")

    # ---- orden observado del DSS
    print("\n### Observed DSS order (log2 of the error ratio)\n")
    print("| forma | " + " | ".join(f"{a}->{b}" for a, b in zip(NS[:-1], NS[1:])) + " |")
    print("|---|" + "---:|" * (len(NS) - 1))
    for tag in ("circle", "ellipse"):
        e = [abs(res[(tag, n)]["DSS"]) for n in NS]
        cells = " | ".join(f"{math.log2(e[i] / e[i + 1]):+.2f}" if e[i + 1] > 0 else "inf"
                           for i in range(len(e) - 1))
        print(f"| {tag} | {cells} |")
    e = [disp_tbl[(n, "DSS")][0] for n in NS]
    cells = " | ".join(f"{math.log2(e[i] / e[i + 1]):+.2f}" if e[i + 1] > 0 else "inf"
                       for i in range(len(e) - 1))
    print(f"| square (worst) | {cells} |")

    if not have_cv:
        print("\n_opencv not available: the cv2.arcLength column is omitted._")


NS_DEEP = (200, 400, 800, 1600, 3200, 6400, 12800)


def deep():
    """
    Deep convergence of DSS alone, down to h = 2.5e-4.

    The question this table answers is the only one that matters: does the error
    TEND TO ZERO or plateau. It goes to 12800 because at 3200 DSS is already
    0.003 % and two more octaves are needed to tell convergence from a plateau.
    """
    ref = {"circle": P_CIRCLE, "ellipse": P_ELLIPSE, "square 30deg": P_SQUARE}
    mk = {"circle": mask_circle, "ellipse": mask_ellipse,
          "square 30deg": lambda n: mask_square(n, deg=30.0)}
    err = {k: [] for k in ref}
    print("\n### Deep convergence of DSS (relative error, %)\n")
    print("| n | h | " + " | ".join(ref) + " |")
    print("|---:|---:|" + "---:|" * len(ref))
    for n in NS_DEEP:
        cells = []
        for k in ref:
            m, h = mk[k](n)
            e = 100.0 * (dss_length(m, h) / ref[k] - 1.0)
            err[k].append(abs(e))
            cells.append(f"{e:+.5f}")
            del m
        print(f"| {n} | {3.2 / (n - 1):.6f} | " + " | ".join(cells) + " |", flush=True)

    lh = np.log(np.array([3.2 / (n - 1) for n in NS_DEEP]))
    print("\nObserved order (least squares of log|err| against log h):")
    for k in ref:
        p = float(np.polyfit(lh, np.log(np.array(err[k])), 1)[0])
        print(f"  {k:16s} orden {p:.2f}")


# --------------------------------------------------------------------------
# 8 · The project leaf
# --------------------------------------------------------------------------
def leaf():
    # The root is derived from the location of THIS file: it used to be an
    # absolute path with the username, which does not run on another machine.
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from foliar import synth

    cfg = synth.LeafConfig()          # seed 7 by default
    sample = synth.generate(cfg)
    m = sample.mask_blade
    h = 1.0 / sample.px_per_cm
    ref = sample.truth.projected_perimeter_cm

    print(f"\nleaf seed {cfg.seed}: {m.shape[1]} x {m.shape[0]} px at "
          f"{sample.px_per_cm:g} px/cm")
    print(f"reference (zero level, projected): {ref:.4f} cm\n")
    print("| estimator | perimeter (cm) | error (%) |")
    print("|---|---:|---:|")
    for name, fn in ESTIMATORS:
        v = fn(m, h)
        if v is None:
            continue
        print(f"| {name} | {v:.3f} | {100.0 * (v / ref - 1.0):+.3f} |")
    print(f"\nsegmentos DSS: {dss_nsegments(m)}   "
          f"vertices del crack: {sum(len(l) - 1 for l in crack_contours(m))}   "
          f"curvas cerradas: {len(crack_contours(m))}")

    # The reference does NOT depend on px_per_cm (it comes from the intrinsic
    # grid), so this column measures real convergence and not a moving target.
    print("\n### Leaf at several resolutions (fixed reference "
          f"{ref:.4f} cm)\n")
    print("| px/cm | imagen (px) | DSS (cm) | err % | MS bin (cm) | err % | "
          "Crofton-4 (cm) | err % |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for ppc in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        s = synth.generate(synth.LeafConfig(px_per_cm=ppc))
        hh = 1.0 / ppc
        r = s.truth.projected_perimeter_cm
        d = dss_length(s.mask_blade, hh)
        b = ms_binary_length(s.mask_blade, hh)
        cf = sk_crofton4(s.mask_blade, hh)
        print(f"| {ppc:g} | {s.mask_blade.shape[1]}x{s.mask_blade.shape[0]} | "
              f"{d:.3f} | {100 * (d / r - 1):+.3f} | {b:.3f} | "
              f"{100 * (b / r - 1):+.3f} | {cf:.3f} | {100 * (cf / r - 1):+.3f} |",
              flush=True)
        del s

    # five seeds: the sign and the order of magnitude cannot be an accident
    print("\n### Five seeds at 4 px/cm\n")
    print("| seed | ref (cm) | DSS | err % | MS bin | err % | Crofton-4 | err % |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    acc = {"DSS": [], "MS bin": [], "Crofton-4": []}
    for sd in (7, 8, 9, 10, 11):
        s = synth.generate(synth.LeafConfig(seed=sd))
        hh = 1.0 / s.px_per_cm
        r = s.truth.projected_perimeter_cm
        d = dss_length(s.mask_blade, hh)
        b = ms_binary_length(s.mask_blade, hh)
        cf = sk_crofton4(s.mask_blade, hh)
        for k, v in (("DSS", d), ("MS bin", b), ("Crofton-4", cf)):
            acc[k].append(100.0 * (v / r - 1.0))
        print(f"| {sd} | {r:.3f} | {d:.3f} | {100 * (d / r - 1):+.3f} | {b:.3f} | "
              f"{100 * (b / r - 1):+.3f} | {cf:.3f} | {100 * (cf / r - 1):+.3f} |",
              flush=True)
        del s
    print()
    for k, v in acc.items():
        a = np.array(v)
        print(f"  {k:10s} {a.mean():+.3f} +- {a.std(ddof=1):.3f} %")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(1 if selftest() else 0)
    elif cmd == "bench":
        bench()
    elif cmd == "deep":
        deep()
    elif cmd == "leaf":
        leaf()
    else:
        print(__doc__)
