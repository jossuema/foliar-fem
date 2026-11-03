#!/usr/bin/env python3
"""
3D extension of the perimeter bias result: SURFACE AREA.

Why this file exists
--------------------
Reviewer question 5. In clinical radiomics the descriptor is not the 2D
perimeter but the 3D SURFACE AREA, and the standard estimator (PyRadiomics
shape_SurfaceArea, recommended by IBSI) is marching cubes on the BINARY mask.
If the failure mechanism the paper documents in 2D survives in 3D, the scope of
the argument changes from one leaf image to any shape biomarker.

What is measured
----------------
  1. Exact areas: sphere 4 pi r^2 and ellipsoid by Legendre's formula with
     incomplete elliptic integrals, VERIFIED against Gauss-Legendre quadrature
     (an independent reference, agreeing to 2e-14 relative).
  2. Convergence with resolution, n = 50..1600 per side in [-1.6,1.6]^3
     (4.1e9 voxels in the largest case, processed in blocks of z slices):
       - MC-bin : marching cubes on the binary indicator, level 0.5
                  (this is what PyRadiomics does)
       - MC-0   : marching cubes on the signed field, level 0.0
                  (the 3D analogue of the paper's zero level contour)
       - Crofton-13 : Crofton discretised over the 13 directions of the 3D
                  neighbourhood (analogue of Crofton-4 in 2D)
       - Faces  : face counting on the digital boundary (the naive estimator,
                  with a known asymptotic bias: E|u|_1 over the normals)
  3. ORIENTATION spread, with the SAME definition as the paper's 2D one:
     max(error) - min(error) in PERCENTAGE POINTS over ten rotations from 0 to
     45 degrees in steps of 5. Cube and ellipsoid, rotating about z and about
     (1,1,1).
  4. The MC-bin plateau averaged over nine grids, to separate the asymptotic
     bias from the grid's phase fluctuation.
  5. IBSI / PyRadiomics shape descriptors: sphericity (36 pi V^2)^(1/3) / A
     with V by voxel counting, which is convergent.
  6. A convergent estimator that DOES run on the binary mask (integral
     invariant); the 3D analogue of DSS.
  7. The 3D analogue of Kulpa's constant: f(u) is measured, the ratio
     binary-MC-area / true-area of a digital HALF-SPACE of normal u, and
     averaged over u uniform on S^2. That average predicts the sphere's
     asymptotic bias; averaged over the ellipsoid's normals it predicts the
     ellipsoid's. Exact control of the method: for face counting
     f(u) = |ux|+|uy|+|uz|, whose mean is known in closed form.

Short result: in 3D the bias does NOT disappear under refinement. MC-bin stays
at +8.80 % (sphere) and +8.87 % (ellipsoid) with observed order 0.00 between
n=100 and n=1600, while MC-0 falls at order 2.00. PyRadiomics' sphericity on a
perfect digital sphere gives 0.920 instead of 1.000, at every resolution.

Usage
-----
    python3 surf3d.py quick      # n <= 400          (~1 min)
    python3 surf3d.py            # n <= 800          (~6 min)
    python3 surf3d.py full       # n <= 1600         (~25 min)
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.special import ellipkinc, ellipeinc
from skimage import measure

HALF = 1.6           # half-side of the domain [-1.6, 1.6]^3, same as in 2D
BLOCK = None         # layers per block; None = pick one to bound memory


def auto_block(n, target=3e7):
    """Slices per block so that one block stays under ~target voxels."""
    return max(2, int(target / (n * n)))


# ==========================================================================
# 1. Exact areas
# ==========================================================================

def sphere_area(r=1.0):
    return 4.0 * np.pi * r * r


def ellipsoid_area_legendre(a, b, c):
    """Legendre closed form, a >= b >= c, with incomplete F and E."""
    a, b, c = sorted((a, b, c), reverse=True)
    phi = np.arccos(c / a)
    m = (a * a * (b * b - c * c)) / (b * b * (a * a - c * c))   # m = k^2
    s, cs = np.sin(phi), np.cos(phi)
    return 2 * np.pi * c * c + (2 * np.pi * a * b / s) * (
        ellipeinc(phi, m) * s * s + ellipkinc(phi, m) * cs * cs)


def ellipsoid_area_quad(a, b, c, nq=400):
    """INDEPENDENT reference: Gauss-Legendre quadrature of the area element of
    the (theta, psi) parametrisation. It shares not a single line with
    Legendre's formula."""
    xt, wt = np.polynomial.legendre.leggauss(nq)
    th = 0.5 * np.pi * (xt + 1.0)
    wth = 0.5 * np.pi * wt
    ps = np.pi * (xt + 1.0)
    wps = np.pi * wt
    TH, PS = np.meshgrid(th, ps, indexing="ij")
    W = np.outer(wth, wps)
    st, ct = np.sin(TH), np.cos(TH)
    sp, cp = np.sin(PS), np.cos(PS)
    # r = (a st cp, b st sp, c ct);  |r_th x r_ps|
    ex = b * c * st * st * cp
    ey = a * c * st * st * sp
    ez = a * b * st * ct
    return float(np.sum(W * np.sqrt(ex * ex + ey * ey + ez * ez)))


# ==========================================================================
# 2. Signed fields
# ==========================================================================

def rot_matrix(axis, deg):
    ax = np.asarray(axis, float)
    ax /= np.linalg.norm(ax)
    t = np.deg2rad(deg)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def make_field(kind, gx, gy, gz, par=None, Rm=None):
    """Signed field (>0 inside) on the product grid gx x gy x gz.
    Built by broadcasting: it never materialises a meshgrid."""
    X = gx[:, None, None]
    Y = gy[None, :, None]
    Z = gz[None, None, :]
    if Rm is not None:
        Rt = Rm.T
        Xr = Rt[0, 0] * X + Rt[0, 1] * Y + Rt[0, 2] * Z
        Yr = Rt[1, 0] * X + Rt[1, 1] * Y + Rt[1, 2] * Z
        Zr = Rt[2, 0] * X + Rt[2, 1] * Y + Rt[2, 2] * Z
        X, Y, Z = Xr, Yr, Zr
    if kind == "sphere":
        return 1.0 - np.sqrt(X * X + Y * Y + Z * Z)
    if kind == "ellipsoid":
        a, b, c = par
        return 1.0 - np.sqrt((X / a) ** 2 + (Y / b) ** 2 + (Z / c) ** 2)
    if kind == "cube":
        s = par
        return np.minimum(np.minimum(s - np.abs(X), s - np.abs(Y)),
                          s - np.abs(Z))
    raise ValueError(kind)


# ==========================================================================
# 3. Estimators, all in blocks along z
# ==========================================================================

# 13 directions of the 3D neighbourhood and their solid-angle weights
# (Legland, Kieu & Devaux 2007; they sum to 0.5 = half sphere).
DIRS13 = [((1, 0, 0), 0.04577789115), ((0, 1, 0), 0.04577789115),
          ((0, 0, 1), 0.04577789115),
          ((1, 1, 0), 0.03698062462), ((1, -1, 0), 0.03698062462),
          ((1, 0, 1), 0.03698062462), ((1, 0, -1), 0.03698062462),
          ((0, 1, 1), 0.03698062462), ((0, 1, -1), 0.03698062462),
          ((1, 1, 1), 0.03519563264), ((1, 1, -1), 0.03519563264),
          ((1, -1, 1), 0.03519563264), ((-1, 1, 1), 0.03519563264)]


def _mc_area(vol, level, h):
    """Marching cubes area; 0 if the block contains no isosurface."""
    if not (vol.min() < level < vol.max()):
        return 0.0
    v, f, _, _ = measure.marching_cubes(vol, level, spacing=(h, h, h))
    return float(measure.mesh_surface_area(v, f))


def estimate_all(kind, n, par=None, Rm=None, half=HALF, block=None,
                 want=("mcbin", "mc0")):
    """Areas by marching cubes. Processes the volume in blocks of z slices;
    adjacent blocks sharing ONE single slice cover DISJOINT cells, so the areas
    add up without duplication. skimage's marching_cubes does not close the
    surface at the array border, so no spurious caps appear at the cut either.
    Verified in selftest (c)."""
    block = auto_block(n) if block is None else block
    g = np.linspace(-half, half, n)
    h = float(g[1] - g[0])
    a_mcbin = a_mc0 = 0.0
    k0 = 0
    while k0 < n - 1:
        k1 = min(k0 + block, n - 1)
        gz = g[k0:k1 + 1]
        phi = make_field(kind, g, g, gz, par=par, Rm=Rm).astype(np.float32)
        if "mc0" in want:
            a_mc0 += _mc_area(phi, 0.0, h)
        if "mcbin" in want:
            mb = (phi > 0).astype(np.float32)
            a_mcbin += _mc_area(mb, 0.5, h)
            del mb
        del phi
        k0 = k1
    out = {"h": h, "n": n}
    if "mc0" in want:
        out["mc0"] = a_mc0
    if "mcbin" in want:
        out["mcbin"] = a_mcbin
    return out


def _sl(k, N):
    """Range of valid indices for the starting point along one axis."""
    if k > 0:
        return slice(0, N - k)
    if k < 0:
        return slice(-k, N)
    return slice(0, N)


def _count_dir(mb, lo, n, k0, k1, d):
    """Transitions between p and p+d with p_z in [k0, k1) and both points
    inside the full volume of n slices. `mb` is the block of slices [lo, lo+len)."""
    dx, dy, dz = d
    nx, ny, nzl = mb.shape
    zlo = max(k0, -dz if dz < 0 else 0)
    zhi = min(k1, n - dz if dz > 0 else n)
    if zhi <= zlo:
        return 0
    ax, ay = _sl(dx, nx), _sl(dy, ny)
    bx = slice(ax.start + dx, ax.stop + dx)
    by = slice(ay.start + dy, ay.stop + dy)
    a = mb[ax, ay, zlo - lo:zhi - lo]
    b = mb[bx, by, zlo - lo + dz:zhi - lo + dz]
    return int(np.count_nonzero(a != b))


def crofton_faces(kind, n, par=None, Rm=None, half=HALF, block=None):
    """Crofton-13 and face counting, in blocks with one halo slice on each
    side; every pair of the full volume is counted exactly once because only
    pairs whose STARTING point falls in the block's own part are admitted.
    Verified in selftest (d)."""
    block = auto_block(n) if block is None else block
    g = np.linspace(-half, half, n)
    h = float(g[1] - g[0])
    trans = {d: 0 for d, _ in DIRS13}
    k0 = 0
    while k0 < n:
        k1 = min(k0 + block, n)
        lo, hi = max(0, k0 - 1), min(n, k1 + 1)
        mb = make_field(kind, g, g, g[lo:hi], par=par, Rm=Rm) > 0
        for d, _w in DIRS13:
            trans[d] += _count_dir(mb, lo, n, k0, k1, d)
        del mb
        k0 = k1
    S_cro = 4.0 * sum(w * trans[d] * h * h / float(np.linalg.norm(d))
                      for d, w in DIRS13)
    S_fac = (trans[(1, 0, 0)] + trans[(0, 1, 0)] + trans[(0, 0, 1)]) * h * h
    return {"crofton": S_cro, "faces": S_fac}


# ==========================================================================
# 4. Digital half-space: f(u) and the 3D analogue of Kulpa's constant
# ==========================================================================

def halfspace_ratio(u, L=64, seed=0, nrep=3):
    """Ratio binary-MC-area / true-area for the digital half-space of normal u.
    u_z = max|u_i| is required (allowed by the cubic symmetry of the
    estimator). The surface is a graph z = (d - ux x - uy y)/uz, so it is
    measured over a window in (x, y) and divided by its true area L_w^2 / u_z.
    An interior window with margin removes the border."""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    assert abs(u[2]) >= max(abs(u[0]), abs(u[1])) - 1e-12
    rng = np.random.default_rng(seed)
    x = np.arange(L + 1, dtype=float)
    y = np.arange(L + 1, dtype=float)
    m = L // 6                                # window margin
    vals = []
    for _ in range(nrep):
        d = float(rng.random()) * 1.0
        zc = (d - u[0] * x[:, None] - u[1] * y[None, :]) / u[2]
        z0 = np.floor(zc.min()) - 3
        z1 = np.ceil(zc.max()) + 3
        z = np.arange(z0, z1 + 1, dtype=float)
        phi = (d - (u[0] * x[:, None, None] + u[1] * y[None, :, None]
                    + u[2] * z[None, None, :]))
        mb = (phi > 0).astype(np.float32)
        del phi
        v, f, _, _ = measure.marching_cubes(mb, 0.5, spacing=(1.0, 1.0, 1.0))
        cen = v[f].mean(axis=1)
        keep = ((cen[:, 0] >= m) & (cen[:, 0] <= L - m) &
                (cen[:, 1] >= m) & (cen[:, 1] <= L - m))
        tri = v[f[keep]]
        ar = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                           tri[:, 2] - tri[:, 0]), axis=1)
        A = float(ar.sum())
        Atrue = (L - 2 * m) ** 2 / abs(u[2])
        vals.append(A / Atrue)
    return float(np.mean(vals)), float(np.std(vals))


def fold(u):
    """Map u into the fundamental domain of the cubic group: components >= 0
    and sorted, with the largest along z."""
    v = np.sort(np.abs(np.asarray(u, float)))
    return np.array([v[0], v[1], v[2]])


# ==========================================================================
# selftest
# ==========================================================================

def selftest():
    print("=" * 74)
    print("SELFTEST")
    print("=" * 74)
    a, b, c = 1.3, 1.0, 0.7
    SL = ellipsoid_area_legendre(a, b, c)
    SQ = ellipsoid_area_quad(a, b, c)
    print("(a) ellipsoid 1.3x1.0x0.7")
    print("    Legendre (incomplete F,E)  : %.10f" % SL)
    print("    Gauss-Legendre quadrature  : %.10f" % SQ)
    print("    relative difference        : %.3e   %s"
          % (abs(SL / SQ - 1), "OK" if abs(SL / SQ - 1) < 1e-10 else "FAIL"))
    print("(b) Legendre against quadrature on other shapes, and the spherical limit")
    for tri in [(2.0, 1.0, 0.5), (1.0, 1.0, 0.5), (1.0, 0.6, 0.6),
                (1.0, 1.0, 1.0 - 1e-9)]:
        L = ellipsoid_area_legendre(*tri)
        Q = ellipsoid_area_quad(*tri)
        print("    %-22s Legendre %.10f  quad %.10f  rel %.2e"
              % (str(tri), L, Q, abs(L / Q - 1)))
    SS = ellipsoid_area_legendre(1.0, 1.0, 1.0 - 1e-9)
    print("    spherical limit vs 4pi     : %.10f vs %.10f  (%.2e)"
          % (SS, 4 * np.pi, abs(SS / (4 * np.pi) - 1)))

    print("(c) block sum == whole volume (sphere, n=120)")
    r_full = estimate_all("sphere", 120, block=10 ** 6,
                          want=("mcbin", "mc0"))
    r_blk = estimate_all("sphere", 120, block=17, want=("mcbin", "mc0"))
    # MC-bin agrees to 1e-16 (the vertices are exact midpoints).
    # MC-0 agrees only to ~4e-10: skimage interpolates in float32, and the
    # absolute z index of the vertex changes between the whole volume and the
    # block, so it rounds differently. That is 4e-10 of noise against errors
    # of interest from 1e-4 upwards; the declared tolerance is 1e-8, not 1e-12.
    for k, tol in (("mcbin", 1e-12), ("mc0", 1e-8)):
        print("    %-6s whole %.10f  blocks %.10f  diff %.2e  (tol %.0e) %s"
              % (k, r_full[k], r_blk[k], abs(r_full[k] / r_blk[k] - 1), tol,
                 "OK" if abs(r_full[k] / r_blk[k] - 1) < tol else "FAIL"))

    print("(d) Crofton/faces in blocks == in one pass (sphere, n=120)")
    c_full = crofton_faces("sphere", 120, block=10 ** 6)
    c_blk = crofton_faces("sphere", 120, block=13)
    for k in ("crofton", "faces"):
        print("    %-8s whole %.8f  blocks %.8f  diff %.2e  %s"
              % (k, c_full[k], c_blk[k], abs(c_full[k] / c_blk[k] - 1),
                 "OK" if abs(c_full[k] / c_blk[k] - 1) < 1e-12 else "FAIL"))

    print("(e) half-space of normal (0,0,1): f must be exactly 1")
    f, s = halfspace_ratio((0, 0, 1), L=48, nrep=2)
    print("    f = %.6f  (sd %.1e)  %s" % (f, s, "OK" if abs(f - 1) < 1e-9
                                           else "FAIL"))
    print("(f) half-space of normal (1,1,0)/sqrt2 -> the 2D analogue at 45 deg")
    f, s = halfspace_ratio((1, 0, 1), L=48, nrep=2)
    print("    f = %.6f  (2D at 45 deg predicts 1.000000)" % f)
    print("(g) axis-aligned cube, exact area 15.36")
    cf = crofton_faces("cube", 200, par=0.8)
    r = estimate_all("cube", 200, par=0.8)
    print("    faces     %.6f  (%+.4f %%)  <- side quantisation only"
          % (cf["faces"], 100 * (cf["faces"] / 15.36 - 1)))
    print("    MC-bin    %.6f  (%+.4f %%)"
          % (r["mcbin"], 100 * (r["mcbin"] / 15.36 - 1)))
    print("    MC-0      %.6f  (%+.4f %%)"
          % (r["mc0"], 100 * (r["mc0"] / 15.36 - 1)))
    print("    Crofton13 %.6f  (%+.4f %%)"
          % (cf["crofton"], 100 * (cf["crofton"] / 15.36 - 1)))
    print()


# ==========================================================================
# experiments
# ==========================================================================

def convergence(ns):
    a, b, c = 1.3, 1.0, 0.7
    cases = [("sphere r=1", "sphere", None, sphere_area(1.0)),
             ("ellipsoid 1.3x1.0x0.7", "ellipsoid", (a, b, c),
              ellipsoid_area_legendre(a, b, c))]
    out = {}
    for name, kind, par, exact in cases:
        print("=" * 86)
        print("%s   exact area = %.8f" % (name, exact))
        print("=" * 86)
        print("%6s %9s %12s %9s %12s %9s %12s %9s %12s %9s"
              % ("n", "h", "MC-bin", "err %", "MC-0", "err %",
                 "Crofton13", "err %", "faces", "err %"))
        rows = []
        for n in ns:
            t0 = time.time()
            r = estimate_all(kind, n, par=par, want=("mcbin", "mc0"))
            cf = crofton_faces(kind, n, par=par)
            e = lambda v: 100 * (v / exact - 1)
            print("%6d %9.6f %12.5f %+9.4f %12.5f %+9.4f %12.5f %+9.4f "
                  "%12.5f %+9.4f    (%.0fs)"
                  % (n, r["h"], r["mcbin"], e(r["mcbin"]), r["mc0"],
                     e(r["mc0"]), cf["crofton"], e(cf["crofton"]),
                     cf["faces"], e(cf["faces"]), time.time() - t0))
            rows.append((n, r["h"], r["mcbin"], r["mc0"], cf["crofton"],
                         cf["faces"]))
            sys.stdout.flush()
        print()
        print("  observed order  log2(|err_n| / |err_2n|):")
        print("  %10s %10s %10s %10s %10s" % ("n->2n", "MC-bin", "MC-0",
                                              "Crofton13", "faces"))
        for i in range(len(rows) - 1):
            def order(j):
                e1 = abs(rows[i][j] / exact - 1)
                e2 = abs(rows[i + 1][j] / exact - 1)
                if e1 == 0 or e2 == 0:
                    return float("nan")
                return np.log2(e1 / e2)
            print("  %10s %10.2f %10.2f %10.2f %10.2f"
                  % ("%d->%d" % (rows[i][0], rows[i + 1][0]),
                     order(2), order(3), order(4), order(5)))
        print()
        out[name] = (rows, exact)
    return out


def orientation(n=400):
    """Orientation spread, SAME definition as the paper's 2D one:
    max(err) - min(err) in percentage points over 0..45 degrees, step 5."""
    ANGS = [float(a) for a in range(0, 50, 5)]
    a, b, c = 1.3, 1.0, 0.7
    S_ELL = ellipsoid_area_legendre(a, b, c)
    cases = [("cube side 1.6, rotating about z", "cube", 0.8, 15.36, (0, 0, 1)),
             ("cube side 1.6, rotating about (1,1,1)", "cube", 0.8, 15.36,
              (1, 1, 1)),
             ("ellipsoid 1.3x1.0x0.7, rotating about z", "ellipsoid", (a, b, c),
              S_ELL, (0, 0, 1)),
             ("ellipsoid 1.3x1.0x0.7, rotating about (1,1,1)", "ellipsoid",
              (a, b, c), S_ELL, (1, 1, 1))]
    print("=" * 86)
    print("ORIENTATION SPREAD   n = %d" % n)
    print("   definition: max(err) - min(err) in pp, angles 0..45 step 5")
    print("=" * 86)
    for name, kind, par, exact, axis in cases:
        eb, e0, ec = [], [], []
        for ang in ANGS:
            Rm = rot_matrix(axis, ang)
            r = estimate_all(kind, n, par=par, Rm=Rm)
            cf = crofton_faces(kind, n, par=par, Rm=Rm)
            eb.append(100 * (r["mcbin"] / exact - 1))
            e0.append(100 * (r["mc0"] / exact - 1))
            ec.append(100 * (cf["crofton"] / exact - 1))
        eb, e0, ec = np.array(eb), np.array(e0), np.array(ec)
        print("%s   (exact %.6f)" % (name, exact))
        print("   %-10s %s" % ("MC-bin", " ".join("%+.3f" % v for v in eb)))
        print("   %-10s %s" % ("MC-0", " ".join("%+.3f" % v for v in e0)))
        print("   %-10s %s" % ("Crofton13", " ".join("%+.3f" % v for v in ec)))
        print("   spread (pp):  MC-bin %.4f   MC-0 %.4f   Crofton13 %.4f"
              % (eb.max() - eb.min(), e0.max() - e0.min(), ec.max() - ec.min()))
        print("   factor MC-bin / MC-0 : %.0f"
              % ((eb.max() - eb.min()) / max(e0.max() - e0.min(), 1e-12)))
        if kind == "cube" and axis == (0, 0, 1):
            # Closure with the 2D case: rotating the cube about z, the 4 side
            # faces behave like the 2D square and the 2 caps stay fixed with
            # f = 1, so the 3D curve has to be (2/3) of the 2D one. The
            # measured 2D spread is 8.13 pp.
            th = np.deg2rad(np.array(ANGS))
            f2 = np.cos(th) + (np.sqrt(2) - 1) * np.sin(th)
            pred = (2.0 / 3.0) * 100 * (f2 - 1) + eb[0]
            print("   prediction from the 2D case, (2/3)*(f2(t)-1) + err(0):")
            print("   %-10s %s" % ("pred", " ".join("%+.3f" % v
                                                    for v in pred)))
            print("   predicted spread %.4f pp against %.4f measured"
                  % (pred.max() - pred.min(), eb.max() - eb.min()))
        sys.stdout.flush()
    print()


# --------------------------------------------------------------------------
# f(u) on the digital half-space, the 3D constant, and the PREDICTION of a
# smooth body's bias from the distribution of its normals.
# --------------------------------------------------------------------------

def ellipsoid_normal_sample(a, b, c, N, seed, nq=200):
    """Sample N ellipsoid surface normals with AREA weighting."""
    xt, wt = np.polynomial.legendre.leggauss(nq)
    th = 0.5 * np.pi * (xt + 1.0)
    wth = 0.5 * np.pi * wt
    ps = np.pi * (xt + 1.0)
    wps = np.pi * wt
    TH, PS = np.meshgrid(th, ps, indexing="ij")
    W = np.outer(wth, wps)
    st, ct = np.sin(TH), np.cos(TH)
    sp, cp = np.sin(PS), np.cos(PS)
    ex, ey, ez = b * c * st * st * cp, a * c * st * st * sp, a * b * st * ct
    J = np.sqrt(ex * ex + ey * ey + ez * ez)
    p = (W * J).ravel()
    S = p.sum()
    p = p / S
    idx = np.random.default_rng(seed).choice(p.size, size=N, p=p)
    nrm = np.stack([(ex / J).ravel()[idx], (ey / J).ravel()[idx],
                    (ez / J).ravel()[idx]], axis=1)
    # E[|nx|+|ny|+|nz|] exact by quadrature: analytic control of the method
    L1 = float((W * J * (np.abs(ex / J) + np.abs(ey / J)
                         + np.abs(ez / J))).sum() / S)
    return nrm, L1


def fbar(dirs, L=40, seed0=1000):
    fs = np.array([halfspace_ratio(fold(dirs[i]), L=L, seed=seed0 + i,
                                   nrep=1)[0] for i in range(len(dirs))])
    return fs


def kulpa3d(ndir=32000, L=40, seed=11):
    print("=" * 78)
    print("f(u) OF THE DIGITAL HALF-SPACE AND THE 3D ANALOGUE OF KULPA'S CONST")
    print("=" * 78)
    print("f(u) = binary-MC area / true area of a half-space of")
    print("normal u. In 2D the analogue is f2(t) = cos t + (sqrt2 - 1) sin t.")
    print()
    named = [((0, 0, 1), "axis       (0,0,1)"),
             ((1, 0, 1), "face-diag  (1,0,1)/sqrt2"),
             ((1, 1, 1), "cube-diag  (1,1,1)/sqrt3"),
             ((1, 1, 2), "           (1,1,2)/sqrt6"),
             ((1, 1, 3), "           (1,1,3)/sqrt11"),
             ((1, 2, 3), "           (1,2,3)/sqrt14"),
             ((2, 3, 5), "           (2,3,5)/sqrt38")]
    for u, lab in named:
        f, s = halfspace_ratio(fold(u), L=64, nrep=4)
        print("  %-28s f = %.5f   (sd across phases %.1e)" % (lab, f, s))
    print("  The THREE symmetry directions give f = 1 exactly; the bias")
    print("  appears between them. It is the structure of f2 in 2D, which is 1")
    print("  at 0 and at 45 degrees and rises to 1.0824 at 22.5.")
    print()

    t0 = time.time()
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(ndir, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    fs = fbar(v, L=L, seed0=1000)
    m, se = fs.mean(), fs.std(ddof=1) / np.sqrt(ndir)
    print("  mean over %d uniform directions on S^2 (L = %d):" % (ndir, L))
    print("     fbar_3D = %.6f +- %.6f    [min %.4f, max %.4f]"
          % (m, se, fs.min(), fs.max()))
    print("     -> predicted asymptotic bias for the SPHERE: %+.4f +- %.4f pp"
          % (100 * (m - 1), 100 * se))
    print()
    print("  in 2D:  fbar_2D = (4/pi)(2 sqrt2 - 2) = %.6f  (+%.4f %%)"
          % ((4 / np.pi) * (2 * np.sqrt(2) - 2),
             100 * ((4 / np.pi) * (2 * np.sqrt(2) - 2) - 1)))
    print("  NO closed form is known for fbar_3D; the value above is")
    print("  measured, not derived.")
    print("  (%.0f s)" % (time.time() - t0))
    print()

    # prediction for a body whose normal distribution is NOT uniform
    a, b, c = 1.3, 1.0, 0.7
    nrm, L1 = ellipsoid_normal_sample(a, b, c, ndir, seed=5)
    fe = fbar(nrm, L=L, seed0=7000)
    me, see = fe.mean(), fe.std(ddof=1) / np.sqrt(ndir)
    print("  PREDICTION for the ellipsoid 1.3x1.0x0.7 (normals NOT uniform)")
    print("     fbar over its normals = %.6f +- %.6f -> %+.4f +- %.4f pp"
          % (me, see, 100 * (me - 1), 100 * see))
    print("     analytic control of the method: for FACE counting f is")
    print("     exactly |ux|+|uy|+|uz|, whose mean over the ellipsoid's")
    print("     normals is %.6f -> %+.4f %% (sphere: 3/2 -> +50 %% exact)"
          % (L1, 100 * (L1 - 1)))
    return fs, fe


def sphere_plateau(ns=(400, 500, 600, 700, 800, 900, 1000, 1100, 1200)):
    """The MC-bin bias does not drop: it is averaged over n to separate the
    plateau from the grid's phase fluctuation."""
    T = sphere_area(1.0)
    print("=" * 78)
    print("THE MC-bin PLATEAU ON THE SPHERE (it is not slow convergence)")
    print("=" * 78)
    vals = []
    for n in ns:
        r = estimate_all("sphere", n, want=("mcbin",))
        e = 100 * (r["mcbin"] / T - 1)
        vals.append(e)
        print("  n = %5d   %+.5f %%" % (n, e))
        sys.stdout.flush()
    v = np.array(vals)
    print("  plateau = %.5f pp   sd %.5f   se %.5f   (no trend in n)"
          % (v.mean(), v.std(ddof=1), v.std(ddof=1) / np.sqrt(len(v))))
    print()
    return v


def radiomics(ns=(200, 400, 800)):
    """What happens to the IBSI / PyRadiomics shape descriptors.

    The VOLUME by voxel counting IS multigrid-convergent, so the area bias
    passes entirely into any descriptor that divides by it.
    IBSI sphericity: (36 pi V^2)^(1/3) / A, which is 1 for a sphere.
    Surface-to-volume ratio: A / V."""
    a, b, c = 1.3, 1.0, 0.7
    cases = [("sphere r=1", "sphere", None, sphere_area(1.0),
              4 * np.pi / 3),
             ("ellipsoid 1.3x1.0x0.7", "ellipsoid", (a, b, c),
              ellipsoid_area_legendre(a, b, c), 4 * np.pi * a * b * c / 3)]
    print("=" * 78)
    print("IBSI / PyRadiomics SHAPE DESCRIPTORS")
    print("=" * 78)
    for name, kind, par, A_true, V_true in cases:
        sph_true = (36 * np.pi * V_true ** 2) ** (1 / 3) / A_true
        print("%s   A = %.6f  V = %.6f  exact sphericity = %.6f"
              % (name, A_true, V_true, sph_true))
        print("  %6s %12s %10s %12s %10s %12s %10s"
              % ("n", "V voxel", "err %", "spher MC-bin", "err %",
                 "spher MC-0", "err %"))
        for n in ns:
            g = np.linspace(-HALF, HALF, n)
            h = float(g[1] - g[0])
            cnt = 0
            k0 = 0
            blk = auto_block(n)
            while k0 < n:
                k1 = min(k0 + blk, n)
                cnt += int(np.count_nonzero(
                    make_field(kind, g, g, g[k0:k1], par=par) > 0))
                k0 = k1
            V = cnt * h ** 3
            r = estimate_all(kind, n, par=par)
            num = (36 * np.pi * V ** 2) ** (1 / 3)
            s_bin, s_0 = num / r["mcbin"], num / r["mc0"]
            print("  %6d %12.6f %+10.4f %12.6f %+10.4f %12.6f %+10.4f"
                  % (n, V, 100 * (V / V_true - 1), s_bin,
                     100 * (s_bin / sph_true - 1), s_0,
                     100 * (s_0 / sph_true - 1)))
            sys.stdout.flush()
        print()


# ==========================================================================
# 5. A CONVERGENT estimator that runs on the binary mask
# ==========================================================================
#
# MC-0 converges but needs the signed field: it is not a replacement for
# PyRadiomics, it is the 3D analogue of the paper's zero level reference.
# The 3D analogue of DSS -- convergent and on the MASK -- is the integral
# invariant estimator: the normal at each surfel is estimated from the
# displacement of the centroid of X inside a ball of radius r = k h^(1/3),
# and the area follows from
#     S = h^2 * sum_surfels |n_est . e_j| ,
# which in the continuum gives sum_j int n_j^2 dS = S. References:
# Coeurjolly, Lachaud & Levallois; Lachaud & Thibert. The theoretical
# convergence rate is O(h^(1/3)), slow, and that shows in the numbers.

def _ball_offsets(rv):
    R = int(np.ceil(rv))
    a = np.arange(-R, R + 1)
    dx, dy, dz = np.meshgrid(a, a, a, indexing="ij")
    m = (dx ** 2 + dy ** 2 + dz ** 2) <= rv * rv
    return (dx[m].astype(np.int64), dy[m].astype(np.int64),
            dz[m].astype(np.int64))


def ii_area(kind, n, par=None, Rm=None, k=0.25, half=HALF, chunk=None):
    """Area by integral invariant on the binary mask."""
    g = np.linspace(-half, half, n)
    h = float(g[1] - g[0])
    X = make_field(kind, g, g, g, par=par, Rm=Rm) > 0
    rv = max(2.0, k * (h ** (1.0 / 3.0)) / h)
    ox, oy, oz = _ball_offsets(rv)
    # the index block costs chunk * |ball| * 8 bytes per axis; it is capped at
    # ~4e6 entries so large n does not push the machine into swap
    if chunk is None:
        chunk = max(500, int(4e6 / len(ox)))
    Xp = np.pad(X, 1)
    tot = 0.0
    for j in (0, 1, 2):
        for sgn in (+1, -1):
            m = Xp & (~np.roll(Xp, -sgn, axis=j))
            ii = np.array(np.nonzero(m[1:-1, 1:-1, 1:-1]))
            N = ii.shape[1]
            for c0 in range(0, N, chunk):
                c = ii[:, c0:c0 + chunk]
                px = np.clip(c[0][:, None] + ox[None, :], 0, n - 1)
                py = np.clip(c[1][:, None] + oy[None, :], 0, n - 1)
                pz = np.clip(c[2][:, None] + oz[None, :], 0, n - 1)
                v = X[px, py, pz]
                cnt = v.sum(axis=1).astype(float)
                cnt[cnt == 0] = 1.0
                nn = np.stack([-(v * ox[None, :]).sum(axis=1) / cnt,
                               -(v * oy[None, :]).sum(axis=1) / cnt,
                               -(v * oz[None, :]).sum(axis=1) / cnt], axis=1)
                L = np.linalg.norm(nn, axis=1)
                L[L == 0] = 1.0
                tot += float(np.abs(nn[:, j] / L).sum())
    return tot * h * h


def convergent_on_mask(ns=(100, 200, 400), k=0.25):
    # n = 800 was measured partially and costs >20 min per body (1.2e6 surfels
    # x 4100 ball voxels, dominated by cache misses); it is not part of the
    # default run.
    a, b, c = 1.3, 1.0, 0.7
    print("=" * 78)
    print("A CONVERGENT ESTIMATOR THAT DOES RUN ON THE BINARY MASK")
    print("   integral invariant, r = %.2f h^(1/3)" % k)
    print("=" * 78)
    for name, kind, par, exact in [
            ("sphere r=1", "sphere", None, sphere_area(1.0)),
            ("ellipsoid 1.3x1.0x0.7", "ellipsoid", (a, b, c),
             ellipsoid_area_legendre(a, b, c))]:
        print("  %s   exact %.6f" % (name, exact))
        prev = None
        for n in ns:
            t0 = time.time()
            A = ii_area(kind, n, par=par, k=k)
            e = 100 * (A / exact - 1)
            ordo = "" if prev is None else "  order %.2f" % np.log2(
                abs(prev / e))
            print("    n = %4d   %12.6f   %+8.4f %%%s   (%.0fs)"
                  % (n, A, e, ordo, time.time() - t0))
            prev = e
            sys.stdout.flush()
        print()
    # orientation spread of the cube, the hard test
    ANGS = [float(x) for x in range(0, 50, 5)]
    es = []
    for ang in ANGS:
        es.append(100 * (ii_area("cube", 200, par=0.8, k=k,
                                 Rm=rot_matrix((0, 0, 1), ang)) / 15.36 - 1))
    es = np.array(es)
    print("  cube side 1.6, rotating about z, n = 200:")
    print("    %s" % " ".join("%+.3f" % v for v in es))
    print("    spread %.4f pp  (MC-bin gives 5.25 pp at n=400)"
          % (es.max() - es.min()))
    print()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "std"
    selftest()
    if mode == "quick":
        convergence([50, 100, 200, 400])
        orientation(n=200)
        sphere_plateau((200, 300, 400))
        radiomics((100, 200, 400))
        convergent_on_mask((100, 200))
        kulpa3d(ndir=2000)
    elif mode == "full":
        convergence([50, 100, 200, 400, 800, 1600])
        orientation(n=400)
        sphere_plateau()
        radiomics((200, 400, 800, 1600))
        convergent_on_mask((100, 200, 400))
        kulpa3d(ndir=32000)
    else:
        convergence([50, 100, 200, 400, 800])
        orientation(n=400)
        sphere_plateau((400, 600, 800))
        radiomics((200, 400, 800))
        convergent_on_mask((100, 200, 400))
        kulpa3d(ndir=8000)


if __name__ == "__main__":
    main()
