#!/usr/bin/env python3
"""
Numerical validation of the prototype.

    python3 validate.py

1. Unit tests of the FEM operators against exact solutions.
2. Area convergence under mesh refinement.
3. Comparison of the three geodesic methods against the ground truth.
4. Sensitivity to the contour simplification tolerance.

None of this leans on the pipeline: these are independent checks that must run
green before any number from the demo is believed.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.sparse.linalg as spla
import triangle as tr
from scipy import ndimage
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ellipe
from skimage import measure

from foliar import fem, infection, latency, meshing, pipeline, synth

OK, BAD = "\033[92mOK\033[0m", "\033[91mFAIL\033[0m"
_fails = []


def check(name, cond, detail=""):
    print(f"  [{OK if cond else BAD}] {name}   {detail}")
    if not cond:
        _fails.append(name)


def unit_square(n=40, L=1.0):
    pts = ([[L * i / n, 0] for i in range(n)] + [[L, L * i / n] for i in range(n)] +
           [[L - L * i / n, L] for i in range(n)] + [[0, L - L * i / n] for i in range(n)])
    N = len(pts)
    o = tr.triangulate(dict(vertices=np.array(pts),
                            segments=np.array([[i, (i + 1) % N] for i in range(N)])),
                       # :.16f and not :g -- with n >= 57 the general format emits
                       # scientific notation, Triangle's flag parser does not read
                       # the exponent, discards the area constraint SILENTLY and
                       # returns a mesh coarser than the previous level's
                       f"pq30a{(L / n) ** 2 * 0.5:.16f}")
    V = np.column_stack([o["vertices"], np.zeros(len(o["vertices"]))])
    return V, o["triangles"]


# --------------------------------------------------------------------------
def test_operators():
    print("\n\033[1m1 · FEM operators against exact solutions\033[0m")
    V, F = unit_square(40)
    G, A = fem.shape_gradients(V, F)

    gu = np.einsum("ta,tai->ti", V[F][:, :, 0], G)
    check("grad(x) = (1,0,0) on every triangle",
          np.abs(gu - np.array([1.0, 0, 0])).max() < 1e-10,
          f"max err {np.abs(gu - np.array([1.0, 0, 0])).max():.2e}")

    check("area of the unit square = 1", abs(A.sum() - 1.0) < 1e-12,
          f"{A.sum():.15f}")

    S = fem.stiffness_matrix(V, F)
    check("S @ 1 = 0 (constant function has no energy)",
          np.abs(S @ np.ones(len(V))).max() < 1e-10,
          f"max {np.abs(S @ np.ones(len(V))).max():.2e}")

    M = fem.mass_matrix(V, F)
    check("mass matrix sums to the area", abs(M.sum() - 1.0) < 1e-12,
          f"{M.sum():.15f}")
    Ml = fem.mass_matrix(V, F, lumped=True)
    check("lumped mass preserves the area", abs(Ml.sum() - 1.0) < 1e-12)

    u = V[:, 0]
    check("Dirichlet energy of u=x over [0,1]^2 = 1",
          abs(u @ (S @ u) - 1.0) < 1e-8, f"{u @ (S @ u):.12f}")

    check("perimeter of the unit square = 4",
          abs(fem.boundary_length(V, F) - 4.0) < 1e-10,
          f"{fem.boundary_length(V, F):.12f}")

    # ---- exact geodesic: source at x=0 => phi = x
    src = np.flatnonzero(V[:, 0] < 1e-9)
    print("\n  geodesic distance on the square (exact: phi = x)")
    # tolerance per method: the metrication bias of Dijkstra on edges is known
    # and is precisely why the default is the 2-ring
    for meth, kw, tol in [("dijkstra", {}, 0.060), ("dijkstra2", {}, 0.010),
                          ("heat", dict(m=0.5), 0.020)]:
        d = fem.geodesic(V, F, src, method=meth, **kw)
        err = np.abs(d - V[:, 0])
        check(f"    {meth:10s} mean error < {100 * tol:.0f} % of the side",
              err.mean() < tol,
              f"mean {err.mean():.5f}  max {err.max():.5f}  phi_max {d.max():.5f}")


# --------------------------------------------------------------------------
def test_convergence(sample):
    """
    Two different things that used to be mixed into a single check.

    (a) The area of the FLAT mesh is exactly the area of the polygon and is
        INVARIANT under interior refinement: refining subdivides triangles, it
        does not change the area of a polygon. That is an exact identity, not a
        convergence, and it has to be tested as such. The previous version of
        this test presented it as convergence, which meant it could not fail.

    (b) The area of the LIFTED mesh does converge, because flat triangles
        approximate a cylinder and the chord error decays as O(h^2). That is the
        only legitimate convergence test here.
    """
    print("\n\033[1m2 · Area: exact identity in the plane, convergence once lifted\033[0m")
    gt = sample.truth
    px, org = sample.px_per_cm, sample.origin_cm
    bp = meshing.mask_to_polygon(sample.mask_blade, org, px)
    lp = meshing.mask_to_polygon(sample.mask_lesion, org, px, 0.18)
    mp = meshing.mask_to_polygon(sample.mask_midrib, org, px)

    print(f"  {'edge':>8} {'triangles':>11} {'flat area':>14} {'lifted area':>16} "
          f"{'change':>11}")
    flat, lift = [], []
    prev = None
    for edge in [2.0, 1.0, 0.5]:
        m = meshing.build_mesh(bp, lp, mp, px, target_edge_cm=edge,
                               lesion_edge_cm=min(edge, 0.45))
        ml = meshing.lift_to_curl(m, sample.curl_radius_cm)
        a_f = fem.surface_area(m.V, m.F)
        a_l = fem.surface_area(ml.V, ml.F)
        flat.append(a_f)
        lift.append(a_l)
        ch = "" if prev is None else f"{a_l - prev:+11.5f}"
        print(f"  {edge:8.2f} {len(m.F):11d} {a_f:14.5f} {a_l:16.5f} {ch:>11}")
        prev = a_l

    check("the flat mesh area is EXACTLY the polygon area",
          abs(flat[0] / bp.area - 1.0) < 1e-12, f"{flat[0]:.6f} against {bp.area:.6f}")
    check("and it is invariant under refinement (identity, not convergence)",
          max(abs(a / flat[0] - 1.0) for a in flat) < 1e-12,
          f"spread {max(flat) - min(flat):.3e} cm2")

    d1, d2 = abs(lift[1] - lift[0]), abs(lift[2] - lift[1])
    check("the LIFTED area converges under refinement (the change shrinks)",
          d2 < 0.5 * d1, f"change {d1:.4f} -> {d2:.4f} cm2 (factor {d1/max(d2,1e-12):.1f})")
    rel = 100.0 * d2 / lift[2]
    check("and the last relative change is below 0.002 %", rel < 0.002,
          f"{rel:.5f} %")
    e = 100.0 * (lift[2] / gt.blade_area_cm2 - 1.0)
    print(f"  the residual against the intrinsic truth ({e:+.4f} %) is the contour\n"
          f"  digitisation error, which does NOT drop by refining the mesh")


# --------------------------------------------------------------------------
def test_geodesics(sample):
    print("\n\033[1m3 · Geodesics against the reference eikonal field\033[0m")
    g = sample.truth._grid
    R = sample.curl_radius_cm
    px, org = sample.px_per_cm, sample.origin_cm

    blade = meshing.mask_to_polygon(sample.mask_blade, org, px)
    lesion = meshing.mask_to_polygon(sample.mask_lesion, org, px, 0.18)
    midrib = meshing.mask_to_polygon(sample.mask_midrib, org, px)
    mesh = meshing.build_mesh(blade, lesion, midrib, px)
    ml = meshing.lift_to_curl(mesh, R)
    src = np.array(sorted(set(mesh.vertices_in(meshing.REG_MIDRIB).tolist()) &
                          set(mesh.vertices_in(meshing.REG_HEALTHY).tolist())), dtype=int)

    D = g["dist"].copy()
    idx = ndimage.distance_transform_edt(~np.isfinite(D), return_distances=False,
                                         return_indices=True)
    itp = RegularGridInterpolator((g["v"], g["u"]), D[tuple(idx)],
                                  bounds_error=False, fill_value=None)
    V2 = mesh.V[:, :2]
    ref = itp(np.column_stack([R * np.arcsin(np.clip(V2[:, 1] / R, -1, 1)), V2[:, 0]]))

    A = fem.triangle_areas(ml.V, ml.F)
    les = ml.attr == meshing.REG_LESION

    def lesion_mean(dn):
        dt = fem.interpolate_to_triangles(dn, ml.F)
        k = les & np.isfinite(dt)
        return float((dt[k] * A[k]).sum() / A[k].sum())

    base = lesion_mean(ref)
    print(f"  reference (eikonal on a fine intrinsic grid): {base:.3f} cm")
    print(f"  {'method':>12} {'lesion mean':>16} {'err %':>9} {'RMSE cm':>9} {'max cm':>8}")
    best = None
    # the heat method runs at its MEASURED optimum (m = 0.30, the default), not at
    # an arbitrary value: comparing it badly tuned would be a straw man
    heat_rmse = None
    for meth, kw in [("dijkstra", {}), ("dijkstra2", {}), ("heat", {})]:
        d = fem.geodesic(ml.V, ml.F, src, method=meth, **kw)
        ok = np.isfinite(d) & (ref > 5.0)
        rmse = float(np.sqrt(np.mean((d[ok] - ref[ok]) ** 2)))
        e = 100.0 * (lesion_mean(d) / base - 1.0)
        print(f"  {meth:>12} {lesion_mean(d):16.3f} {e:+9.2f} {rmse:9.3f} "
              f"{np.nanmax(d):8.2f}")
        if meth == "dijkstra2":
            best = (abs(e), rmse)
        if meth == "heat":
            heat_rmse = rmse
    check("default method (dijkstra2) within 0.6 % and RMSE < 0.15 cm",
          best[0] < 0.6 and best[1] < 0.15, f"err {best[0]:.2f} %, RMSE {best[1]:.3f} cm")
    check("and it beats the heat method even with heat at its measured optimum",
          best[1] < 0.5 * heat_rmse,
          f"{best[1]:.3f} against {heat_rmse:.3f} cm (factor {heat_rmse/best[1]:.1f})")


# --------------------------------------------------------------------------
def test_simplification(sample):
    print("\n\033[1m4 · Contour simplification tolerance\033[0m")
    gt = sample.truth
    print(f"  {'simplify_px':>12} {'triangles':>11} {'min ang':>8} {'-S cap':>8} "
          f"{'area err':>10} {'perimeter err':>14} {'severity err':>14}")
    rows = []
    for sp in [1.2, 0.6, 0.30, 0.15]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # the warning is reported in the table
            r = pipeline.run(sample, curl_radius_cm=sample.curl_radius_cm,
                             scale_cm=19.0, simplify_px=sp, geodesic_method="dijkstra")
        C = r.level_C
        ea = 100 * (C["area_cm2"] / gt.blade_area_cm2 - 1)
        ep = 100 * (C["perimeter_cm"] / gt.perimeter_cm - 1)
        es = 100 * (C["severity_pct"] / gt.severity_pct - 1)
        rows.append((sp, ea, ep, es))
        cap = "YES" if r.mesh_flat.steiner_capped else "no"
        print(f"  {sp:12.2f} {r.quality['n_triangles']:11d} "
              f"{r.mesh_flat.min_angle_deg:8.2f} {cap:>8} {ea:+9.3f}% "
              f"{ep:+13.3f}% {es:+13.3f}%")
    print("  The 1.20 px row is not a geometric explosion from slivers: it is the")
    print("  Steiner cap firing and returning a degenerate mesh. The 'min ang'")
    print("  column gives it away.")
    a_span = max(abs(r[1]) for r in rows)
    p_span = max(abs(r[2]) for r in rows)
    check("the area is insensitive to the tolerance (< 0.5 %)", a_span < 0.5,
          f"worst {a_span:.3f} %")
    check("the perimeter is NOT: it is the parameter that sets the tolerance",
          p_span > 1.0, f"range {p_span:.2f} %")
    dflt = [r for r in rows if r[0] == 0.30][0]
    # This check used to demand |error| < 0.5 % and passed with -0.014 %. It was
    # green through CANCELLATION OF BIASES, not accuracy: the ground truth used
    # the same binary marching squares as mask_to_polygon, so the two +4.5 %
    # biases subtracted. With the corrected ground truth (zero level contour, see
    # section 9) the real residual appears and it is +4.5 %.
    #
    # It is an OPEN defect of the input path, not of the mesh: extracting a
    # contour from a BINARY mask with marching squares carries that bias, and the
    # binary mask is the real data of a segmented photo. The value it has today is
    # asserted, with a band, so that a change moves it and shows up.
    check("the level C perimeter carries the binary contour bias, "
          "+4.5 % +- 1", abs(dflt[2] - 4.5) < 1.0, f"{dflt[2]:+.3f} %")
    print("  That +4.5 % is the bias of marching squares on a binary mask in")
    print("  mask_to_polygon. Measured, a Gaussian smoothing of the blade before")
    print("  extracting the contour brings it down to -1.0 %, but it introduces a")
    print("  sigma that drifts with resolution: documented and NOT enabled by default.")


# --------------------------------------------------------------------------
def structured_grid(nx: int, ny: int, Lx: float, Ly: float):
    """
    Structured grid of right triangles over [0,Lx] x [0,Ly].

    Used on purpose instead of Triangle: on this grid the angles opposite each
    interior edge add up to 90 or 180 degrees, so the cotangent weights are
    non-negative BY CONSTRUCTION and the hypothesis of the Ciarlet-Raviart
    theorem holds exactly. On a Triangle mesh it does not hold, and the test
    would stop being an exact property.
    """
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    X, Y = np.meshgrid(x, y)
    V = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    idx = lambda i, j: j * (nx + 1) + i          # noqa: E731
    F = []
    for j in range(ny):
        for i in range(nx):
            F.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            F.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return V, np.asarray(F, int)


def test_infection(sample):
    print("\n\033[1m5 · Infection PDE: exact identities\033[0m")

    # ---- 5a. algebraic identities of the anisotropic operator ----
    # The hint has to be explicit: without it, triangle_frames falls back to the
    # first edge, which on this grid alternates between +x and +y, and the tensor
    # would stop being axis aligned.
    V, F = structured_grid(24, 16, 6.0, 4.0)
    xhint = np.tile(np.array([1.0, 0.0, 0.0]), (len(F), 1))
    e1, e2, N = infection.triangle_frames(V, F, e1_hint=xhint)
    S = fem.stiffness_matrix(V, F)
    c = 2.7
    K = infection.stiffness_anisotropic(V, F, infection.diffusion_tensor(e1, e2, c, c))
    err = np.abs((K - c * S).toarray()).max()
    check("K_D with D = c*P reduces to c*S entry by entry", err < 1e-13,
          f"max {err:.2e}")

    Dt = infection.diffusion_tensor(e1, e2, 3.0, 0.4)
    Kd = infection.stiffness_anisotropic(V, F, Dt)
    r = np.abs(Kd @ np.ones(len(V))).max()
    check("K_D @ 1 = 0 for arbitrary anisotropic D", r < 1e-12 * np.abs(Kd).max(),
          f"max {r:.2e}")
    asym = np.abs((Kd - Kd.T).toarray()).max()
    check("K_D symmetric", asym < 1e-14, f"max {asym:.2e}")

    # Dirichlet energy of u = x with D = diag(Dv, Du): exactly Dv * area
    Dv, Du = 3.0, 0.4
    u = V[:, 0]
    e = float(u @ (Kd @ u))
    exact = Dv * 6.0 * 4.0
    check("energy of u=x with the tensor = D_v * area", abs(e / exact - 1) < 1e-10,
          f"{e:.6f} against {exact:.6f}")

    # ---- 5b. maximum principle on a structured grid ----
    # Theorem (Ciarlet-Raviart): if K has no positive off-diagonal entries,
    # (M_lump + s K) is an M-matrix and preserves [0,1] for every s > 0.
    ML = fem.mass_matrix(V, F, lumped=True)
    Ko = Kd.tocoo()
    npos = int(((Ko.data > 1e-14) & (Ko.row != Ko.col)).sum())
    rng = np.random.default_rng(0)
    u0 = (rng.random(len(V)) < 0.5).astype(float)
    worst = 0.0
    for s in np.geomspace(1e-4, 1e2, 25):
        u1 = spla.spsolve((ML + s * Kd).tocsc(), ML @ u0)
        worst = min(worst, u1.min())
    check("no positive off-diagonals on the structured grid", npos == 0,
          f"{npos} entries")
    check("preserves [0,1] across 25 time scales (lumped mass)",
          worst >= -1e-12, f"min {worst:.2e}")

    # ---- 5c. the second moment recovers the tensor ----
    # For -div(D grad Phi) + Phi = delta, Fourier gives <x_i x_j> = 2 D_ij exactly.
    # Domain of 30 decay lengths on each axis and h = 0.25 (i.e. h/ell_u = 0.62);
    # measured, with this the identity is recovered to -0.00 %.
    ell_v, ell_u = 1.0, 0.4
    Lx, Ly = 30.0, 12.0
    Vg, Fg = structured_grid(240, 96, 2 * Lx, 2 * Ly)
    Vg[:, 0] -= Lx
    Vg[:, 1] -= Ly
    ghint = np.tile(np.array([1.0, 0.0, 0.0]), (len(Fg), 1))
    eg1, eg2, _ = infection.triangle_frames(Vg, Fg, e1_hint=ghint)
    Kg = infection.stiffness_anisotropic(
        Vg, Fg, infection.diffusion_tensor(eg1, eg2, ell_v ** 2, ell_u ** 2))
    MLg = fem.mass_matrix(Vg, Fg, lumped=True)
    b = np.zeros(len(Vg))
    b[int(np.argmin(np.linalg.norm(Vg[:, :2], axis=1)))] = 1.0
    phi = spla.spsolve((MLg + Kg).tocsc(), b)
    w = MLg.diagonal() * phi
    mx = float((w * Vg[:, 0] ** 2).sum() / w.sum())
    my = float((w * Vg[:, 1] ** 2).sum() / w.sum())
    check("second moment <x^2> = 2 ell_v^2",
          abs(mx / (2 * ell_v ** 2) - 1) < 1e-3, f"{mx:.5f} against {2*ell_v**2:.5f}")
    check("second moment <y^2> = 2 ell_u^2",
          abs(my / (2 * ell_u ** 2) - 1) < 1e-3, f"{my:.5f} against {2*ell_u**2:.5f}")
    # The ratio is what fixes the calibration rule aniso_ratio = elongation^2
    check("the tensor ratio is recovered from the moments",
          abs((mx / my) / (ell_v ** 2 / ell_u ** 2) - 1) < 1e-3,
          f"{mx/my:.4f} against {ell_v**2/ell_u**2:.4f}")

    # ---- 5d. the exponential integrator is exact and bounded ----
    rho = np.abs(rng.normal(0.4, 0.3, 500))
    I0 = rng.random(500)
    dt, nst = 0.05, 40
    I = I0.copy()
    for _ in range(nst):
        I = 1.0 - (1.0 - I) * np.exp(-dt * rho)
    exact_I = 1.0 - (1.0 - I0) * np.exp(-(nst * dt) * rho)
    check("composition of N steps = one step of N*dt",
          np.abs(I - exact_I).max() < 1e-12, f"max {np.abs(I - exact_I).max():.2e}")
    check("0 <= I <= 1 and monotone", I.min() >= 0.0 and I.max() <= 1.0
          and (I >= I0 - 1e-15).all(), f"[{I.min():.3f}, {I.max():.3f}]")

    print("\n\033[1m6 · Infection PDE on the leaf\033[0m")
    px, org = sample.px_per_cm, sample.origin_cm
    mesh = meshing.build_mesh(meshing.mask_to_polygon(sample.mask_blade, org, px),
                              meshing.mask_to_polygon(sample.mask_lesion, org, px, 0.18),
                              meshing.mask_to_polygon(sample.mask_midrib, org, px), px)
    ml = meshing.lift_to_curl(mesh, sample.curl_radius_cm)
    src = np.array(sorted(set(mesh.vertices_in(meshing.REG_MIDRIB).tolist()) &
                          set(mesh.vertices_in(meshing.REG_HEALTHY).tolist())), int)
    frame = infection.vein_frame(ml, src)

    # ---- 6a. the vein frame against the exact cylinder ----
    # By the isometry of the embedding, the intrinsic v direction on the cylinder
    # is exactly sign(v)*(0, cos(v/R), sin(v/R)). That is analytic ground truth.
    R = sample.curl_radius_cm
    C = ml.V[ml.F].mean(axis=1)
    vv = R * np.arcsin(np.clip(C[:, 1] / R, -1, 1))
    exact_dir = np.column_stack([np.zeros(len(C)), np.cos(vv / R),
                                 np.sin(vv / R)]) * np.sign(vv)[:, None]
    dot = np.einsum("ti,ti->t", frame.e_v, exact_dir)
    ang = np.degrees(np.arccos(np.clip(np.abs(dot), -1, 1)))
    A = fem.triangle_areas(ml.V, ml.F)
    far = np.abs(vv) > 2.0                       # away from the midrib, where v is defined
    ang_mean = float(np.average(ang[far], weights=A[far]))
    check("vein frame against the cylinder: mean error < 1.5 degrees",
          ang_mean < 1.5, f"{ang_mean:.3f} degrees  (max {ang[far].max():.2f})")
    check("zero triangles with a flipped frame",
          int(((dot < 0) & far).sum()) == 0, f"{int(((dot < 0) & far).sum())}")
    ortho = np.abs(np.einsum("ti,ti->t", frame.e_v, frame.normal)).max()
    check("e_v tangent to the triangle (|e_v . N| = 0)", ortho < 1e-12,
          f"max {ortho:.2e}")

    # ---- 6b. transport identities on the leaf ----
    op = infection.InfectionOperator(ml, frame, infection.InfectionModel())
    one = np.ones(len(ml.V))
    err1 = np.abs(op.transport(one) - 1.0).max()
    check("Phi = 1 when I = 1 (preservation of constants)", err1 < 1e-12,
          f"max {err1:.2e}")
    I0 = infection.initial_infection(ml)
    ratio = (op.lump * op.transport(I0)).sum() / (op.lump * I0).sum()
    check("conservation 1'M Phi = 1'M I (zero flux on the whole boundary)",
          abs(ratio - 1.0) < 1e-12, f"{ratio:.15f}")

    sev0 = op.severity_pct(I0)
    sev_mesh = 100.0 * fem.surface_area(ml.V, ml.F, ml.attr == meshing.REG_LESION) \
        / fem.surface_area(ml.V, ml.F)
    check("the initial condition introduces no area bias",
          abs(sev0 / sev_mesh - 1) < 1e-3, f"{sev0:.4f} % against {sev_mesh:.4f} %")

    # ---- 6c. bounds on I over the real mesh, with no clipping ----
    _, sts = infection.simulate(ml, frame, infection.InfectionModel(gamma=1.0),
                                I0, 42.0, dt_day=0.5, record_every_day=1e9)
    fin = sts[-1]
    check("0 <= I <= 1 after 84 steps on the real mesh, NO clipping",
          fin.I.min() >= 0.0 and fin.I.max() <= 1.0,
          f"[{fin.I.min():.3e}, {fin.I.max():.9f}]")
    diag = op.diagnostics(I0)
    check("the Phi undershoot stays negligible", diag["reliable"],
          f"min/max = {diag['undershoot_rel']:.2e}  "
          f"({diag['n_offdiag_positive']} of {diag['n_offdiag']} positive off-diag)")

    # ---- 6d. order 2 in dt ----
    sev = {}
    for dt in [2.0, 1.0, 0.5, 0.25]:
        _, ss = infection.simulate(ml, frame, infection.InfectionModel(gamma=0.0),
                                   I0, 42.0, dt_day=dt, record_every_day=1e9)
        sev[dt] = ss[-1].severity_pct
    d1 = abs(sev[2.0] - sev[1.0])
    d2 = abs(sev[1.0] - sev[0.5])
    d3 = abs(sev[0.5] - sev[0.25])
    order = np.log2(np.mean([d1 / max(d2, 1e-30), d2 / max(d3, 1e-30)]))
    check("the predictor-corrector converges at order 2 in dt", order > 1.8,
          f"observed order {order:.2f}")

    # ---- 6e. the statistic separates the hypotheses at equal final severity ----
    # Every gamma is calibrated to the SAME final severity: otherwise changing
    # gamma changes the global rate and Psi from two runs is not comparable, one
    # would be measuring how much the lesion grew and not where.
    psi_ref = op.state(0.0, I0).psi
    target, horizon = 2.50, 21.0
    I_null, _, _ = infection.null_dilation(ml, I0, target)
    psi_null = op.wsi(I_null) / op.severity_pct(I_null)
    exc = {}
    print(f"  {'gamma':>7} {'beta0 calib':>12} {'Psi model':>11} {'excess pp':>11}")
    for g in [-1.0, 0.0, 1.0]:
        mod = infection.InfectionModel(gamma=g)
        b0 = infection.calibrate_beta0(ml, frame, mod, I0, horizon, target)
        mod = infection.InfectionModel(gamma=g, beta0_per_day=b0)
        _, ss = infection.simulate(ml, frame, mod, I0, horizon, dt_day=0.5,
                                   record_every_day=1e9)
        exc[g] = infection.excess_pp(ss[-1].psi, psi_null, psi_ref)
        print(f"  {g:+7.1f} {b0:12.5f} {ss[-1].psi:11.4f} {exc[g]:+11.2f}")
    check("the excess is monotonically increasing in gamma",
          exc[-1.0] < exc[0.0] < exc[1.0],
          f"{exc[-1.0]:+.2f} < {exc[0.0]:+.2f} < {exc[1.0]:+.2f} pp")
    check("and it separates the hypotheses by more than 10 pp",
          exc[1.0] - exc[-1.0] > 10.0, f"{exc[1.0] - exc[-1.0]:.1f} pp")
    return ml, frame, op, I0


# --------------------------------------------------------------------------
def test_latency(ml, frame, op, I0):
    """
    The latent compartment anchors the time axis to a measurable quantity.

    Everything tested here is an EXACT identity of the closed-form kernel and of
    the compartment structure, not memorised values.
    """
    print("\n\033[1m7 · Latency: the closed-form kernel\033[0m")
    lat = latency.LatencyModel()
    kap, k = lat.kappa_per_day, lat.k_stages
    print(f"  tau_E = {lat.tau_E_day:g} d, k = {k}, kappa = {kap:.5f}/d, "
          f"delay sd = {lat.sd_day:.1f} d, source weights "
          f"{np.round(lat.source_weights(), 3)}")

    # ---- 7a. semigroup: exp(A dt)^2 = exp(A 2dt), algebraic identity ----
    rho = np.concatenate([[0.0, kap, kap * (1 + 1e-12)],
                          np.geomspace(1e-8, 5.0, 200)])
    E = np.tile(np.linspace(0.05, 0.25, k), (len(rho), 1))
    I = np.full(len(rho), 0.3)
    worst = 0.0
    for dt in [0.05, 0.25, 0.5, 1.0, 2.0]:
        E1, I1 = latency.apply_kernel(E, I, *latency.latent_kernel(rho, dt, kap, k))
        E2, I2 = latency.apply_kernel(E1, I1, *latency.latent_kernel(rho, dt, kap, k))
        Eb, Ib = latency.apply_kernel(E, I, *latency.latent_kernel(rho, 2 * dt, kap, k))
        worst = max(worst, np.abs(E2 - Eb).max(), np.abs(I2 - Ib).max())
    check("exp(A dt)^2 = exp(A 2dt) over 1015 combinations of (rho, dt)",
          worst < 1e-12, f"max {worst:.2e}")

    # ---- 7b. the confluence rho = kappa, where the naive formula breaks ----
    # There the integral solves in closed form: W_j = (kappa t)^j e^{-kappa t}/j!
    worst = 0.0
    for kk in [1, 2, 3, 4, 6]:
        for tau in [1.0, 38.0, 300.0]:
            kp = kk / tau
            for dt in [0.05, 0.5, 2.0]:
                _, W, _ = latency.latent_kernel(np.array([kp]), dt, kp, kk)
                exact = np.array([np.exp(-kp * dt) * (kp * dt) ** j / math.factorial(j)
                                  for j in range(1, kk + 1)])
                worst = max(worst, np.abs(W[0] - exact).max())
    check("at the confluence rho = kappa the kernel gives the exact closed form",
          worst < 1e-14, f"max {worst:.2e}")

    # ---- 7c. the delay is exactly Erlang(k, k/tau_E) ----
    # This is what ties tau_E to "mean days from infection to visible symptom".
    worst = 0.0
    for kk in [1, 2, 3, 4]:
        for tau in [10.0, 38.0]:
            kp = kk / tau
            E = np.zeros((1, kk))
            E[0, 0] = 1.0
            Ic = np.zeros(1)
            zero = np.zeros(1)
            for step_i in range(200):
                E, Ic = latency.apply_kernel(E, Ic,
                                             *latency.latent_kernel(zero, 0.5, kp, kk))
                t = (step_i + 1) * 0.5
                cdf = 1.0 - sum(np.exp(-kp * t) * (kp * t) ** mm / math.factorial(mm)
                                for mm in range(kk))
                worst = max(worst, abs(float(Ic[0]) - cdf))
    check("the implemented delay is exactly Erlang(k, k/tau_E)",
          worst < 1e-13, f"max {worst:.2e}")

    print("\n\033[1m8 · Latency on the leaf\033[0m")
    lop = latency.LatentOperator(ml, frame, infection.InfectionModel(gamma=1.0), lat)
    r_obs = latency.growth_rate_from_pair(1.2370, 2.50, 21.0)
    E0, info = latency.initial_latent(lop, I0, r_obs)
    st0 = lop.state(0.0, E0, I0)

    # ---- 8a. the initial latent mass is what the exponential phase says ----
    ratio = st0.latent_pct / st0.severity_pct
    exact = (1.0 + r_obs / kap) ** k - 1.0
    check("the initial latent pool is (1 + r/kappa)^k - 1 times the visible one",
          abs(ratio / exact - 1) < 1e-6, f"{ratio:.6f} against {exact:.6f}")
    check("and the target mass is placed in full", abs(info["deficit_rel"]) < 1e-10,
          f"relative deficit {info['deficit_rel']:.2e}, "
          f"{info['n_saturated']} saturated nodes")

    # ---- 8b. bounds and conservation after 84 steps, NO clipping ----
    _, sts = latency.simulate_latent(ml, frame, infection.InfectionModel(gamma=1.0),
                                     lat, I0, E0, 42.0, dt_day=0.5,
                                     record_every_day=1e9)
    f = sts[-1]
    tot = f.S + f.E.sum(axis=1) + f.I
    check("S + sum E + I = 1 after 84 steps (stochastic transition matrix)",
          np.abs(tot - 1.0).max() < 1e-13, f"max deviation {np.abs(tot-1).max():.2e}")
    check("no negative compartment, without clipping",
          f.E.min() > -1e-13 and f.I.min() > -1e-13,
          f"min E {f.E.min():.2e}, min I {f.I.min():.2e}")
    check("I remains monotone (dI/dt = kappa E_k >= 0)",
          (f.I >= sts[0].I - 1e-13).all() and f.I.max() <= 1.0,
          f"max I {f.I.max():.12f}")
    q = lop.source(f.E, f.I)
    ratio = (op.lump * op.transport(q)).sum() / (op.lump * q).sum()
    check("transport conservation with the latent source Q = I + sum w_j E_j",
          abs(ratio - 1.0) < 1e-12, f"{ratio:.15f}")

    # ---- 8c. tau_E -> 0 recovers the latency-free scheme ----
    # This is what guarantees the module GENERALISES infection.py, not replaces it.
    mod0 = infection.InfectionModel(gamma=1.0)
    _, ref = infection.simulate(ml, frame, mod0, I0, 10.0, dt_day=0.5,
                                record_every_day=1e9)
    errs = []
    for tau in [1e-2, 1e-3]:
        l0 = latency.LatencyModel(tau_E_day=tau, k_stages=1, tau_asex_day=None)
        _, ss = latency.simulate_latent(ml, frame, mod0, l0, I0,
                                        np.zeros((len(ml.V), 1)), 10.0,
                                        dt_day=0.5, record_every_day=1e9)
        errs.append(abs(ss[-1].severity_pct - ref[-1].severity_pct))
    check("tau_E -> 0 converges to the latency-free scheme, at order 1",
          0.05 < errs[1] / max(errs[0], 1e-30) < 0.2,
          f"error {errs[0]:.2e} -> {errs[1]:.2e}, ratio {errs[1]/errs[0]:.3f}")

    # ---- 8d. order 2 in dt on the full scheme ----
    sev = {}
    for dt in [2.0, 1.0, 0.5, 0.25]:
        _, ss = latency.simulate_latent(ml, frame, infection.InfectionModel(gamma=0.0),
                                        lat, I0, E0, 42.0, dt_day=dt,
                                        record_every_day=1e9)
        sev[dt] = ss[-1].severity_pct
    d1, d2, d3 = (abs(sev[2.0] - sev[1.0]), abs(sev[1.0] - sev[0.5]),
                  abs(sev[0.5] - sev[0.25]))
    order = np.log2(np.mean([d1 / max(d2, 1e-30), d2 / max(d3, 1e-30)]))
    check("the predictor-corrector with latency is still order 2",
          order > 1.8, f"observed order {order:.2f}")

    # ---- 8e. the finding that changes the field protocol ----
    b0_lat, ident = latency.calibrate_beta0_latent(
        ml, frame, infection.InfectionModel(gamma=1.0), lat, I0, E0, 21.0, 2.50)
    b0_no = infection.calibrate_beta0(ml, frame, infection.InfectionModel(gamma=1.0),
                                      I0, 21.0, 2.50)
    print(f"  calibrated beta_0: {b0_no:.5f} without latency against {b0_lat:.5f} with "
          f"latency (x{b0_lat/b0_no:.2f})")
    check("beta_0 with latency is LARGER, not smaller (it is an infection rate, "
          "not a necrosis rate)", b0_lat > b0_no, f"x{b0_lat/b0_no:.2f}")
    mod_free = infection.InfectionModel(gamma=1.0, beta0_per_day=0.0)
    _, ss0 = latency.simulate_latent(ml, frame, mod_free, lat, I0, E0, 21.0,
                                     dt_day=0.5, record_every_day=1e9)
    frac = 100.0 * ((ss0[-1].severity_pct - st0.severity_pct) /
                    (2.50 - st0.severity_pct))
    print(f"  with beta_0 = 0 draining the latent pool already explains "
          f"{frac:.1f} % of the observed increment")
    check("and that is why at 21 days beta_0 is BADLY identified (the pool explains >50 %)",
          frac > 50.0, f"{frac:.1f} % of the increment does not come from new infection")


# --------------------------------------------------------------------------
def _contour_len(field, x, y, level):
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    tot = 0.0
    for c in measure.find_contours(field, level):
        pts = np.column_stack([x[0] + c[:, 1] * dx, y[0] + c[:, 0] * dy])
        tot += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return tot


def _grid(n, half=1.6):
    g = np.linspace(-half, half, n)
    XX, YY = np.meshgrid(g, g)
    return g, g, XX, YY


def _leaf_true_perimeter_no_tears(cfg):
    """
    Exact perimeter of the blade with no tears: arc length of v = +-W(u) by
    quadrature. W' diverges as s^(a-1) at the tips, so each half is integrated
    with the substitution u = (L/2) t^2, which absorbs the integrable
    singularity.
    """
    L, H, A = cfg.length_cm, cfg.halfwidth_cm, 0.42     # 0.42 = profile exponent
    #                                                     in halfwidth()

    def dW(u, side):
        s = np.clip(u / L, 1e-300, 1.0 - 1e-300)
        k = H / (0.5 ** (2 * A)) * (cfg.asymmetry if side < 0 else 1.0)
        return k * (A / L) * s ** (A - 1) * (1 - s) ** (A - 1) * (1 - 2 * s)

    total = 0.0
    for side in (+1, -1):
        f = lambda u, side=side: np.sqrt(1.0 + dW(u, side) ** 2)
        # epsabs 1e-10 and not 1e-12: with 1e-12 quad warns about roundoff in the
        # extrapolation table without gaining digits. 1e-10 over 500 cm is 1e-13
        # relative, seven orders below the test tolerance.
        total += integrate.quad(lambda t: f(0.5 * L * t * t) * (L * t), 0, 1,
                                limit=400, epsabs=1e-10, epsrel=1e-10)[0]
        total += integrate.quad(lambda t: f(L - 0.5 * L * t * t) * (L * t), 0, 1,
                                limit=400, epsabs=1e-10, epsrel=1e-10)[0]
    return total


# --------------------------------------------------------------------------
def test_perimeter():
    print("\n\033[1m9 · The ground truth perimeter estimator\033[0m")

    # -- 1. shapes with an analytic perimeter ------------------------------
    # The zero level estimator is order 2 on smooth boundaries. With n = 800 the
    # measured error is 1e-6 %; the 0.01 % tolerance leaves four orders of margin
    # and is still a thousand times stricter than the binary bias.
    x, y, XX, YY = _grid(800)
    e_cir = abs(_contour_len(1.0 - np.hypot(XX, YY), x, y, 0.0) / (2 * np.pi) - 1)
    check("circle r=1: perimeter = 2*pi to within 0.01 %", e_cir < 1e-4,
          f"{100*e_cir:.5f} %")

    a_e, b_e = 1.3, 0.7
    P_ell = 4 * a_e * float(ellipe(1.0 - (b_e / a_e) ** 2))
    e_ell = abs(_contour_len(1.0 - np.hypot(XX / a_e, YY / b_e), x, y, 0.0) / P_ell - 1)
    check("ellipse 1.3x0.7: perimeter = 4a*E(e^2) to within 0.01 %", e_ell < 1e-4,
          f"{100*e_ell:.5f} %")

    # -- 2. invariance under orientation -----------------------------------
    # A consistent estimator cannot depend on how the boundary falls relative to
    # the grid axes. A square of perimeter 6.4 is measured at five angles. The
    # error per corner is O(h) and there are four corners, hence the tolerance
    # (0.15 %) is looser than the one for smooth curves.
    x, y, XX, YY = _grid(1600)
    vs, vb = [], []
    for ang in (0.0, 10.0, 20.0, 30.0, 45.0):
        t = np.deg2rad(ang)
        c, s = np.cos(t), np.sin(t)
        xr, yr = c * XX + s * YY, -s * XX + c * YY
        phi = np.minimum(np.minimum(0.8 - xr, 0.8 + xr),
                         np.minimum(0.8 - yr, 0.8 + yr))
        vs.append(_contour_len(phi, x, y, 0.0))
        vb.append(_contour_len((phi > 0).astype(float), x, y, 0.5))
    vs, vb = np.array(vs), np.array(vb)
    check("rotated square: the zero level gives 6.4 at every angle (< 0.15 %)",
          np.abs(vs / 6.4 - 1).max() < 1.5e-3,
          f"worst {100*np.abs(vs/6.4-1).max():.3f} %")
    check("the binary estimator is NOT invariant under orientation",
          (vb.max() / vb.min() - 1) > 0.05,
          f"spread {100*(vb.max()/vb.min()-1):.2f} % against "
          f"{100*(vs.max()/vs.min()-1):.3f} % for the zero level")

    # -- 3. the sign reproduces in_blade exactly ---------------------------
    cfg = synth.LeafConfig()
    leaf = synth.SyntheticLeaf(cfg)
    ug = np.linspace(-0.03 * cfg.length_cm, 1.03 * cfg.length_cm, 3000)
    vg = np.linspace(-1.03 * cfg.halfwidth_cm, 1.03 * cfg.halfwidth_cm, 1100)
    UG, VG = np.meshgrid(ug, vg)
    n_bad = int(((leaf.blade_signed(UG, VG) > 0) != leaf.in_blade(UG, VG)).sum())
    del UG, VG
    # EXACT property, not approximate: the min/max algebra preserves the set
    check("{phi > 0} matches in_blade without a single discrepancy",
          n_bad == 0, f"{n_bad} discrepancies over 3.3e6 points")

    # -- 4. tear-free leaf against the analytic quadrature -----------------
    # The only case of the project's REAL geometry with a closed-form answer.
    # Measured error 0.002 %; tolerance 0.05 %, 25 times looser.
    cfg0 = synth.LeafConfig(n_tears=0, n_lesions=0)
    leaf0 = synth.SyntheticLeaf(cfg0)
    P_exact = _leaf_true_perimeter_no_tears(cfg0)
    P_est, _, _, _ = synth.perimeter_levelset(leaf0, cfg0)
    e0 = abs(P_est / P_exact - 1)
    check("tear-free leaf: matches the arc-length quadrature (< 0.05 %)",
          e0 < 5e-4, f"{100*e0:+.5f} %  ({P_est:.3f} against {P_exact:.3f} cm)")

    # -- 5. multigrid consistency on the whole leaf ------------------------
    # The property the old estimator does NOT have: halving h has to move the
    # value only slightly and towards the same limit. It is compared with the
    # binary one on the SAME grid so it is clear this is not a resolution issue.
    R = cfg.curl_radius_cm
    vals_s, vals_b = [], []
    for h in (0.2, 0.1, 0.05):
        nu = int(round(1.06 * cfg.length_cm / h)) + 1
        nv = int(round(2.06 * cfg.halfwidth_cm / h)) + 1
        ug = np.linspace(-0.03 * cfg.length_cm, 1.03 * cfg.length_cm, nu)
        vg = np.linspace(-1.03 * cfg.halfwidth_cm, 1.03 * cfg.halfwidth_cm, nv)
        UG, VG = np.meshgrid(ug, vg)
        phi = leaf.blade_signed(UG, VG)
        del UG, VG
        vals_s.append(_contour_len(phi, ug, vg, 0.0))
        vals_b.append(_contour_len((phi > 0).astype(float), ug, vg, 0.5))
        del phi
    vals_s, vals_b = np.array(vals_s), np.array(vals_b)
    d_s = abs(vals_s[2] / vals_s[1] - 1)
    # the binary one converges, but to ANOTHER value: the signal is not that it
    # moves but that its limit sits well above the zero level's
    gap = vals_b[2] / vals_s[2] - 1
    # Tolerance 0.15 %: the measured movement from 0.1 to 0.05 cm is 0.077 %,
    # half of the one from 0.2 to 0.1 cm. Twice that margin is left.
    check("zero level: halving h moves the perimeter by less than 0.15 %",
          d_s < 1.5e-3, f"{100*d_s:+.4f} %  ({vals_s[1]:.2f} -> {vals_s[2]:.2f} cm)")
    # The property that justifies order-1 Richardson extrapolation: successive
    # differences are divided by 2 when h is halved. If the order were 2 they
    # would be divided by 4 (ratio 0.25) and if it did not converge the ratio
    # would be around 1. The window [0.35, 0.75] covers orders between 0.4 and 1.5.
    ratio = (vals_s[2] - vals_s[1]) / (vals_s[1] - vals_s[0])
    check("successive differences are divided by ~2: the order is 1, not 0",
          0.35 < ratio < 0.75, f"ratio {ratio:.3f} -> order {np.log2(1/ratio):.2f}")
    check("the binary one converges to a value > 3 % longer (bias, not noise)",
          gap > 0.03, f"{100*gap:+.2f} %  ({vals_b[2]:.2f} against {vals_s[2]:.2f} cm)")

    # -- 6. regression guard on the reported value -------------------------
    # Not a mathematical property: it is a lock. The value comes from the
    # extrapolation on grids 8 times finer than the ground truth's (744.25 cm)
    # and synth.py's estimator must reproduce it to 0.1 %.
    P_leaf, P_proj, _, _ = synth.perimeter_levelset(leaf, cfg)
    check("seed 7: intrinsic perimeter = 744.25 +- 0.75 cm",
          abs(P_leaf - 744.25) < 0.75, f"{P_leaf:.3f} cm")
    check("seed 7: projected perimeter = 687.00 +- 0.70 cm",
          abs(P_proj - 687.00) < 0.70, f"{P_proj:.3f} cm")
    # the projected one has to be shorter: the Jacobian cos(v/R) compresses
    check("the projected perimeter is shorter than the intrinsic one",
          P_proj < P_leaf, f"{P_proj:.2f} < {P_leaf:.2f} cm")


# --------------------------------------------------------------------------
def main():
    print("\033[1mNumerical validation\033[0m")
    test_operators()
    print("\n  generating the reference leaf ...")
    sample = synth.generate(synth.LeafConfig())
    test_convergence(sample)
    test_geodesics(sample)
    test_simplification(sample)
    ml, frame, op, I0 = test_infection(sample)
    test_latency(ml, frame, op, I0)
    test_perimeter()

    print()
    if _fails:
        print(f"\033[91m{len(_fails)} check(s) failed:\033[0m " + ", ".join(_fails))
        raise SystemExit(1)
    print("\033[92mAll checks green.\033[0m")


if __name__ == "__main__":
    main()
