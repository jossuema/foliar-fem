"""
Quasi-static SI model of lesion expansion on the mesh.

This is the repository's first real PDE: up to here the FEM built geometry and
distances, and here it solves an equation.

    (1)  -div_g( D(x) grad_g Phi )  +  Phi  =  I        transport, quasi-static
    (2)  dI/dt = [ beta(x) Phi + alpha(x) ] (1 - I)     reaction, closed form

I(x,t) in [0,1] is the necrotic tissue fraction, i.e. exactly what segmentation
sees as a lesion. Phi is the inoculum pressure: dimensionless, with no initial
condition, recomputed from I at every step.

The diffusion tensor is ANISOTROPIC and follows the secondary vein frame:

    D(x) = ell_v^2 (e_v (x) e_v)  +  ell_u^2 (e_u (x) e_u)      [cm^2]

with e_v the midrib -> margin direction, obtained from the gradient of the
geodesic distance to the midrib. That is what ties the PDE to what the pipeline
already computes: the same geodesic that gives the distance to the midrib gives
the vein orientation.

THE HYPOTHESIS LIVES IN gamma, AND ITS SIGN IS FREE

    beta(x) = beta_0 exp( -gamma * d_midrib(x) / lambda )

    gamma > 0  tissue gets infected faster NEAR THE MIDRIB (midrib = resource
               source; this is the hypothesis that motivates the module)
    gamma = 0  the rate does not depend on distance to the midrib (null)
    gamma < 0  faster NEAR THE MARGIN (marginal senescence)

The term alpha(x) = alpha_0 exp(-d_margin/lambda_m) is the COMPETING mechanism:
exogenous spore deposition through the edge. The two exponentials point in
opposite directions on purpose. A model that can only confirm the hypothesis is
useless for testing it.

WHAT CAN AND CANNOT BE CLAIMED WITH THIS

Phase 1 (synthetic data) measures whether the NUMERICAL SCHEME works, not
whether the biology is true. The anisotropy ratio comes from (elongation)^2 and
the elongation is a parameter of the synthetic generator: using it as biological
evidence would be circular. The only real test of the model is a pair of photos
of the same leaf 14-21 days apart.

Units: centimetres and days throughout the module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import fem
from .meshing import REG_LESION, Mesh

# Smoothing of the vein frame. L = 2.0 cm is the measured optimum on the
# reference mesh (mean h 1.09 cm, i.e. L ~ 1.8 h): the mean angular error is
# 1.68 / 1.22 / 0.85 / 0.76 / 0.85 / 1.98 / 3.71 degrees for L = 0.3 / 0.8 / 1.5
# / 2.0 / 3.0 / 6.0 / 9.0 cm. The minimum is flat between 1.5 and 3.0. Below it
# the Dijkstra graph metrication noise remains; above it the filter crosses the
# midrib and the tears. This is a MESH length, not a physical scale.
SMOOTH_LEN_CM = 2.0
SMOOTH_STEPS = 3

# If min(Phi)/max(Phi) falls below this, the mesh violates the maximum principle
# enough not to trust the run. Measured on the reference mesh: -8.5e-4, i.e. an
# order of magnitude of margin.
UNDERSHOOT_ABORT = -1e-2


# --------------------------------------------------------------------------
# Vein frame
# --------------------------------------------------------------------------
@dataclass
class VeinFrame:
    """Orthonormal frame of the secondary veins, per triangle."""
    e_v: np.ndarray            # (m, 3) midrib -> margin, unit and TANGENT
    e_u: np.ndarray            # (m, 3) along the midrib, = N x e_v
    normal: np.ndarray         # (m, 3) triangle normal
    coherence: np.ndarray      # (m,)   (lam1-lam2)/lam1 of the structure tensor
    grad_norm: np.ndarray      # (m,)   |grad d|; should be 1
    d_nerve_cm: np.ndarray     # (n,)   geodesic to the midrib, nan on loose strips
    d_margin_cm: np.ndarray    # (n,)   geodesic to the free edge of the blade


def _tri_to_node(vals: np.ndarray, F: np.ndarray, A: np.ndarray, n: int):
    """Area-weighted average of a per-triangle field onto the nodes."""
    extra = (None,) * (vals.ndim - 1)
    num = np.zeros((n,) + vals.shape[1:])
    den = np.zeros(n)
    for a in range(3):
        np.add.at(num, F[:, a], A[(slice(None),) + extra] * vals)
        np.add.at(den, F[:, a], A)
    return num / np.maximum(den, 1e-300)[(slice(None),) + extra]


def _unit_gradient(V: np.ndarray, F: np.ndarray, u: np.ndarray):
    """Per-triangle unit gradient of a nodal field, and its modulus."""
    G, _ = fem.shape_gradients(V, F)
    g = np.einsum("ta,tai->ti", np.nan_to_num(u, nan=0.0)[F], G)
    nrm = np.linalg.norm(g, axis=1)
    out = np.zeros_like(g)
    ok = nrm > 0
    out[ok] = g[ok] / nrm[ok, None]
    return out, nrm


def triangle_frames(V: np.ndarray, F: np.ndarray, e1_hint: np.ndarray | None = None):
    """
    Orthonormal frame (e1, e2, N) per triangle.

    e1_hint is PROJECTED onto the tangent plane and normalised. The projection is
    not optional: without it the tensor D diffuses OFF the surface. Measured, the
    vein field hint leaves the plane by 0.45 degrees on average and 4.16 at most,
    which gives |D.N| = 0.49 cm2/day and breaks the exact isotropic reduction
    (from 2.7e-15 to 6.6e-3). Where the projection vanishes it falls back to the
    first edge, so as not to leave a degenerate frame.
    """
    V = fem._as3d(V)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    nrm_vec = np.cross(v1 - v0, v2 - v0)
    twoA = np.linalg.norm(nrm_vec, axis=1)
    N = nrm_vec / np.maximum(twoA, 1e-300)[:, None]

    fallback = v1 - v0
    e1 = fallback.copy() if e1_hint is None else np.array(e1_hint, float, copy=True)
    e1 = e1 - np.einsum("ti,ti->t", e1, N)[:, None] * N
    nrm = np.linalg.norm(e1, axis=1)
    deg = nrm < 1e-12 * np.maximum(np.linalg.norm(fallback, axis=1), 1e-300)
    if deg.any():
        f = fallback[deg] - np.einsum("ti,ti->t", fallback[deg], N[deg])[:, None] * N[deg]
        e1[deg] = f
        nrm[deg] = np.linalg.norm(f, axis=1)
    e1 = e1 / np.maximum(nrm, 1e-300)[:, None]
    return e1, np.cross(N, e1), N


def boundary_nodes(F: np.ndarray) -> np.ndarray:
    """Nodes on the free edge: outer margin and tear lips."""
    return np.unique(fem.boundary_edges(F).ravel())


def vein_frame(mesh: Mesh, source_nodes, length_cm: float = SMOOTH_LEN_CM,
               steps: int = SMOOTH_STEPS, method: str = "dijkstra2") -> VeinFrame:
    """
    Secondary vein direction field on the mesh.

    Measured recipe, and all three pieces are necessary:

      raw gradient of the geodesic        5.64 degrees mean error, 178.4 max,
                                          30 flipped triangles
      + STRUCTURE TENSOR smoothing        0.76 degrees mean
      + sign against a DIFFUSED average   0.70 mean, 4.4 max,
        of the gradient                   0 flipped triangles
      + projection onto the tangent plane |e_v . N| = exactly 0

    Why the tensor and not the vector: a vein direction is a LINE, not an arrow.
    Diffusing g cancels the opposite vectors on the two sides of the midrib;
    diffusing T = g g^T is sign invariant. And the sign is recovered against a
    DIFFUSED average of g, not against the local g, because on triangles where
    |grad d| ~ 0.5 the raw gradient points the wrong way and would contaminate
    the smoothed field.

    |grad d| is returned as SELF-DIAGNOSTIC: the eikonal condition says it is 1,
    so where it departs the frame is not trustworthy. It is computable on a real
    leaf, where no truth is known (measured: 0.516 on the bad triangles against
    1.007 globally).
    """
    V, F = mesh.V, mesh.F
    n = len(V)
    A = fem.triangle_areas(V, F)

    d_nerve = fem.geodesic(V, F, source_nodes, method=method)
    d_margin = fem.geodesic(V, F, boundary_nodes(F), method=method)
    g, gnorm = _unit_gradient(V, F, d_nerve)

    S = fem.stiffness_matrix(V, F).tocsc()
    M = fem.mass_matrix(V, F, lumped=True).tocsc()
    tau = (length_cm ** 2) / max(steps, 1)          # L = sqrt(tau * steps)
    lu = spla.splu((M + tau * S).tocsc())

    # --- structure tensor diffusion (6 independent components) ---
    Tn = _tri_to_node(np.einsum("ti,tj->tij", g, g), F, A, n)
    for _ in range(steps):
        for i in range(3):
            for j in range(i, 3):
                x = lu.solve(M @ Tn[:, i, j])
                Tn[:, i, j] = Tn[:, j, i] = x

    # --- vector diffusion, ONLY to fix the sign ---
    gn = _tri_to_node(g, F, A, n)
    for _ in range(steps):
        for i in range(3):
            gn[:, i] = lu.solve(M @ gn[:, i])

    Tt = Tn[F].mean(axis=1)
    Tt = 0.5 * (Tt + np.transpose(Tt, (0, 2, 1)))
    ev, evec = np.linalg.eigh(Tt)
    e1 = evec[:, :, 2]
    s = np.sign(np.einsum("ti,ti->t", e1, gn[F].mean(axis=1)))
    s[s == 0] = 1.0
    e1 = e1 * s[:, None]

    e_v, e_u, N = triangle_frames(V, F, e1_hint=e1)
    coh = (ev[:, 2] - ev[:, 1]) / np.maximum(ev[:, 2], 1e-300)
    return VeinFrame(e_v=e_v, e_u=e_u, normal=N, coherence=coh,
                     grad_norm=gnorm, d_nerve_cm=d_nerve, d_margin_cm=d_margin)


# --------------------------------------------------------------------------
# Tensor operator
# --------------------------------------------------------------------------
def diffusion_tensor(e_v: np.ndarray, e_u: np.ndarray, d_par, d_perp) -> np.ndarray:
    """
    D_T = d_par (e_v (x) e_v) + d_perp (e_u (x) e_u), of shape (m, 3, 3).

    Symmetric, positive semidefinite, with its kernel along the normal:
    D_T . N = 0, i.e. no diffusion off the surface. d_par and d_perp can be
    scalars or vectors of length m.
    """
    d_par = np.broadcast_to(np.asarray(d_par, float), (len(e_v),))
    d_perp = np.broadcast_to(np.asarray(d_perp, float), (len(e_v),))
    return (d_par[:, None, None] * np.einsum("ti,tj->tij", e_v, e_v) +
            d_perp[:, None, None] * np.einsum("ti,tj->tij", e_u, e_u))


def stiffness_anisotropic(V: np.ndarray, F: np.ndarray, Dt: np.ndarray) -> sp.csr_matrix:
    """
    Anisotropic P1 stiffness:  K_ab = sum_T A_T (D_T grad phi_a) . grad phi_b.

    Valid in R^2 and in R^3, same structure as fem.stiffness_matrix.
    Exact properties, not asymptotic ones:
      K @ 1 = 0 for ANY D_T, because sum_a grad(phi_a) = 0 per triangle;
      with D_T = c * P (P the tangent projector) it returns
      c * fem.stiffness_matrix entry by entry;
      symmetric and positive semidefinite if D_T is.
    """
    V = fem._as3d(V)
    G, A = fem.shape_gradients(V, F)
    Dt = np.asarray(Dt, float)
    if Dt.ndim == 2:
        Dt = np.broadcast_to(Dt, (len(F), 3, 3))
    DG = np.einsum("tij,taj->tai", Dt, G)

    n = len(V)
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            rows.append(F[:, a])
            cols.append(F[:, b])
            vals.append(A * np.einsum("ti,ti->t", DG[:, a], G[:, b]))
    return sp.coo_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n)).tocsr()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class InfectionModel:
    """
    Model parameters. All in cm and days.

    ell_v_cm       Length of the dispersal kernel along the veins. It is
                   sqrt(D*tau): in the quasi-static limit D and tau are NOT
                   separately identifiable, so the model carries ONE length
                   scale and not two diffusivities. 2.6 cm is an INVENTED
                   working value, chosen so the kernel is resolved by the mesh
                   (ell_v/h = 2.4). It is the weakest coefficient in the model.
                   Real calibration: 50th percentile of the geodesic distance
                   from each new lesion in photo 2 to the nearest pre-existing
                   lesion in photo 1.
    aniso_ratio    ell_v^2/ell_u^2 = (mean elongation)^2, measured on the
                   SURFACE and not on the projection. Diffusion happens on the
                   surface, so the elongation that calibrates it is the level C
                   one: 3.7003, hence 13.69. Using the projected value (3.2155,
                   ratio 10.34) underestimates the anisotropy by 24 %, because
                   the projection compresses exactly the lesion's major axis.
                   IN PHASE 1 THIS IS CIRCULAR: the elongation comes from the
                   synthetic generator, so it validates the scheme, not biology.
    gamma          Resource gradient exponent. SIGN IS FREE. This is the
                   hypothesis parameter; see the module docstring.
    beta0_per_day  Infection rate per unit inoculum pressure, evaluated on the
                   midrib. Not a free fit: it is found by bisection so that the
                   simulated severity matches the one observed in the second
                   photo.
    lambda_cm      Resource decay scale. FIXED at 19.0 = halfwidth/2, the same
                   scale_cm used by demo.py and the ground truth WSI. It is not
                   fitted: lambda and gamma trade off against each other and
                   leaving both free destroys the identifiability of the
                   hypothesis.
    alpha0_per_day Exogenous inoculum deposition through the margin. It is the
                   COMPETING mechanism to gamma; omitting it biases the result
                   in favour of the hypothesis.
    lambda_m_cm    Scale of the exogenous deposition. INVENTED in phase 1. It is
                   calibrated separately and BEFORE gamma, against the histogram
                   of distance-to-margin of the isolated new lesions.
    """
    ell_v_cm: float = 2.6
    aniso_ratio: float = 13.69
    gamma: float = 0.0
    beta0_per_day: float = 0.043
    lambda_cm: float = 19.0
    alpha0_per_day: float = 0.0
    lambda_m_cm: float = 6.0

    @property
    def ell_u_cm(self) -> float:
        return float(self.ell_v_cm / np.sqrt(self.aniso_ratio))


@dataclass
class InfectionState:
    """State at one instant."""
    t_day: float
    I: np.ndarray              # (n,) nodal necrotic fraction, in [0, 1]
    phi: np.ndarray            # (n,) inoculum pressure
    severity_pct: float
    wsi: float
    psi: float                 # wsi / severity_pct; > 1 = enriched at the midrib


def initial_infection(mesh: Mesh) -> np.ndarray:
    """
    I(x, 0) by LUMPED MASS projection of the lesion indicator.

        b_a = sum_{T in lesion} A_T/3 ;   I0_a = clip( b_a / M_L[a,a], 0, 1 )

    That is, the lesioned area fraction of each node's support. The raw nodal
    indicator is NOT used: measured, the projection reproduces the mesh lesion
    area with zero bias, while the raw indicator overestimates it by 70 %
    because it inflates every lesion by a one-triangle ring.
    """
    A = fem.triangle_areas(mesh.V, mesh.F)
    n = len(mesh.V)
    b = np.zeros(n)
    np.add.at(b, mesh.F[mesh.attr == REG_LESION].ravel(),
              np.repeat(A[mesh.attr == REG_LESION] / 3.0, 3))
    lump = np.zeros(n)
    np.add.at(lump, mesh.F.ravel(), np.repeat(A / 3.0, 3))
    return np.clip(b / np.maximum(lump, 1e-300), 0.0, 1.0)


class InfectionOperator:
    """
    Precomputed operators of the model.

    Nothing depends on time, so it is assembled and factorised ONCE. Measured:
    51x faster than calling spsolve at every step, and the reused factor gives
    a bit-for-bit identical result, because spsolve internally calls the same
    splu with the same ordering.
    """

    def __init__(self, mesh: Mesh, frame: VeinFrame, model: InfectionModel):
        self.mesh = mesh
        self.frame = frame
        self.model = model

        A = fem.triangle_areas(mesh.V, mesh.F)
        n = len(mesh.V)
        lump = np.zeros(n)
        np.add.at(lump, mesh.F.ravel(), np.repeat(A / 3.0, 3))
        self.lump = lump                                # diagonal of M_L
        self.area_cm2 = float(lump.sum())

        ell_v2 = model.ell_v_cm ** 2
        ell_u2 = model.ell_u_cm ** 2
        Dt = diffusion_tensor(frame.e_v, frame.e_u, ell_v2, ell_u2)
        self.K = stiffness_anisotropic(mesh.V, mesh.F, Dt)
        self.lu = spla.splu((sp.diags(lump) + self.K).tocsc())

        # --- rate fields, per node ---
        d = frame.d_nerve_cm
        detached = ~np.isfinite(d)
        # On a detached strip there is no midrib, so gamma cannot have a SPATIAL
        # effect there: the resource factor is frozen at the median value of the
        # connected blade. That is precisely what turns the strips into an
        # internal control of the experiment.
        d_eff = np.where(detached, np.nanmedian(d) if np.isfinite(d).any() else 0.0, d)
        self.beta = model.beta0_per_day * np.exp(-model.gamma * d_eff / model.lambda_cm)
        self.alpha = model.alpha0_per_day * np.exp(
            -np.nan_to_num(frame.d_margin_cm, nan=0.0) / model.lambda_m_cm)
        self.detached = detached

        # WSI weights: the same exp(-d/lambda) that params.analyse uses
        self.w_wsi = np.where(detached, 0.0, np.exp(-d_eff / model.lambda_cm))

    # ---- transport -------------------------------------------------------
    def transport(self, I: np.ndarray) -> np.ndarray:
        """Phi = (M_L + K_D)^-1 M_L I. One forward and backward substitution."""
        return self.lu.solve(self.lump * I)

    # ---- observables -----------------------------------------------------
    def severity_pct(self, I: np.ndarray) -> float:
        return float(100.0 * (self.lump * I).sum() / self.area_cm2)

    def wsi(self, I: np.ndarray) -> float:
        w = self.lump * self.w_wsi
        return float(100.0 * (w * I).sum() / max(w.sum(), 1e-300))

    def state(self, t_day: float, I: np.ndarray) -> InfectionState:
        sev = self.severity_pct(I)
        w = self.wsi(I)
        return InfectionState(t_day=float(t_day), I=I.copy(),
                              phi=self.transport(I), severity_pct=sev, wsi=w,
                              psi=float(w / sev) if sev > 1e-12 else float("nan"))

    # ---- diagnostics -----------------------------------------------------
    def diagnostics(self, I: np.ndarray | None = None) -> dict:
        """
        Numerical health of the run. ALWAYS reported, not only in the tests.

        The relative undershoot of Phi is the direct measure of how far the mesh
        violates the discrete maximum principle. If it falls below
        UNDERSHOOT_ABORT the run is not trustworthy.
        """
        I = initial_infection(self.mesh) if I is None else I
        phi = self.transport(I)
        K = self.K.tocoo()
        off = K.row != K.col
        pos_off = int(((K.data > 0) & off).sum())
        return dict(
            phi_min=float(phi.min()), phi_max=float(phi.max()),
            undershoot_rel=float(phi.min() / max(phi.max(), 1e-300)),
            reliable=bool(phi.min() / max(phi.max(), 1e-300) > UNDERSHOOT_ABORT),
            n_offdiag_positive=pos_off,
            n_offdiag=int(off.sum()),
            grad_norm_mean=float(np.mean(self.frame.grad_norm)),
            coherence_mean=float(np.mean(self.frame.coherence)),
            n_detached_nodes=int(self.detached.sum()),
        )


# --------------------------------------------------------------------------
# Time integration
# --------------------------------------------------------------------------
def step(op: InfectionOperator, I: np.ndarray, dt_day: float,
         order: int = 2) -> np.ndarray:
    """
    One time step.

    The reaction is integrated in CLOSED form with rho frozen:

        I <- 1 - (1 - I) exp(-dt * max(rho, 0))

    which guarantees 0 <= I <= 1 and dI/dt >= 0 for any dt and any mesh,
    WITHOUT clipping and without relying on the matrix maximum principle. The
    protection comes from the integrator, not from the operator: it is the only
    thing that survives the 20.4 % obtuse triangles of this mesh.

    order=2 is predictor-corrector on Phi (two solves per step, same
    factorisation); order=1 is Lie splitting. Error in severity at 42 days
    measured on the reference leaf:

        Lie                  -2.49 / -1.23 / -0.58 %   for dt = 1.0 / 0.5 / 0.25
        predictor-corrector  -0.0365 / -0.0092 / -0.0023 %   (order 2.00)

    At dt = 0.5 day the error drops 130-fold for twice the cost, so the default
    is 2. Note: measured on Psi the difference is barely visible, because Psi is
    a ratio and cancels the splitting error; on severity, which is the agronomic
    deliverable, it is visible.
    """
    rho1 = np.maximum(op.beta * op.transport(I) + op.alpha, 0.0)
    if order == 1:
        return 1.0 - (1.0 - I) * np.exp(-dt_day * rho1)
    I_pred = 1.0 - (1.0 - I) * np.exp(-dt_day * rho1)
    rho2 = np.maximum(op.beta * op.transport(I_pred) + op.alpha, 0.0)
    return 1.0 - (1.0 - I) * np.exp(-dt_day * 0.5 * (rho1 + rho2))


def simulate(mesh: Mesh, frame: VeinFrame, model: InfectionModel,
             I0: np.ndarray, horizon_day: float, dt_day: float = 0.5,
             order: int = 2, record_every_day: float = 1.0):
    """
    Run the model and return (operator, list of InfectionState).

    horizon_day should be the REAL interval between two photos of the same leaf
    (14-21 days). Extrapolating much beyond that is a product, not a test.

    Measured cost (11,015 nodes, 20,917 triangles): vein frame 89 ms, assembly
    11 ms, factorisation 18 ms, 84 steps 69 ms. Total ~0.19 s, three orders of
    magnitude below meshing, which takes ~35 s. Cost is not a design criterion
    here.
    """
    op = InfectionOperator(mesh, frame, model)
    I = np.clip(np.asarray(I0, float), 0.0, 1.0)
    states = [op.state(0.0, I)]
    n_steps = int(round(horizon_day / dt_day))
    next_rec = record_every_day
    for k in range(n_steps):
        I = step(op, I, dt_day, order=order)
        t = (k + 1) * dt_day
        if t >= next_rec - 1e-12 or k == n_steps - 1:
            states.append(op.state(t, I))
            next_rec += record_every_day
    return op, states


def calibrate_beta0(mesh: Mesh, frame: VeinFrame, model: InfectionModel,
                    I0: np.ndarray, horizon_day: float,
                    target_severity_pct: float, dt_day: float = 0.5,
                    tol: float = 1e-4, iters: int = 40) -> float:
    """
    Find beta_0 by bisection so that the simulated severity at the horizon
    MATCHES the observed one. Returns the calibrated beta_0.

    This is not a free fit: it is one parameter against one observed scalar. And
    it is MANDATORY before comparing values of gamma. Without it, changing gamma
    changes the global infection rate and the final severity, and then Psi from
    two runs is not comparable: one would be measuring how much the lesion grew,
    not where it went. Measured without calibration, gamma = -2 reaches 63 %
    severity against 1.9 % for gamma = +1.5, and the statistic stops being
    monotone.
    """
    def sev(b0):
        m = InfectionModel(**{**model.__dict__, "beta0_per_day": b0})
        _, ss = simulate(mesh, frame, m, I0, horizon_day, dt_day=dt_day,
                         record_every_day=1e9)
        return ss[-1].severity_pct

    lo, hi = 0.0, max(model.beta0_per_day, 1e-3)
    while sev(hi) < target_severity_pct and hi < 1e4:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if sev(mid) < target_severity_pct:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(hi, 1e-12):
            break
    return float(0.5 * (lo + hi))


# --------------------------------------------------------------------------
# The paired null: half the statistic, not an optional control
# --------------------------------------------------------------------------
def null_dilation(mesh: Mesh, I0: np.ndarray, target_severity_pct: float,
                  method: str = "dijkstra2", tol: float = 1e-3, iters: int = 60):
    """
    Geometric null: ISOTROPIC geodesic dilation of the initial lesions, with the
    radius found by bisection until it matches the model's final severity.

    It exists because the naive version of the statistic is CONFOUNDED: with
    gamma = 0, i.e. no preference for the midrib at all, Psi still rises,
    because the leaf edge rectifies isotropic growth inwards. Reporting "the
    lesions moved towards the midrib" without this null is reporting geometry
    and calling it biology.

    Returns (I_null, radius_cm, achieved_severity_pct).
    """
    A = fem.triangle_areas(mesh.V, mesh.F)
    n = len(mesh.V)
    lump = np.zeros(n)
    np.add.at(lump, mesh.F.ravel(), np.repeat(A / 3.0, 3))
    total = lump.sum()

    src = np.flatnonzero(I0 > 0.5)
    if src.size == 0:
        src = np.flatnonzero(I0 > 0)
    if src.size == 0:
        return np.zeros(n), 0.0, 0.0
    d = fem.geodesic(mesh.V, mesh.F, src, method=method)
    d = np.nan_to_num(d, nan=np.inf)

    def sev(r):
        return 100.0 * (lump * np.maximum(I0, (d <= r).astype(float))).sum() / total

    lo, hi = 0.0, 1.0
    while sev(hi) < target_severity_pct and hi < 1e4:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if sev(mid) < target_severity_pct:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    r = 0.5 * (lo + hi)
    return np.maximum(I0, (d <= r).astype(float)), float(r), float(sev(r))


def excess_pp(psi_model: float, psi_null: float, psi_ref: float) -> float:
    """
    Decision statistic, in percentage points:

        excess = 100*(psi_model/psi_ref - 1) - 100*(psi_null/psi_ref - 1)

    Only the PAIRED difference is robust. Measured by perturbing the mask, the
    ABSOLUTE excess moves ~5.9 pp depending on the perturbation (a third of the
    effect), but the separation between hypotheses holds, because the null is
    computed on the SAME noisy segmentation and the error cancels in the
    subtraction. Reporting the absolute excess without the null is reporting the
    quality of the segmenter.

    The level (B flat or C lifted) has to be the SAME on both sides: mixing them
    manufactures false signal.
    """
    return float(100.0 * (psi_model / psi_ref - 1.0) -
                 100.0 * (psi_null / psi_ref - 1.0))
