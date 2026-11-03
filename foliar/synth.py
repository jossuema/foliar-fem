"""
Synthetic generator of banana leaves with black sigatoka, with exact ground truth.

Geometric model
---------------
The blade is defined in INTRINSIC coordinates (u, v):

    u : distance along the midrib               [0, L]
    v : distance over the surface, measured from the midrib towards the margin

In that system the leaf is flat: areas and distances in (u, v) are the true
ones, because a banana blade curls about the midrib axis without stretching
(developable surface).

The 3D embedding wraps the blade onto a cylinder of radius R about the midrib
axis:

    x = u
    y = R * sin(v / R)
    z = R * (1 - cos(v / R))

which is an exact isometry: (dy/dv)^2 + (dz/dv)^2 = 1.

The camera is orthographic looking down -z, so the image records (x, y).
The inversion is closed form:

    v = R * arcsin(Y / R)          z = R - sqrt(R^2 - Y^2)

The Jacobian of (u,v) -> (X,Y) is cos(v/R) <= 1: the projected area ALWAYS
underestimates the true area. That is the bias the lifted mesh corrects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class LeafConfig:
    # --- blade, in centimetres over the surface (not projected) ---
    length_cm: float = 220.0
    halfwidth_cm: float = 38.0
    asymmetry: float = 0.88          # left half-width / right half-width
    midrib_halfwidth_cm: float = 1.3

    # --- curl about the midrib axis ---
    curl_radius_cm: float = 38.0     # R; smaller = more tightly curled

    # --- sampling ---
    px_per_cm: float = 4.0
    margin_cm: float = 3.0
    gt_samples_u: int = 2200         # fine intrinsic grid for ground truth
    gt_samples_v: int = 760

    # --- wind tears ---
    n_tears: int = 9
    tear_depth_frac: tuple[float, float] = (0.25, 0.90)   # fraction of half-width
    tear_width_cm: tuple[float, float] = (1.2, 6.5)
    tear_tip_cm: float = 0.35        # blunt tip: avoids null input angles

    # --- sigatoka lesions ---
    n_lesions: int = 55
    lesion_length_cm: tuple[float, float] = (1.2, 6.0)    # major axis (v direction)
    lesion_aspect: tuple[float, float] = (0.16, 0.40)     # minor axis / major axis
    lesion_distal_bias: float = 2.2                       # >1 concentrates towards the apex
    lesion_margin_bias: float = 2.6                       # >1 concentrates towards the margin

    seed: int = 7


@dataclass
class GroundTruth:
    """True quantities, measured on the unrolled blade."""
    blade_area_cm2: float
    projected_area_cm2: float
    lesion_area_cm2: float
    projected_lesion_area_cm2: float
    severity_pct: float
    projected_severity_pct: float
    perimeter_cm: float
    projected_perimeter_cm: float
    n_lesions: int
    torn_area_cm2: float
    mean_geodesic_to_midrib_cm: float     # weighted by lesion area
    weighted_severity_index: float
    curl_radius_cm: float
    px_per_cm: float
    _grid: dict = field(default_factory=dict, repr=False)


@dataclass
class LeafSample:
    mask_blade: np.ndarray       # bool, whole blade (includes midrib and lesions)
    mask_lesion: np.ndarray      # bool, sigatoka lesions
    mask_midrib: np.ndarray      # bool, midrib
    px_per_cm: float
    curl_radius_cm: float
    origin_cm: tuple[float, float]   # (X, Y) in cm of the centre of pixel (0, 0)
    truth: GroundTruth

    def pixel_xy_cm(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y) coordinates in cm of every pixel centre of the image."""
        h, w = self.mask_blade.shape
        x0, y0 = self.origin_cm
        X = x0 + np.arange(w) / self.px_per_cm
        Y = y0 + np.arange(h) / self.px_per_cm
        return X, Y


# --------------------------------------------------------------------------
# Intrinsic geometry
# --------------------------------------------------------------------------
class SyntheticLeaf:
    """Defines the leaf as membership functions over (u, v)."""

    def __init__(self, cfg: LeafConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        self._make_tears(rng)
        self._make_lesions(rng)

    # ---- blade profile ---------------------------------------------------
    def halfwidth(self, u: np.ndarray, side: np.ndarray | float) -> np.ndarray:
        """Blade half-width at u. side = +1 (right) or -1 (left)."""
        cfg = self.cfg
        s = np.clip(np.asarray(u, float) / cfg.length_cm, 0.0, 1.0)
        # blunt lanceolate profile, maximum at s = 0.5
        w = (s ** 0.42) * ((1.0 - s) ** 0.42) / (0.5 ** 0.84)
        w = cfg.halfwidth_cm * w
        factor = np.where(np.asarray(side) < 0, cfg.asymmetry, 1.0)
        return w * factor

    # ---- tears -----------------------------------------------------------
    def _make_tears(self, rng):
        cfg = self.cfg
        tears = []
        for _ in range(cfg.n_tears):
            u_i = rng.uniform(0.08, 0.96) * cfg.length_cm
            side = rng.choice([-1.0, 1.0])
            W_i = float(self.halfwidth(u_i, side))
            depth = rng.uniform(*cfg.tear_depth_frac)
            inner = W_i * (1.0 - depth)          # |v| where the tear starts
            width = rng.uniform(*cfg.tear_width_cm)
            slant = rng.normal(0.0, 0.18)        # du per unit of |v|
            tears.append(dict(u=u_i, side=side, inner=inner, W=W_i,
                              width=width, slant=slant))
        self.tears = tears

    def in_tear(self, u, v):
        out = np.zeros(np.broadcast(u, v).shape, dtype=bool)
        for t in self.tears:
            av = np.abs(v)
            same_side = (np.sign(v) == t["side"])
            beyond = av > t["inner"]
            span = max(t["W"] - t["inner"], 1e-6)
            frac = np.clip((av - t["inner"]) / span, 0.0, 1.4)
            # wedge opening towards the margin, with a blunt tip: a zero-width
            # tip generates a 0 degree input angle and makes Ruppert's
            # refinement fail to terminate.
            half = 0.5 * self.cfg.tear_tip_cm + 0.5 * t["width"] * frac
            u_axis = t["u"] + t["slant"] * (av - t["inner"])
            out |= same_side & beyond & (np.abs(u - u_axis) < half)
        return out

    # ---- lesions ---------------------------------------------------------
    def _make_lesions(self, rng):
        cfg = self.cfg
        lesions = []
        tries = 0
        while len(lesions) < cfg.n_lesions and tries < 40 * cfg.n_lesions:
            tries += 1
            # distal bias: sigatoka advances from the apex
            s = rng.beta(cfg.lesion_distal_bias, 1.35)
            u_c = float(np.clip(s, 0.04, 0.985)) * cfg.length_cm
            side = rng.choice([-1.0, 1.0])
            W = float(self.halfwidth(u_c, side))
            # margin bias: infection enters through the edge
            frac = rng.beta(cfg.lesion_margin_bias, 1.4)
            v_c = side * frac * W
            if abs(v_c) < cfg.midrib_halfwidth_cm * 1.6:
                continue
            L_les = rng.uniform(*cfg.lesion_length_cm)
            asp = rng.uniform(*cfg.lesion_aspect)
            a_v = 0.5 * L_les                     # major axis: along secondary veins
            a_u = a_v * asp
            theta = np.deg2rad(rng.normal(0.0, 12.0))
            # irregular contour: low-order Fourier perturbation
            ks = np.array([2, 3, 4, 5])
            amps = rng.uniform(0.04, 0.16, size=ks.size) / ks
            phases = rng.uniform(0, 2 * np.pi, size=ks.size)
            lesions.append(dict(u=u_c, v=v_c, a_u=a_u, a_v=a_v, theta=theta,
                                ks=ks, amps=amps, phases=phases))
        self.lesions = lesions

    def _lesion_mask_one(self, les, u, v):
        du = u - les["u"]
        dv = v - les["v"]
        c, s = np.cos(les["theta"]), np.sin(les["theta"])
        du_r = c * du + s * dv
        dv_r = -s * du + c * dv
        xx = du_r / les["a_u"]
        yy = dv_r / les["a_v"]
        rho = np.sqrt(xx * xx + yy * yy)
        phi = np.arctan2(yy, xx)
        radius = np.ones_like(rho)
        for k, a, p in zip(les["ks"], les["amps"], les["phases"]):
            radius = radius + a * np.cos(k * phi + p)
        return rho < radius

    def in_lesion(self, u, v):
        out = np.zeros(np.broadcast(u, v).shape, dtype=bool)
        for les in self.lesions:
            # cheap clip: only evaluate near the lesion
            r = 1.6 * max(les["a_u"], les["a_v"])
            near = (np.abs(u - les["u"]) < r) & (np.abs(v - les["v"]) < r)
            if not near.any():
                continue
            sub = self._lesion_mask_one(les, u, v)
            out |= sub & near
        return out

    def in_midrib(self, u, v):
        return (np.abs(v) < self.cfg.midrib_halfwidth_cm) & \
               (u >= 0) & (u <= self.cfg.length_cm)

    # ---- whole blade -----------------------------------------------------
    def in_blade(self, u, v):
        cfg = self.cfg
        inside_u = (u >= 0.0) & (u <= cfg.length_cm)
        W = self.halfwidth(u, np.sign(v) + (np.sign(v) == 0))
        inside_v = np.abs(v) <= W
        return inside_u & inside_v & ~self.in_tear(u, v)

    def blade_signed(self, u, v):
        """
        SIGNED field whose set {phi > 0} is exactly in_blade and whose set
        {phi = 0} is exactly its boundary.

        in_blade is a conjunction of boolean conditions. It is translated to a
        continuous field with the standard level-set algebra

            AND = min       OR = max        NOT = negation

        which preserves the zero set of every operation. The field is NOT a
        signed distance (it does not satisfy |grad phi| = 1) and it does not
        need to be: marching squares only needs continuity and the zero to sit
        where the boundary is.

        WHY THIS AND NOT THE BINARY INDICATOR. Marching squares on a binary
        indicator places every crossing at the MIDPOINT of the cell edge,
        because the only two possible values are 0 and 1. The resulting polygon
        is a midpoint staircase and its length does NOT converge to the
        perimeter: it converges to a biased limit that depends on the
        ORIENTATION of the boundary relative to the grid. Measured: 0 % with
        axis-parallel edges, +5.5 % on a circle, +7.3 % on a square rotated 30
        degrees. With a continuous field the crossing is interpolated where it
        belongs and the length converges: order 2 on smooth boundary, order 1
        where there are corners.
        """
        cfg = self.cfg
        u = np.asarray(u, float)
        v = np.asarray(v, float)

        # inside in u: signed distance to the interval [0, L]
        s_u = np.minimum(u, cfg.length_cm - u)
        # inside in v: half-width of the matching side, minus |v|
        side = np.sign(v) + (np.sign(v) == 0)
        s_v = self.halfwidth(u, side) - np.abs(v)

        # outside every tear: -max_i (intersection of the three conditions of
        # wedge i). Same numbers as in_tear, without the step to boolean.
        av = np.abs(v)
        s_tear = None
        for t in self.tears:
            span = max(t["W"] - t["inner"], 1e-6)
            frac = np.clip((av - t["inner"]) / span, 0.0, 1.4)
            half = 0.5 * cfg.tear_tip_cm + 0.5 * t["width"] * frac
            u_axis = t["u"] + t["slant"] * (av - t["inner"])
            s_i = np.minimum(np.minimum(t["side"] * v,       # same side
                                        av - t["inner"]),    # beyond the start
                             half - np.abs(u - u_axis))      # inside the wedge
            s_tear = s_i if s_tear is None else np.maximum(s_tear, s_i)

        phi = np.minimum(s_u, s_v)
        if s_tear is not None:
            phi = np.minimum(phi, -s_tear)
        return phi


# --------------------------------------------------------------------------
# Reference perimeter
# --------------------------------------------------------------------------
def perimeter_levelset(leaf: SyntheticLeaf, cfg: LeafConfig, refine: int = 3):
    """
    True blade perimeter: intrinsic and projected.

    Measures the length of the ZERO level contour of the signed field, on two
    grids whose spacing differs by a factor `refine`, and extrapolates.

    WHY EXTRAPOLATION IS NEEDED. The estimator is order 2 where the boundary is
    smooth, but drops to order 1 at every CORNER: the polyline cuts the corner
    inside the cell containing it and loses O(h) of length there. The number of
    corners is fixed -- the tear wedges -- so the total error is O(h) with a
    constant proportional to that number. Order-1 Richardson:

        p_inf = (r * p_fine - p_coarse) / (r - 1),    r = h_coarse / h_fine

    Measured on the default leaf: the base grid gives -0.15 %, the refined one
    -0.07 %, and the extrapolation 0.000 % against the limit obtained with grids
    8 times finer. With no tears -- the only case with a closed analytic answer,
    see validate.py -- the estimator is already right to 0.002 % WITHOUT
    extrapolating, which confirms the O(h) comes entirely from the corners.

    refine = 3 and not 2: with 2 the worst error over 5 seeds is 0.037 % and it
    costs 0.9 s; with 3 it drops to 0.012 % and costs 1.5 s; with 4 it does not
    improve and costs 3.2 s.

    The grid carries a 3 % margin so the zero set never touches the border: the
    maximum half-width is exactly halfwidth_cm.

    Returns (intrinsic, projected, uncertainty_int, uncertainty_proj), where the
    uncertainty is |extrapolated - fine|: if the order were exactly 1 it would
    be zero, so it measures how far it departs from the assumed error model.
    """
    from skimage import measure

    R = cfg.curl_radius_cm
    L = cfg.length_cm
    Hv = cfg.halfwidth_cm * 1.03

    def measure_once(nu, nv):
        ug = np.linspace(-0.03 * L, 1.03 * L, nu)
        vg = np.linspace(-Hv, Hv, nv)
        du, dv = ug[1] - ug[0], vg[1] - vg[0]
        UG, VG = np.meshgrid(ug, vg)
        phi = leaf.blade_signed(UG, VG)
        p_int = p_prj = 0.0
        for c in measure.find_contours(phi, 0.0):
            u_c = ug[0] + c[:, 1] * du
            v_c = vg[0] + c[:, 0] * dv
            p_int += float(np.sum(np.linalg.norm(
                np.diff(np.column_stack([u_c, v_c]), axis=0), axis=1)))
            p_prj += float(np.sum(np.linalg.norm(
                np.diff(np.column_stack([u_c, R * np.sin(v_c / R)]), axis=0), axis=1)))
        return p_int, p_prj

    nu0 = int(round(cfg.gt_samples_u * 1.06))
    nv0 = int(round(cfg.gt_samples_v * 1.03))
    c_int, c_prj = measure_once(nu0, nv0)
    f_int, f_prj = measure_once((nu0 - 1) * refine + 1, (nv0 - 1) * refine + 1)
    r = float(refine)
    p_int = (r * f_int - c_int) / (r - 1.0)
    p_prj = (r * f_prj - c_prj) / (r - 1.0)
    return p_int, p_prj, abs(p_int - f_int), abs(p_prj - f_prj)


# --------------------------------------------------------------------------
# Rasterisation + ground truth
# --------------------------------------------------------------------------
def generate(cfg: LeafConfig | None = None) -> LeafSample:
    cfg = cfg or LeafConfig()
    leaf = SyntheticLeaf(cfg)
    R = cfg.curl_radius_cm

    # ---------- image (orthographic projection) --------------------------
    y_max = R * np.sin(min(cfg.halfwidth_cm / R, np.pi / 2))
    x0 = -cfg.margin_cm
    y0 = -(y_max + cfg.margin_cm)
    w_px = int(np.ceil((cfg.length_cm + 2 * cfg.margin_cm) * cfg.px_per_cm))
    h_px = int(np.ceil(2 * (y_max + cfg.margin_cm) * cfg.px_per_cm))

    X = x0 + np.arange(w_px) / cfg.px_per_cm
    Y = y0 + np.arange(h_px) / cfg.px_per_cm
    XX, YY = np.meshgrid(X, Y)

    # exact inversion of the projection
    ratio = np.clip(YY / R, -1.0, 1.0)
    VV = R * np.arcsin(ratio)
    UU = XX

    blade = leaf.in_blade(UU, VV)
    lesion = leaf.in_lesion(UU, VV) & blade
    midrib = leaf.in_midrib(UU, VV) & blade

    # ---------- ground truth on the intrinsic grid -----------------------
    ug = np.linspace(0.0, cfg.length_cm, cfg.gt_samples_u)
    vg = np.linspace(-cfg.halfwidth_cm, cfg.halfwidth_cm, cfg.gt_samples_v)
    du = ug[1] - ug[0]
    dv = vg[1] - vg[0]
    UG, VG = np.meshgrid(ug, vg)

    g_blade = leaf.in_blade(UG, VG)
    g_lesion = leaf.in_lesion(UG, VG) & g_blade
    g_midrib = leaf.in_midrib(UG, VG) & g_blade

    cell = du * dv
    area_true = float(g_blade.sum() * cell)
    area_lesion = float(g_lesion.sum() * cell)

    # area with no tears, to quantify the lost blade
    W_r = leaf.halfwidth(UG, 1.0)
    W_l = leaf.halfwidth(UG, -1.0)
    intact = np.where(VG >= 0, VG <= W_r, np.abs(VG) <= W_l)
    area_intact = float(intact.sum() * cell)

    # true projected areas: Jacobian cos(v/R).
    # Lesions concentrate near the margin, which is exactly where the projection
    # compresses most, so they compress more than the average blade: the
    # severity measured in the image comes out biased DOWNWARD.
    area_proj = float((g_blade * np.cos(VG / R)).sum() * cell)
    area_lesion_proj = float((g_lesion * np.cos(VG / R)).sum() * cell)

    # True perimeter from the ZERO level contour of the signed field.
    # Both are computed: the real one (over the surface) and the one the camera
    # would see (same contour, pushed through the projection). Judging an image
    # estimator against the REAL perimeter would be cheating: the image can only
    # aspire to the projected one as long as the curl is unknown.
    #
    # Marching squares on the binary indicator is NOT used. That estimator is
    # not multigrid consistent and on this leaf gave +4.56 % (intrinsic) and
    # +4.18 % (projected). Worse: mask_to_polygon uses the SAME estimator, so
    # the -0.01 % agreement that used to be reported for the level C perimeter
    # did not measure accuracy, it measured two implementations of the same bias
    # agreeing. See perimeter_levelset and section 9 of validate.py.
    perim, perim_proj, _, _ = perimeter_levelset(leaf, cfg)

    # ---------- true geodesics (eikonal on the intrinsic grid) -----------
    import skfmm
    phi = np.where(g_midrib, -1.0, 1.0)
    obstacle = ~g_blade
    dist = skfmm.distance(np.ma.MaskedArray(phi, obstacle), dx=[dv, du])
    gt_dist = np.ma.filled(dist, np.nan)
    gt_dist = np.where(g_blade & ~g_midrib, gt_dist, np.where(g_midrib, 0.0, np.nan))

    finite = g_lesion & np.isfinite(gt_dist)
    mean_geo = float(np.nanmean(gt_dist[finite])) if finite.any() else float("nan")

    # weighted index: a lesion weighs more the closer it is to the midrib
    half = 0.5 * cfg.halfwidth_cm
    w_field = np.exp(-gt_dist / half)
    wsi = float(np.nansum(np.where(g_lesion, w_field, 0.0)) * cell /
                max(np.nansum(np.where(g_blade, w_field, 0.0)) * cell, 1e-9)) * 100.0

    from scipy import ndimage
    n_les = int(ndimage.label(g_lesion)[1])

    truth = GroundTruth(
        blade_area_cm2=area_true,
        projected_area_cm2=area_proj,
        lesion_area_cm2=area_lesion,
        projected_lesion_area_cm2=area_lesion_proj,
        severity_pct=100.0 * area_lesion / area_true,
        projected_severity_pct=100.0 * area_lesion_proj / area_proj,
        perimeter_cm=perim,
        projected_perimeter_cm=perim_proj,
        n_lesions=n_les,
        torn_area_cm2=area_intact - area_true,
        mean_geodesic_to_midrib_cm=mean_geo,
        weighted_severity_index=wsi,
        curl_radius_cm=R,
        px_per_cm=cfg.px_per_cm,
        _grid=dict(u=ug, v=vg, blade=g_blade, lesion=g_lesion,
                   midrib=g_midrib, dist=gt_dist),
    )

    return LeafSample(
        mask_blade=blade,
        mask_lesion=lesion,
        mask_midrib=midrib,
        px_per_cm=cfg.px_per_cm,
        curl_radius_cm=R,
        origin_cm=(x0, y0),
        truth=truth,
    )
