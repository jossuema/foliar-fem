"""
Extraction of agronomic parameters from the mesh + FEM fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from . import fem
from .meshing import REG_HEALTHY, REG_LESION, REG_MIDRIB, Mesh

# --------------------------------------------------------------------------
# Stover scale as modified by Gauhl (1990), the standard for banana and
# plantain. Thresholds on % of affected leaf area. Grade 1 also carries the
# "up to 10 lesions" condition.
# --------------------------------------------------------------------------
STOVER_GAUHL = [
    (0, 0.0, "Healthy leaf, no visible symptoms"),
    (1, 1.0, "Up to 10 lesions or streaks; less than 1% of leaf area"),
    (2, 5.0, "More than 10 lesions; 1 to 5% of leaf area"),
    (3, 15.0, "6 to 15% of leaf area affected"),
    (4, 33.0, "16 to 33% of leaf area affected"),
    (5, 50.0, "34 to 50% of leaf area affected"),
    (6, 100.0, "More than 50% of leaf area affected"),
]


def stover_gauhl_grade(severity_pct: float, n_lesions: int) -> tuple[int, str]:
    if severity_pct <= 0.0:
        return 0, STOVER_GAUHL[0][2]
    if severity_pct < 1.0:
        if n_lesions <= 10:
            return 1, STOVER_GAUHL[1][2]
        # under 1% but more than 10 lesions: the grade is 2 by the count
        # condition, but the canonical description of grade 2 says "1 to 5%",
        # which contradicts the measured severity. The matching one is returned.
        return 2, "More than 10 lesions; less than 1% of leaf area"
    for grade, upper, desc in STOVER_GAUHL[2:]:
        if severity_pct <= upper:
            return grade, desc
    return 6, STOVER_GAUHL[6][2]


@dataclass
class LesionRecord:
    id: int
    area_cm2: float
    centroid: tuple[float, float]
    length_cm: float          # major axis (4 sigma of the second moment, in 3D)
    width_cm: float           # minor axis
    elongation: float
    orientation_deg: float    # 0 = parallel to the midrib, 90 = perpendicular
    dist_to_midrib_cm: float  # geodesic, area-weighted mean
    dist_min_cm: float
    detour_ratio: float       # geodesic / straight-line Euclidean
    detached: bool            # on a strip with no connection to the midrib


def lesion_components(mesh: Mesh) -> list[np.ndarray]:
    """Group lesion triangles sharing an edge. Coalesced lesions = one."""
    idx = np.flatnonzero(mesh.attr == REG_LESION)
    if idx.size == 0:
        return []
    F = mesh.F[idx]
    # edge -> triangles map
    edges = {}
    for t, tri in enumerate(F):
        for a in range(3):
            e = tuple(sorted((int(tri[a]), int(tri[(a + 1) % 3]))))
            edges.setdefault(e, []).append(t)
    rows, cols = [], []
    for e, ts in edges.items():
        for k in range(len(ts) - 1):
            rows.append(ts[k])
            cols.append(ts[k + 1])
    n = len(F)
    G = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, lab = csgraph.connected_components(G, directed=False)
    return [idx[lab == c] for c in np.unique(lab)]


def _second_moments(mesh: Mesh, tris: np.ndarray):
    """
    Principal axes of a group of triangles, area weighted.

    Computed in 3D and then projected onto the lesion's own tangent plane. It
    used to be computed on (x, y) directly, which meant that on the LIFTED mesh
    level C returned PROJECTED morphometry: the same as level B, compressed by
    cos(v/R), which near the margin reaches 0.54.

    Each triangle's own inertia is added with its exact formula. For a triangle
    with vertices p_i and centroid g, the covariance of the uniform distribution
    is (1/12) sum_i (p_i - g)(p_i - g)^T; verified against the unit triangle,
    where Var(x) = 1/18 and Cov(x,y) = -1/36. A crude isotropic correction with
    the mean area was used before, which widened thin lesions and underestimated
    their elongation.

    Returns the centroid (in the image plane, so it can be plotted), the major
    and minor axes as 4 sigma, and the orientation in degrees with 0 = parallel
    to the midrib and 90 = perpendicular.
    """
    V = mesh.V
    A = fem.triangle_areas(V, mesh.F)[tris]
    P = V[mesh.F[tris]]                       # (t, 3, 3) vertices per triangle
    C = P.mean(axis=1)                        # centroids in 3D
    w = A / max(A.sum(), 1e-12)

    mu = (C * w[:, None]).sum(axis=0)
    d = C - mu
    cov = (w[:, None, None] * np.einsum("ti,tj->tij", d, d)).sum(axis=0)
    # own inertia of each triangle, exact (parallel axis theorem)
    q = P - C[:, None, :]
    cov += (w[:, None, None] * np.einsum("tai,taj->tij", q, q) / 12.0).sum(axis=0)

    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # the first two eigenvectors span the lesion's tangent plane; the third is
    # the normal and its eigenvalue only measures local curvature
    length = 4.0 * np.sqrt(max(vals[0], 0.0))
    width = 4.0 * np.sqrt(max(vals[1], 0.0))
    axis = vecs[:, 0]
    # 0 degrees = parallel to the midrib (x axis), 90 = perpendicular
    ang = np.degrees(np.arccos(np.clip(abs(axis[0]) / max(np.linalg.norm(axis), 1e-12),
                                       0.0, 1.0)))
    return mu[:2], length, width, ang


def analyse(mesh: Mesh, dist_nodes: np.ndarray, scale_cm: float,
            straight_nodes: np.ndarray | None = None) -> dict:
    """
    Full set of foliar parameters.

    mesh           : mesh (flat or lifted; areas come from whichever is passed)
    dist_nodes     : nodal geodesic distance to the midrib
    scale_cm       : decay scale of the weighted index
    straight_nodes : straight-line distance to the midrib; if given, the
                     geodesic/straight detour is reported, which is what reveals
                     a tear forcing a path around
    """
    A_tri = fem.triangle_areas(mesh.V, mesh.F)
    total_area = float(A_tri.sum())
    lesion_area = float(A_tri[mesh.attr == REG_LESION].sum())
    midrib_area = float(A_tri[mesh.attr == REG_MIDRIB].sum())

    d_tri = fem.interpolate_to_triangles(dist_nodes, mesh.F)
    reachable = np.isfinite(d_tri)
    s_tri = (fem.interpolate_to_triangles(straight_nodes, mesh.F)
             if straight_nodes is not None else None)

    groups = lesion_components(mesh)
    records: list[LesionRecord] = []
    for i, tris in enumerate(groups):
        a = float(A_tri[tris].sum())
        mu, length, width, ang = _second_moments(mesh, tris)
        dsub = d_tri[tris]
        ok = np.isfinite(dsub)
        if ok.any():
            wgt = A_tri[tris][ok]
            dmean = float((dsub[ok] * wgt).sum() / wgt.sum())
            dmin = float(np.nanmin(dsub[ok]))
            detached = False
        else:
            dmean = dmin = float("nan")
            detached = True
        if s_tri is not None and np.isfinite(dmean):
            ssub = s_tri[tris]
            sok = np.isfinite(ssub)
            smean = float((ssub[sok] * A_tri[tris][sok]).sum() /
                          max(A_tri[tris][sok].sum(), 1e-12)) if sok.any() else np.nan
            ratio = float(dmean / smean) if smean > 1e-6 else float("nan")
        else:
            ratio = float("nan")
        records.append(LesionRecord(
            id=i, area_cm2=a, centroid=(float(mu[0]), float(mu[1])),
            length_cm=float(length), width_cm=float(width),
            elongation=float(length / max(width, 1e-9)),
            orientation_deg=float(ang),
            dist_to_midrib_cm=dmean, dist_min_cm=dmin,
            detour_ratio=ratio, detached=detached,
        ))

    severity = 100.0 * lesion_area / max(total_area, 1e-9)
    grade, desc = stover_gauhl_grade(severity, len(records))

    # severity index weighted by proximity to the midrib
    w = np.where(reachable, np.exp(-np.nan_to_num(d_tri, nan=1e9) / scale_cm), 0.0)
    num = float((w * A_tri * (mesh.attr == REG_LESION)).sum())
    den = float((w * A_tri).sum())
    wsi = 100.0 * num / max(den, 1e-9)

    les_area_reach = A_tri[(mesh.attr == REG_LESION) & reachable]
    d_les = d_tri[(mesh.attr == REG_LESION) & reachable]
    mean_geo = float((d_les * les_area_reach).sum() / max(les_area_reach.sum(), 1e-9)) \
        if les_area_reach.size else float("nan")

    return dict(
        area_cm2=total_area,
        lesion_area_cm2=lesion_area,
        midrib_area_cm2=midrib_area,
        severity_pct=severity,
        stover_gauhl_grade=grade,
        stover_gauhl_desc=desc,
        n_lesions=len(records),
        mean_geodesic_to_midrib_cm=mean_geo,
        weighted_severity_index=wsi,
        n_detached_lesions=sum(r.detached for r in records),
        detached_area_cm2=float(A_tri[~reachable].sum()),
        n_lesions_with_detour=sum(1 for r in records
                                  if np.isfinite(r.detour_ratio) and r.detour_ratio > 1.10),
        max_detour_ratio=float(np.nanmax([r.detour_ratio for r in records]))
            if records and np.isfinite([r.detour_ratio for r in records]).any() else float("nan"),
        perimeter_cm=fem.boundary_length(mesh.V, mesh.F),
        mean_lesion_elongation=float(np.mean([r.elongation for r in records]))
            if records else float("nan"),
        mean_lesion_orientation_deg=float(np.mean([r.orientation_deg for r in records]))
            if records else float("nan"),
        lesions=records,
    )


def perimeter_from_polygon(poly) -> float:
    """Perimeter of the continuous contour, with no digitisation bias."""
    if poly.is_empty:
        return 0.0
    geoms = poly.geoms if hasattr(poly, "geoms") else [poly]
    return float(sum(g.exterior.length + sum(r.length for r in g.interiors)
                     for g in geoms))
