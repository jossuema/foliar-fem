"""
Mask -> subpixel contour -> clean planar partition -> CDT mesh.

The non-trivial step is that blade, lesions and midrib share boundaries: a
lesion touching the margin produces duplicate segments and breaks the
triangulation. The fix is to build a planar partition with shapely first
(unary_union nodes every intersection, polygonize rebuilds the faces) and only
then assemble the PSLG.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import shapely
import triangle as tr
from shapely.geometry import MultiLineString, Point, Polygon
from shapely.ops import polygonize, unary_union
from skimage import measure

# region attributes
REG_HEALTHY, REG_LESION, REG_MIDRIB = 0, 1, 2


@dataclass
class Mesh:
    V: np.ndarray            # (n, 3) vertices; z = 0 if the mesh is flat
    F: np.ndarray            # (m, 3) triangles
    attr: np.ndarray         # (m,) region attribute per triangle
    px_per_cm: float
    lifted: bool = False
    min_angle_deg: float = float("nan")   # effective quality, not the requested one
    steiner_capped: bool = False          # True if the -S cap limited refinement

    @property
    def V2(self) -> np.ndarray:
        return self.V[:, :2]

    def is_lesion(self) -> np.ndarray:
        return self.attr == REG_LESION

    def is_midrib(self) -> np.ndarray:
        return self.attr == REG_MIDRIB

    def vertices_in(self, attr_value: int) -> np.ndarray:
        """Indices of vertices belonging to triangles of that region."""
        sel = self.F[self.attr == attr_value]
        return np.unique(sel.ravel()) if sel.size else np.array([], dtype=int)


# --------------------------------------------------------------------------
# Mask -> polygons
# --------------------------------------------------------------------------
def mask_to_polygon(mask: np.ndarray, origin_cm, px_per_cm: float,
                    simplify_px: float = 0.30,
                    min_area_px: float = 2.0) -> shapely.MultiPolygon | Polygon:
    """Subpixel contours (marching squares) -> polygon with holes, in cm."""
    if not mask.any():
        return Polygon()

    padded = np.pad(mask.astype(float), 1)
    min_area_cm2 = min_area_px / (px_per_cm ** 2)
    rings = []
    for c in measure.find_contours(padded, 0.5):
        rc = c - 1.0                                    # undo the pad
        x = origin_cm[0] + rc[:, 1] / px_per_cm
        y = origin_cm[1] + rc[:, 0] / px_per_cm
        if len(x) < 4:
            continue
        p = Polygon(np.column_stack([x, y]))
        if not p.is_valid:
            p = p.buffer(0)
            if p.geom_type == "MultiPolygon":       # buffer(0) can split it
                p = max(p.geoms, key=lambda g: g.area)
        if p.is_empty or p.geom_type != "Polygon" or p.area < min_area_cm2:
            continue
        rings.append(p)

    if not rings:
        return Polygon()

    # ---- even-odd rule ----
    # Depth is measured with a VERTEX of the ring, not with its representative
    # point. Marching squares contours at the same level never touch, so a
    # vertex of p lies strictly inside q if and only if p is nested in q. With
    # representative_point() the probe of a shell with a hole falls INSIDE its
    # own hole, the shell gets classified as a hole and the hole as a shell, and
    # the function returned an EMPTY polygon for any mask with an interior hole.
    rings.sort(key=lambda p: p.area, reverse=True)
    probes = [Point(p.exterior.coords[0]) for p in rings]
    depth = [sum(1 for j, q in enumerate(rings) if j != i and q.contains(probes[i]))
             for i in range(len(rings))]

    # Each shell (even depth) is assembled with its IMMEDIATE holes (odd depth,
    # one level down, contained in it). Doing union(shells).difference(union(holes))
    # is wrong as soon as there is nesting three levels deep or more: the
    # interior island would be lost.
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

    result = unary_union(pieces) if pieces else Polygon()

    tol = simplify_px / px_per_cm
    if tol > 0:
        result = result.simplify(tol, preserve_topology=True)
    if not result.is_valid:
        result = result.buffer(0)
    return result


# --------------------------------------------------------------------------
# Planar partition -> PSLG
# --------------------------------------------------------------------------
def _pslg_from_lines(lines) -> tuple[np.ndarray, np.ndarray]:
    geoms = lines.geoms if hasattr(lines, "geoms") else [lines]
    index: dict[tuple[float, float], int] = {}
    verts: list[tuple[float, float]] = []
    segs: list[tuple[int, int]] = []

    def vid(p):
        key = (round(p[0], 9), round(p[1], 9))
        if key not in index:
            index[key] = len(verts)
            verts.append(key)
        return index[key]

    seen = set()
    for g in geoms:
        cs = list(g.coords)
        for a, b in zip(cs[:-1], cs[1:]):
            ia, ib = vid(a), vid(b)
            if ia == ib:
                continue
            e = (min(ia, ib), max(ia, ib))
            if e in seen:
                continue
            seen.add(e)
            segs.append(e)

    return np.asarray(verts, float), np.asarray(segs, int)


def build_mesh(blade_poly, lesion_poly, midrib_poly, px_per_cm: float,
               target_edge_cm: float = 2.0,
               lesion_edge_cm: float = 0.45,
               midrib_edge_cm: float = 0.8,
               min_angle: float = 28.0,
               max_steiner: int = 400_000) -> Mesh:
    """Constrained Delaunay triangulation with lesion / midrib subdomains."""
    lesion_poly = lesion_poly.intersection(blade_poly) if not lesion_poly.is_empty else lesion_poly
    midrib_poly = midrib_poly.intersection(blade_poly) if not midrib_poly.is_empty else midrib_poly

    boundaries = [blade_poly.boundary]
    if not lesion_poly.is_empty:
        boundaries.append(lesion_poly.boundary)
    if not midrib_poly.is_empty:
        boundaries.append(midrib_poly.boundary)

    # unary_union nodes the input: no two segments cross without sharing a vertex
    lines = unary_union([b for b in boundaries if not b.is_empty])
    if isinstance(lines, MultiLineString) is False and lines.geom_type == "LineString":
        lines = MultiLineString([lines])

    verts, segs = _pslg_from_lines(lines)

    # Faces of the planar partition -> regions and holes.
    # Classification goes by area overlap, not by contains() of the
    # representative point: on very thin faces that point lands on the boundary
    # and contains() returns False, which would open a spurious hole in the mesh.
    faces = list(polygonize(lines))
    regions, holes = [], []
    a_healthy = float(np.sqrt(3) / 4 * target_edge_cm ** 2)
    a_lesion = float(np.sqrt(3) / 4 * lesion_edge_cm ** 2)
    a_midrib = float(np.sqrt(3) / 4 * midrib_edge_cm ** 2)

    def covers(poly, face):
        if poly.is_empty or face.area <= 0:
            return False
        return face.intersection(poly).area > 0.5 * face.area

    for f in faces:
        pt = f.representative_point()
        if not covers(blade_poly, f):
            holes.append([pt.x, pt.y])
            continue
        if covers(midrib_poly, f):
            regions.append([pt.x, pt.y, REG_MIDRIB, a_midrib])
        elif covers(lesion_poly, f):
            regions.append([pt.x, pt.y, REG_LESION, a_lesion])
        else:
            regions.append([pt.x, pt.y, REG_HEALTHY, a_healthy])

    pslg = dict(vertices=verts, segments=segs)
    if holes:
        pslg["holes"] = np.asarray(holes, float)
    if regions:
        pslg["regions"] = np.asarray(regions, float)

    # -S bounds the Steiner points. Ruppert does not terminate if the PSLG
    # carries very small input angles, and a real mask always carries one.
    flags = f"pq{min_angle:g}aAS{max_steiner:d}"
    out = tr.triangulate(pslg, flags)

    V2 = np.asarray(out["vertices"], float)
    F = np.asarray(out["triangles"], int)
    attr = np.asarray(out.get("triangle_attributes",
                              np.zeros((len(F), 1)))).ravel().astype(int)

    V = np.column_stack([V2, np.zeros(len(V2))])
    mesh = Mesh(V=V, F=F, attr=attr, px_per_cm=px_per_cm, lifted=False)

    # When the -S cap fires, Triangle returns a mesh that does NOT meet the
    # requested minimum angle, with no error and no warning. That is worse than
    # a hang: the degenerate mesh keeps running and contaminates everything
    # downstream. It is checked and reported, and recorded on the object so that
    # whoever consumes the mesh can decide.
    mesh.min_angle_deg = float(mesh_quality(mesh)["min_angle_deg"])
    mesh.steiner_capped = bool(mesh.min_angle_deg < min_angle - 1.0)
    if mesh.steiner_capped:
        warnings.warn(
            f"the mesher did not reach the requested minimum angle: {mesh.min_angle_deg:.2f} "
            f"degrees against {min_angle:g} requested, with {len(F)} triangles. "
            f"Likely the Steiner point cap (max_steiner={max_steiner}). "
            f"The mesh is usable but its quality is NOT guaranteed.",
            RuntimeWarning, stacklevel=2)
    return mesh


def lift_to_curl(mesh: Mesh, curl_radius_cm: float) -> Mesh:
    """
    Lift the flat mesh onto the cylinder of radius R about the midrib axis.

        z = R - sqrt(R^2 - y^2)

    This is the exact inverse of the model's orthographic projection. It needs R
    from an external source (stereo pair, depth camera, scale fiducial): R is
    NOT recoverable from a single view.

    The lift is only DEFINED if R is at least the projected half-width of the
    mesh: the square root requires |y| <= R. Below that the geometric model is
    inconsistent, not "less accurate", and an exception is raised. It used to
    clip silently with np.clip, which crushed a whole strip of the margin
    against the singularity where dz/dy -> infinity and produced inflated areas
    that read as a sensitivity to R.
    """
    R = float(curl_radius_cm)
    y_max = float(np.abs(mesh.V[:, 1]).max())
    if not R > y_max:
        raise ValueError(
            f"the lift is undefined: R = {R:.4f} cm is less than or equal to the "
            f"projected half-width of the mesh, {y_max:.4f} cm. The model requires "
            f"|y| <= R. This mesh needs R > {y_max:.4f} cm.")
    y = mesh.V[:, 1]
    V = mesh.V.copy()
    V[:, 2] = R - np.sqrt(np.maximum(R * R - y * y, 0.0))
    out = Mesh(V=V, F=mesh.F.copy(), attr=mesh.attr.copy(),
               px_per_cm=mesh.px_per_cm, lifted=True,
               steiner_capped=mesh.steiner_capped)
    # the lift deforms the triangles, so the minimum angle of the lifted mesh is
    # not that of the flat one: it is measured again
    out.min_angle_deg = float(mesh_quality(out)["min_angle_deg"])
    return out


def mesh_quality(mesh: Mesh) -> dict:
    from .fem import triangle_areas
    V, F = mesh.V, mesh.F
    A = triangle_areas(V, F)
    e = [np.linalg.norm(V[F[:, (i + 1) % 3]] - V[F[:, i]], axis=1) for i in range(3)]
    e = np.stack(e, axis=1)
    # minimum angle per triangle, via the law of cosines
    ang = []
    for i in range(3):
        a, b, c = e[:, i], e[:, (i + 1) % 3], e[:, (i + 2) % 3]
        cosang = np.clip((b * b + c * c - a * a) / (2 * b * c + 1e-30), -1, 1)
        ang.append(np.degrees(np.arccos(cosang)))
    ang = np.stack(ang, axis=1)
    return dict(
        n_vertices=len(V), n_triangles=len(F),
        min_angle_deg=float(ang.min()),
        mean_edge_cm=float(e.mean()),
        area_min_cm2=float(A.min()), area_max_cm2=float(A.max()),
    )
