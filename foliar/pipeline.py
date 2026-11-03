"""
Full pipeline: masks -> polygons -> mesh -> FEM -> parameters.

It runs at three levels so that what each ingredient buys stays explicit:

  A  raster         pixel counting + Euclidean distance in the image
  B  flat mesh      continuous geometry; respects tears. Still measures on the
                    projection, so it still cannot see the curl
  C  lifted mesh    also corrects the curl. Needs R from an external source:
                    it is NOT recoverable from a single view
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import baseline, fem, meshing, params


@dataclass
class PipelineResult:
    polygons: dict
    mesh_flat: meshing.Mesh
    mesh_lifted: meshing.Mesh | None
    fields: dict
    level_A: dict
    level_B: dict
    level_C: dict | None
    quality: dict


def run(sample, curl_radius_cm: float | None = None,
        scale_cm: float | None = None,
        geodesic_method: str = "dijkstra2",
        simplify_px: float = 0.30,
        target_edge_cm: float = 2.0,
        lesion_edge_cm: float = 0.45) -> PipelineResult:
    px = sample.px_per_cm
    org = sample.origin_cm

    blade = meshing.mask_to_polygon(sample.mask_blade, org, px, simplify_px)
    lesion = meshing.mask_to_polygon(sample.mask_lesion, org, px, simplify_px * 0.6,
                                     min_area_px=2.0)
    midrib = meshing.mask_to_polygon(sample.mask_midrib, org, px, simplify_px)

    mesh = meshing.build_mesh(blade, lesion, midrib, px,
                              target_edge_cm=target_edge_cm,
                              lesion_edge_cm=lesion_edge_cm)

    # source nodes = midrib / healthy blade interface
    mid_nodes = set(mesh.vertices_in(meshing.REG_MIDRIB).tolist())
    heal_nodes = set(mesh.vertices_in(meshing.REG_HEALTHY).tolist())
    src = np.array(sorted(mid_nodes & heal_nodes), dtype=int)
    if src.size == 0:                                    # isolated midrib
        src = np.array(sorted(mid_nodes), dtype=int)

    if scale_cm is None:
        scale_cm = 0.5 * float(np.nanmax(fem.dijkstra_distance(mesh.V, mesh.F, src, ring=1)))

    # ---------------- A: raster ----------------
    lvl_a = baseline.raster_report(sample, scale_cm=scale_cm)
    lvl_a["perimeter_cm"] = lvl_a["perimeter_crofton_cm"]

    # ---------------- B: flat mesh ----------------
    d_flat = fem.geodesic(mesh.V, mesh.F, src, method=geodesic_method)
    straight_flat = cKDTree(mesh.V[src][:, :2]).query(mesh.V[:, :2], k=1)[0]
    lvl_b = params.analyse(mesh, d_flat, scale_cm, straight_nodes=straight_flat)

    # ---------------- C: lifted mesh ----------------
    lvl_c = None
    mesh_l = None
    d_lift = None
    if curl_radius_cm is not None:
        mesh_l = meshing.lift_to_curl(mesh, curl_radius_cm)
        d_lift = fem.geodesic(mesh_l.V, mesh_l.F, src, method=geodesic_method)
        # The reference straight line has to be measured in the SAME metric as
        # the geodesic, i.e. in 3D on the lifted mesh. Comparing a 3D arc
        # against a 2D chord gives a ratio that is already v/(R sin(v/R)) from
        # the curl alone -- 1.19 at the margin -- and that was being read as
        # "detour around a tear" when it was nothing of the kind.
        straight_lift = cKDTree(mesh_l.V[src]).query(mesh_l.V, k=1)[0]
        lvl_c = params.analyse(mesh_l, d_lift, scale_cm, straight_nodes=straight_lift)

    return PipelineResult(
        polygons=dict(blade=blade, lesion=lesion, midrib=midrib),
        mesh_flat=mesh, mesh_lifted=mesh_l,
        fields=dict(d_flat=d_flat, d_lift=d_lift, straight=straight_flat,
                    source=src, scale_cm=scale_cm),
        level_A=lvl_a, level_B=lvl_b, level_C=lvl_c,
        quality=meshing.mesh_quality(mesh),
    )
