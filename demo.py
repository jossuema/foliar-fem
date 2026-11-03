#!/usr/bin/env python3
"""
End-to-end demo: synthetic leaf -> masks -> mesh -> FEM -> parameters.

    python3 demo.py

Writes figures and tables to out/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from foliar import fem, meshing, pipeline, synth, viz

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def banner(t):
    print(f"\n\033[1m{t}\033[0m\n" + "-" * len(t))


def main():
    banner("1 · Synthetic leaf")
    cfg = synth.LeafConfig()
    t0 = time.time()
    sample = synth.generate(cfg)
    gt = sample.truth
    print(f"image            {sample.mask_blade.shape[1]} x {sample.mask_blade.shape[0]} px "
          f"at {sample.px_per_cm:g} px/cm   ({time.time() - t0:.1f} s)")
    print(f"true area        {gt.blade_area_cm2:9.1f} cm2")
    print(f"projected area   {gt.projected_area_cm2:9.1f} cm2   "
          f"({100 * (gt.projected_area_cm2 / gt.blade_area_cm2 - 1):+.1f} % from the curl)")
    print(f"true severity    {gt.severity_pct:9.3f} %      "
          f"(projected {gt.projected_severity_pct:.3f} %)")
    print(f"lesions          {gt.n_lesions:9d}")
    print(f"torn blade       {gt.torn_area_cm2:9.1f} cm2")

    banner("2 · Pipeline")
    t0 = time.time()
    res = pipeline.run(sample, curl_radius_cm=sample.curl_radius_cm,
                       scale_cm=0.5 * cfg.halfwidth_cm)
    print(f"mesh             {res.quality['n_triangles']} triangles, "
          f"{res.quality['n_vertices']} nodes, minimum angle "
          f"{res.quality['min_angle_deg']:.1f} degrees   ({time.time() - t0:.1f} s)")

    A, B, C = res.level_A, res.level_B, res.level_C

    banner("3 · Agronomic parameters (level C, lifted mesh)")
    print(f"leaf area                      {C['area_cm2']:9.1f} cm2")
    print(f"perimeter                      {C['perimeter_cm']:9.1f} cm")
    print(f"severity                       {C['severity_pct']:9.3f} %")
    print(f"Stover-Gauhl grade             {C['stover_gauhl_grade']:9d}  -> {C['stover_gauhl_desc']}")
    print(f"lesions detected               {C['n_lesions']:9d}")
    print(f"mean geodesic dist. to midrib  {C['mean_geodesic_to_midrib_cm']:9.2f} cm")
    print(f"proximity-weighted index       {C['weighted_severity_index']:9.3f}")
    print(f"mean lesion elongation         {C['mean_lesion_elongation']:9.2f}")
    print(f"mean orientation               {C['mean_lesion_orientation_deg']:9.1f} degrees "
          f"(90 = perpendicular to the midrib, i.e. along the secondary veins)")
    print(f"lesions forced to detour       {C['n_lesions_with_detour']:9d}  "
          f"(maximum detour x{C['max_detour_ratio']:.2f})")
    print(f"lesions on loose strips        {C['n_detached_lesions']:9d}")

    # ---------------------------------------------------------------- table
    banner("4 · Error against the ground truth")
    rows = [
        ("Leaf area (cm2)", gt.blade_area_cm2,
         A["area_cm2"], B["area_cm2"], C["area_cm2"]),
        ("Perimeter (cm)", gt.perimeter_cm,
         A["perimeter_crofton_cm"], B["perimeter_cm"], C["perimeter_cm"]),
        ("Severity (%)", gt.severity_pct,
         A["severity_pct"], B["severity_pct"], C["severity_pct"]),
        ("Dist. to midrib (cm)", gt.mean_geodesic_to_midrib_cm,
         A["mean_dist_euclidean_cm"], B["mean_geodesic_to_midrib_cm"],
         C["mean_geodesic_to_midrib_cm"]),
        ("Weighted index", gt.weighted_severity_index,
         A["wsi_euclidean"], B["weighted_severity_index"], C["weighted_severity_index"]),
    ]
    tab = pd.DataFrame(rows, columns=["parameter", "truth",
                                      "A_raster", "B_flat_mesh", "C_lifted_mesh"])
    for col in ["A_raster", "B_flat_mesh", "C_lifted_mesh"]:
        tab[f"err_{col}_%"] = 100.0 * (tab[col] / tab["truth"] - 1.0)
    show = tab[["parameter", "truth", "A_raster", "err_A_raster_%",
                "B_flat_mesh", "err_B_flat_mesh_%",
                "C_lifted_mesh", "err_C_lifted_mesh_%"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
    tab.to_csv(OUT / "comparison.csv", index=False)

    # ------------------------------------------------------ sensitivity to R
    banner("5 · Sensitivity to the external curvature input")
    # The lift is only DEFINED if R exceeds the projected half-width of the mesh.
    # Below that lift_to_curl raises ValueError instead of clipping silently, so
    # those rows are reported as undefined and not as a sensitivity. They used to
    # produce numbers that looked like a measurement.
    y_max = float(np.abs(res.mesh_flat.V[:, 1]).max())
    print(f"  the projected half-width of the mesh is {y_max:.2f} cm, so the model")
    print(f"  requires R > {y_max:.2f} cm ({100 * (y_max / sample.curl_radius_cm - 1):+.1f} % "
          f"error in R)\n")
    radii, areas = [], []
    for f in [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]:
        R = sample.curl_radius_cm * f
        lbl = f"  R = {R:6.2f} cm ({100 * (R / sample.curl_radius_cm - 1):+5.0f} %)"
        try:
            ml = meshing.lift_to_curl(res.mesh_flat, R)
        except ValueError:
            print(f"{lbl}  -> UNDEFINED (R below the projected half-width)")
            continue
        a = fem.surface_area(ml.V, ml.F)
        radii.append(R)
        areas.append(a)
        print(f"{lbl}  -> area {a:9.1f} cm2  "
              f"({100 * (a / gt.blade_area_cm2 - 1):+6.2f} %)")

    # -------------------------------------------------------------- lesions
    les = pd.DataFrame([{
        "id": r.id, "area_cm2": r.area_cm2,
        "x_cm": r.centroid[0], "y_cm": r.centroid[1],
        "length_cm": r.length_cm, "width_cm": r.width_cm,
        "elongation": r.elongation, "orientation_deg": r.orientation_deg,
        "dist_midrib_cm": r.dist_to_midrib_cm, "dist_min_cm": r.dist_min_cm,
        "detour": r.detour_ratio, "detached": r.detached,
    } for r in C["lesions"]]).sort_values("area_cm2", ascending=False)
    les.to_csv(OUT / "lesions.csv", index=False)

    banner("6 · Figures")
    for p in [
        viz.figure_pipeline(sample, res, OUT / "01_pipeline.png"),
        viz.figure_comparison(sample, res, OUT / "02_comparison.png"),
        viz.figure_curl_sensitivity(sample, res, radii, areas, OUT / "03_curl_sensitivity.png"),
    ]:
        print(f"  {p}")
    print(f"  {OUT / 'comparison.csv'}")
    print(f"  {OUT / 'lesions.csv'}")

    summary = {k: v for k, v in C.items() if k != "lesions"}
    summary["ground_truth"] = {k: v for k, v in gt.__dict__.items()
                               if not k.startswith("_")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"  {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
