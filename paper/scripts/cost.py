#!/usr/bin/env python3
"""
Computational cost of the three levels on the reference leaf.

    python3 cost.py [--repeats N]

The paper claims the biased reference inverted the ranking in favour of the
EXPENSIVE method. That sentence cannot be rhetoric: it is measured here.

What is measured is the cost of turning masks into parameters, not that of
generating the synthetic leaf, which does not exist in a real deployment.
Generation is kept off the clock on purpose.

    A  raster        mask -> Crofton + fast marching
    B  flat mesh     mask -> contour -> CDT -> FEM geodesic -> params
    C  lifted mesh   B + cylindrical lift + a second geodesic

ONLY TIME IS REPORTED, on purpose. An earlier version of this script also
reported peak memory and those numbers did NOT reproduce: whichever level runs
first pays the lazy imports of skimage and scipy plus the first-touch page
faults, so they were charged to that level. Three consecutive runs gave
22.1 / 6.3 / 0.6 MB for level A. Time does reproduce because it has warm-up and
a median over N runs. Before adding memory back it needs the same treatment,
and even then the number supports no argument: these are tens of MB.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial import cKDTree

from foliar import baseline, fem, meshing, params, synth

# The weighted index scale is fixed from outside so that level A does not depend
# on level B's mesh. demo.py uses this same value.
SCALE_FRAC = 0.5


def build_sample():
    cfg = synth.LeafConfig()                      # seed 7 by default
    return synth.generate(cfg), SCALE_FRAC * cfg.halfwidth_cm


def stage_A(sample, scale_cm):
    """Level A: pixel counting, 4-direction Crofton, fast marching."""
    return baseline.raster_report(sample, scale_cm=scale_cm)


def stage_B(sample, scale_cm):
    """Level B: subpixel contour -> planar partition -> CDT -> FEM."""
    px, org = sample.px_per_cm, sample.origin_cm
    blade = meshing.mask_to_polygon(sample.mask_blade, org, px, 0.30)
    lesion = meshing.mask_to_polygon(sample.mask_lesion, org, px, 0.30 * 0.6,
                                     min_area_px=2.0)
    midrib = meshing.mask_to_polygon(sample.mask_midrib, org, px, 0.30)
    mesh = meshing.build_mesh(blade, lesion, midrib, px,
                              target_edge_cm=2.0, lesion_edge_cm=0.45)
    mid = set(mesh.vertices_in(meshing.REG_MIDRIB).tolist())
    heal = set(mesh.vertices_in(meshing.REG_HEALTHY).tolist())
    src = np.array(sorted(mid & heal), dtype=int)
    if src.size == 0:
        src = np.array(sorted(mid), dtype=int)
    d = fem.geodesic(mesh.V, mesh.F, src, method="dijkstra2")
    straight = cKDTree(mesh.V[src][:, :2]).query(mesh.V[:, :2], k=1)[0]
    out = params.analyse(mesh, d, scale_cm, straight_nodes=straight)
    return out, mesh, src


def stage_C_extra(mesh, src, sample, scale_cm):
    """What level C adds ON TOP of level B, not the whole of level C."""
    lifted = meshing.lift_to_curl(mesh, sample.curl_radius_cm)
    d = fem.geodesic(lifted.V, lifted.F, src, method="dijkstra2")
    straight = cKDTree(lifted.V[src]).query(lifted.V, k=1)[0]
    return params.analyse(lifted, d, scale_cm, straight_nodes=straight)


def timeit(fn, repeats):
    """Median of `repeats` runs after one warm-up."""
    fn()                                          # warm-up: imports and caches
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    sample, scale_cm = build_sample()
    r = args.repeats

    tA = timeit(lambda: stage_A(sample, scale_cm), r)
    tB = timeit(lambda: stage_B(sample, scale_cm), r)

    _, mesh, src = stage_B(sample, scale_cm)
    tCx = timeit(lambda: stage_C_extra(mesh, src, sample, scale_cm), r)

    q = meshing.mesh_quality(mesh)
    tC = (tB[0] + tCx[0], tB[1] + tCx[1], tB[2] + tCx[2])

    print(f"leaf seed 7 · {sample.mask_blade.shape[1]} x "
          f"{sample.mask_blade.shape[0]} px at {sample.px_per_cm:g} px/cm")
    print(f"mesh · {q['n_triangles']} triangles, {q['n_vertices']} nodes, "
          f"minimum angle {q['min_angle_deg']:.1f} degrees")
    print(f"median of {r} runs after warm-up\n")
    print(f"{'level':<22}{'median (s)':>13}{'min-max (s)':>18}{'x A':>8}")
    print("-" * 61)
    for name, t in (("A · raster", tA),
                    ("B · flat mesh", tB),
                    ("C · lifted mesh", tC)):
        print(f"{name:<22}{t[0]:>13.3f}{f'{t[1]:.3f}-{t[2]:.3f}':>18}"
              f"{t[0] / tA[0]:>8.1f}")
    print(f"\nthe lift adds {tCx[0]:.3f} s on top of level B")
    print(f"level C / level A = {tC[0] / tA[0]:.1f}x in time")


if __name__ == "__main__":
    main()
