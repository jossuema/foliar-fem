#!/usr/bin/env python3
"""
Leaf perimeter: zero level set vs. binary indicator, intrinsic and projected.

The multigrid table at the end is the point: the zero level set settles, the
binary estimator also settles -- but on a different value. Convergence of the
reference is not evidence that the reference is right.
"""

import os
import sys

import numpy as np
from skimage import measure

# scripts/ lives two levels below the repository root: the root is derived from
# the location of THIS file, not from an absolute path on one machine.
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from foliar import synth  # noqa: E402


def measure_once(leaf, cfg, nu, nv, binary):
    """Contour length on one grid, intrinsic and pushed through the projection."""
    R = cfg.curl_radius_cm
    L = cfg.length_cm
    Hv = cfg.halfwidth_cm * 1.03
    ug = np.linspace(-0.03 * L, 1.03 * L, nu)
    vg = np.linspace(-Hv, Hv, nv)
    du, dv = ug[1] - ug[0], vg[1] - vg[0]
    UG, VG = np.meshgrid(ug, vg)
    if binary:
        fld, lvl = leaf.in_blade(UG, VG).astype(float), 0.5
    else:
        fld, lvl = leaf.blade_signed(UG, VG), 0.0
    p_int = p_prj = 0.0
    for c in measure.find_contours(fld, lvl):
        u_c = ug[0] + c[:, 1] * du
        v_c = vg[0] + c[:, 0] * dv
        p_int += float(np.sum(np.linalg.norm(
            np.diff(np.column_stack([u_c, v_c]), axis=0), axis=1)))
        p_prj += float(np.sum(np.linalg.norm(
            np.diff(np.column_stack([u_c, R * np.sin(v_c / R)]), axis=0), axis=1)))
    return p_int, p_prj


for seed in (7,):
    cfg = synth.LeafConfig(seed=seed)
    leaf = synth.SyntheticLeaf(cfg)
    nu0 = int(round(cfg.gt_samples_u * 1.06))
    nv0 = int(round(cfg.gt_samples_v * 1.03))
    print(f"seed {seed}  base grid {nu0} x {nv0}  "
          f"(h_u={1.06 * cfg.length_cm / (nu0 - 1):.4f} cm, "
          f"h_v={2 * 1.03 * cfg.halfwidth_cm / (nv0 - 1):.4f} cm)")

    p_int, p_prj, u_int, u_prj = synth.perimeter_levelset(leaf, cfg)
    print(f"  ZERO LEVEL (Richardson): intrinsic {p_int:.4f} cm  "
          f"projected {p_prj:.4f} cm")
    print(f"     uncertainty |extrap - fine|: {u_int:.4f} / {u_prj:.4f} cm")

    # binary on the same base grid and on the refined one
    b_c = measure_once(leaf, cfg, nu0, nv0, True)
    b_f = measure_once(leaf, cfg, (nu0 - 1) * 3 + 1, (nv0 - 1) * 3 + 1, True)
    b_R = ((3 * b_f[0] - b_c[0]) / 2, (3 * b_f[1] - b_c[1]) / 2)
    print(f"  BINARY base grid       : intrinsic {b_c[0]:.4f} cm  "
          f"projected {b_c[1]:.4f} cm")
    print(f"  BINARY grid x3         : intrinsic {b_f[0]:.4f} cm  "
          f"projected {b_f[1]:.4f} cm")
    print(f"  BINARY Richardson      : intrinsic {b_R[0]:.4f} cm  "
          f"projected {b_R[1]:.4f} cm")
    for nm, v in (("base", b_c), ("x3", b_f), ("Richardson", b_R)):
        print(f"    binary bias {nm:<11s} intrinsic {100 * (v[0] / p_int - 1):+.3f} %   "
              f"projected {100 * (v[1] / p_prj - 1):+.3f} %")

    print("  multigrid convergence on the leaf (h in cm, intrinsic contour):")
    print(f"   {'h':>8} {'n_u x n_v':>10} {'zero level':>14} {'binary':>14}")
    for h in (0.4, 0.2, 0.1, 0.05, 0.025):
        nu = int(round(1.06 * cfg.length_cm / h)) + 1
        nv = int(round(2.06 * cfg.halfwidth_cm / h)) + 1
        s = measure_once(leaf, cfg, nu, nv, False)
        b = measure_once(leaf, cfg, nu, nv, True)
        print(f"   {h:8.3f} {f'{nu}x{nv}':>10} {s[0]:14.4f} {b[0]:14.4f}   "
              f"(binary {100 * (b[0] / s[0] - 1):+0.3f} % over zero level)")
