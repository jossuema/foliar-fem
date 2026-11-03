#!/usr/bin/env python3
"""
The original, biased ground truth: marching squares on the binary indicator.

Reproduces what synth.py used to report as the true perimeter, on the exact
ground truth grid, and measures its bias against the zero level set estimator
that replaced it. This is the incorporation bias of the paper, in the one place
where it mattered most: the reference itself.
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

# Zero level set reference for seed 7, from grids 8 times finer than the ground
# truth's. These are the values synth.perimeter_levelset reproduces.
REF_INTRINSIC = 744.2254917143233
REF_PROJECTED = 686.9815602950247

cfg = synth.LeafConfig()
leaf = synth.SyntheticLeaf(cfg)
R = cfg.curl_radius_cm

ug = np.linspace(0.0, cfg.length_cm, cfg.gt_samples_u)
vg = np.linspace(-cfg.halfwidth_cm, cfg.halfwidth_cm, cfg.gt_samples_v)
du, dv = ug[1] - ug[0], vg[1] - vg[0]
UG, VG = np.meshgrid(ug, vg)

blade = leaf.in_blade(UG, VG).astype(float)
p_int = p_prj = 0.0
for c in measure.find_contours(blade, 0.5):
    u = ug[0] + c[:, 1] * du
    v = vg[0] + c[:, 0] * dv
    p_int += float(np.sum(np.linalg.norm(
        np.diff(np.column_stack([u, v]), axis=0), axis=1)))
    p_prj += float(np.sum(np.linalg.norm(
        np.diff(np.column_stack([u, R * np.sin(v / R)]), axis=0), axis=1)))

print(f"exact GT grid {cfg.gt_samples_u}x{cfg.gt_samples_v} "
      f"(du={du:.4f} dv={dv:.4f})")
print(f"  binary indicator: intrinsic {p_int:.3f} cm   projected {p_prj:.3f} cm")
print(f"  zero level set:   intrinsic {REF_INTRINSIC:.3f} cm   "
      f"projected {REF_PROJECTED:.3f} cm")
print(f"  bias: {100 * (p_int / REF_INTRINSIC - 1):+.3f} % intrinsic, "
      f"{100 * (p_prj / REF_PROJECTED - 1):+.3f} % projected")
