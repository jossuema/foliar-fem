#!/usr/bin/env python3
"""
Orientation spread of EVERY estimator, under one single definition.

    spread = max(error) - min(error),  in PERCENTAGE POINTS,
    over rotations from 0 to 45 degrees in steps of 5, at the stated resolution.

Why it had to be redefined
--------------------------
The old definition, max(value)/min(value) - 1, fails for two reasons:

1. It means nothing when the errors cross zero. Crofton-4 runs from -5.18 % to
   +2.52 %: that is 7.71 POINTS of travel, not "8.13 %".

2. It is scale invariant, so two estimators differing by a constant factor give
   exactly the same number. That is what happened with Crofton and marching
   squares, and it was no coincidence: on any mask with NO saddle configuration
   (a 2x2 window with the two pixels on a diagonal),

       ms_binary_length  ==  (4/pi)*(2*sqrt(2)-2) * skimage.perimeter_crofton(4)

   EXACTLY, the constant being Kulpa's factor 1.05478618. Verified by exhaustive
   enumeration in `dss.py selftest` (check g) and on the leaf mask, which has 0
   saddles over 251,937 windows. With the definition in percentage points the
   relation is visible: spread(Crofton) = spread(MS) / 1.05478618.

Test shape: square of side 1.6 centred at the origin, domain [-1.6, 1.6]^2,
NODE grid (linspace with endpoints), h = 3.2/(n-1).

Usage
-----
    python3 spread.py           # n = 1600 and n = 3200
    python3 spread.py 3200      # one resolution only
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dss as D  # noqa: E402

HALF = D.HALF          # 1.6
SIDE = 1.6
P_SQUARE = 6.4
ANGLES = [float(a) for a in range(0, 50, 5)]


def phi_square(XX, YY, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    xr, yr = c * XX + s * YY, -s * XX + c * YY
    q = 0.5 * SIDE
    return np.minimum(np.minimum(q - xr, q + xr), np.minimum(q - yr, q + yr))


def contour_len(field, g, level):
    d = float(g[1] - g[0])
    tot = 0.0
    for c in measure.find_contours(field, level):
        pts = np.column_stack([g[0] + c[:, 1] * d, g[0] + c[:, 0] * d])
        tot += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return tot


def levelset(n, deg):
    g = np.linspace(-HALF, HALF, n)
    XX, YY = np.meshgrid(g, g)
    return contour_len(phi_square(XX, YY, deg), g, 0.0)


def levelset_richardson(n, deg, r=3):
    """Order-1 Richardson between h and h/r, as in synth.perimeter_levelset."""
    pc = levelset(n, deg)
    pf = levelset((n - 1) * r + 1, deg)
    return (r * pf - pc) / (r - 1)


RASTER = [("MS binary 0.5", D.ms_binary_length),
          ("skimage.measure.perimeter", D.sk_perimeter),
          ("skimage.perimeter_crofton(4)", D.sk_crofton4),
          ("cv2.arcLength", D.cv_arclength),
          ("DSS", D.dss_length)]
FIELD = [("zero level set", levelset),
         ("zero level set + Richardson", levelset_richardson)]


def run(n):
    t0 = time.time()
    vals = {}
    for a in ANGLES:
        m, h = D.mask_square(n, side=SIDE, deg=a)
        for nm, fn in RASTER:
            v = fn(m, h)
            if v is not None:
                vals.setdefault(nm, {})[a] = v
        del m
        for nm, fn in FIELD:
            vals.setdefault(nm, {})[a] = fn(n, a)

    h = 3.2 / (n - 1)
    print(f"\n=== n = {n}   h = {h:.6f}   side/h = {SIDE / h:.0f} px   "
          f"({time.time() - t0:.0f} s)")
    print(f"{'estimator':<30} {'min err %':>10} {'max err %':>10} "
          f"{'spread pp':>11} {'max|err| %':>11}")
    print("-" * 76)
    ref = None
    for nm in [k for k, _ in RASTER] + [k for k, _ in FIELD]:
        if nm not in vals:
            continue
        v = np.array([vals[nm][a] for a in ANGLES])
        e = 100.0 * (v / P_SQUARE - 1.0)
        sp = float(e.max() - e.min())
        if ref is None:
            ref = sp
        print(f"{nm:<30} {e.min():+10.4f} {e.max():+10.4f} {sp:11.4f} "
              f"{np.abs(e).max():11.4f}")
    print()
    k = 4.0 / math.pi * (2.0 * math.sqrt(2.0) - 2.0)
    ms = np.array([vals["MS binary 0.5"][a] for a in ANGLES])
    cr = np.array([vals["skimage.perimeter_crofton(4)"][a] for a in ANGLES])
    print(f"  Kulpa identity: max|MS - {k:.8f}*Crofton| = "
          f"{np.abs(ms - k * cr).max():.2e} cm over the {len(ANGLES)} angles")
    for nm in ("DSS", "zero level set"):
        v = np.array([vals[nm][a] for a in ANGLES])
        e = 100.0 * (v / P_SQUARE - 1.0)
        print(f"  factor MS / {nm:<16s} = {ref / (e.max() - e.min()):.0f}")
    return vals


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1:]] or [1600, 3200]
    for n in ns:
        run(n)
