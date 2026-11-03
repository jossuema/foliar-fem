#!/usr/bin/env python3
"""
Convergence of the perimeter estimator: signed field vs. binary field.

Three shapes with a closed-form perimeter, five resolutions each, and then the
orientation spread. The zero level set converges to the right value; the binary
indicator converges to a different one.
"""

import numpy as np
from scipy.special import ellipe
from skimage import measure


def contour_len(field, x, y, level):
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    tot = 0.0
    for c in measure.find_contours(field, level):
        pts = np.column_stack([x[0] + c[:, 1] * dx, y[0] + c[:, 0] * dy])
        tot += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return tot


def grid(n, half=1.6):
    g = np.linspace(-half, half, n)
    XX, YY = np.meshgrid(g, g)
    return g, XX, YY


A_E, B_E = 1.3, 0.7


def phi_circle(XX, YY):
    return 1.0 - np.hypot(XX, YY)


def phi_ellipse(XX, YY):
    return 1.0 - np.hypot(XX / A_E, YY / B_E)


def phi_sq(XX, YY, ang_deg):
    t = np.deg2rad(ang_deg)
    c, s = np.cos(t), np.sin(t)
    xr, yr = c * XX + s * YY, -s * XX + c * YY
    return np.minimum(np.minimum(0.8 - xr, 0.8 + xr),
                      np.minimum(0.8 - yr, 0.8 + yr))


P_CIR = 2 * np.pi
P_ELL = 4 * A_E * float(ellipe(1.0 - (B_E / A_E) ** 2))
P_SQ = 6.4

NS = [200, 400, 800, 1600, 3200]

# SINGLE DEFINITION: spread = max(error) - min(error), in PERCENTAGE POINTS,
# over rotations from 0 to 45 degrees in steps of 5.
#
# The previous definition, max(value)/min(value) - 1, is wrong for two reasons:
# it means nothing when the errors cross zero (Crofton runs from -5.18 % to
# +2.52 %, which is 7.71 POINTS of travel, not "8.13 %"), and it is scale
# invariant, so two estimators differing by a constant factor give the SAME
# number. See note (g) of dss.py selftest: on any mask with no saddle
# configuration, marching squares == 1.0547862 * Crofton-4 EXACTLY.
ANGS = tuple(float(a) for a in range(0, 50, 5))

print(f"exact P_circle  = {P_CIR:.10f}")
print(f"exact P_ellipse = {P_ELL:.10f}   (4a E(e^2), a=1.3 b=0.7)")
print(f"exact P_square  = {P_SQ:.10f}   (side 1.6, rotated 30 degrees)")
print()

cases = [("circle r=1", phi_circle, P_CIR),
         ("ellipse 1.3x0.7", phi_ellipse, P_ELL),
         ("square rotated 30", lambda X, Y: phi_sq(X, Y, 30.0), P_SQ)]

for name, fn, exact in cases:
    print(f"=== {name}   exact = {exact:.8f}")
    print(f"{'n':>6} {'zero level':>14} {'err %':>11} {'binary':>14} {'err %':>11}")
    for n in NS:
        g, XX, YY = grid(n)
        phi = fn(XX, YY)
        ps = contour_len(phi, g, g, 0.0)
        pb = contour_len((phi > 0).astype(float), g, g, 0.5)
        print(f"{n:6d} {ps:14.6f} {100 * (ps / exact - 1):+11.5f} "
              f"{pb:14.6f} {100 * (pb / exact - 1):+11.5f}")
    print()

# --- orientation spread ---------------------------------------------------
print("=== orientation spread (square side 1.6, P = 6.4)")
print(f"    definition: max(err) - min(err) in pp, angles {ANGS}")
for n in (800, 1600, 3200):
    g, XX, YY = grid(n)
    vs, vb = [], []
    for ang in ANGS:
        phi = phi_sq(XX, YY, ang)
        vs.append(contour_len(phi, g, g, 0.0))
        vb.append(contour_len((phi > 0).astype(float), g, g, 0.5))
    es = 100 * (np.array(vs) / P_SQ - 1)
    eb = 100 * (np.array(vb) / P_SQ - 1)
    sp_s, sp_b = es.max() - es.min(), eb.max() - eb.min()
    print(f" n={n}")
    print(f"   spread (pp): zero level {sp_s:.4f}   binary {sp_b:.4f}   "
          f"(factor {sp_b / sp_s:.0f})")
    print(f"   zero level err range [{es.min():+.4f}, {es.max():+.4f}] %   "
          f"binary [{eb.min():+.4f}, {eb.max():+.4f}] %")
print()

# fine angle sweep at n = 1600
g, XX, YY = grid(1600)
vs, vb = [], []
for ang in ANGS:
    phi = phi_sq(XX, YY, ang)
    vs.append(contour_len(phi, g, g, 0.0))
    vb.append(contour_len((phi > 0).astype(float), g, g, 0.5))
es = 100 * (np.array(vs) / P_SQ - 1)
eb = 100 * (np.array(vb) / P_SQ - 1)
print("fine sweep n=1600, angles 0..45 step 5:")
for a, s, b in zip(ANGS, es, eb):
    print(f"  {a:5.1f} deg  zero level {s:+.4f} %   binary {b:+.4f} %")
print(f"  global spread (pp): zero level {es.max() - es.min():.4f}   "
      f"binary {eb.max() - eb.min():.4f}")

# circle: what the binary estimator converges to
print("\ncircle: asymptotic bias of the binary estimator")
for n in NS:
    g, XX, YY = grid(n)
    pb = contour_len((phi_circle(XX, YY) > 0).astype(float), g, g, 0.5)
    print(f"  n={n:5d}  binary {pb:.6f}  ({100 * (pb / P_CIR - 1):+.4f} % over 2pi)")
