#!/usr/bin/env python3
"""
Marching squares on a binary mask against Proffitt & Rosen (1979).

Measures the bias of the estimator on a circle at four resolutions and on a
square at six rotation angles, and compares the square against the closed form

    L_est / L_true = cos(theta) + (sqrt(2) - 1) sin(theta)

with theta reduced to [0, 45] degrees.
"""

import numpy as np
from skimage import measure

HALF = 2.0                      # domain [-2, 2]^2
SQRT2M1 = np.sqrt(2.0) - 1.0


def perimeter(mask, h):
    """Length of the marching squares contour of a binary mask, scaled by h."""
    total = 0.0
    for c in measure.find_contours(mask.astype(float), 0.5):
        d = np.diff(c, axis=0)
        total += np.hypot(d[:, 0], d[:, 1]).sum()
    return total * h


def grid(n):
    h = 2.0 * HALF / n
    y, x = np.mgrid[0:n, 0:n]
    return (x + 0.5) * h - HALF, (y + 0.5) * h - HALF, h


def pr_factor(angle_deg):
    """Eq. (1), with the angle folded into [0, 45] degrees."""
    a = angle_deg % 90.0
    t = np.deg2rad(min(a, 90.0 - a))
    return np.cos(t) + SQRT2M1 * np.sin(t)


print("== circle r = 1, true perimeter 6.283185 ==")
print(f"{'n':>6} {'marching sq':>14} {'bias':>10}")
for n in (200, 400, 800, 1600):
    X, Y, h = grid(n)
    p = perimeter(X ** 2 + Y ** 2 <= 1.0, h)
    print(f"{n:6d} {p:14.6f} {100 * (p / (2 * np.pi) - 1):+9.3f}%")

print(f"\n== square side 1, rotated, true perimeter 4 ==")
print(f"{'angle':>6} {'marching sq':>14} {'bias':>10} {'Eq. (1)':>10} {'residual':>10}")
for ang in (0.0, 10.0, 20.0, 22.5, 30.0, 45.0):
    X, Y, h = grid(1600)
    t = np.deg2rad(ang)
    Xr = X * np.cos(t) + Y * np.sin(t)
    Yr = -X * np.sin(t) + Y * np.cos(t)
    p = perimeter((np.abs(Xr) <= 0.5) & (np.abs(Yr) <= 0.5), h)
    meas = 100 * (p / 4.0 - 1)
    pred = 100 * (pr_factor(ang) - 1)
    print(f"{ang:6.1f} {p:14.5f} {meas:+9.3f}% {pred:+9.3f}% {meas - pred:+9.3f} pp")

print(f"\nEq. (1) is cos(t) + {SQRT2M1:.4f} sin(t). The bias is a property of the")
print("estimator and of the boundary orientation, not of the resolution.")
