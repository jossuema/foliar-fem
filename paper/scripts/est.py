#!/usr/bin/env python3
"""
The three perimeter estimators the default stack ships, side by side.

    skimage.measure.perimeter            4-connectivity border counting
    skimage.measure.perimeter_crofton    discrete Crofton, 4 directions
    marching squares on the binary mask  find_contours at level 0.5

The point of the table is that none of the three converges to the true
perimeter, and that the Crofton column is not independent of the marching
squares column: see crofton.py for the exact identity.
"""

import numpy as np
from skimage import measure

HALF = 2.0


def grid(n):
    h = 2.0 * HALF / n
    y, x = np.mgrid[0:n, 0:n]
    return (x + 0.5) * h - HALF, (y + 0.5) * h - HALF, h


def marching_squares(mask, h):
    return sum(np.hypot(*np.diff(c, axis=0).T).sum()
               for c in measure.find_contours(mask.astype(float), 0.5)) * h


def row(mask, h, true):
    p1 = measure.perimeter(mask) * h
    p2 = measure.perimeter_crofton(mask, directions=4) * h
    ms = marching_squares(mask, h)
    return (100 * (p1 / true - 1), 100 * (p2 / true - 1), 100 * (ms / true - 1))


print("== circle r = 1, true perimeter 6.283185 ==")
print(f"{'n':>6} {'skim.perimeter':>15} {'crofton4':>10} {'marchsq':>10}")
for n in (200, 400, 800, 1600, 3200):
    X, Y, h = grid(n)
    e1, e2, e3 = row(X ** 2 + Y ** 2 <= 1.0, h, 2 * np.pi)
    print(f"{n:6d} {e1:+14.3f}% {e2:+9.3f}% {e3:+9.3f}%")

print("\n== square side 1, rotated, true perimeter 4 ==")
print(f"{'angle':>5} {'skim.perimeter':>15} {'crofton4':>10} {'marchsq':>10}")
for ang in (0.0, 10.0, 20.0, 22.5, 30.0, 45.0):
    X, Y, h = grid(3200)
    t = np.deg2rad(ang)
    Xr = X * np.cos(t) + Y * np.sin(t)
    Yr = -X * np.sin(t) + Y * np.cos(t)
    e1, e2, e3 = row((np.abs(Xr) <= 0.5) & (np.abs(Yr) <= 0.5), h, 4.0)
    print(f"{ang:5.1f} {e1:+14.3f}% {e2:+9.3f}% {e3:+9.3f}%")
