#!/usr/bin/env python3
"""
The orientation average of Eq. (1): uniform vs. arc-length weighted.

Why this file exists
--------------------
Section 2 used to say that averaging Eq. (1) uniformly over orientations is what
"a smooth convex curve" does, and that is false except for the circle. A
reviewer pointed it out and the measurement agrees with the reviewer.

Arc length weights each tangent direction by the RADIUS OF CURVATURE of the
curve there, not uniformly:

    <f>_arc = integral f(theta(s)) ds / integral ds

and theta(s) sweeps the directions at a constant rate only if the curvature is
constant, i.e. on the circle. For any other smooth convex curve the arc-length
weighted average departs from the uniform one, and Kulpa's constant -- which is
exactly the uniform average -- stops being the right correction.

What is measured here is that the arc-length weighted average PREDICTS the error
of the binary estimator across a family of ellipses, and the uniform one does not.

Usage
-----
    python3 avg.py            # circle + ellipse family (~20 s)
    python3 avg.py fast       # coarser grid, for iterating (~3 s)
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.special import ellipe
from skimage import measure

# Domain and grid. half = 1.6 and n = 3200 are those of Table 1 of the paper, so
# that the figures from this file are comparable with the ones there.
HALF = 1.6
N_GRID = 3200
N_QUAD = 4_000_000   # samples of the arc-length average quadrature

KULPA = 4.0 / np.pi * (2.0 * np.sqrt(2.0) - 2.0)
"""UNIFORM average of Eq. (1) over [0, 90): 4/pi (2 sqrt2 - 2) = 1.05479.

It is a constant: it does not depend on the shape. Hence it cannot be the
arc-length average of every smooth convex curve, whose value does depend on the
shape.
"""

# Ellipse family: semi-axes (a, b) with a*b fixed near 0.5 so the four shapes
# have comparable size and the only thing that changes is the eccentricity.
FAMILY = (("circle r=1", 1.0, 1.0),
          ("ellipse 1.3 x 0.7", 1.3, 0.7),
          ("ellipse 1.5 x 0.3", 1.5, 0.3),
          ("ellipse 1.55 x 0.1", 1.55, 0.1))


def pr_factor(theta_rad):
    """Eq. (1): L_est/L_true = cos(t) + (sqrt2 - 1) sin(t), t folded into [0,45]."""
    deg = np.rad2deg(theta_rad) % 90.0
    t = np.deg2rad(np.minimum(deg, 90.0 - deg))
    return np.cos(t) + (np.sqrt(2.0) - 1.0) * np.sin(t)


def exact_perimeter(a, b):
    """Exact perimeter of the ellipse: 4 a E(e^2), with a the MAJOR semi-axis."""
    a_maj, b_min = max(a, b), min(a, b)
    return 4.0 * a_maj * ellipe(1.0 - (b_min / a_maj) ** 2)


def arclength_mean(a, b, n_quad=N_QUAD):
    """Arc-length weighted average of Eq. (1), and the total arc length.

    Returning the total arc is not decorative: it is an independent check of the
    quadrature against the closed form E(e^2).
    """
    u = np.linspace(0.0, 2.0 * np.pi, n_quad, endpoint=False)
    dx, dy = -a * np.sin(u), b * np.cos(u)          # tangent vector
    ds = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)                      # tangent direction
    du = u[1] - u[0]
    return float(np.sum(pr_factor(theta) * ds) / np.sum(ds)), float(np.sum(ds) * du)


def measured_bias(a, b, n=N_GRID):
    """Error of the binary marching squares estimator on the ellipse."""
    g = np.linspace(-HALF, HALF, n)
    X, Y = np.meshgrid(g, g)
    h = float(g[1] - g[0])
    mask = ((X / a) ** 2 + (Y / b) ** 2 < 1.0).astype(float)
    L = sum(np.hypot(*np.diff(c, axis=0).T).sum()
            for c in measure.find_contours(mask, 0.5)) * h
    return L / exact_perimeter(a, b) - 1.0


def main(n_grid=N_GRID, n_quad=N_QUAD):
    print("=" * 86)
    print("ORIENTATION AVERAGE OF EQ. (1): UNIFORM VS. ARC-LENGTH WEIGHTED")
    print("=" * 86)
    print(f"grid {n_grid}^2 over [-{HALF:.1f}, {HALF:.1f}]^2    "
          f"quadrature {n_quad} samples")
    print(f"the uniform average (Kulpa) is {100.0 * (KULPA - 1.0):+.3f} % for EVERY "
          f"shape, by construction\n")
    hdr = (f"{'shape':<20} {'exact P':>11} {'unif %':>11} {'arc %':>11} "
           f"{'measured %':>11} {'arc-meas':>9}")
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    for name, a, b in FAMILY:
        P = exact_perimeter(a, b)
        mean_arc, P_quad = arclength_mean(a, b, n_quad)
        # check: the quadrature has to reproduce the closed-form perimeter
        assert abs(P_quad / P - 1.0) < 1e-6, (name, P_quad, P)
        meas = measured_bias(a, b, n_grid)
        gap = 100.0 * (mean_arc - 1.0) - 100.0 * meas
        worst = max(worst, abs(gap))
        print(f"{name:<20} {P:11.6f} {100.0 * (KULPA - 1.0):+10.3f}% "
              f"{100.0 * (mean_arc - 1.0):+10.3f}% {100.0 * meas:+10.3f}% "
              f"{gap:+8.3f} pp")
    print()
    print(f"the arc-length average predicts the measurement to {worst:.3f} pp "
          f"in the worst case;")
    print(f"the uniform one departs by up to "
          f"{max(abs(100.0 * (KULPA - 1.0) - 100.0 * measured_bias(a, b, n_grid)) for _, a, b in FAMILY):.2f} pp "
          f"over the same family.")
    print()
    print("Reading: Kulpa's constant is calibrated to the circle -- the only case")
    print("where the curvature is constant and arc length sweeps the directions")
    print("uniformly -- so a circle is NO test for an estimator that applies it.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fast":
        main(n_grid=800, n_quad=200_000)
    else:
        main()
