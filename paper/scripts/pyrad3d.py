#!/usr/bin/env python3
"""
Closes the one gap of the 3D experiment: until now the PyRadiomics path was
MODELLED with skimage.measure.marching_cubes. Here PyRadiomics itself is run, on
exactly the same masks, and the two are compared.

What is verified:
  - original_shape_SurfaceArea  against the exact area (4*pi*r^2 / Legendre)
  - original_shape_Sphericity   against 1.000000 on the sphere, which is its
                                value by construction, and against the exact
                                value for the ellipsoid
  - MeshVolume and VoxelVolume  to confirm the volume DOES converge and that
                                therefore the area bias passes entirely into
                                the sphericity
  - the difference between PyRadiomics and our skimage inference

PyRadiomics 3.0.1 requires numpy<2, so it lives in its own venv.

Usage:  python3 pyrad3d.py [n1 n2 ...]     (default 100 200 400)
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import radiomics
import SimpleITK as sitk
from radiomics.shape import RadiomicsShape
from scipy.special import ellipeinc, ellipkinc

radiomics.logger.setLevel(logging.ERROR)          # no progress noise

HALF = 1.6
ELL = (1.3, 1.0, 0.7)


# --------------------------------------------------------------------------
# Exact areas and volumes. Identical to surf3d.py, rewritten here so that this
# script is INDEPENDENT: if they shared an implementation we would be committing
# exactly the error the paper reports.
# --------------------------------------------------------------------------

def sphere_area(r=1.0):
    return 4.0 * np.pi * r * r


def sphere_volume(r=1.0):
    return 4.0 / 3.0 * np.pi * r ** 3


def ellipsoid_area_legendre(a, b, c):
    """Legendre with incomplete F and E, a >= b >= c."""
    a, b, c = sorted((a, b, c), reverse=True)
    phi = np.arccos(c / a)
    m = (a * a * (b * b - c * c)) / (b * b * (a * a - c * c))
    s, cs = np.sin(phi), np.cos(phi)
    return 2 * np.pi * c * c + (2 * np.pi * a * b / s) * (
        ellipeinc(phi, m) * s * s + ellipkinc(phi, m) * cs * cs)


def ellipsoid_volume(a, b, c):
    return 4.0 / 3.0 * np.pi * a * b * c


def exact_sphericity(vol, area):
    """The IBSI definition, which is the one PyRadiomics uses."""
    return (36.0 * np.pi * vol ** 2) ** (1.0 / 3.0) / area


# --------------------------------------------------------------------------
# Masks: identical to surf3d.py's (NODE grid, phi > 0)
# --------------------------------------------------------------------------

def mask_and_h(kind, n, half=HALF):
    g = np.linspace(-half, half, n)
    h = float(g[1] - g[0])
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    if kind == "sphere":
        phi = 1.0 - np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    elif kind == "ellipsoid":
        a, b, c = ELL
        phi = 1.0 - np.sqrt((X / a) ** 2 + (Y / b) ** 2 + (Z / c) ** 2)
    else:
        raise ValueError(kind)
    return (phi > 0).astype(np.uint8), h


def pyradiomics_shape(mask, h):
    """Run PyRadiomics' shape extractor with isotropic spacing h.

    The intensity image is irrelevant for shape features, but PyRadiomics
    requires one; a zero volume of the same size is passed.
    """
    img = sitk.GetImageFromArray(np.zeros_like(mask, dtype=np.float32))
    msk = sitk.GetImageFromArray(mask)
    for im in (img, msk):
        im.SetSpacing((h, h, h))
        im.SetOrigin((0.0, 0.0, 0.0))
    shape = RadiomicsShape(img, msk, label=1)
    shape.enableAllFeatures()
    shape.execute()
    return {k: float(v) for k, v in shape.featureValues.items()}


# --------------------------------------------------------------------------

def run(ns=(100, 200, 400)):
    bodies = [
        ("sphere", sphere_area(), sphere_volume(), 1.0),
        ("ellipsoid", ellipsoid_area_legendre(*ELL), ellipsoid_volume(*ELL), None),
    ]
    for name, A_exact, V_exact, sph_hint in bodies:
        S_exact = exact_sphericity(V_exact, A_exact)
        if sph_hint is not None:
            assert abs(S_exact - sph_hint) < 1e-12, (S_exact, sph_hint)
        print("=" * 92)
        print(f"{name.upper()}   exact area {A_exact:.7f}   "
              f"exact volume {V_exact:.7f}   exact sphericity {S_exact:.7f}")
        print("=" * 92)
        print(f"{'n':>6} {'h':>9} | {'SurfaceArea':>12} {'err %':>9} | "
              f"{'MeshVolume':>12} {'err %':>8} | {'VoxelVolume':>12} {'err %':>8} | "
              f"{'Sphericity':>11} {'err %':>8}")
        print("-" * 92)
        for n in ns:
            mask, h = mask_and_h(name, n)
            f = pyradiomics_shape(mask, h)
            A, Vm, Vv = f["SurfaceArea"], f["MeshVolume"], f["VoxelVolume"]
            S = f["Sphericity"]
            print(f"{n:>6} {h:>9.6f} | {A:>12.6f} {100*(A/A_exact-1):>+9.4f} | "
                  f"{Vm:>12.6f} {100*(Vm/V_exact-1):>+8.4f} | "
                  f"{Vv:>12.6f} {100*(Vv/V_exact-1):>+8.4f} | "
                  f"{S:>11.6f} {100*(S/S_exact-1):>+8.4f}")
            del mask
        print()

    print("=" * 92)
    print("CONTRAST WITH THE skimage INFERENCE THE PAPER USED TO CARRY")
    print("=" * 92)
    print("  surf3d.py modelled the PyRadiomics path with skimage.measure.marching_cubes")
    print("  on the binary indicator at level 0.5, and reported:")
    print("    sphere     +8.82 % (n=400)   sphericity 0.918987  (-8.10 %)")
    print("    ellipsoid  +8.86 % (n=400)   sphericity 0.872475  (-8.14 %)")
    print("  The rows above are PyRadiomics itself. If they agree, the inference")
    print("  was correct and the paper can cite PyRadiomics by name.")
    print()
    print(f"  pyradiomics {radiomics.__version__}   numpy {np.__version__}   "
          f"SimpleITK {sitk.__version__}")


if __name__ == "__main__":
    ns = tuple(int(a) for a in sys.argv[1:]) or (100, 200, 400)
    run(ns)
