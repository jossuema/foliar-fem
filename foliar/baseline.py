"""
Raster baselines, so the mesh is compared against an honest rival.

The GOOD variants of pixel counting are included -- Crofton perimeter, fast
marching geodesics -- not just the naive ones: if meshing wins, let it win
against the best available competitor.
"""

from __future__ import annotations

import numpy as np
import skfmm
from scipy import ndimage
from skimage import measure


def pixel_area_cm2(mask: np.ndarray, px_per_cm: float) -> float:
    return float(mask.sum()) / (px_per_cm ** 2)


def pixel_perimeter_naive_cm(mask: np.ndarray, px_per_cm: float) -> float:
    """Count border pixels. Known upward bias, ~4/pi on diagonal borders."""
    er = ndimage.binary_erosion(mask, np.ones((3, 3), bool))
    return float((mask & ~er).sum()) / px_per_cm


def pixel_perimeter_crofton_cm(mask: np.ndarray, px_per_cm: float) -> float:
    """Crofton estimator (skimage): the best correction available on a raster."""
    return float(measure.perimeter_crofton(mask, directions=4)) / px_per_cm


def euclidean_distance_to_midrib(mask_blade, mask_midrib, px_per_cm: float) -> np.ndarray:
    """
    Euclidean distance in the projected image. Ignores two things at once:
    the tears (it crosses thin air) and the curl (it measures the chord, not
    the arc).
    """
    d = ndimage.distance_transform_edt(~mask_midrib) / px_per_cm
    return np.where(mask_blade, d, np.nan)


def raster_geodesic_to_midrib(mask_blade, mask_midrib, px_per_cm: float) -> np.ndarray:
    """
    Fast marching geodesic on the raster, with the blade as the domain.
    Corrects for the tears. Does NOT correct for the curl: it still measures
    on the projection.
    """
    phi = np.where(mask_midrib, -1.0, 1.0)
    # Explicit float(): skfmm rejects an np.float64 with "dx must be a 1D array
    # of doubles". With the synthetic leaf px_per_cm is a Python float and this
    # went unnoticed; a scale DERIVED from real data arrives as np.float64.
    dx = float(1.0 / float(px_per_cm))
    d = skfmm.distance(np.ma.MaskedArray(phi, ~mask_blade), dx=dx)
    d = np.ma.filled(d, np.nan)
    return np.where(mask_blade & ~mask_midrib, d,
                    np.where(mask_midrib, 0.0, np.nan))


def raster_report(sample, scale_cm: float) -> dict:
    """Every parameter computable without leaving the raster."""
    px = sample.px_per_cm
    blade, lesion, midrib = sample.mask_blade, sample.mask_lesion, sample.mask_midrib

    area = pixel_area_cm2(blade, px)
    les_area = pixel_area_cm2(lesion, px)

    d_eu = euclidean_distance_to_midrib(blade, midrib, px)
    d_ge = raster_geodesic_to_midrib(blade, midrib, px)

    def weighted_index(d):
        w = np.exp(-d / scale_cm)
        num = np.nansum(np.where(lesion, w, 0.0))
        den = np.nansum(np.where(blade, w, 0.0))
        return 100.0 * float(num) / max(float(den), 1e-9)

    n_les = int(ndimage.label(lesion)[1])

    return dict(
        area_cm2=area,
        lesion_area_cm2=les_area,
        severity_pct=100.0 * les_area / max(area, 1e-9),
        perimeter_naive_cm=pixel_perimeter_naive_cm(blade, px),
        perimeter_crofton_cm=pixel_perimeter_crofton_cm(blade, px),
        n_lesions=n_les,
        mean_dist_euclidean_cm=float(np.nanmean(d_eu[lesion])) if lesion.any() else np.nan,
        mean_dist_geodesic_cm=float(np.nanmean(d_ge[lesion])) if lesion.any() else np.nan,
        wsi_euclidean=weighted_index(d_eu),
        wsi_geodesic=weighted_index(d_ge),
        _fields=dict(euclidean=d_eu, geodesic=d_ge),
    )
