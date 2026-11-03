"""Segmentation -> meshing -> FEM -> foliar parameters. Synthetic prototype."""

from . import baseline, fem, infection, latency, meshing, params, synth  # noqa: F401

__all__ = ["synth", "meshing", "fem", "params", "baseline", "infection", "latency"]
