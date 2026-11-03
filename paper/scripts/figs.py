#!/usr/bin/env python3
"""
Publication figures for the NeurIPS (LatinX in AI) workshop paper.

Seven figures, each written as a vector PDF and a 200 dpi PNG:

    fig1_orientation.pdf/.png   orientation dependence of the binary estimator
    fig2_convergence.pdf/.png   non-convergence: more pixels do not fix the bias
    fig3_circular.pdf/.png      the circular validation (shared implementation)
    fig4_inversion.pdf/.png     the ranking inversion
    fig5_selfconv.pdf/.png      why a self-convergence study certifies the bias
    fig6_surf3d.pdf/.png        the same failure in 3D, and in PyRadiomics
    fig7_crofton.pdf/.png       more Crofton directions helps and never suffices

Figures 1, 2, 5 and 7 are computed from scratch at run time (nothing is
hard-coded): 5 drives the estimators in twogrid.py, 7 drives crofton.py.
Figures 3, 4 and 6 report numbers measured elsewhere -- the foliar_fem
pipeline, scripts/surf3d.py and PyRadiomics 3.0.1 via scripts/pyrad3d.py --
and list them as named constants at the top of their sections.

Run:
    python figs.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from skimage import measure

# The paper includes fig/<stem>.pdf, so write there when the directory exists;
# fall back to the script's own directory otherwise.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIGDIR = os.path.normpath(os.path.join(_HERE, os.pardir, "fig"))
OUT = _FIGDIR if os.path.isdir(_FIGDIR) else _HERE

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
# Single-column width for the NeurIPS style file is ~3.3 in; full width ~6.8 in.
COL = 2.7   # REAL placement width: a 0.48*5.5in minipage. It used to be 3.3,
FULL = 5.5  # and \textwidth in neurips_2025 is 5.5in, not 6.8, so figures were
            # being scaled down by 19%.

# neurips_2025.sty sets \textwidth = 5.5 in (single column).  Figures 5-7 are
# built at the size they are placed at, so 8 pt type reaches the page as 8 pt:
# TEXTW spans \textwidth, HALFW fits the 0.45\textwidth minipage pair used for
# figures 1 and 2.
TEXTW = 5.5
HALFW = 2.7

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.columnspacing": 1.1,
    "legend.labelspacing": 0.35,
    "axes.linewidth": 0.6,
    "axes.unicode_minus": False,          # ASCII hyphen: safest for embedding
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.1,
    "lines.markersize": 3.2,
    "grid.linewidth": 0.4,
    "grid.color": "#cfcfcf",
    "hatch.linewidth": 0.5,
    "patch.linewidth": 0.7,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito: colour-blind safe, and the pairs used here also separate in
# greyscale because line style / marker fill / hatch carry the same contrast.
C_BIN = "#D55E00"   # vermillion  -- binary marching squares
C_LVL = "#0072B2"   # blue        -- zero level set
C_THEORY = "#000000"
C_A = "#0072B2"     # level A (raster)
C_B = "#E69F00"     # level B (flat mesh)
C_C = "#009E73"     # level C (lifted mesh)
C_NEUTRAL_FC = "#F2F2F2"
C_NEUTRAL_EC = "#4D4D4D"
C_HL_FC = "#FBE3D0"   # highlighted (shared estimator) fill
C_HL_EC = C_BIN
C_FIX_FC = "#DCEAF5"  # corrected path fill
C_FIX_EC = C_LVL
GREY = "#777777"


def _fit_width(fig, target_w, tol=0.004, it=10):
    """Grow/shrink the canvas so the *cropped* output is exactly `target_w` in.

    savefig.bbox='tight' trims the figure to its ink, so the file that reaches
    LaTeX is generally not the figsize that was asked for.  Placing it with
    width=\\columnwidth then rescales it and the 8 pt type stops being 8 pt.
    The margins are near-constant in inches, so an additive fixed point on the
    canvas width converges in two or three passes.
    """
    pad = plt.rcParams["savefig.pad_inches"]
    fig.canvas.draw()
    for _ in range(it):
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w = bb.width + 2.0 * pad
        if abs(w - target_w) < tol:
            break
        W, H = fig.get_size_inches()
        fig.set_size_inches(W + (target_w - w), H)
        fig.canvas.draw()
    return w


def save(fig, stem, target_w=None):
    if target_w is not None:
        _fit_width(fig, target_w)
    pdf = os.path.join(OUT, stem + ".pdf")
    png = os.path.join(OUT, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("  wrote", pdf)
    print("  wrote", png)


# ----------------------------------------------------------------------------
# Shared numerics
# ----------------------------------------------------------------------------
def contour_length(field, level, h):
    """Polyline length of the `level` iso-contour of `field`, scaled by h."""
    total = 0.0
    for c in measure.find_contours(field, level):
        d = np.diff(c, axis=0)
        total += np.hypot(d[:, 0], d[:, 1]).sum()
    return total * h


def node_grid(n, half):
    """n x n nodes on [-half, half]^2 (linspace convention, endpoints included)."""
    v = np.linspace(-half, half, n)
    h = v[1] - v[0]
    X, Y = np.meshgrid(v, v)
    return X, Y, h


def square_phi(X, Y, angle_deg, side):
    """Signed field for a square of `side`, rotated by `angle_deg`; >0 inside.

    min/max of the four half-planes: the zero set is *exactly* the square
    boundary, and the field is piecewise linear along grid lines, so the
    level-0 contour is recovered to machine-ish precision.
    """
    t = np.deg2rad(angle_deg)
    Xr = X * np.cos(t) + Y * np.sin(t)
    Yr = -X * np.sin(t) + Y * np.cos(t)
    a = 0.5 * side
    return np.minimum(a - np.abs(Xr), a - np.abs(Yr))


def circle_phi(X, Y, r):
    return r - np.hypot(X, Y)


def proffitt_rosen(angle_deg):
    """Digitised-length bias factor L_est / L_true for a straight edge.

        f(t) = cos(t) + (sqrt(2) - 1) * sin(t),     t = reduced angle in [0, 45]

    Since sqrt(2)-1 = tan(22.5 deg), this is f(t) = cos(t - 22.5) / cos(22.5):
    a cosine of amplitude sec(22.5) = 1.08239 centred on 22.5 deg, equal to 1
    at t = 0 and t = 45.

    For a SQUARE rotated by t the four edges sit at t and t+90 with respect to
    the pixel axes.  The reduction to the fundamental domain is
    t -> min(t mod 90, 90 - t mod 90); it maps t and t+90 to the SAME reduced
    angle, so the edge-length-weighted average over the four sides collapses
    to f(reduced(t)).  No averaging term survives.
    """
    t = np.deg2rad(np.minimum(np.asarray(angle_deg) % 90.0,
                              90.0 - np.asarray(angle_deg) % 90.0))
    return np.cos(t) + (np.sqrt(2) - 1.0) * np.sin(t)


# ============================================================================
# FIG 1 -- orientation dependence
# ============================================================================
def fig1():
    print("fig1: orientation dependence")
    side, exact = 1.6, 6.4
    n, half = 1600, 1.6
    X, Y, h = node_grid(n, half)

    angles = np.arange(0.0, 45.001, 1.0)
    err_bin = np.empty_like(angles)
    err_lvl = np.empty_like(angles)
    for i, a in enumerate(angles):
        phi = square_phi(X, Y, a, side)
        err_bin[i] = 100.0 * (contour_length((phi > 0).astype(float), 0.5, h) / exact - 1.0)
        err_lvl[i] = 100.0 * (contour_length(phi, 0.0, h) / exact - 1.0)

    a_fine = np.linspace(0.0, 45.0, 361)
    err_th = 100.0 * (proffitt_rosen(a_fine) - 1.0)

    mk = np.isin(angles, np.arange(0.0, 45.001, 5.0))
    i_max = int(np.argmax(err_bin))

    print("    binary  @5deg steps :", " ".join("%+.3f" % v for v in err_bin[mk]))
    print("    levelset@5deg steps :", " ".join("%+.4f" % v for v in err_lvl[mk]))
    print("    theory  @5deg steps :",
          " ".join("%+.3f" % v for v in 100.0 * (proffitt_rosen(angles[mk]) - 1.0)))
    print("    level-set spread: mean|e| %.4f %%, max|e| %.4f %%, sd %.4f %%"
          % (np.abs(err_lvl).mean(), np.abs(err_lvl).max(), err_lvl.std()))
    print("    binary max %+.3f %% at %.1f deg (theory peak 22.5 deg, %+.3f %%)"
          % (err_bin[i_max], angles[i_max], 100.0 * (proffitt_rosen(22.5) - 1.0)))
    print("    binary rms deviation from theory: %.3f pp"
          % np.sqrt(np.mean((err_bin - 100.0 * (proffitt_rosen(angles) - 1.0)) ** 2)))

    fig, ax = plt.subplots(figsize=(COL, 2.55))
    ax.axhline(0.0, color="black", lw=0.7, zorder=1)

    ax.plot(a_fine, err_th, color=C_THEORY, ls=(0, (1.4, 1.4)), lw=1.0, zorder=3,
            label="Proffitt-Rosen (1979) closed form")
    ax.plot(angles, err_bin, color=C_BIN, lw=1.2, zorder=4)
    ax.plot(angles[mk], err_bin[mk], color=C_BIN, ls="none", marker="o", ms=3.4,
            mfc=C_BIN, mec=C_BIN, zorder=5, label="Binary marching squares")
    ax.plot(angles, err_lvl, color=C_LVL, lw=1.2, ls="--", zorder=4)
    ax.plot(angles[mk], err_lvl[mk], color=C_LVL, ls="none", marker="s", ms=3.0,
            mfc="white", mec=C_LVL, mew=0.9, zorder=5,
            label="Zero level set (signed field)")

    ax.annotate("max %+.2f %%  near 22.5$^\\circ$" % err_bin[i_max],
                xy=(angles[i_max], err_bin[i_max]), xytext=(22.5, 11.3),
                fontsize=6.8, color=C_BIN, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_BIN, lw=0.7,
                                shrinkA=1.0, shrinkB=2.5,
                                connectionstyle="arc3,rad=0.0"))
    ax.annotate("|error| $\\leq$ %.2f %%  (mean %.3f %%)"
                % (np.abs(err_lvl).max(), np.abs(err_lvl).mean()),
                xy=(31.0, err_lvl[31]), xytext=(25.0, -2.55),
                fontsize=6.8, color=C_LVL, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_LVL, lw=0.7,
                                shrinkA=1.0, shrinkB=1.5,
                                connectionstyle="arc3,rad=0.22"))

    ax.set_xlabel("square rotation angle (degrees)")
    ax.set_ylabel("perimeter error (%)")
    ax.set_xlim(-1.2, 46.2)
    ax.set_ylim(-4.3, 12.6)
    ax.set_xticks(np.arange(0, 46, 5))
    ax.set_yticks(np.arange(-4, 13, 2))
    ax.grid(axis="y", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    save(fig, "fig1_orientation", target_w=COL)


# ============================================================================
# FIG 2 -- non-convergence
# ============================================================================
def fig2():
    print("fig2: non-convergence")
    ns = np.array([200, 400, 800, 1600, 3200])
    half = 1.6
    P_CIRC = 2.0 * np.pi
    P_SQ = 6.4

    res = {k: [] for k in ("cb", "cl", "sb", "sl")}
    for n in ns:
        X, Y, h = node_grid(int(n), half)
        phi = circle_phi(X, Y, 1.0)
        res["cb"].append(100.0 * (contour_length((phi > 0).astype(float), 0.5, h) / P_CIRC - 1.0))
        res["cl"].append(100.0 * (contour_length(phi, 0.0, h) / P_CIRC - 1.0))
        phi = square_phi(X, Y, 30.0, 1.6)
        res["sb"].append(100.0 * (contour_length((phi > 0).astype(float), 0.5, h) / P_SQ - 1.0))
        res["sl"].append(100.0 * (contour_length(phi, 0.0, h) / P_SQ - 1.0))
    res = {k: np.array(v) for k, v in res.items()}

    for k, lab in [("cb", "circle   binary  "), ("sb", "square30 binary  "),
                   ("cl", "circle   levelset"), ("sl", "square30 levelset")]:
        print("    %s:" % lab, " ".join("%+.5f" % v for v in res[k]))
    for k in ("cl", "sl"):
        p = np.polyfit(np.log(ns), np.log(np.abs(res[k])), 1)[0]
        print("    fitted order (%s): %.2f" % (k, -p))

    fig, ax = plt.subplots(figsize=(COL, 2.75))

    ax.loglog(ns, np.abs(res["cb"]), color=C_BIN, ls="-", marker="o", ms=3.4,
              mfc="white", mec=C_BIN, mew=0.9, label="Binary MS, circle")
    ax.loglog(ns, np.abs(res["sb"]), color=C_BIN, ls="-", marker="s", ms=3.2,
              mfc=C_BIN, mec=C_BIN, label="Binary MS, square 30$^\\circ$")
    ax.loglog(ns, np.abs(res["cl"]), color=C_LVL, ls="--", marker="o", ms=3.4,
              mfc="white", mec=C_LVL, mew=0.9, label="Zero level set, circle")
    ax.loglog(ns, np.abs(res["sl"]), color=C_LVL, ls="--", marker="s", ms=3.2,
              mfc=C_LVL, mec=C_LVL, label="Zero level set, square 30$^\\circ$")

    # Guide lines, each drawn parallel to the series whose order it reads off:
    # slope -2 next to the circle level set, slope -1 next to the square level
    # set (the re-entrant corners of the square drop the order to 1).
    for slope, ref_y0, factor, lab in [(-2.0, abs(res["cl"][0]), 7.0, "slope $-$2"),
                                       (-1.0, abs(res["sl"][0]), 0.20, "slope $-$1")]:
        g = np.array([215.0, 1250.0])
        y = ref_y0 * factor * (g / 200.0) ** slope
        ax.loglog(g, y, color=GREY, lw=0.8, ls=(0, (3.5, 2)), zorder=1)
        ax.text(g[1] * 1.10, y[1], lab, fontsize=6.5, color=GREY,
                ha="left", va="center")

    ax.text(215, 2.6, "bias floor: converges,\nbut to the wrong value",
            fontsize=6.6, color=C_BIN, ha="left", va="top")

    ax.set_xlabel("grid resolution $n$  ($n \\times n$ pixels)")
    ax.set_ylabel("|perimeter error| (%)")
    ax.set_xlim(160, 6200)
    ax.set_ylim(4e-6, 40)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(v)) for v in ns])
    ax.minorticks_off()
    ax.grid(which="major", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=1,
              fontsize=6.5, handlelength=1.6, columnspacing=0.8, borderpad=0.3)
    save(fig, "fig2_convergence", target_w=COL)


# ============================================================================
# FIG 3 -- the circular validation
# ============================================================================
P_REF_BIASED = 778.17     # ground truth, binary marching squares
P_LEVEL_C = 778.06        # level C, measured (same estimator)
P_REF_CONSISTENT = 744.23  # ground truth, zero-level-set estimator


def _box(ax, cx, cy, w, h, text, fc, ec, lw=0.8, fs=6.5, tc="black",
         weight="normal", pad=0.02, zorder=3, ls="solid",
         mx=2.4, my=1.8):
    """Labelled box. `w` and `h` are MINIMA: the box grows if the text does not fit.

    They used to be fixed sizes with the text laid out separately, with no
    relation between the two: "Ground-truth generator" and "Segmentation" ran
    out of their box and overlapped the next one. Here the already-composed
    text is measured and the box adjusts, so the defect cannot come back
    silently when a label or the figure size changes.
    """
    t = ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
                zorder=zorder + 1, linespacing=1.25, fontweight=weight)
    # the text extent only exists with a renderer; under Agg the whole figure
    # does not have to be drawn to get it
    bb = (t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
          .transformed(ax.transData.inverted()))
    w = max(w, bb.width + 2.0 * mx)
    h = max(h, bb.height + 2.0 * my)
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2.0, cy - h / 2.0), w, h,
        boxstyle="round,pad=%.3f,rounding_size=1.2" % pad,
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))
    return w, h


def _arrow(ax, p0, p1, color="#4D4D4D", lw=0.8, style="-|>", rad=0.0, zorder=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=6.5, color=color, lw=lw,
        shrinkA=0.0, shrinkB=0.0, zorder=zorder,
        connectionstyle="arc3,rad=%.3f" % rad))


def _panel_a(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")

    c0, c1, c2, c3, c4 = 10.0, 30.0, 50.0, 70.5, 91.0
    bw, bh = 17.5, 11.0
    y_top, y_bot = 52.5, 32.5

    # source
    _box(ax, c0, 42.5, bw, bh, "parametric\nleaf model", C_NEUTRAL_FC, C_NEUTRAL_EC)

    # upper path -- the reference
    _box(ax, (c1 + c2) / 2.0, y_top, 30.0, bh,
         "ground-truth generator", C_NEUTRAL_FC, C_NEUTRAL_EC)
    # lower path -- the method under test
    _box(ax, c1, y_bot, bw, bh, "segmented\nmask", C_NEUTRAL_FC, C_NEUTRAL_EC)
    _box(ax, c2, y_bot, bw, bh, "mesh pipeline\n(level C)", C_NEUTRAL_FC, C_NEUTRAL_EC)

    # the shared estimator, twice
    for y in (y_top, y_bot):
        _box(ax, c3, y, bw, bh, "binary\nmarching\nsquares", C_HL_FC, C_HL_EC,
             lw=1.3, fs=6.3, tc=C_HL_EC, weight="bold", zorder=4)

    # outputs
    _box(ax, c4, y_top, bw, bh, "reference\nperimeter\n778.17 cm",
         "white", C_NEUTRAL_EC, fs=6.3)
    _box(ax, c4, y_bot, bw, bh, "measured\nperimeter\n778.06 cm",
         "white", C_NEUTRAL_EC, fs=6.3)

    # wiring
    _arrow(ax, (c0 + bw / 2.0, 45.5), (c1 - bw / 2.0, y_top - 1.5), rad=-0.12)
    _arrow(ax, (c0 + bw / 2.0, 39.5), (c1 - bw / 2.0, y_bot + 1.5), rad=0.12)
    _arrow(ax, (c2 + bw / 2.0, y_top), (c3 - bw / 2.0, y_top))
    _arrow(ax, (c1 + bw / 2.0, y_bot), (c2 - bw / 2.0, y_bot))
    _arrow(ax, (c2 + bw / 2.0, y_bot), (c3 - bw / 2.0, y_bot))
    _arrow(ax, (c3 + bw / 2.0, y_top), (c4 - bw / 2.0, y_top))
    _arrow(ax, (c3 + bw / 2.0, y_bot), (c4 - bw / 2.0, y_bot))

    # the dashed frame that ties the two estimator boxes together
    ax.add_patch(FancyBboxPatch(
        (c3 - 11.0, y_bot - 8.0), 22.0, (y_top - y_bot) + 16.0,
        boxstyle="round,pad=0.0,rounding_size=1.6",
        facecolor="none", edgecolor=C_HL_EC, linewidth=0.85,
        linestyle=(0, (2.6, 2.0)), zorder=2))
    ax.text(c3, y_top + 9.6, "shared implementation", ha="center", va="bottom",
            fontsize=6.6, color=C_HL_EC, style="italic")

    # the comparison
    _arrow(ax, (c4, y_top - bh / 2.0 - 0.3), (c4, y_bot + bh / 2.0 + 0.3),
           color=C_HL_EC, lw=0.9, style="<|-|>")

    # verdicts
    _box(ax, 65.0, 17.0, 70.0, 8.6,
         "reported error  $-$0.01 %   (looks exact)",
         "white", C_HL_EC, lw=1.0, fs=7.2, tc=C_HL_EC, weight="bold")
    _arrow(ax, (c4, y_bot - bh / 2.0 - 0.4), (c4, 21.6), color=C_HL_EC, lw=0.9)

    _box(ax, 50.0, 5.6, 62.0, 9.6,
         "corrected: a zero-level-set reference gives 744.23 cm\n"
         r"$\Rightarrow$  true error  $+$4.55 %",
         C_FIX_FC, C_FIX_EC, lw=1.0, fs=6.6, tc=C_FIX_EC)

    ax.set_title("The method and its reference shared one estimator", loc="left",
                 fontsize=8.0, pad=4)


def fig3():
    print("fig3: the circular validation")
    print("    biased reference %.2f, level C %.2f -> %+.2f %%"
          % (P_REF_BIASED, P_LEVEL_C, 100.0 * (P_LEVEL_C / P_REF_BIASED - 1.0)))
    print("    consistent reference %.2f, level C %.2f -> %+.2f %%"
          % (P_REF_CONSISTENT, P_LEVEL_C, 100.0 * (P_LEVEL_C / P_REF_CONSISTENT - 1.0)))

    # A single panel, full width. This used to be a panel (a) with the diagram
    # and a panel (b) with three bars, and the diagram was supposed to fit in
    # 3.6 in. It did not: with five columns of boxes the column pitch is
    # smaller than the width of "Mesh pipeline", so the boxes ate the arrows
    # and the text ran off the sides. Panel (b) was removed rather than
    # squeezing the diagram, because it was redundant -- its three numbers
    # (778.17, 778.06, 744.23) are in the prose accompanying the figure.
    fig = plt.figure(figsize=(FULL, 2.35))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.92])
    _panel_a(ax)
    save(fig, "fig3_circular", target_w=FULL)
# ============================================================================
# FIG 4 -- the ranking inversion
# ============================================================================
LEVELS = ["A\nraster (Crofton)", "B\nflat mesh", "C\nlifted mesh"]

# Measured perimeters, seed 7 (cm).  Everything below is derived from these,
# so the panels cannot drift apart from each other again.
P_A, P_B, P_C = 677.9006, 713.5357, 778.0640
P_BIASED_INTRINSIC = 778.1718     # binary marching squares on the GT indicator
P_TRUE_INTRINSIC = 744.2255       # zero level set + Richardson
P_TRUE_PROJECTED = 686.9816       # same contour, pushed through the projection

# (a) AS ORIGINALLY REPORTED.  The published table had ONE ground-truth
# perimeter -- truth.perimeter_cm, the intrinsic one -- and every level was
# divided by it, level C included.  That is the reading reproduced here.
# (The previous version of this figure mixed references: it divided A and B by
# the CONSISTENT intrinsic reference and only C by the biased one, which is
# neither the published reading nor a corrected one.)
BIASED = [round(100.0 * (p / P_BIASED_INTRINSIC - 1.0), 2) for p in (P_A, P_B, P_C)]

# (c) AGAINST THE REFERENCE EACH LEVEL CAN ATTAIN.  A and B measure on the
# projection, so their ceiling is the projected perimeter; only C can aim at
# the intrinsic one.
ATTAIN = [round(100.0 * (P_A / P_TRUE_PROJECTED - 1.0), 2),
          round(100.0 * (P_B / P_TRUE_PROJECTED - 1.0), 2),
          round(100.0 * (P_C / P_TRUE_INTRINSIC - 1.0), 2)]

SD5 = [0.435, 0.374, 0.392]                    # sd over seeds 7..11
MEAN5 = [-1.349, +3.794, +4.472]               # mean over seeds 7..11


def _bars(ax, vals, errs=None):
    xs = np.arange(3)
    cols = [C_A, C_B, C_C]
    hats = ["", "///", "..."]
    for x, v, c, ht in zip(xs, vals, cols, hats):
        ax.bar([x], [v], width=0.60, color=c, edgecolor="black", linewidth=0.7,
               hatch=ht, zorder=3)
    if errs is not None:
        ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor="black",
                    elinewidth=0.8, capsize=2.6, capthick=0.8, zorder=5)
    ax.axhline(0.0, color="black", lw=0.7, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(LEVELS, fontsize=7.0)
    ax.set_xlim(-0.62, 2.62)
    ax.grid(axis="y", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig4():
    print("fig4: the ranking inversion")
    print("    |err| ranking vs biased reference    :",
          " < ".join("ABC"[i] for i in np.argsort(np.abs(BIASED))))
    print("    |err| ranking vs attainable reference:",
          " < ".join("ABC"[i] for i in np.argsort(np.abs(ATTAIN))))
    print("    B / A ratio (attainable): %.2f" % (abs(ATTAIN[1]) / abs(ATTAIN[0])))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(FULL, 2.85), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.16, wspace=0.34)

    _bars(axl, BIASED)
    _bars(axr, ATTAIN, errs=SD5)

    for ax, vals, errs in [(axl, BIASED, [0.0, 0.0, 0.0]), (axr, ATTAIN, SD5)]:
        for x, (v, e) in enumerate(zip(vals, errs)):
            # a bar that barely leaves the axis gets its label above it, so the
            # text does not collide with the zero line and the "0" tick
            up = v >= 0.0 or abs(v) < 0.6
            off = 0.55 + e
            ax.text(x, v + (off if up else -off), "%+.2f" % v,
                    ha="center", va="bottom" if up else "top",
                    fontsize=7.2, fontweight="bold", zorder=6)

    axl.set_ylim(-16.2, 7.8)
    axl.set_yticks(np.arange(-16, 8, 2))
    axl.set_ylabel("perimeter error (%)")

    # kept to <= ~40 characters per line so the titles do not overhang the
    # axes; an overhanging title makes the tight bbox wider than the column
    axl.set_title("(a)  As reported: every level against the\n"
                  "       single biased reference, 778.17 cm\n"
                  "|error| ranking:   C  <  B  <  A", loc="left", fontsize=8.0,
                  pad=4, linespacing=1.4)
    axr.set_title("(b)  Corrected: each level against the\n"
                  "       consistent reference it can attain\n"
                  "|error| ranking:   A  <  B  <  C", loc="left", fontsize=8.0,
                  pad=4, linespacing=1.4)

    # Notes go wherever the bars leave room: above zero on the left panel
    # (every bar is negative there), below zero on the right panel.
    axl.text(1.0, 5.6, "C looks exact;  B looks %.1f$\\times$ better than A"
             % (abs(BIASED[0]) / abs(BIASED[1])),
             ha="center", va="center", fontsize=6.9, style="italic",
             color="#333333", zorder=6)
    axl.text(0.02, 0.022, "all three vs 778.17 cm (intrinsic)",
             transform=axl.transAxes, ha="left", va="bottom", fontsize=6.4,
             color="#555555")
    axr.text(1.0, -10.6, "A wins;  B is %.1f$\\times$ worse than A"
             % (abs(ATTAIN[1]) / abs(ATTAIN[0])),
             ha="center", va="center", fontsize=6.9, style="italic",
             color="#333333", zorder=6)
    axr.text(0.02, 0.022, "A, B vs 686.98 cm (projected);  C vs 744.23 cm (intrinsic)",
             transform=axr.transAxes, ha="left", va="bottom", fontsize=6.4,
             color="#555555")
    axr.text(0.02, 0.97, "error bars: $\\pm$1 s.d., seeds 7-11",
             transform=axr.transAxes, ha="left", va="top", fontsize=6.4,
             color="#555555")

    # the inversion, marked in the gap between the panels
    axl.annotate("", xy=(1.245, 0.50), xytext=(1.045, 0.50),
                 xycoords="axes fraction", annotation_clip=False,
                 arrowprops=dict(arrowstyle="-|>", color=C_BIN, lw=1.1,
                                 mutation_scale=9))
    axl.text(1.145, 0.545, "ranking\ninverts", transform=axl.transAxes,
             ha="center", va="bottom", fontsize=7.0, color=C_BIN,
             fontweight="bold", linespacing=1.2, clip_on=False)

    save(fig, "fig4_inversion", target_w=FULL)


# ============================================================================
# FIG 5 -- the self-convergence trap
# ============================================================================
#
# Everything plotted here is recomputed at run time from twogrid.py, which owns
# the prescribed shape (square at 22.5 deg, P = 4a exactly) and the five
# estimators.  Nothing is hard-coded: the numbers quoted in the annotations are
# derived from the same arrays that are drawn.

def _load_sibling(name):
    """Import a script that sits next to this one, independently of the cwd."""
    import importlib.util
    import sys
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve their annotations through sys.modules, so the module
    # has to be registered BEFORE it is executed (Python 3.10 raises otherwise)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# (legend label, key in twogrid.ESTIMATORS, colour, line style, marker, filled)
# Filled markers on solid lines = the family that fails; open markers on broken
# lines = the family that converges.  That grouping is what survives greyscale.
FIG5_SERIES = [
    ("Marching squares (binary)",      "marching squares",   C_BIN,     "-",   "o", True),
    ("skimage.perimeter",              "skimage.perimeter",  "#CC79A7", "-",   "^", True),
    ("skimage.perimeter_crofton (4)",  "skimage.crofton(4)", "#E69F00", "-",   "D", True),
    ("Zero level set (signed field)",  "zero level set",     C_LVL,     "--",  "s", False),
    ("DSS (same binary mask)",         "DSS",                "#009E73", "-.",  "v", False),
]


def fig5():
    print("fig5: the self-convergence trap")
    tg = _load_sibling("twogrid")

    ns = np.array(tg.GRIDS, dtype=float)          # 400, 800, 1600
    P = tg.CANON.exact
    samples = {int(n): tg.CANON.sample(int(n)) for n in ns}
    L = {name: np.array([float(fn(samples[int(n)])) for n in ns])
         for name, fn, _ in tg.ESTIMATORS}

    # (a) what you can measure with no exact perimeter: successive differences
    # (b) what you cannot measure without one: the error itself
    self_d = {k: 100.0 * np.abs(np.diff(v)) / v[1:] for k, v in L.items()}
    err = {k: 100.0 * np.abs(v / P - 1.0) for k, v in L.items()}
    p_self = {k: np.log2(v[0] / v[1]) for k, v in self_d.items()}
    p_true = {k: np.log2(err[k][0] / err[k][-1]) / 2.0 for k in err}

    print("    shape %s, P = %.8f, grids %s" % (tg.CANON.name, P, tg.GRIDS))
    print("    %-20s %10s %11s %13s" % ("estimator", "p_self", "true order", "final error"))
    for _, k, _, _, _, _ in FIG5_SERIES:
        print("    %-20s %10.2f %11.2f %12.4f %%"
              % (k, p_self[k], p_true[k], err[k][-1]))

    k_bin = "marching squares"
    k_lvl = "zero level set"
    floor = err[k_bin][-1]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(TEXTW, 2.10), sharey=True)
    fig.subplots_adjust(left=0.093, right=0.995, top=0.865, bottom=0.175, wspace=0.075)

    for ax in (axa, axb):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(300.0, 2250.0)
        ax.set_xticks([400, 800, 1600])
        ax.set_xticklabels(["400", "800", "1600"])
        ax.minorticks_off()
        ax.grid(which="major", lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for label, key, col, ls, mk, filled in FIG5_SERIES:
        kw = dict(color=col, ls=ls, marker=mk, ms=3.4, mew=0.9,
                  mfc=col if filled else "white", mec=col, zorder=4)
        # Crofton(4) is marching squares divided by Kulpa's constant, so in
        # panel (a) -- which is scale free -- the two curves are IDENTICAL.
        # Draw it as a halo underneath so both remain visible.
        if key == "skimage.crofton(4)":
            axa.plot(ns[:-1], self_d[key], color=col, ls="-", lw=2.8, alpha=0.55,
                     solid_capstyle="round", zorder=2)
            axa.plot(ns[:-1], self_d[key], label=label, **dict(kw, ls="none"))
        else:
            axa.plot(ns[:-1], self_d[key], label=label, **kw)
        axb.plot(ns, err[key], **kw)

    # ---- (a) what the study reads off -------------------------------------
    # The two annotated numbers ARE the slopes of the two segments: with three
    # grids a self-convergence study has exactly this much information.
    # the level set ends ABOVE marching squares here, so its label goes on top:
    # crossing the two leaders would suggest the ranking runs the other way
    axa.annotate("zero level set:\napparent order %.2f" % p_self[k_lvl],
                 xy=(ns[1], self_d[k_lvl][1]), xytext=(1040.0, 0.150),
                 fontsize=6.3, color=C_LVL, ha="left", va="center",
                 linespacing=1.25,
                 arrowprops=dict(arrowstyle="-|>", color=C_LVL, lw=0.7,
                                 shrinkA=2.0, shrinkB=2.5,
                                 connectionstyle="arc3,rad=0.20"))
    axa.annotate("marching squares:\napparent order %.2f" % p_self[k_bin],
                 xy=(ns[1], self_d[k_bin][1]), xytext=(1040.0, 0.0270),
                 fontsize=6.3, color=C_BIN, ha="left", va="center",
                 linespacing=1.25,
                 arrowprops=dict(arrowstyle="-|>", color=C_BIN, lw=0.7,
                                 shrinkA=2.0, shrinkB=2.5,
                                 connectionstyle="arc3,rad=-0.20"))
    axa.text(325.0, 0.0072,
             "Crofton(4) $\\equiv$ MS$/\\kappa$, so the two coincide exactly here",
             fontsize=6.0, color="#8A6100", ha="left", va="bottom")

    # ---- (b) the floor the study cannot see -------------------------------
    axb.axhline(floor, color=C_BIN, lw=0.8, ls=(0, (1.6, 1.6)), zorder=2)
    axb.text(2150.0, floor * 1.35, "$+$%.1f %% bias floor" % floor,
             fontsize=6.8, color=C_BIN, ha="right", va="bottom",
             fontweight="bold")
    axb.text(2150.0, 0.60,
             "true order:  MS %.2f,   level set %+.2f"
             % (p_true[k_bin], p_true[k_lvl]),
             fontsize=6.2, color="#444444", ha="right", va="bottom")

    axa.set_ylim(0.0055, 26.0)
    axa.set_ylabel("relative change / error (%)")
    axa.set_xlabel("coarser grid $n$ of the pair ($n$, $2n$)")
    axb.set_xlabel("grid resolution $n$  ($n \\times n$ pixels)")

    axa.set_title("(a)  What a self-convergence study shows:\n"
                  "       $|L_n - L_{2n}| / L_{2n}$", loc="left", fontsize=7.8,
                  pad=3, linespacing=1.35)
    axb.set_title("(b)  What the true error does:\n"
                  "       $|L_n / P_{\\mathrm{exact}} - 1|$", loc="left",
                  fontsize=7.8, pad=3, linespacing=1.35)

    axa.legend(loc="upper left", bbox_to_anchor=(-0.015, 1.015), ncol=1,
               fontsize=6.2, handlelength=2.4, labelspacing=0.28,
               borderaxespad=0.0)
    save(fig, "fig5_selfconv", target_w=TEXTW)


# ============================================================================
# FIG 6 -- 3D, where radiomics lives
# ============================================================================
# Panel (a): surface-area series measured by scripts/surf3d.py (`full`, ~25 min);
#   the reference output is scripts/surf3d_output.txt.
# Panel (b): sphericity.  The binary row is PyRadiomics 3.0.1 itself, run on the
#   same masks -- scripts/pyrad3d.py, output in scripts/pyrad3d_output.txt.  The
#   signed-field row is surf3d.radiomics((100, 200, 400)), i.e. the SAME grids,
#   so the two rows of the panel are comparable resolution by resolution.
FIG6_N = np.array([50, 100, 200, 400, 800, 1600])
FIG6_MCBIN_SPH = np.array([+8.1059, +9.0009, +8.6838, +8.8211, +8.7936, +8.7953])
FIG6_MC0_SPH = np.array([-0.1335, -0.0327, -0.0081, -0.0020, -0.0005, -0.0001])
FIG6_MCBIN_ELL = np.array([+8.2476, +8.8015, +8.8173, +8.8597, +8.8726, +8.8720])
FIG6_MC0_ELL = np.array([-0.1576, -0.0386, -0.0096, -0.0024, -0.0006, -0.0001])
FIG6_PLATEAU = 8.80269          # surf3d.py, n = 400..1200, sd 0.017, no trend

FIG6_SPH_N = np.array([100, 200, 400])
FIG6_SPH_PYRAD = np.array([0.918272, 0.920189, 0.918968])   # PyRadiomics 3.0.1
FIG6_SPH_MC0 = np.array([1.001601, 1.000262, 1.000073])     # signed field
FIG6_SPH_TRUE = 1.0             # exact for a sphere, by construction


def _fig6_a(ax):
    # No legend: with four series in a 2.4 in panel a legend box costs more
    # room than it buys.  Colour+line style names the estimator, marker fill
    # names the shape, and both are said in words on the plot.
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(FIG6_N, np.abs(FIG6_MCBIN_SPH), color=C_BIN, ls="-", marker="o",
            ms=3.4, mfc="white", mec=C_BIN, mew=0.9, zorder=4)
    ax.plot(FIG6_N, np.abs(FIG6_MCBIN_ELL), color=C_BIN, ls="-", marker="s",
            ms=3.0, mfc=C_BIN, mec=C_BIN, zorder=4)
    ax.plot(FIG6_N, np.abs(FIG6_MC0_SPH), color=C_LVL, ls="--", marker="o",
            ms=3.4, mfc="white", mec=C_LVL, mew=0.9, zorder=4)
    ax.plot(FIG6_N, np.abs(FIG6_MC0_ELL), color=C_LVL, ls="--", marker="s",
            ms=3.0, mfc=C_LVL, mec=C_LVL, zorder=4)

    ax.axhline(FIG6_PLATEAU, color=C_BIN, lw=0.8, ls=(0, (1.6, 1.6)), zorder=2)
    ax.text(46.0, 95.0, "Binary marching cubes\n(the PyRadiomics route)",
            fontsize=6.5, color=C_BIN, ha="left", va="top", fontweight="bold",
            linespacing=1.25)
    ax.text(2200.0, 2.2, "$+$%.2f %% plateau,  order 0.00" % FIG6_PLATEAU,
            fontsize=6.4, color=C_BIN, ha="right", va="top")

    ax.text(47.0, 1.4e-4, "Marching cubes on the\nsigned field: order 2.00,\n"
            "down to $-$0.0001 %",
            fontsize=6.5, color=C_LVL, ha="left", va="bottom", linespacing=1.25)

    g = np.array([115.0, 620.0])
    ax.loglog(g, 0.055 * (g / 100.0) ** -2.0, color=GREY, lw=0.8,
              ls=(0, (3.5, 2)), zorder=1)
    ax.text(115.0, 0.055 * 1.3, "slope $-$2", fontsize=6.4, color=GREY,
            ha="left", va="bottom")
    ax.text(47.0, 3.9e-5, "sphere: open markers    ellipsoid: filled",
            fontsize=6.0, color="#555555", ha="left", va="bottom")

    ax.set_xlim(42.0, 2400.0)
    ax.set_ylim(3.2e-5, 320.0)
    ax.set_xticks(FIG6_N)
    ax.set_xticklabels([str(int(v)) for v in FIG6_N])
    ax.minorticks_off()
    ax.set_xlabel("grid resolution $n$  ($n^3$ voxels)")
    ax.set_ylabel("|surface-area error| (%)")
    ax.grid(which="major", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(a)  Surface area, sphere and ellipsoid",
                 loc="left", fontsize=7.8, pad=3)


def _fig6_b(ax):
    # Markers, not bars: bars would fill the panel from the axis floor up and
    # leave nowhere to put the legend, and the quantity of interest is the gap
    # to the truth line, not the height of a column.
    xs = np.arange(len(FIG6_SPH_N), dtype=float)

    ax.axhline(FIG6_SPH_TRUE, color="black", lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    ax.text(2.40, FIG6_SPH_TRUE + 0.0026,
            "truth $=$ 1, exact by construction", fontsize=6.4,
            ha="right", va="bottom")

    ax.plot(xs, FIG6_SPH_PYRAD, color=C_BIN, ls="-", marker="o", ms=4.0,
            mfc=C_BIN, mec="black", mew=0.5, zorder=5)
    ax.plot(xs, FIG6_SPH_MC0, color=C_LVL, ls="--", marker="s", ms=3.6,
            mfc="white", mec=C_LVL, mew=1.0, zorder=5)

    for i, (x, v) in enumerate(zip(xs, FIG6_SPH_PYRAD)):
        # the last point carries the gap arrow, so its label steps aside
        ax.text(x - (0.10 if i == 2 else 0.0), v + 0.0030, "%.4f" % v,
                ha="right" if i == 2 else "center", va="bottom",
                fontsize=6.2, color=C_BIN)
    for x, v in zip(xs, FIG6_SPH_MC0):
        ax.text(x, v - 0.0030, "%.4f" % v, ha="center", va="top",
                fontsize=6.2, color=C_LVL)

    # named on the plot rather than in a legend box: at this width the two
    # labels are wider than any legend that would still be legible
    ax.text(-0.34, 0.9735, "PyRadiomics 3.0.1\n(binary marching cubes)",
            fontsize=6.3, color=C_BIN, ha="left", va="top", linespacing=1.25,
            fontweight="bold")
    ax.text(-0.34, 0.9375, "Marching cubes on the\nsigned field",
            fontsize=6.3, color=C_LVL, ha="left", va="top", linespacing=1.25)

    # the gap, drawn on the finest resolution
    gap = 100.0 * (FIG6_SPH_PYRAD.mean() / FIG6_SPH_TRUE - 1.0)
    ax.annotate("", xy=(xs[2], FIG6_SPH_TRUE), xytext=(xs[2], FIG6_SPH_PYRAD[2]),
                arrowprops=dict(arrowstyle="<|-|>", color=C_BIN, lw=0.9,
                                mutation_scale=6.5, shrinkA=0.0, shrinkB=0.0),
                zorder=6)
    ax.text(xs[2] - 0.10, 0.5 * (FIG6_SPH_TRUE + FIG6_SPH_PYRAD[2]),
            "$-$%.1f %%" % abs(gap), fontsize=8.5, color=C_BIN, ha="right",
            va="center", fontweight="bold", zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels(["$n = %d$" % v for v in FIG6_SPH_N], fontsize=7.0)
    ax.set_xlim(-0.40, 2.40)
    ax.set_ylim(0.9055, 1.0125)
    ax.set_yticks([0.92, 0.94, 0.96, 0.98, 1.00])
    ax.set_ylabel("IBSI sphericity")
    ax.grid(axis="y", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # axis-break marks: the y axis does not start at zero
    for dy in (0.012, 0.026):
        ax.plot([-0.020, 0.020], [dy, dy + 0.018], transform=ax.transAxes,
                color="black", lw=0.7, clip_on=False, zorder=7)

    ax.set_title("(b)  IBSI sphericity, perfect digital sphere",
                 loc="left", fontsize=7.8, pad=3)


def fig6():
    print("fig6: 3D, where radiomics lives")
    print("    sphere    MC-bin plateau %.4f %%, MC-0 at n=1600 %+.4f %%"
          % (FIG6_MCBIN_SPH[-1], FIG6_MC0_SPH[-1]))
    print("    ellipsoid MC-bin plateau %.4f %%, MC-0 at n=1600 %+.4f %%"
          % (FIG6_MCBIN_ELL[-1], FIG6_MC0_ELL[-1]))
    for k, v in (("sphere", FIG6_MC0_SPH), ("ellipsoid", FIG6_MC0_ELL)):
        p = np.log2(np.abs(v[:-1]) / np.abs(v[1:]))
        print("    MC-0 observed order (%s): %s"
              % (k, " ".join("%.2f" % x for x in p)))
    print("    PyRadiomics sphericity err %%: %s   (mean %+.3f %%)"
          % (" ".join("%+.4f" % (100.0 * (v - 1.0)) for v in FIG6_SPH_PYRAD),
             100.0 * (FIG6_SPH_PYRAD.mean() - 1.0)))
    print("    signed-field sphericity err %%: %s"
          % " ".join("%+.4f" % (100.0 * (v - 1.0)) for v in FIG6_SPH_MC0))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(TEXTW, 2.10))
    fig.subplots_adjust(left=0.088, right=0.995, top=0.865, bottom=0.155,
                        wspace=0.30)
    _fig6_a(axa)
    _fig6_b(axb)
    save(fig, "fig6_surf3d", target_w=TEXTW)


# ============================================================================
# FIG 7 -- Crofton: more directions is not enough
# ============================================================================
# Both series are recomputed here from crofton.py: the measured one rasterises
# ten rotations of the square at n = 3200, the closed-form one is the h -> 0
# limit (1/2) sum_k dtheta_k |sin(theta_k - phi)|, predicted analytically.
FIG7_DSS_PP = 0.060        # orientation spread of DSS,            same definition
FIG7_LVL_PP = 0.023        # orientation spread of the zero level set
FIG7_N = 3200


def fig7():
    print("fig7: Crofton, more directions is not enough")
    cr = _load_sibling("crofton")

    ks = np.array(cr.KS, dtype=float)
    fams = {k: cr.direction_family(k) for k in cr.KS}

    lim = np.array([max(100.0 * (cr.asymptotic_square(k, a, fams[k]) - 1.0)
                        for a in cr.ANGS)
                    - min(100.0 * (cr.asymptotic_square(k, a, fams[k]) - 1.0)
                          for a in cr.ANGS) for k in cr.KS])

    _, h = cr.grid(FIG7_N)
    errs = {k: [] for k in cr.KS}
    for a in cr.ANGS:
        m = cr.mask_square(FIG7_N, a)
        for k in cr.KS:
            errs[k].append(100.0 * (cr.crofton_k(m, k, h, fams[k]) / cr.P_SQ - 1.0))
        del m
    meas = np.array([max(errs[k]) - min(errs[k]) for k in cr.KS])

    slope, icept = np.polyfit(np.log(ks), np.log(lim), 1)
    k_dss = float(np.exp((np.log(FIG7_DSS_PP) - icept) / slope))
    # measured density of primitive directions: ~1.1 per unit of |v|^2
    side_px = 2 * round(np.sqrt(k_dss / 1.10)) + 1

    print("    K            :", " ".join("%9d" % k for k in cr.KS))
    print("    measured (pp):", " ".join("%9.4f" % v for v in meas))
    print("    limit    (pp):", " ".join("%9.4f" % v for v in lim))
    print("    power law: spread ~ %.3f * K^(%.3f)" % (np.exp(icept), slope))
    print("    crosses DSS (%.3f pp) at K ~ %.0f -> ~%dx%d px window"
          % (FIG7_DSS_PP, k_dss, side_px, side_px))

    fig, ax = plt.subplots(figsize=(HALFW, 2.42))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # the power law, and its extrapolation past the largest K ever rasterised
    kf = np.geomspace(1.7, 32.0, 40)
    ke = np.geomspace(32.0, 420.0, 60)
    ax.plot(kf, np.exp(icept) * kf ** slope, color=GREY, lw=0.9,
            ls=(0, (3.5, 2)), zorder=2)
    ax.plot(ke, np.exp(icept) * ke ** slope, color=GREY, lw=0.9,
            ls=(0, (1.3, 1.6)), zorder=2)
    ax.text(3.0, np.exp(icept) * 3.0 ** slope * 0.40,
            "$\\propto K^{%.2f}$" % slope, fontsize=6.5, color=GREY,
            ha="center", va="top")

    # the reference floors
    # distinct dash patterns, not just distinct colours: in greyscale the two
    # floors sit within a factor of three of each other
    for y, lab, col, dsh in ((FIG7_DSS_PP, "DSS", "#009E73", (0, (4.5, 2.0))),
                             (FIG7_LVL_PP, "zero level set", C_LVL,
                              (0, (5.0, 1.6, 1.0, 1.6)))):
        ax.axhline(y, color=col, lw=0.9, ls=dsh, zorder=3)
        ax.text(1.85, y * 1.22, "%s  %.3f pp" % (lab, y), fontsize=6.2,
                color=col, ha="left", va="bottom")

    # closed-form limit (halo) and the rasterised measurement (markers)
    ax.plot(ks, lim, color=C_THEORY, lw=2.6, alpha=0.28, solid_capstyle="round",
            zorder=4, label="closed-form $h \\to 0$ limit")
    ax.plot(ks, meas, color=C_BIN, lw=1.1, ls="-", marker="o", ms=3.6,
            mfc=C_BIN, mec="black", mew=0.5, zorder=5,
            label="measured, $n = %d$" % FIG7_N)

    # the crossing: extrapolated, and said so
    ax.plot([k_dss], [FIG7_DSS_PP], ls="none", marker="*", ms=8.0, mfc="white",
            mec=C_BIN, mew=1.0, zorder=6)
    ax.annotate("$K \\approx %.0f$, extrapolated\n($\\approx %d \\times %d$ px window)"
                % (k_dss, side_px, side_px),
                xy=(k_dss, FIG7_DSS_PP), xytext=(26.0, 0.0135),
                fontsize=6.2, color=C_BIN, ha="center", va="top",
                arrowprops=dict(arrowstyle="-|>", color=C_BIN, lw=0.7,
                                shrinkA=1.0, shrinkB=3.0,
                                connectionstyle="arc3,rad=0.30"))

    ax.set_xlim(1.7, 460.0)
    ax.set_ylim(4.5e-3, 90.0)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
    ax.minorticks_off()
    ax.set_xlabel("number of Crofton directions $K$")
    ax.set_ylabel("orientation spread (pp)")
    ax.grid(which="major", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.015, 1.015), fontsize=6.2,
              handlelength=1.9, labelspacing=0.28, borderaxespad=0.0)
    save(fig, "fig7_crofton", target_w=HALFW)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("\ndone ->", OUT)
