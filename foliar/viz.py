"""Pipeline figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from . import fem, meshing

BG = "#0e1116"
FG = "#e8eaed"
GRID = "#2a3038"
C_BLADE = "#8fbf6a"
C_LESION = "#c25b3a"
C_MIDRIB = "#e0c15c"
C_MESH = "#5a6472"


def _style(ax, title=None):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors="#9aa3ad", labelsize=8)
    if title:
        ax.set_title(title, color=FG, fontsize=10, loc="left", pad=6)
    ax.set_aspect("equal")


def _tri_collection(mesh, values=None, cmap="viridis", vmin=None, vmax=None,
                    facecolors=None, lw=0.12):
    verts = mesh.V[:, :2][mesh.F]
    pc = PolyCollection(verts, linewidths=lw, edgecolors=C_MESH)
    if values is not None:
        pc.set_array(np.asarray(values))
        pc.set_cmap(cmap)
        pc.set_clim(vmin, vmax)
    if facecolors is not None:
        pc.set_facecolors(facecolors)
    return pc


def figure_pipeline(sample, result, path):
    """Four panels: mask -> contour -> mesh -> geodesic field."""
    fig = plt.figure(figsize=(15, 11), facecolor=BG)
    gs = fig.add_gridspec(4, 2, width_ratios=[60, 1], hspace=0.32, wspace=0.02,
                          left=0.045, right=0.965, top=0.965, bottom=0.05)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    cax = fig.add_subplot(gs[3, 1])
    X, Y = sample.pixel_xy_cm()
    ext = [X[0], X[-1], Y[-1], Y[0]]

    # 1 --- segmentation masks
    ax = axes[0]
    rgb = np.zeros(sample.mask_blade.shape + (3,))
    rgb[sample.mask_blade] = matplotlib.colors.to_rgb(C_BLADE)
    rgb[sample.mask_midrib] = matplotlib.colors.to_rgb(C_MIDRIB)
    rgb[sample.mask_lesion] = matplotlib.colors.to_rgb(C_LESION)
    ax.imshow(rgb, extent=ext, interpolation="nearest")
    _style(ax, "1 · Segmentation masks (pipeline input)")

    # 2 --- simplified subpixel contours
    ax = axes[1]
    ax.imshow(np.where(sample.mask_blade, 0.12, 0.0), extent=ext,
              cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
    for name, col, lwd in [("blade", C_BLADE, 1.1), ("midrib", C_MIDRIB, 0.9),
                           ("lesion", C_LESION, 0.9)]:
        p = result.polygons[name]
        for g in (p.geoms if hasattr(p, "geoms") else [p]):
            if g.is_empty:
                continue
            for ring in [g.exterior, *g.interiors]:
                xy = np.asarray(ring.coords)
                ax.plot(xy[:, 0], xy[:, 1], color=col, lw=lwd)
    nv = sum(len(np.asarray(r.coords))
             for pn in ["blade", "lesion", "midrib"]
             for g in (result.polygons[pn].geoms if hasattr(result.polygons[pn], "geoms")
                       else [result.polygons[pn]]) if not g.is_empty
             for r in [g.exterior, *g.interiors])
    _style(ax, f"2 · Subpixel contours (marching squares + Douglas-Peucker) · {nv} vertices")

    # 3 --- mesh
    ax = axes[2]
    mesh = result.mesh_flat
    cols = np.empty((len(mesh.F), 4))
    cols[mesh.attr == meshing.REG_HEALTHY] = matplotlib.colors.to_rgba("#4a7a3a", 1.0)
    cols[mesh.attr == meshing.REG_LESION] = matplotlib.colors.to_rgba(C_LESION, 1.0)
    cols[mesh.attr == meshing.REG_MIDRIB] = matplotlib.colors.to_rgba(C_MIDRIB, 1.0)
    ax.add_collection(_tri_collection(mesh, facecolors=cols, lw=0.06))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    q = result.quality
    _style(ax, f"3 · CDT mesh with subdomains · {q['n_triangles']} triangles · "
               f"minimum angle {q['min_angle_deg']:.1f} degrees")

    # 4 --- geodesic field
    ax = axes[3]
    d = result.fields["d_lift"] if result.fields["d_lift"] is not None else result.fields["d_flat"]
    dt = fem.interpolate_to_triangles(d, mesh.F)
    finite = np.isfinite(dt)
    pc = _tri_collection(mesh, values=np.nan_to_num(dt, nan=np.nanmax(dt)),
                         cmap="magma", vmin=0, vmax=np.nanmax(dt), lw=0.0)
    ax.add_collection(pc)
    if (~finite).any():
        ax.add_collection(PolyCollection(mesh.V[:, :2][mesh.F[~finite]],
                                         facecolors="#3a3f47", linewidths=0))
    # lesion contours on top
    p = result.polygons["lesion"]
    for g in (p.geoms if hasattr(p, "geoms") else [p]):
        if not g.is_empty:
            xy = np.asarray(g.exterior.coords)
            ax.plot(xy[:, 0], xy[:, 1], color="#7dd3fc", lw=0.8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label("geodesic distance to the midrib (cm)", color=FG, fontsize=8)
    cb.ax.tick_params(colors="#9aa3ad", labelsize=7)
    cb.outline.set_edgecolor(GRID)
    _style(ax, "4 · Geodesic distance to the midrib on the lifted mesh "
               "(blue = lesion contour, grey = strip with no connection)")

    for a in axes:
        a.set_xlabel("cm", color="#9aa3ad", fontsize=8)
    fig.savefig(path, dpi=125, facecolor=BG)
    plt.close(fig)
    return path


def figure_comparison(sample, result, path):
    """Error bars for each method against the ground truth."""
    gt = sample.truth
    a, b, c = result.level_A, result.level_B, result.level_C

    panels = [
        ("Leaf area (cm2)", gt.blade_area_cm2, [
            ("A · pixel counting", a["area_cm2"]),
            ("B · flat mesh", b["area_cm2"]),
            ("C · lifted mesh", c["area_cm2"] if c else np.nan)]),
        ("Perimeter (cm)", gt.perimeter_cm, [
            ("A · raster Crofton", a["perimeter_crofton_cm"]),
            ("B · mesh contour", b["perimeter_cm"]),
            ("C · lifted contour", c["perimeter_cm"] if c else np.nan)]),
        ("Severity (%)", gt.severity_pct, [
            ("A · pixels", a["severity_pct"]),
            ("B · flat mesh", b["severity_pct"]),
            ("C · lifted mesh", c["severity_pct"] if c else np.nan)]),
        ("Geodesic dist. to midrib (cm)", gt.mean_geodesic_to_midrib_cm, [
            ("A · Euclidean in image", a["mean_dist_euclidean_cm"]),
            ("B · flat mesh geodesic", b["mean_geodesic_to_midrib_cm"]),
            ("C · lifted geodesic", c["mean_geodesic_to_midrib_cm"] if c else np.nan)]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), facecolor=BG)
    for ax, (title, truth, items) in zip(axes, panels):
        names = [n for n, _ in items]
        errs = [100.0 * (v / truth - 1.0) for _, v in items]
        colors = ["#c25b3a" if abs(e) > 5 else "#e0a83a" if abs(e) > 1 else "#4ea86b"
                  for e in errs]
        y = np.arange(len(names))[::-1]
        ax.barh(y, errs, color=colors, height=0.5)
        ax.axvline(0, color=FG, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        lim = max(3.0, 1.30 * max(abs(min(errs)), abs(max(errs))))
        ax.set_xlim(-lim, lim)
        for yi, e, nm in zip(y, errs, names):
            # name above the bar, value at its tip (inside if it fits)
            ax.text(-lim * 0.97, yi + 0.30, nm, va="bottom", ha="left",
                    color="#c8ccd2", fontsize=8)
            inside = abs(e) > 0.30 * lim
            if inside:
                ax.text(e + (-0.02 if e >= 0 else 0.02) * lim, yi, f"{e:+.2f}%",
                        va="center", ha="right" if e >= 0 else "left",
                        color="#141414", fontsize=8.5, fontweight="bold")
            else:
                ax.text(e + (0.02 if e >= 0 else -0.02) * lim, yi, f"{e:+.2f}%",
                        va="center", ha="left" if e >= 0 else "right",
                        color=FG, fontsize=8.5, fontweight="bold")
        ax.set_facecolor(BG)
        ax.set_title(f"{title}\ntruth = {truth:.2f}", color=FG, fontsize=9, loc="left")
        ax.set_xlabel("relative error (%)", color="#9aa3ad", fontsize=8)
        ax.tick_params(colors="#9aa3ad", labelsize=8)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=BG)
    plt.close(fig)
    return path


def figure_curl_sensitivity(sample, result, radii, areas, path):
    """What it costs to get the curl radius wrong."""
    gt = sample.truth
    R0 = gt.curl_radius_cm
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=BG)
    rel = 100.0 * (np.asarray(areas) / gt.blade_area_cm2 - 1.0)
    err_R = 100.0 * (np.asarray(radii) / R0 - 1.0)
    ax.plot(err_R, rel, color="#7dd3fc", lw=1.8, marker="o", ms=4)
    ax.axhline(0, color=FG, lw=0.8)
    ax.axvline(0, color=FG, lw=0.8, ls=":")
    flat = 100.0 * (result.level_B["area_cm2"] / gt.blade_area_cm2 - 1.0)
    ax.axhline(flat, color="#c25b3a", lw=1.2, ls="--",
               label=f"ignoring the curl: {flat:+.1f}%")
    ax.set_xlabel("error in the curl radius R (%)", color=FG, fontsize=9)
    ax.set_ylabel("error in leaf area (%)", color=FG, fontsize=9)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#9aa3ad", labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    lg = ax.legend(facecolor=BG, edgecolor=GRID, fontsize=8)
    for t in lg.get_texts():
        t.set_color(FG)
    ax.set_title("Sensitivity of the correction to the external curvature input",
                 color=FG, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=BG)
    plt.close(fig)
    return path


def figure_infection(mesh, frame, states, path, gamma=None):
    """
    Four panels of the infection model on the lifted mesh.

    Panel 4 is the one that answers the question: where the lesion GREW, which
    is not the same as where it is.
    """
    from . import infection as inf

    V2 = mesh.V[:, :2]
    st0, stf = states[0], states[-1]
    Itri0 = st0.I[mesh.F].mean(axis=1)
    Itrif = stf.I[mesh.F].mean(axis=1)

    fig = plt.figure(figsize=(15, 11), facecolor=BG)
    gs = fig.add_gridspec(4, 2, width_ratios=[60, 1], hspace=0.34, wspace=0.02,
                          left=0.045, right=0.965, top=0.955, bottom=0.05)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    # 1 --- vein frame
    ax = axes[0]
    pc0 = _tri_collection(mesh, lw=0.04)
    pc0.set_facecolors("#1b2029")
    ax.add_collection(pc0)
    C = mesh.V[mesh.F].mean(axis=1)
    k = max(1, len(C) // 700)                     # subsample so it stays legible
    q = ax.quiver(C[::k, 0], C[::k, 1], frame.e_v[::k, 0], frame.e_v[::k, 1],
                  frame.coherence[::k], cmap="viridis", clim=(0.6, 1.0),
                  width=0.0022, scale=34, headwidth=3.2, pivot="mid")
    ax.autoscale_view()
    cb0 = fig.colorbar(q, cax=fig.add_subplot(gs[0, 1]))
    cb0.set_label("frame coherence", color=FG, fontsize=8)
    cb0.ax.tick_params(colors="#9aa3ad", labelsize=7)
    cb0.outline.set_edgecolor(GRID)
    _style(ax, "1 · Secondary vein frame (e_v, from the geodesic gradient) "
               "— the major axis of the diffusion tensor")

    # 2 --- initial state
    ax = axes[1]
    ax.add_collection(_tri_collection(mesh, values=Itri0, cmap="inferno",
                                      vmin=0, vmax=1, lw=0.05))
    ax.autoscale_view()
    _style(ax, f"2 · Observed lesion, I(t=0) — severity {st0.severity_pct:.2f} %")

    # 3 --- final state
    ax = axes[2]
    pc = _tri_collection(mesh, values=Itrif, cmap="inferno", vmin=0, vmax=1, lw=0.05)
    ax.add_collection(pc)
    ax.autoscale_view()
    ttl = f"3 · Prediction at {stf.t_day:.0f} days — severity {stf.severity_pct:.2f} %"
    if gamma is not None:
        ttl += f"   (gamma = {gamma:+.1f})"
    _style(ax, ttl)
    cb = fig.colorbar(pc, cax=fig.add_subplot(gs[2, 1]))
    cb.set_label("necrotic fraction I", color=FG, fontsize=8)
    cb.ax.tick_params(colors="#9aa3ad", labelsize=7)
    cb.outline.set_edgecolor(GRID)

    # 4 --- where it grew
    ax = axes[3]
    grow = Itrif - Itri0
    pc = _tri_collection(mesh, values=grow, cmap="magma", vmin=0,
                         vmax=max(grow.max(), 1e-9), lw=0.05)
    ax.add_collection(pc)
    ax.autoscale_view()
    _style(ax, f"4 · Where it grew, I({stf.t_day:.0f}) - I(0) — "
               f"Psi {st0.psi:.3f} -> {stf.psi:.3f} "
               f"(>0 = the lesion mass shifts towards the midrib)")
    cb = fig.colorbar(pc, cax=fig.add_subplot(gs[3, 1]))
    cb.set_label("increment of I", color=FG, fontsize=8)
    cb.ax.tick_params(colors="#9aa3ad", labelsize=7)
    cb.outline.set_edgecolor(GRID)

    fig.savefig(path, dpi=140, facecolor=BG)
    plt.close(fig)
    return path


def figure_infection_hypothesis(gammas, excesses, series, path):
    """
    Left: the decision statistic against gamma, with the null at zero.
    Right: severity over time for three hypotheses, at equal final severity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=BG)

    ax1.axhline(0, color=FG, lw=0.9)
    ax1.axvline(0, color="#9aa3ad", lw=0.8, ls=":")
    ax1.plot(gammas, excesses, color="#7dd3fc", lw=1.8, marker="o", ms=4.5)
    ax1.fill_between(gammas, 0, excesses, where=np.asarray(excesses) > 0,
                     color="#4ea86b", alpha=0.18)
    ax1.fill_between(gammas, 0, excesses, where=np.asarray(excesses) < 0,
                     color="#c25b3a", alpha=0.18)
    ax1.set_xlabel("gamma  (>0 = the midrib accelerates infection)", color=FG, fontsize=9)
    ax1.set_ylabel("excess over the paired null (pp)", color=FG, fontsize=9)
    ax1.set_title("Decision statistic: only the excess over the null,\n"
                  "at identical final severity", color=FG, fontsize=10, loc="left")

    # Psi(t) goes on the right, not severity: with beta_0 calibrated to the same
    # end point the three severity curves are indistinguishable by eye. That is
    # precisely the argument for why a SPATIAL model is needed -- the aggregate
    # does not separate the hypotheses -- and the figure has to show it, not
    # hide it.
    for (lab, t, sev, psi), col in zip(series, ["#c25b3a", "#e0c15c", "#4ea86b"]):
        ax2.plot(t, psi, color=col, lw=1.8, label=lab)
        ax2.plot(t, np.asarray(sev) / sev[-1] * psi[0], color=col, lw=1.0,
                 ls=":", alpha=0.55)
    ax2.set_xlabel("days", color=FG, fontsize=9)
    ax2.set_ylabel("Psi = WSI / severity", color=FG, fontsize=9)
    ax2.set_title("Solid: Psi(t), separates the hypotheses.  Dotted: rescaled\n"
                  "severity, identical in all three by construction",
                  color=FG, fontsize=10, loc="left")
    lg = ax2.legend(facecolor=BG, edgecolor=GRID, fontsize=8)
    for t_ in lg.get_texts():
        t_.set_color(FG)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#9aa3ad", labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=BG)
    plt.close(fig)
    return path
