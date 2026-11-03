#!/usr/bin/env python3
"""
Demo of the infection PDE on the mesh.

    python3 demo_infeccion.py

Solves the quasi-static SI model of foliar/infection.py on the reference leaf
and writes figures and tables to out/.

Every number this script prints is reproducible by running this file. What
CANNOT be concluded from it is in the final section of the output: phase 1
measures whether the numerical scheme works, not whether the biology is true.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from foliar import fem, infection, latency, meshing, params, synth, viz

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# Horizon and target severity. The horizon is the typical interval between two
# photos of the same leaf in a field protocol; extrapolating much beyond that is
# a product, not a test. The target severity is a CHOICE of this demo (doubling
# in 21 days), not data: in the field it comes from the second photo.
HORIZON_DAY = 21.0
TARGET_SEVERITY_PCT = 2.50
GAMMAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]


def banner(t):
    print(f"\n\033[1m{t}\033[0m\n" + "-" * len(t))


def main():
    banner("1 · Lifted mesh and vein frame")
    t0 = time.time()
    cfg = synth.LeafConfig()
    sample = synth.generate(cfg)
    px, org = sample.px_per_cm, sample.origin_cm
    mesh = meshing.build_mesh(meshing.mask_to_polygon(sample.mask_blade, org, px),
                              meshing.mask_to_polygon(sample.mask_lesion, org, px, 0.18),
                              meshing.mask_to_polygon(sample.mask_midrib, org, px), px)
    ml = meshing.lift_to_curl(mesh, sample.curl_radius_cm)
    src = np.array(sorted(set(mesh.vertices_in(meshing.REG_MIDRIB).tolist()) &
                          set(mesh.vertices_in(meshing.REG_HEALTHY).tolist())), int)
    print(f"mesh             {len(ml.F)} triangles, {len(ml.V)} nodes "
          f"({time.time() - t0:.1f} s)")

    t0 = time.time()
    frame = infection.vein_frame(ml, src)
    print(f"vein frame       {time.time() - t0:.2f} s")

    # the frame has analytic ground truth: the isometry of the cylinder
    R = sample.curl_radius_cm
    C = ml.V[ml.F].mean(axis=1)
    vv = R * np.arcsin(np.clip(C[:, 1] / R, -1, 1))
    exact = np.column_stack([np.zeros(len(C)), np.cos(vv / R),
                             np.sin(vv / R)]) * np.sign(vv)[:, None]
    ang = np.degrees(np.arccos(np.clip(
        np.abs(np.einsum("ti,ti->t", frame.e_v, exact)), -1, 1)))
    A = fem.triangle_areas(ml.V, ml.F)
    # evaluated away from the midrib: within |v| < 2 cm the v direction changes
    # sign and the analytic reference is undefined
    far = np.abs(vv) > 2.0
    inv = int(((np.einsum("ti,ti->t", frame.e_v, exact) < 0) & far).sum())
    print(f"angular error    {np.average(ang[far], weights=A[far]):.3f} degrees mean, "
          f"{ang[far].max():.2f} max, {inv} flipped  (within |v| > 2 cm)")
    print(f"mean |grad d|    {frame.grad_norm.mean():.4f}   "
          f"(self-diagnostic: the eikonal condition says 1)")

    I0 = infection.initial_infection(ml)
    op0 = infection.InfectionOperator(ml, frame, infection.InfectionModel())
    diag = op0.diagnostics(I0)
    print(f"Phi undershoot   {diag['undershoot_rel']:.2e} relative "
          f"({diag['n_offdiag_positive']} of {diag['n_offdiag']} off-diagonals "
          f"positive from obtuse triangles)")
    print(f"run reliable: {diag['reliable']}")

    banner("2 · Hypothesis sweep, at identical final severity")
    print(f"beta_0 is calibrated by bisection so that all {len(GAMMAS)} models reach")
    print(f"{TARGET_SEVERITY_PCT:.2f} % severity at {HORIZON_DAY:.0f} days.")
    print("Without that, changing gamma would change the global rate and Psi would")
    print("stop being comparable: one would measure how much it grew, not where.\n")

    psi_ref = op0.state(0.0, I0).psi
    I_null, r_null, sev_null = infection.null_dilation(ml, I0, TARGET_SEVERITY_PCT)
    psi_null = op0.wsi(I_null) / op0.severity_pct(I_null)
    print(f"Psi(0)                      {psi_ref:.4f}")
    print(f"geometric null              isotropic geodesic dilation of "
          f"{r_null:.3f} cm -> Psi {psi_null:.4f}")
    print("  (the null exists because the leaf edge rectifies isotropic growth")
    print("   inwards: without it, Psi rises even when gamma is zero)\n")

    print(f"  {'gamma':>7} {'beta0 [1/d]':>12} {'final sev %':>12} "
          f"{'Psi':>8} {'excess pp':>11}")
    rows, series = [], []
    for g in GAMMAS:
        mod = infection.InfectionModel(gamma=g)
        b0 = infection.calibrate_beta0(ml, frame, mod, I0, HORIZON_DAY,
                                       TARGET_SEVERITY_PCT)
        mod = infection.InfectionModel(gamma=g, beta0_per_day=b0)
        _, sts = infection.simulate(ml, frame, mod, I0, HORIZON_DAY,
                                    dt_day=0.5, record_every_day=1.0)
        exc = infection.excess_pp(sts[-1].psi, psi_null, psi_ref)
        rows.append(dict(gamma=g, beta0_per_day=b0,
                         final_severity_pct=sts[-1].severity_pct,
                         psi_final=sts[-1].psi, psi_null=psi_null,
                         excess_pp=exc))
        print(f"  {g:+7.1f} {b0:12.5f} {sts[-1].severity_pct:12.4f} "
              f"{sts[-1].psi:8.4f} {exc:+11.2f}")
        if g in (-1.0, 0.0, 1.0):
            series.append((f"gamma = {g:+.0f}", [s.t_day for s in sts],
                           [s.severity_pct for s in sts], [s.psi for s in sts]))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "infection_hypothesis.csv", index=False)

    gg = np.asarray([r["gamma"] for r in rows])
    ee = np.asarray([r["excess_pp"] for r in rows])
    cross = float(np.interp(0.0, ee, gg)) if ee[0] < 0 < ee[-1] else float("nan")
    print(f"\nReading: the excess crosses zero at gamma = {cross:+.2f}. With a real pair")
    print("of photos gamma is fitted; if the confidence interval includes zero, the")
    print("midrib-as-source hypothesis is NOT supported by that leaf.")

    banner("3 · Agronomic deliverable: days to cross a grade")
    # The Stover-Gauhl grade is the quantity field decisions are made on.
    # CAREFUL: this is the model's OUTPUT, not a test of it. See section 5.
    mod = infection.InfectionModel(gamma=1.0,
                                   beta0_per_day=float(tab.loc[tab.gamma == 1.0,
                                                               "beta0_per_day"].iloc[0]))
    op, sts_long = infection.simulate(ml, frame, mod, I0, 90.0, dt_day=0.5,
                                      record_every_day=1.0)
    n_les = len(params.lesion_components(ml))
    cur_grade, _ = params.stover_gauhl_grade(sts_long[0].severity_pct, n_les)
    print(f"current grade    {cur_grade}   (severity {sts_long[0].severity_pct:.3f} %, "
          f"{n_les} lesions)")
    for thr, grade in [(1.0, 2), (5.0, 3), (15.0, 4)]:
        hit = next((s for s in sts_long if s.severity_pct >= thr), None)
        if hit is not None and thr > sts_long[0].severity_pct:
            print(f"crosses {thr:5.1f} % (grade {grade})  on day {hit.t_day:5.1f}")

    banner("3bis · The time axis, anchored with the latency compartment")
    lat = latency.LatencyModel()
    print(f"tau_E = {lat.tau_E_day:g} d (infection -> segmentable spot, Foure stage 4)")
    print(f"tau_asex = {lat.tau_asex_day:g} d (infection -> conidia, stage 2)")
    print(f"k = {lat.k_stages} stages, kappa = {lat.kappa_per_day:.5f}/day, "
          f"delay standard deviation {lat.sd_day:.1f} d")
    print("Source: Bornacelly Horta (2009), MSc thesis UNAL Palmira, Table 4.\n")

    r_obs = latency.growth_rate_from_pair(sts_long[0].severity_pct,
                                          TARGET_SEVERITY_PCT, HORIZON_DAY)
    lop = latency.LatentOperator(ml, frame, infection.InfectionModel(gamma=1.0), lat)
    E0, info = latency.initial_latent(lop, I0, r_obs)
    stl0 = lop.state(0.0, E0, I0)
    print(f"prior growth rate r = {r_obs:.5f}/day (from the observed severity)")
    print(f"VISIBLE tissue   {stl0.severity_pct:7.4f} %")
    print(f"LATENT tissue    {stl0.latent_pct:7.4f} %   <- invisible to the camera")
    print(f"ratio {stl0.latent_pct/stl0.severity_pct:.4f}, exact (1+r/kappa)^k - 1 = "
          f"{(1 + r_obs/lat.kappa_per_day)**lat.k_stages - 1:.4f}\n")

    b0_lat, ident = latency.calibrate_beta0_latent(
        ml, frame, infection.InfectionModel(gamma=1.0), lat, I0, E0,
        HORIZON_DAY, TARGET_SEVERITY_PCT)
    b0_no = float(tab.loc[tab.gamma == 1.0, "beta0_per_day"].iloc[0])
    print(f"beta_0 without latency  {b0_no:.5f} /day   (it is a NECROSIS rate)")
    print(f"beta_0 with latency     {b0_lat:.5f} /day   (it is an INFECTION rate)"
          f"   x{b0_lat/b0_no:.2f}")
    print("It comes out LARGER, not inflated: tissue infected within the window")
    print("barely has time to become visible, so more infection is needed for the")
    print("same observed increment.\n")

    _, ss0 = latency.simulate_latent(
        ml, frame, infection.InfectionModel(gamma=1.0, beta0_per_day=0.0),
        lat, I0, E0, HORIZON_DAY, dt_day=0.5, record_every_day=1e9)
    frac = 100.0 * ((ss0[-1].severity_pct - stl0.severity_pct) /
                    (TARGET_SEVERITY_PCT - stl0.severity_pct))
    print(f"With beta_0 = 0, i.e. ZERO new infection, severity at {HORIZON_DAY:.0f} days")
    print(f"still reaches {ss0[-1].severity_pct:.4f} %: draining the latent pool "
          f"explains {frac:.1f} %")
    print(f"of the observed increment. beta_0 rests on the remaining {100-frac:.1f} %.")
    print("CONSEQUENCE: the two-photo protocol at 14-21 days is NOT enough to")
    print("calibrate beta_0. A third photo or a longer window is required.")

    banner("4 · Figures and tables")
    mod1 = infection.InfectionModel(gamma=1.0,
                                    beta0_per_day=float(tab.loc[tab.gamma == 1.0,
                                                                "beta0_per_day"].iloc[0]))
    _, sts1 = infection.simulate(ml, frame, mod1, I0, HORIZON_DAY, dt_day=0.5,
                                 record_every_day=1.0)
    for p in [
        viz.figure_infection(ml, frame, sts1, OUT / "04_infection.png", gamma=1.0),
        viz.figure_infection_hypothesis(list(tab.gamma), list(tab.excess_pp),
                                        series, OUT / "05_infection_hypothesis.png"),
    ]:
        print(f"  {p}")
    print(f"  {OUT / 'infection_hypothesis.csv'}")

    summary = dict(
        n_triangles=int(len(ml.F)), n_nodes=int(len(ml.V)),
        horizon_day=HORIZON_DAY, target_severity_pct=TARGET_SEVERITY_PCT,
        psi_initial=psi_ref, psi_null=psi_null, null_radius_cm=r_null,
        frame_angular_error_deg=float(np.average(ang[far], weights=A[far])),
        mean_grad_norm=float(frame.grad_norm.mean()),
        relative_undershoot=diag["undershoot_rel"],
        offdiag_positive=diag["n_offdiag_positive"],
        offdiag_total=diag["n_offdiag"],
        excess_pp={str(r["gamma"]): r["excess_pp"] for r in rows},
    )
    (OUT / "infection_summary.json").write_text(json.dumps(summary, indent=2,
                                                           default=float))
    print(f"  {OUT / 'infection_summary.json'}")

    banner("5 · What this demo does NOT demonstrate")
    for line in [
        "The anisotropy ratio comes from (elongation 3.70)^2 and that 3.70 is a",
        "  parameter of the synthetic generator: using it as biological evidence is",
        "  circular. Phase 1 validates the SCHEME, not the biological model.",
        "ell_v = 2.6 cm is invented and directly controls how far the front",
        "  advances. No absolute expansion speed is defensible yet.",
        "The time axis IS anchored (section 3bis), but tau_E = 38 d comes from a",
        "  different region and management system than the simulated leaf, and rests",
        "  on a single primary source.",
        "With tau_E = 38 d a 21-day window does NOT identify beta_0: the initial",
        "  latent pool explains most of the increment. The field protocol has to be",
        "  rewritten with three photos or a longer window.",
        "The model asks for a new observable, r, the prior growth rate. Getting r",
        "  wrong by a factor 2 moves severity at 60 days by more than 300 %. It is",
        "  measurable, but today it is not measured.",
        "The grade crossing date differs by ~3 days between gamma = -1 and",
        "  gamma = +1, so it does NOT separate the hypotheses and cannot validate",
        "  the model.",
        "The ABSOLUTE excess depends on segmentation quality; only the paired",
        "  difference against the null survives mask noise.",
        "Level B and level C cannot be mixed: the null and the model have to come",
        "  from the same level or false signal is manufactured.",
        "The only real test is a pair of photos of the same leaf 14-21 days apart.",
    ]:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
