#!/usr/bin/env python3
"""
Five-seed replication of the digitisation error of levels A, B and C.

The table that matters is the last block: each level against the reference it
can actually reach. A and B measure on the projection, so their reference is the
projected perimeter; only C can aspire to the intrinsic one. The two rows marked
"published table" show what mixing them produced.
"""

import os
import sys
import time
import warnings

import numpy as np

# scripts/ lives two levels below the repository root: the root is derived from
# the location of THIS file, not from an absolute path on one machine.
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from foliar import pipeline, synth  # noqa: E402

SEEDS = (7, 8, 9, 10, 11)

rows = []
for seed in SEEDS:
    t0 = time.time()
    cfg = synth.LeafConfig(seed=seed)
    s = synth.generate(cfg)
    t_gen = time.time() - t0

    t1 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = pipeline.run(s, curl_radius_cm=s.curl_radius_cm)
        warned = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    t_pipe = time.time() - t1

    gt = s.truth
    pa, pb, pc = (r.level_A["perimeter_cm"], r.level_B["perimeter_cm"],
                  r.level_C["perimeter_cm"])
    rows.append(dict(
        seed=seed, p_int=gt.perimeter_cm, p_prj=gt.projected_perimeter_cm,
        pA=pa, pB=pb, pC=pc,
        eA=100 * (pa / gt.projected_perimeter_cm - 1),
        eB=100 * (pb / gt.projected_perimeter_cm - 1),
        eC=100 * (pc / gt.perimeter_cm - 1),
        # errors "against the intrinsic truth", the ones the A/B/C table published
        eA_i=100 * (pa / gt.perimeter_cm - 1),
        eB_i=100 * (pb / gt.perimeter_cm - 1),
        proj=100 * (gt.projected_perimeter_cm / gt.perimeter_cm - 1),
        tri=len(r.mesh_flat.F), nod=len(r.mesh_flat.V),
        ang=r.mesh_flat.min_angle_deg, cap=r.mesh_flat.steiner_capped,
        eareaA=100 * (r.level_A["area_cm2"] / gt.blade_area_cm2 - 1),
        eareaB=100 * (r.level_B["area_cm2"] / gt.blade_area_cm2 - 1),
        eareaC=100 * (r.level_C["area_cm2"] / gt.blade_area_cm2 - 1),
        tgen=t_gen, tpipe=t_pipe, warn=warned))

    print(f"seed {seed}  gen {t_gen:.1f}s  pipe {t_pipe:.1f}s  "
          f"tri {len(r.mesh_flat.F)}  nod {len(r.mesh_flat.V)}  "
          f"ang {r.mesh_flat.min_angle_deg:.2f}  "
          f"cap {r.mesh_flat.steiner_capped}  warn {warned}")
    sys.stdout.flush()

print()
hdr = (f"{'seed':>5} {'P_int':>10} {'P_proj':>10} {'A':>10} {'B':>10} {'C':>10} | "
       f"{'errA%':>8} {'errB%':>8} {'errC%':>8} | {'proj%':>8}")
print(hdr)
for d in rows:
    print(f"{d['seed']:5d} {d['p_int']:10.3f} {d['p_prj']:10.3f} {d['pA']:10.3f} "
          f"{d['pB']:10.3f} {d['pC']:10.3f} | {d['eA']:+8.3f} {d['eB']:+8.3f} "
          f"{d['eC']:+8.3f} | {d['proj']:+8.3f}")

print()
for k, lab in (("eA", "A (Crofton, against projected)"),
               ("eB", "B (flat mesh, against projected)"),
               ("eC", "C (lifted mesh, against intrinsic)"),
               ("proj", "pure projection factor"),
               ("eA_i", "A against intrinsic (published table)"),
               ("eB_i", "B against intrinsic (published table)"),
               ("eareaA", "area A"), ("eareaB", "area B"), ("eareaC", "area C")):
    v = np.array([d[k] for d in rows])
    print(f"{lab:<42s}  mean {v.mean():+7.3f} %  sd {v.std(ddof=1):6.3f}  "
          f"(n={len(v)}, range {v.min():+0.3f} to {v.max():+0.3f})")

t = np.array([d["tri"] for d in rows], float)
n = np.array([d["nod"] for d in rows], float)
print(f"\ntriangles:   {[int(x) for x in t]}  (mean {t.mean():.0f})")
print(f"nodes:       {[int(x) for x in n]}  (mean {n.mean():.0f})")
print(f"min angle:   {['%.2f' % d['ang'] for d in rows]}")
print(f"gen/pipe time (s): {['%.1f' % d['tgen'] for d in rows]} / "
      f"{['%.1f' % d['tpipe'] for d in rows]}")
