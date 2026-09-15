#!/usr/bin/env python3
"""
locality_attribution.py -- is the measured influence physical, or the solver's?

    conda activate tissue-host
    python host/locality_attribution.py            # ~2-3 min, 6 children

DECISION_LOG.md section 13.2. HOST ONLY (children import MPM.mpm3d).

WHY THIS EXISTS
---------------
host/mpm_locality_study.py found the 1% velocity-influence radius of a local kick
at 1/240 s to be ~130 mm for the SOFT material, where the P-wave can only have
travelled c_p * t = 18 mm. Influence beyond the physical domain of dependence
has to be carried by something else, and the two candidates predict different
things:

  net momentum     a kick on 12 of 24,000 particles gives the whole free body a
                   centre-of-mass velocity. Momentum is conserved, so this part
                   does not depend on how the solver discretises time.
  solver stencil   every P2G + G2P transfer spreads a particle's momentum over
                   +-1.5 dx twice. More substeps per frame = more transfers in the
                   same physical time = further numerical spread. This part
                   GROWS with substep count at fixed physical time.

So the experiment holds physical time fixed and changes only the substep count
(55, 110, 220 per 1/240 s frame -- one process each, since dt is baked into the
kernels). Response that grows with substep count is numerical; response that
does not is physical (or the conserved momentum mode).

A MOMENTUM-FREE KICK WAS TRIED AND REJECTED. Removing the patch impulse by
subtracting dv * n_patch / N from every particle's velocity keeps the dynamics
momentum-free -- but that uniform subtraction is itself part of B - A, so it
appears as a response of ~5e-4 m/s (0.07-0.2 of the source) at EVERY distance
from t = 0. Its far field rose instead of falling, for that reason alone. The
centre-of-mass share of the standard kick's response is reported instead.

Reuses the locality study's probe, snapshot/restore and _measure(), so the
numbers are the same measurement at different substep counts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "host"))

import damping as DMP                                    # noqa: E402
import graph_design as GD                                # noqa: E402
import mpm_locality_study as LS                          # noqa: E402

SUBSTEPS = (55, 110, 220)
MATERIALS = ("soft", "stiff")
FRAMES = (1, 2, 3)                   # 1/240, 1/120, 1/80 s
DISTANCES_MM = (20, 50, 100, 150, 200)


def child(mat: str, n_sub: int, out: str) -> None:
    import mpm_energy as ME
    mu, lam, rho = LS.material_corners()[mat]
    inst = ME.Instrument(mu=mu, lam=lam, rho=rho, n_substeps=n_sub,
                         frame_dt=LS.FRAME_DT, gravity=0.0, quiet=True)
    cfg = LS.CONFIGS["clean"]
    inst.init_slab(centre=cfg["centre"])
    snap = LS._snapshot(inst)
    base = snap["x"].astype(np.float64)
    pid = LS._probe_sites(base, inst.dx)["centre"]
    dist = np.linalg.norm(base - base[pid], axis=1)
    patch = dist < LS.PATCH_RADIUS_CELLS * inst.dx

    def traj(dv):
        return LS._trajectory(inst, snap, cfg, (patch, 0, dv) if dv else None)

    A, A2 = traj(0.0), traj(0.0)
    B, Bp, B2 = traj(LS.DV), traj(LS.DV), traj(2 * LS.DV)
    rows = []
    for f in FRAMES:
        k = LS.SNAPSHOT_FRAMES.index(f)
        cell = LS._measure(dist, patch, A, A2, B, Bp, B2, "v", k, inst.substep_dt)
        dvel = B["v"][k] - A["v"][k]
        prof = GD.influence_profile(dist, np.linalg.norm(dvel, axis=1), LS.BIN_WIDTH)
        r10 = GD.influence_radius(prof, ref=cell["ref"], frac=0.10, noise_floor=cell["floor"])
        cen = np.asarray(cell["profile_centres"])
        pm = np.asarray(cell["profile_max"])
        rows.append({
            "frames": f, "t": f * LS.FRAME_DT,
            "r1": cell["radii"]["0.01"], "r10": {"radius": r10["radius"], "status": r10["status"]},
            "com_over_ref": float(np.linalg.norm(dvel.mean(axis=0)) / cell["ref"]),
            "resp_over_ref_at_mm": {str(d): float(pm[np.argmin(np.abs(cen - d / 1e3))])
                                    for d in DISTANCES_MM},
            "linearity_bands": [b["median"] for b in cell["linearity_bands"]],
            "floor_over_ref": cell["floor"] / cell["ref"],
        })
    json.dump({"material": mat, "n_substeps": n_sub, "substep_dt": inst.substep_dt,
               "c_p": float(DMP.p_wave_speed(mu, lam, rho)), "patch": int(patch.sum()),
               "rows": rows}, open(out, "w"), indent=2)


def _fmt(r):
    s = r["status"]
    if s == GD.RESOLVED:
        return f"{r['radius']*1e3:.0f}"
    if s == GD.SATURATED:
        return f">={r['radius']*1e3:.0f}"
    return "n/r" if s == GD.BELOW_NOISE else "none"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "locality", "attribution"))
    ap.add_argument("--child", nargs=3, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        child(args.child[0], int(args.child[1]), args.child[2])
        return 0

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    lines = ["| material | substeps/frame | substep | t | c_p*t | r_10% | r_1% | CoM/ref | "
             + " | ".join(f"resp/ref @{d} mm" for d in DISTANCES_MM) + " |",
             "|" + "---|" * (8 + len(DISTANCES_MM))]
    for mat in MATERIALS:
        for n in SUBSTEPS:
            out = os.path.join(args.out, f"{mat}_{n}.json")
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", mat, str(n), out],
                               capture_output=True, text=True)
            if p.returncode != 0:
                print(f"{mat} {n}: child failed\n{p.stderr[-1500:]}")
                return 1
            r = json.load(open(out))
            for row in r["rows"]:
                lines.append(
                    f"| {mat} | {n} | {r['substep_dt']*1e6:.1f} us | {row['t']*1e3:.1f} ms | "
                    f"{r['c_p']*row['t']*1e3:.0f} mm | {_fmt(row['r10'])} | {_fmt(row['r1'])} | "
                    f"{row['com_over_ref']:.3f} | "
                    + " | ".join(f"{row['resp_over_ref_at_mm'][str(d)]:.1e}" for d in DISTANCES_MM) + " |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "table.md"), "w") as fh:
        fh.write("# Locality attribution (DECISION_LOG.md section 13.2)\n\n" + table + "\n")
    print(table)
    print(f"\nWall time: {time.time() - t0:.0f}s; table at {os.path.join(args.out, 'table.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
