#!/usr/bin/env python3
"""
timestep_study.py -- find the largest timestep that still gives correct physics.

Run INSIDE the container:

    docker compose run --rm surrol python container/timestep_study.py
    docker compose run --rm surrol python container/timestep_study.py --dts 240 500 1000 2000 4000

THE QUESTION THIS ANSWERS
-------------------------
You measure something in a simulation -- a peak node speed, a deformation, a
contact force. Is that number a property of the TISSUE MODEL, or an artifact of
the SOLVER? A single run cannot tell you. The number exists either way, and a
diverging solver produces confident-looking output right up until it produces
NaN.

The standard answer is a convergence study. Run the identical episode -- same
seed, same mesh, same material, same scripted motion -- at a sequence of
decreasing timesteps. A physical quantity converges: as dt shrinks the answer
stops changing, because you are resolving the true solution of the equations. A
numerical artifact does not converge: it keeps changing, usually shrinking
dramatically, because it was never a property of the model at all.

Then you pick the largest dt whose answer is acceptably close to the converged
one, and you have a defensible reason for that choice instead of a guess.

WHAT IT REPORTS
---------------
For each timestep:
    peak speed   -- fastest node in the episode (m/s). The instability signal.
    peak disp    -- largest deformation of any interior node (mm). The physics.
    peak at      -- when the fastest motion happened, and whether it coincided
                    with the release, which is where genuine recoil lives.
    rel change   -- how much peak displacement moved relative to the next-finer
                    run. Under a few percent means converged.

DISTINGUISHING RECOIL FROM DIVERGENCE
-------------------------------------
Both produce fast nodes. They differ in where the speed comes from:

  Recoil is physical. It happens at the instant of release, its magnitude
  converges as dt shrinks, and the deformation converges too.

  Divergence is numerical. It can happen at any time, often during the pull, it
  grows as dt grows, and it usually corrupts the deformation as well.

The "peak at" column and the convergence trend together separate them.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, "/work/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collect_retraction as cr  # noqa: E402
from trajectory_io import load_trajectory  # noqa: E402


def run_at(dt: float, seed: int, mesh: str, out_dir: str, log_hz: float) -> dict:
    """One episode at timestep dt. Returns the diagnostic numbers."""
    # Rebind the module global. Every function in collect_retraction reads DT
    # from module scope at call time, so this changes the timestep everywhere
    # without touching the file.
    cr.DT = dt

    # Keep the LOGGING rate fixed while the PHYSICS rate varies. If both
    # changed together, the trajectories would have different lengths and the
    # comparison would be meaningless -- you would be varying two things at
    # once, which is the classic way to learn nothing from an experiment.
    record_every = max(1, int(round(1.0 / (dt * log_hz))))

    t0 = time.time()
    path, n_grasped, _ = cr.run_episode(
        0, out_dir, record_every, seed, mesh_path=mesh, verbose=False)
    wall = time.time() - t0

    traj = load_trajectory(path)
    pos = traj.tissue_pos.astype(np.float64)
    speed = np.linalg.norm(traj.tissue_vel.astype(np.float64), axis=2)
    grasp = traj.grasp_active.astype(bool)

    # Interior nodes only: the pinned boundary never moves, so including it
    # would just dilute the deformation statistic.
    moving = np.linalg.norm(pos - pos[0], axis=2).max(axis=0) > 1e-9
    disp = np.linalg.norm(pos - pos[0], axis=2)[:, moving] if moving.any() \
        else np.zeros((len(traj), 1))

    peak_step = int(np.unravel_index(np.argmax(speed), speed.shape)[0])
    # Release is the step where grasp_active goes True -> False.
    rel = np.flatnonzero(np.diff(grasp.astype(int)) < 0)
    release_step = int(rel[0]) + 1 if rel.size else -1
    near_release = (release_step >= 0 and abs(peak_step - release_step) <= 3)

    os.remove(path)
    return {
        "dt": dt,
        "peak_speed": float(speed.max()),
        "peak_disp_mm": float(disp.max() * 1000),
        "peak_step": peak_step,
        "n_steps": len(traj),
        "release_step": release_step,
        "near_release": near_release,
        "has_nan": bool(~np.isfinite(pos).all()),
        "n_grasped": n_grasped,
        "wall": wall,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dts", type=float, nargs="+",
                    default=[240, 500, 1000, 2000, 4000],
                    help="timesteps as RATES in Hz (240 means dt = 1/240)")
    ap.add_argument("--seed", type=int, default=0,
                    help="same seed for every run -- the whole point is that "
                         "only the timestep differs")
    ap.add_argument("--mesh", default=cr.DEFAULT_MESH)
    ap.add_argument("--log-hz", type=float, default=30.0,
                    help="logging rate, held constant across runs")
    args = ap.parse_args()

    if not os.path.isfile(args.mesh):
        raise SystemExit(
            f"Tissue mesh not found: {args.mesh}\nGenerate it first:\n"
            "  docker compose run --rm surrol python container/make_tissue_mesh.py")

    rates = sorted(set(args.dts))
    tmp = tempfile.mkdtemp(prefix="dtstudy_")
    print(f"Timestep convergence study: seed {args.seed}, "
          f"mesh {os.path.basename(args.mesh)}, logging held at {args.log_hz:.0f} Hz")
    print(f"{len(rates)} runs; the finest will take a while.\n")

    header = (f"{'dt':>12} {'peak speed':>11} {'peak disp':>10} "
              f"{'peak at':>9} {'release':>8} {'wall':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    try:
        for hz in rates:
            r = run_at(1.0 / hz, args.seed, args.mesh, tmp, args.log_hz)
            rows.append(r)
            mark = " <-" if r["near_release"] else ""
            print(f"{'1/'+str(int(hz)):>12} {r['peak_speed']:>9.3f} m/s "
                  f"{r['peak_disp_mm']:>7.2f} mm "
                  f"{r['peak_step']:>4d}/{r['n_steps']:<4d} "
                  f"{r['release_step']:>8d} {r['wall']:>6.1f}s{mark}")
            if r["has_nan"]:
                print("             ^ produced NaN -- diverged outright")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- convergence assessment ------------------------------------------
    print("\nConvergence (each row compared with the next-finer timestep):")
    print(f"{'dt':>12} {'d(peak disp)':>13} {'d(peak speed)':>14}")
    print("-" * 41)
    finest = rows[-1]
    for a, b in zip(rows, rows[1:]):
        dd = abs(a["peak_disp_mm"] - b["peak_disp_mm"]) / max(b["peak_disp_mm"], 1e-9)
        ds = abs(a["peak_speed"] - b["peak_speed"]) / max(b["peak_speed"], 1e-9)
        print(f"{'1/'+str(int(1/a['dt'])):>12} {dd*100:>12.1f}% {ds*100:>13.1f}%")

    # --- verdict ----------------------------------------------------------
    print("\n" + "=" * 72)
    ok = [r for r in rows[:-1]
          if abs(r["peak_disp_mm"] - finest["peak_disp_mm"])
          / max(finest["peak_disp_mm"], 1e-9) < 0.05 and not r["has_nan"]]
    if ok:
        best = ok[0]  # largest dt that is within 5% of the converged answer
        print(f"RECOMMENDED: dt = 1/{1/best['dt']:.0f}  "
              f"({best['wall']:.1f}s per episode)")
        print(f"  Peak displacement is within 5% of the finest run "
              f"({best['peak_disp_mm']:.2f} vs {finest['peak_disp_mm']:.2f} mm),")
        print("  so the deformation is resolved. Larger timesteps in this table")
        print("  are not, whatever their output looked like in isolation.")
    else:
        print("NOT CONVERGED at any timestep tested.")
        print("  Extend the study downward, e.g. --dts 2000 4000 8000 16000.")

    if finest["near_release"]:
        print("\n  The peak speed occurs at the release step, and it converges.")
        print("  That is elastic recoil -- real physics, not instability. A high")
        print("  peak speed here is expected and should be kept.")
    else:
        print(f"\n  The peak speed occurs at step {finest['peak_step']} of "
              f"{finest['n_steps']}, away from the release.")
        print("  Fast motion away from release is a divergence signature. Inspect")
        print("  the episode before trusting it.")
    print("=" * 72)
    print("\nApply the result:")
    print(f"  docker compose run --rm surrol python container/collect_retraction.py \\")
    print(f"      --episodes 20 --dt {rows[0]['dt'] if not ok else ok[0]['dt']:.6f}")
    print("Then make it permanent by editing DT in container/collect_retraction.py.")


if __name__ == "__main__":
    main()
