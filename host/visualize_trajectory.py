#!/usr/bin/env python3
"""
visualize_trajectory.py -- look at what was actually recorded.

Run on the HOST:

    conda activate tissue-host
    python host/visualize_trajectory.py data/retraction_0000.npz
    python host/visualize_trajectory.py data/retraction_0000.npz --save out.mp4

WHY THIS AND NOT JUST THE SIMULATOR VIDEO
-----------------------------------------
`collect_retraction.py --video` renders the simulator. This renders the .npz
file. They are not the same thing, and the difference is the point.

If the sim looks right but your model cannot learn, the bug is between the two:
wrong node ordering, a units error, velocities finite-differenced across the
wrong interval, the render mesh logged instead of the simulation mesh. A
simulator video cannot show you any of that, because it never reads the file.
This script draws exactly the numbers your model will be trained on. If the
motion looks wrong here, the dataset is wrong -- regardless of how good the
simulator video looked.

WHAT YOU GET
------------
Left:  3D animation of the tissue nodes and the end-effector. Grasped nodes are
       highlighted, so you can see whether the grasp caught the patch you meant.
Right: three diagnostics against time, with the grasp window shaded and a
       cursor tracking the animation.
         - max node displacement from the initial state (the retraction profile)
         - max node speed (the instability detector -- see below)
         - end-effector height

READING THE SPEED PLOT
----------------------
This is the plot that earns its keep. Mass-spring systems go unstable when the
timestep is too large for the stiffness, and the failure is not subtle: node
speeds blow up over a few frames and the mesh scatters. A run that "completed
successfully" can still be numerical garbage. Physiologically, retraction speeds
are on the order of centimetres per second; anything above ~1 m/s here means the
solver diverged and the episode should be discarded, not trained on.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from trajectory_io import list_trajectories, load_trajectory  # noqa: E402

UNSTABLE_SPEED = 1.0  # m/s -- above this, treat the episode as diverged


def diagnose(traj) -> dict:
    """Numbers worth knowing before you look at any picture."""
    pos = traj.tissue_pos.astype(np.float64)
    disp = np.linalg.norm(pos - pos[0], axis=2)       # (T,N) distance from start
    speed = np.linalg.norm(traj.tissue_vel.astype(np.float64), axis=2)
    grasp = traj.grasp_active.astype(bool)

    peak_step = int(np.argmax(speed.max(axis=1)))
    # Release = the step where the grasp flag goes True -> False. Genuine
    # elastic recoil lives there; fast motion anywhere else is suspicious.
    rel = np.flatnonzero(np.diff(grasp.astype(int)) < 0)
    release_step = int(rel[0]) + 1 if rel.size else -1

    # Gripper speed, so tissue speed can be judged against something. Tissue
    # moving far faster than the thing pulling it has to be explained.
    ee = traj.ee_pose[:, :3].astype(np.float64)
    ee_speed = np.linalg.norm(np.diff(ee, axis=0), axis=1) / float(traj.dt) \
        if len(traj) > 1 else np.zeros(1)

    return {
        "max_disp": disp.max(axis=1),                 # (T,)
        "max_speed": speed.max(axis=1),               # (T,)
        "ee_z": traj.ee_pose[:, 2].astype(np.float64),
        "grasp": grasp,
        "peak_disp_mm": disp.max() * 1000,
        "peak_speed": speed.max(),
        "peak_step": peak_step,
        "release_step": release_step,
        "near_release": release_step >= 0 and abs(peak_step - release_step) <= 3,
        "ee_peak_speed": float(ee_speed.max()),
        "n_grasped": int(traj.grasp_ids(int(np.argmax(grasp))).size)
        if grasp.any() else 0,
    }


def report(traj, d: dict) -> None:
    print(traj)
    print(f"  notes         : {traj.notes}")
    print(f"  peak displacement : {d['peak_disp_mm']:.1f} mm")
    print(f"  peak node speed   : {d['peak_speed']:.3f} m/s "
          f"at step {d['peak_step']}/{len(traj)-1}")
    print(f"  gripper max speed : {d['ee_peak_speed']:.3f} m/s "
          f"({d['peak_speed']/max(d['ee_peak_speed'],1e-9):.0f}x slower than tissue)"
          if d["ee_peak_speed"] > 0 else "")
    print(f"  release at step   : {d['release_step']}")
    print(f"  grasped nodes     : {d['n_grasped']}")
    print(f"  grasp active for  : {d['grasp'].sum()}/{len(traj)} steps")

    # Loud, specific warnings beat a pretty plot you have to interpret.
    if d["peak_speed"] > UNSTABLE_SPEED or d["peak_speed"] > 10 * d["ee_peak_speed"]:
        if d["near_release"]:
            print(f"\n  NOTE: peak speed {d['peak_speed']:.2f} m/s occurs AT THE "
                  "RELEASE (step "
                  f"{d['release_step']}).\n"
                  "  That is where elastic recoil lives, so this may well be real\n"
                  "  physics rather than instability. The two look identical in a\n"
                  "  single run -- confirm with a convergence study:\n"
                  "    docker compose run --rm surrol python container/timestep_study.py\n"
                  "  Recoil converges as the timestep shrinks; divergence does not.")
        else:
            print(f"\n  WARNING: peak speed {d['peak_speed']:.2f} m/s occurs at step "
                  f"{d['peak_step']}, away from the release.\n"
                  "  Fast motion that is not recoil is a divergence signature. Run\n"
                  "    docker compose run --rm surrol python container/timestep_study.py\n"
                  "  to find a timestep where the answer stops changing. Do not\n"
                  "  train on this episode until it does.")
    if d["n_grasped"] == 0 and d["grasp"].any():
        print("\n  WARNING: grasp phase ran but caught 0 nodes -- the gripper "
              "never reached the sheet.\n  Increase GRASP_RADIUS or lower "
              "grasp_z in collect_retraction.py.")
    if d["peak_disp_mm"] < 1.0:
        print("\n  WARNING: peak displacement under 1 mm. The tissue barely "
              "moved, so this\n  episode carries almost no dynamics information.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="an .npz file (default: first in data/)")
    ap.add_argument("--save", metavar="OUT.mp4", help="write to file instead of showing")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1,
                    help="draw every Nth frame -- raise it for long episodes")
    args = ap.parse_args()

    path = args.path or (list_trajectories("data") or [None])[0]
    if path is None:
        raise SystemExit("No trajectory given and none found in data/. Collect some:\n"
                         "  docker compose run --rm surrol python "
                         "container/collect_retraction.py --episodes 3")

    traj = load_trajectory(path)
    d = diagnose(traj)
    report(traj, d)

    pos = traj.tissue_pos
    ee = traj.ee_pose[:, :3]
    frames = range(0, len(traj), args.stride)
    t = np.arange(len(traj)) * float(traj.dt)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.2)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_d = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1], sharex=ax_d)
    ax_z = fig.add_subplot(gs[2, 1], sharex=ax_d)

    # --- 3D panel ---------------------------------------------------------
    # Equal axis limits, computed once over the whole episode. Autoscaling per
    # frame would make a stationary sheet appear to writhe, and would hide the
    # magnitude of the actual deformation.
    lo, hi = pos.reshape(-1, 3).min(0), pos.reshape(-1, 3).max(0)
    ctr, span = (lo + hi) / 2, max((hi - lo).max(), 0.05) / 2 * 1.15
    ax3d.set_xlim(ctr[0] - span, ctr[0] + span)
    ax3d.set_ylim(ctr[1] - span, ctr[1] + span)
    ax3d.set_zlim(max(0, ctr[2] - span), ctr[2] + span)
    ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
    ax3d.view_init(elev=22, azim=-60)

    free = ax3d.scatter([], [], [], s=18, c="#4a90d9", depthshade=True, label="tissue")
    held = ax3d.scatter([], [], [], s=42, c="#d94a4a", depthshade=False, label="grasped")
    tool = ax3d.scatter([], [], [], s=140, c="#2b2b2b", marker="s", label="end-effector")
    trail, = ax3d.plot([], [], [], lw=1.2, c="#2b2b2b", alpha=0.45)
    ax3d.legend(loc="upper left", fontsize=8)

    # --- diagnostic panels ------------------------------------------------
    for ax, y, label, colour in (
            (ax_d, d["max_disp"] * 1000, "max displacement (mm)", "#4a90d9"),
            (ax_s, d["max_speed"], "max node speed (m/s)", "#d98d4a"),
            (ax_z, d["ee_z"] * 1000, "end-effector height (mm)", "#2b2b2b")):
        ax.plot(t, y, c=colour, lw=1.4)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
        # Shade every contiguous grasp window.
        g = d["grasp"].astype(int)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], g, [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color="#d94a4a", alpha=0.10)

    if d["peak_speed"] > UNSTABLE_SPEED:
        ax_s.axhline(UNSTABLE_SPEED, ls="--", lw=1, c="#c00")
        ax_s.text(0.98, 0.9, "unstable", ha="right", va="top", color="#c00",
                  fontsize=9, transform=ax_s.transAxes)
    ax_z.set_xlabel("time (s)", fontsize=9)
    cursors = [ax.axvline(0, c="k", lw=0.9, alpha=0.6) for ax in (ax_d, ax_s, ax_z)]

    fig.suptitle(f"{os.path.basename(path)}  --  {traj.simulator} / {traj.task}",
                 fontsize=11)

    def update(i):
        ids = traj.grasp_ids(i)
        mask = np.zeros(traj.n_nodes, bool)
        if ids.size:
            mask[ids] = True
        f, h = pos[i][~mask], pos[i][mask]
        free._offsets3d = (f[:, 0], f[:, 1], f[:, 2])
        held._offsets3d = (h[:, 0], h[:, 1], h[:, 2])
        tool._offsets3d = ([ee[i, 0]], [ee[i, 1]], [ee[i, 2]])
        trail.set_data(ee[:i + 1, 0], ee[:i + 1, 1])
        trail.set_3d_properties(ee[:i + 1, 2])
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        ax3d.set_title(f"t = {t[i]:.2f} s   step {i}/{len(traj)-1}"
                       f"{'   GRASPED' if d['grasp'][i] else ''}", fontsize=10)
        return free, held, tool, trail, *cursors

    anim = FuncAnimation(fig, update, frames=frames,
                         interval=1000 / args.fps, blit=False, repeat=True)

    if args.save:
        print(f"\nwriting {args.save} ...")
        anim.save(args.save, fps=args.fps, dpi=110)
        print("done")
    else:
        plt.show()


if __name__ == "__main__":
    main()
