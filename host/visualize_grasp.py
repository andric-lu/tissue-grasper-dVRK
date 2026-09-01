#!/usr/bin/env python3
"""
visualize_grasp.py -- render a --task grasp episode with the PSM's ACTUAL
mesh geometry (psm_Si_model/psm_si_surrol.urdf), not a single point marker.

    conda activate tissue-host
    python host/visualize_grasp.py data_mpm_grasp/mpm_0000.npz --save out.gif

WHY A SEPARATE SCRIPT FROM visualize_trajectory.py. That script renders
ee_pose as one square marker because it predates the PSM -- every episode
before DECISION_LOG.md section 10 was ee_pose-only, no robot to draw. Reusing
its 3D panel wholesale would mean bolting mesh rendering onto a function that
also has to keep working for robot-less episodes; this script exists for one
purpose -- render --task grasp episodes with real tool geometry -- and
imports diagnose()/report() from visualize_trajectory.py rather than
duplicating them.

WHAT "REAL GEOMETRY" MEANS HERE. Only the wrist+jaw assembly downstream of
tool_main_link is rendered (RENDERED_LINKS below) -- fully determined by
joint_pos's 7 recorded values plus the fixed j2=j3=0 pin and the
jaw_joint_2=-jaw_joint_1 mimic (host/psm.py never drives anything else).
link_0 through link_4 (the arm base) sit ~0.5 m from the tissue at
psm.DEFAULT_BASE_POSITION and are never near it; including them would force
the view to zoom out until the tissue is a speck. If that base placement
ever changes, this assumption should be revisited.

Mesh vertices are loaded once via trimesh (already a host/environment.yml
dependency), transformed by each link's <visual><origin> ONCE -- parsed
directly from the URDF XML, not hand-transcribed -- and then by that link's
live FK transform every frame, read from a throwaway PyBullet DIRECT
connection. Not host/psm.py's PSM class: pure FK/rendering needs no
Taichi/MPM.mpm3d at all, and importing MPM.mpm3d is a ~10s kernel compile for
nothing this script uses.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "host"))

from trajectory_io import load_trajectory              # noqa: E402
from visualize_trajectory import diagnose, report       # noqa: E402
import psm                                               # noqa: E402

# The tool's business end -- everything that can plausibly reach the tissue.
# link_0..link_4 (the arm base and decorative parallelogram linkage) are
# excluded; see the module docstring.
RENDERED_LINKS = ("tool_main_link", "tool_roll_link", "tool_pitch_shaft_link",
                  "tool_pitch_link", "tool_yaw_link",
                  "tool_gripper_link_1", "tool_gripper_link_2")

TOOL_COLOR = "#b0883a"


def _load_link_meshes(urdf_path: str, link_names) -> dict:
    """{link_name: (verts_in_link_frame (V,3) float64, faces (F,3) int64)}.

    Thin wrapper over psm.mesh_vertices_in_link_frame() (element="visual" --
    this is presentation-only, unlike host/psm.py's own use of that helper
    for the physically-consequential jaw collision proxies, which reads
    <collision> instead)."""
    out = {name: psm.mesh_vertices_in_link_frame(urdf_path, name, element="visual")
           for name in link_names}
    missing = set(link_names) - out.keys()
    if missing:
        raise ValueError(f"URDF is missing link(s): {sorted(missing)}")
    return out


def _reconstruct_joint_state(body, joint_index, joint_pos_row) -> None:
    """joint_pos_row is (7,) in psm.DRIVEN_JOINTS order (what's recorded).
    Reproduces the FULL joint state that produced it: DRIVEN_JOINTS from the
    recording, j2/j3 pinned to 0 (host/psm.py never drives them -- they
    aren't ancestors of the tool), jaw_joint_2 = -jaw_joint_1 (the one
    <mimic> with physical consequence, enforced by hand since PyBullet has
    no native <mimic> support)."""
    for i, name in enumerate(psm.DRIVEN_JOINTS):
        p.resetJointState(body, joint_index[name], float(joint_pos_row[i]))
    for name in psm.INERT_JOINTS:
        p.resetJointState(body, joint_index[name], 0.0)
    jaw1 = float(joint_pos_row[psm.DRIVEN_JOINTS.index("jaw_joint_1")])
    p.resetJointState(body, joint_index[psm.MIMIC_JAW_JOINT], -jaw1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a --task grasp .npz file (needs joint_pos)")
    ap.add_argument("--save", metavar="OUT.gif", help="write to file instead of showing")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1,
                    help="draw every Nth frame -- raise it for long episodes")
    args = ap.parse_args()

    traj = load_trajectory(args.path)
    if traj.joint_pos.size == 0:
        raise SystemExit(
            f"{args.path} has no joint_pos -- this is a passive (--task settle) "
            "episode with no robot to render. Use visualize_trajectory.py for "
            "those, or collect one with --task grasp.")
    d = diagnose(traj)
    report(traj, d)

    print("loading meshes ...")
    meshes = _load_link_meshes(psm.URDF_PATH, RENDERED_LINKS)

    client = p.connect(p.DIRECT)
    body = p.loadURDF(psm.URDF_PATH,
                      basePosition=list(psm.DEFAULT_BASE_POSITION),
                      baseOrientation=list(psm.DEFAULT_BASE_ORIENTATION),
                      useFixedBase=True)
    joint_index = {p.getJointInfo(body, i)[1].decode(): i
                  for i in range(p.getNumJoints(body))}
    link_index = {p.getJointInfo(body, i)[12].decode(): i
                 for i in range(p.getNumJoints(body))}

    def world_verts_at(t):
        _reconstruct_joint_state(body, joint_index, traj.joint_pos[t])
        out = {}
        for name, (verts_link, faces) in meshes.items():
            pos_, quat_ = p.getLinkState(body, link_index[name],
                                         computeForwardKinematics=1)[4:6]
            R = psm.rotmat_from_quat(np.array(quat_))
            out[name] = ((R @ verts_link.T).T + np.array(pos_), faces)
        return out

    # Precompute world vertices for every recorded frame -- a few thousand
    # triangles x T frames, no Taichi/GPU involved -- so axis limits can be
    # computed over the tool's full swept volume, not just the tissue.
    print("computing tool geometry for every frame ...")
    all_frames = [world_verts_at(t) for t in range(len(traj))]
    p.disconnect(client)

    pos = traj.tissue_pos
    tool_pts = np.concatenate(
        [wv for frame in all_frames for wv, _ in frame.values()], axis=0)
    all_pts = np.concatenate([pos.reshape(-1, 3), tool_pts], axis=0)

    t = np.arange(len(traj)) * float(traj.dt)
    frames = range(0, len(traj), args.stride)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.2)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_d = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1], sharex=ax_d)
    ax_z = fig.add_subplot(gs[2, 1], sharex=ax_d)

    # --- 3D panel -----------------------------------------------------------
    # Axis limits over BOTH tissue and the tool's full swept volume, computed
    # once over the whole episode -- see visualize_trajectory.py's identical
    # reasoning for the tissue-only case.
    lo, hi = all_pts.min(0), all_pts.max(0)
    ctr, span = (lo + hi) / 2, max((hi - lo).max(), 0.05) / 2 * 1.15
    ax3d.set_xlim(ctr[0] - span, ctr[0] + span)
    ax3d.set_ylim(ctr[1] - span, ctr[1] + span)
    ax3d.set_zlim(max(0, ctr[2] - span), ctr[2] + span)
    ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
    ax3d.view_init(elev=22, azim=-60)

    free = ax3d.scatter([], [], [], s=18, c="#4a90d9", depthshade=True, label="tissue")
    held = ax3d.scatter([], [], [], s=42, c="#d94a4a", depthshade=False, label="grasped")
    tool_collections = {}
    for name in RENDERED_LINKS:
        # facecolorS (plural): shade=True's constructor-time validation
        # checks this exact kwarg name before alias resolution, so the
        # singular "facecolor" (which PolyCollection normally accepts) fails
        # its own pre-check here.
        pc = Poly3DCollection([np.zeros((3, 3))], facecolors=[TOOL_COLOR],
                              edgecolors=["none"], alpha=0.9, shade=True)
        ax3d.add_collection3d(pc)
        tool_collections[name] = pc
    ax3d.scatter([], [], [], s=1, c=TOOL_COLOR, label="PSM (real geometry)")  # legend swatch only
    ax3d.legend(loc="upper left", fontsize=8)

    # --- diagnostic panels, identical to visualize_trajectory.py -----------
    for ax, y, label, colour in (
            (ax_d, d["max_disp"] * 1000, "max displacement (mm)", "#4a90d9"),
            (ax_s, d["max_speed"], "max node speed (m/s)", "#d98d4a"),
            (ax_z, d["ee_z"] * 1000, "end-effector height (mm)", "#2b2b2b")):
        ax.plot(t, y, c=colour, lw=1.4)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
        g = d["grasp"].astype(int)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], g, [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color="#d94a4a", alpha=0.10)
    ax_z.set_xlabel("time (s)", fontsize=9)
    cursors = [ax.axvline(0, c="k", lw=0.9, alpha=0.6) for ax in (ax_d, ax_s, ax_z)]

    fig.suptitle(f"{os.path.basename(args.path)}  --  {traj.simulator} / {traj.task} "
                f"(real PSM geometry)", fontsize=11)

    def update(i):
        ids = traj.grasp_ids(i)
        mask = np.zeros(traj.n_nodes, bool)
        if ids.size:
            mask[ids] = True
        f, h = pos[i][~mask], pos[i][mask]
        free._offsets3d = (f[:, 0], f[:, 1], f[:, 2])
        held._offsets3d = (h[:, 0], h[:, 1], h[:, 2])
        for name, (wv, faces) in all_frames[i].items():
            tool_collections[name].set_verts(wv[faces])
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        ax3d.set_title(f"t = {t[i]:.2f} s   step {i}/{len(traj)-1}"
                       f"{'   GRASPED' if d['grasp'][i] else ''}", fontsize=10)
        return free, held, *tool_collections.values(), *cursors

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
