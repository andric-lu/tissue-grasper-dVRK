#!/usr/bin/env python3
"""
render_video.py -- a video of a whole episode: the graph over time.

    conda activate tissue-sim
    python host/render_video.py data_pilot/grasp_classic/mpm_0000.npz \\
        --ids configs/particle_ids_3000.npy --out runs/video/grasp_classic.mp4

Left panel: 3D view. Right panel: x-z slice through the jaws.
Nodes are coloured by displacement from rest. Reference edges blue,
spatial-only edges orange, tool edges red, jaw boxes black.

Reuses visualize_graph.py for node selection and src/graph_builder.py for the
edges, so the video shows the same graph that script checks. No statistics are
computed here (that is what made visualize_graph.py slow), so it is mostly the
cost of drawing: roughly 0.3-1 s per frame for the 3D+slice layout.

Needs matplotlib and ffmpeg. If ffmpeg is missing it tries imageio-ffmpeg, then
falls back to a GIF.   brew install ffmpeg    or    pip install imageio-ffmpeg
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import visualize_graph as vg                              # noqa: E402  (also puts src/ on sys.path)

gb, gd, cc = vg.gb, vg.gd, vg.cc


def pick_writer(path, fps):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import animation
    if path.lower().endswith((".mp4", ".mov")):
        ff = shutil.which("ffmpeg")
        if ff is None:
            try:
                import imageio_ffmpeg
                ff = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ff = None
        if ff:
            matplotlib.rcParams["animation.ffmpeg_path"] = ff
            return animation.FFMpegWriter(
                fps=fps, codec="libx264",
                extra_args=["-pix_fmt", "yuv420p", "-crf", "20",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]), path
        path = os.path.splitext(path)[0] + ".gif"
        print("no ffmpeg found (brew install ffmpeg, or pip install imageio-ffmpeg): "
              f"writing a GIF instead -> {path}")
    return animation.PillowWriter(fps=fps), path


def box_segments(tr, t):
    """(12*K, 2, 3) line segments of every jaw box at row t."""
    segs = []
    for k in range(tr.jaw_names.size):
        c = vg._box_corners(tr.jaw_box_pose[t, k], tr.jaw_half_extents[k])
        segs += [np.stack([c[a], c[b]]) for a, b in vg._BOX_EDGES]
    return np.array(segs)


def global_limits(view, stride):
    """Fixed axis limits over the whole episode so the camera never jumps, and the
    largest node displacement for the colour scale."""
    tr = view.tr
    P = tr.tissue_pos[:, view.cols].astype(np.float64)
    lo, hi = P.min(axis=(0, 1)), P.max(axis=(0, 1))
    dmax = float(np.linalg.norm(P - view.X[None], axis=2).max())
    if tr.has_jaws:
        allc = np.concatenate([box_segments(tr, t).reshape(-1, 3)
                               for t in range(0, len(tr), stride)])
        lo, hi = np.minimum(lo, allc.min(axis=0)), np.maximum(hi, allc.max(axis=0))
    pad = 0.04 * (hi - lo) + 1e-6
    return lo - pad, hi + pad, max(dmax, 1e-9)


class Renderer:
    def __init__(self, view, ref, r_sp, r_c, max_edges, slice_only, dpi, stride):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.view, self.ref, self.r_sp, self.r_c = view, ref, r_sp, r_c
        self.max_edges = max_edges
        self.n = view.n
        self.ref_keys = ref[0].astype(np.int64) * self.n + ref[1]
        self.ref_draw = self._sub(ref, np.random.default_rng(0))   # fixed -> no flicker
        self.lo, self.hi, self.dmax = global_limits(view, stride)
        self.h = float(gd.median_nn_distance(view.X))
        if slice_only:
            self.fig = plt.figure(figsize=(7.2, 5.4), dpi=dpi)
            self.ax3 = None
            self.ax2 = self.fig.add_subplot(1, 1, 1)
        else:
            self.fig = plt.figure(figsize=(12.8, 5.4), dpi=dpi)
            self.ax3 = self.fig.add_subplot(1, 2, 1, projection="3d")
            self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.11, wspace=0.1)

    def _sub(self, e, rng):
        e = e[:, e[0] < e[1]]                         # one line per undirected pair
        if e.shape[1] > self.max_edges:
            e = e[:, rng.choice(e.shape[1], self.max_edges, replace=False)]
        return e

    def frame(self, t):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.collections import LineCollection
        view = self.view
        tr, n = view.tr, view.n
        x = view.x(t)
        sp = gb.spatial_edges(x, self.r_sp)
        sp_keys = sp[0].astype(np.int64) * n + sp[1]
        sp_only = sp[:, ~np.isin(sp_keys, self.ref_keys)]
        box = box_segments(tr, t) if tr.has_jaws else None
        te = np.zeros((2, 0), np.int64)
        if tr.has_jaws and self.r_c is not None:
            te = gb.tool_edges(x, view.v(t), tr.jaw_box_pose[t], tr.jaw_box_vel[t],
                               tr.jaw_half_extents, self.r_c)["edge_index"]
        centres = tr.jaw_box_pose[t, :, :3] if tr.has_jaws else None
        disp = np.linalg.norm(x - view.X, axis=1)
        sp_draw = self._sub(sp_only, np.random.default_rng(t))
        lo, hi = self.lo, self.hi

        if self.ax3 is not None:
            ax = self.ax3
            ax.cla()
            for e, col, lw in ((self.ref_draw, "tab:blue", 0.3), (sp_draw, "tab:orange", 0.6)):
                if e.size:
                    ax.add_collection3d(Line3DCollection(
                        np.stack([x[e[0]], x[e[1]]], 1), colors=col, linewidths=lw, alpha=0.5))
            ax.scatter(*x.T, s=1.5, c=disp, cmap="viridis", vmin=0, vmax=self.dmax)
            if box is not None:
                ax.add_collection3d(Line3DCollection(box, colors="k", linewidths=1.2))
                if te.size:
                    ax.add_collection3d(Line3DCollection(
                        np.stack([centres[te[0]], x[te[1]]], 1), colors="tab:red", linewidths=0.6))
            ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi - lo)
            ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")

        ax2 = self.ax2
        ax2.cla()
        y0 = float(tr.jaw_box_pose[t, :, 1].mean()) if tr.has_jaws else float(np.median(x[:, 1]))
        inslab = np.abs(x[:, 1] - y0) < self.h
        for e, col, lw in ((self.ref, "tab:blue", 0.4), (sp_only, "tab:orange", 0.8)):
            if e.size:
                keep = inslab[e[0]] & inslab[e[1]] & (e[0] < e[1])
                if keep.any():
                    segs = np.stack([x[e[0, keep]][:, [0, 2]], x[e[1, keep]][:, [0, 2]]], 1)
                    ax2.add_collection(LineCollection(segs, colors=col, linewidths=lw, alpha=0.7))
        if te.size:
            k = inslab[te[1]]
            if k.any():
                segs = np.stack([centres[te[0, k]][:, [0, 2]], x[te[1, k]][:, [0, 2]]], 1)
                ax2.add_collection(LineCollection(segs, colors="tab:red", linewidths=0.6))
        ax2.scatter(x[inslab, 0], x[inslab, 2], s=5, c=disp[inslab], cmap="viridis",
                    vmin=0, vmax=self.dmax)
        if box is not None:
            ax2.add_collection(LineCollection(box[:, :, [0, 2]], colors="k", linewidths=1))
        ax2.set_xlim(lo[0], hi[0]), ax2.set_ylim(lo[2], hi[2])
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("x [m]"), ax2.set_ylabel("z [m]")
        ax2.set_title(f"x-z slice, |y - {y0:.3f}| < {self.h*1e3:.1f} mm "
                      f"({int(inslab.sum())} nodes)", fontsize=9)

        phase = cc.PHASES[int(tr.phase[t])] if tr.phase.size else ""
        grasp = "   GRASP" if (tr.has_jaws and tr.grasp_active.size and tr.grasp_active[t]) else ""
        self.fig.suptitle(
            f"{os.path.basename(tr.path)}   row {t}/{len(tr) - 1}   t = {t*float(tr.dt):.2f} s   "
            f"{phase}{grasp}\n{n} nodes   reference {self.ref.shape[1]}   spatial {sp.shape[1]}   "
            f"tool {te.shape[1]}   colour = displacement from rest (0-{self.dmax*1e3:.1f} mm)",
            fontsize=10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("episode")
    ap.add_argument("--ids", default=None, help=".npy solver particle ids (full-record episodes)")
    ap.add_argument("--sampler", choices=("random", "fps", "voxel"), default=None)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--r-dx", type=float, default=1.5, help="r_ref = r_spatial = this x grid_dx")
    ap.add_argument("--r-ref", type=float, default=None, help="override, metres")
    ap.add_argument("--r-spatial", type=float, default=None, help="override, metres")
    ap.add_argument("--r-c", type=float, default=0.02, help="tool edge radius, metres")
    ap.add_argument("--stride", type=int, default=2, help="rows per video frame (1 = every row)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-edges", type=int, default=2500, help="drawn per edge type in 3D")
    ap.add_argument("--slice-only", action="store_true", help="skip the 3D panel (much faster)")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--out", default=None, help="output .mp4 (default runs/video/<episode>.mp4)")
    args = ap.parse_args(argv)

    tr = vg.load_trajectory(args.episode)
    if tr.rest_pos.size == 0:
        raise SystemExit(f"{args.episode} has no rest_pos; collect a schema-3.0 episode")
    full = tr.particle_ids.size == 0 or tr.particle_ids.size == int(tr.n_particles_simulated)
    ids = np.load(args.ids) if args.ids else None
    if full and ids is None and args.sampler is None and tr.n_nodes > 5000:
        default = os.path.join(vg.REPO, "configs", "particle_ids_3000.npy")
        if not os.path.exists(default):
            raise SystemExit(f"episode holds all {tr.n_nodes} particles: pass --ids or --sampler")
        ids = np.load(default)
        print(f"episode holds all {tr.n_nodes} particles; using {default}")
    view = vg.NodeView(tr, ids, args.sampler, args.n)

    dx = float(tr.grid_dx) or 1 / 64
    r_ref = args.r_ref or args.r_dx * dx
    r_sp = args.r_spatial or args.r_dx * dx
    r_c = args.r_c if tr.has_jaws else None
    ref = gb.reference_edges(view.X, r_ref)

    stem = os.path.splitext(os.path.basename(args.episode))[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(args.episode)))
    out = args.out or os.path.join(vg.REPO, "runs", "video", f"{parent}_{stem}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    writer, out = pick_writer(out, args.fps)

    rend = Renderer(view, ref, r_sp, r_c, args.max_edges, args.slice_only, args.dpi, args.stride)
    T = len(tr)
    frames = list(range(0, T, args.stride))
    if frames[-1] != T - 1:
        frames.append(T - 1)
    speed = args.fps * args.stride * float(tr.dt)
    print(f"{view.n} nodes, {ref.shape[1]} reference edges, {len(frames)} frames "
          f"({T} rows, stride {args.stride}); plays at {speed:.2f}x real time")

    t0 = time.time()
    with writer.saving(rend.fig, out, args.dpi):
        for i, t in enumerate(frames):
            rend.frame(t)
            writer.grab_frame()
            if i == 2 or (i + 1) % 25 == 0:
                per = (time.time() - t0) / (i + 1)
                print(f"  frame {i + 1}/{len(frames)}  {per:.2f} s/frame  "
                      f"~{per * (len(frames) - i - 1):.0f} s left")
    print(f"wrote {out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
