#!/usr/bin/env python3
"""
visualize_graph.py -- to-do Item 4: look at the graph before training on it.

    conda activate tissue-host
    python host/visualize_graph.py data_pilot/grasp/mpm_0000.npz \\
        --ids configs/particle_ids_3000.npy --r-dx 1.5 --r-c 0.02 --out runs/graph_viz/grasp
    python host/visualize_graph.py data_pilot/settle/mpm_0000.npz --r-dx 1.5 --out runs/graph_viz/settle

matplotlib + numpy (host). Statistics come from src/graph_builder.py and
src/graph_design.py on the FULL graph; only the DRAWING subsamples edges.

WHAT IT PRODUCES (Item 4's deliverable)
---------------------------------------
    rows_<phase>.png     the graph at rest, contact, grasp, strong retraction
                         and release (settle episodes: rest, peak motion, end):
                         a 3D view and an x-z slice through the grasp point.
                         Reference edges blue, spatial-only edges orange,
                         tool edges red, jaw boxes black.
    stats_over_time.csv / .png   per-row edge counts, components, isolated
                         nodes, degree, reference-edge stretch, spatial edges
                         gained/lost, tool edges
    checklist.md         Item 4's checks as PASS / WARN / FAIL with numbers
    graph.gif            (--video) the x-z slice animated over the episode

WHICH NODES
-----------
If the episode recorded every particle (a pilot with --n-record all), the
nodes are `--ids` (or a fresh `--sampler` draw from rest_pos_simulated) and
their trajectories are read out of the full record. Otherwise the episode's
own recorded nodes are used and `--ids` must be absent.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import collection_config as cc                            # noqa: E402
import graph_builder as gb                                # noqa: E402
import graph_design as gd                                 # noqa: E402
import tool_geometry as tg                                # noqa: E402
from trajectory_io import load_trajectory                 # noqa: E402

DEGREE_BLOWUP = 2.0      # max degree during the episode vs at rest -> WARN
FOLD_FACTOR = 2.0        # spatial edge between nodes > FOLD_FACTOR*r_ref apart at rest
TEAR_FACTOR = 3.0        # a reference edge stretched beyond this x its rest length
                         # is treated as BROKEN when judging fragmentation. 3x is
                         # far beyond tissue's elastic range and well below the
                         # 13x measured on the first grasp pilot's torn strand.


class NodeView:
    """The node set's trajectories, whichever way the episode stored them."""

    def __init__(self, tr, ids=None, sampler=None, n=3000, seed=0):
        full = tr.particle_ids.size == 0 or \
            tr.particle_ids.size == int(tr.n_particles_simulated)
        if not full and (ids is not None or sampler):
            raise SystemExit("this episode holds a subset; its own nodes are used "
                             "(drop --ids/--sampler)")
        if full and (ids is not None or sampler):
            if ids is None:
                import sampling_study
                ids = sampling_study.choose_nodes(sampler, tr.rest_pos_simulated.astype(
                    np.float64), n, seed)
            # Recorded column of each requested solver particle.
            order = tr.particle_ids if tr.particle_ids.size else np.arange(tr.n_nodes)
            cols = np.searchsorted(order, ids)
            if not np.array_equal(order[cols], ids):
                raise SystemExit("--ids names particles not in this episode")
            self.cols = cols
        else:
            self.cols = np.arange(tr.n_nodes)
        self.tr = tr
        self.X = tr.rest_pos[self.cols].astype(np.float64)
        self.n = self.cols.size

    def x(self, t):
        return self.tr.tissue_pos[t, self.cols].astype(np.float64)

    def v(self, t):
        return self.tr.tissue_vel[t, self.cols].astype(np.float64)


def key_rows(tr):
    """{label: row} for Item 4's phases. Grasp episodes use the recorded phase
    labels and grasp flags; settle episodes use motion."""
    T = len(tr)
    rows = {"rest": 0}
    if tr.has_jaws:
        contact = tr.node_contact.any(axis=1) if tr.node_contact.size else np.zeros(T, bool)
        grasp = tr.grasp_active.astype(bool)
        if contact.any():
            rows["contact"] = int(np.argmax(contact))
        if grasp.any():
            g0 = int(np.argmax(grasp))
            rows["grasp"] = g0
            if tr.phase.size:
                retract = np.flatnonzero(tr.phase == cc.PHASE_ID["retract"])
                if retract.size:
                    rows["strong_retraction"] = int(retract[-1])
            last = T - 1 - int(np.argmax(grasp[::-1]))
            if last < T - 1:
                rows["release"] = last + 1
        rows["end"] = T - 1
    else:
        speed = np.linalg.norm(tr.tissue_vel, axis=2).max(axis=1)
        rows["peak_motion"] = int(np.argmax(speed))
        rows["end"] = T - 1
    return rows


def per_row_stats(view, ref, r_spatial, r_ref, r_c, stride=1):
    tr = view.tr
    X = view.X
    L0 = np.linalg.norm(X[ref[1]] - X[ref[0]], axis=1)
    rest_deg_max = int(gd.degrees(gb.combined_pairs(ref, gb.spatial_edges(X, r_spatial)),
                                  view.n).max())
    out, prev = [], None
    for t in range(0, len(tr), stride):
        x = view.x(t)
        sp = gb.spatial_edges(x, r_spatial)
        comb = gb.combined_pairs(ref, sp)
        deg = gd.degrees(comb, view.n)
        L = np.linalg.norm(x[ref[1]] - x[ref[0]], axis=1)
        stretch = L / np.maximum(L0, 1e-12)
        sp_set = set(map(tuple, sp.T.tolist()))
        gained = len(sp_set - prev) if prev is not None else 0
        lost = len(prev - sp_set) if prev is not None else 0
        prev = sp_set
        rest_d = np.linalg.norm(X[sp[1]] - X[sp[0]], axis=1)
        n_tool = 0
        tool_problems = []
        if tr.has_jaws and r_c is not None:
            te = gb.tool_edges(x, view.v(t), tr.jaw_box_pose[t], tr.jaw_box_vel[t],
                               tr.jaw_half_extents, r_c)
            n_tool = te["edge_index"].shape[1]
            ei = te["edge_index"]
            if ei.size and (ei[0].max() >= tr.jaw_names.size or ei[1].max() >= view.n):
                tool_problems.append("tool edge index out of range")
        out.append({
            "row": t, "time_s": t * float(tr.dt),
            "phase": cc.PHASES[int(tr.phase[t])] if tr.phase.size else "",
            "ref_edges": ref.shape[1], "spatial_edges": sp.shape[1],
            "combined_edges": comb.shape[1], "tool_edges": n_tool,
            "spatial_components": gd.connected_components(sp, view.n)[0],
            "combined_components": gd.connected_components(comb, view.n)[0],
            # WHY: the combined graph contains the FIXED reference edges, so it
            # can never come apart -- checking its connectivity cannot detect
            # tearing. Material fragmentation is reference connectivity with the
            # torn edges removed: what is still physically joined.
            "intact_components": gd.connected_components(
                ref[:, stretch <= TEAR_FACTOR], view.n)[0] if stretch.size else view.n,
            "torn_edges": int((stretch > TEAR_FACTOR).sum()),
            "combined_isolated": int((deg == 0).sum()),
            "combined_deg_max": int(deg.max()) if deg.size else 0,
            "rest_deg_max": rest_deg_max,
            # An empty reference graph (radius below the node spacing) is a
            # finding to report, not a crash: NaN stretch, FAIL downstream.
            "ref_stretch_max": float(stretch.max()) if stretch.size else float("nan"),
            "ref_stretch_min": float(stretch.min()) if stretch.size else float("nan"),
            "spatial_gained": gained, "spatial_lost": lost,
            "fold_edges": int((rest_d > FOLD_FACTOR * r_ref).sum()),
            "index_problems": "; ".join(gb.check_edge_index(sp, view.n) + tool_problems),
        })
    return out


def checklist(view, ref, stats, r_spatial, r_ref, r_c):
    """Item 4's checks. Returns [(status, check, detail)]."""
    tr, n = view.tr, view.n
    items = []

    probs = gb.check_edge_index(ref, n)
    bad_rows = [s["row"] for s in stats if s["index_problems"]]
    items.append(("FAIL" if probs or bad_rows else "PASS", "indexing",
                  f"reference: {probs or 'ok'}; spatial/tool problems at rows "
                  f"{bad_rows[:5] or 'none'}; node ids follow particle_ids order"))

    # Independent recomputation: full distance matrix at a few rows.
    sample = [s["row"] for s in stats][:: max(1, len(stats) // 4)][:5]
    mismatch = []
    for t in sample:
        x = view.x(t)
        d = np.linalg.norm(x[:, None] - x[None], axis=-1) if n <= 4000 else None
        if d is None:
            continue
        want = {(i, j) for i, j in zip(*np.nonzero(d < r_spatial)) if i != j}
        got = set(map(tuple, gb.spatial_edges(x, r_spatial).T.tolist()))
        if want != got:
            mismatch.append(t)
    moved = float(np.linalg.norm(view.x(len(tr) - 1) - view.x(0), axis=1).max())
    changes = sum(s["spatial_gained"] + s["spatial_lost"] for s in stats)
    status = "FAIL" if mismatch else ("WARN" if changes == 0 and moved > 0.1 * r_spatial
                                      else "PASS")
    items.append((status, "spatial edges appear/disappear with proximity",
                  f"brute-force recomputation matches at rows {sample} "
                  f"(mismatch {mismatch or 'none'}); {changes} edge changes over the "
                  f"episode; peak node travel {moved*1e3:.1f} mm"))

    s_max = max(s["ref_stretch_max"] for s in stats)
    s_min = min(s["ref_stretch_min"] for s in stats)
    const = len({s["ref_edges"] for s in stats}) == 1
    items.append(("PASS" if const else "FAIL", "reference connections survive stretching",
                  f"{ref.shape[1]} reference edges at every row; their length ranges "
                  f"{s_min:.3f}-{s_max:.3f} x rest"))

    folds = max(s["fold_edges"] for s in stats)
    items.append(("WARN" if folds else "PASS", "unintended cross-object / fold edges",
                  f"one material body in this scene; max {folds} spatial edge(s) join "
                  f"nodes > {FOLD_FACTOR:g} x r_ref apart at rest (folding or "
                  "self-contact). Inspect the images if non-zero"))

    dmax = max(s["combined_deg_max"] for s in stats)
    rest = stats[0]["rest_deg_max"]
    items.append(("WARN" if dmax > DEGREE_BLOWUP * rest else "PASS",
                  "excessive degree during folding/compression",
                  f"max combined degree {dmax} vs {rest} at rest "
                  f"(limit {DEGREE_BLOWUP:g}x)"))

    n_ref_comp, _ = gd.connected_components(ref, n)
    ref_iso = int((gd.degrees(ref, n) == 0).sum())
    torn_rows = [s["row"] for s in stats if s["intact_components"] != 1]
    worst = max(stats, key=lambda s: s["intact_components"])
    items.append(("FAIL" if n_ref_comp != 1 or ref_iso or torn_rows else "PASS",
                  "no material fragmentation",
                  f"reference graph at rest: {n_ref_comp} component(s), {ref_iso} "
                  f"isolated; with edges stretched > {TEAR_FACTOR:g}x rest treated as "
                  f"torn, material splits into {worst['intact_components']} piece(s) "
                  f"({worst['torn_edges']} torn edges) at row {worst['row']}, and is "
                  f"not one piece at {len(torn_rows)} row(s), first {torn_rows[:1] or 'none'}. "
                  "(The spatial graph alone need not stay connected: "
                  f"max {max(s['spatial_components'] for s in stats)} components.)"))

    if tr.has_jaws and r_c is not None:
        missed = 0
        for s in stats:
            t = s["row"]
            if tr.node_contact.size:
                flagged = np.flatnonzero(tr.node_contact[t][view.cols]) \
                    if view.cols.size == tr.n_nodes else np.zeros(0)
                if flagged.size:
                    te = gb.tool_edges(view.x(t), view.v(t), tr.jaw_box_pose[t],
                                       tr.jaw_box_vel[t], tr.jaw_half_extents, r_c)
                    missed += np.setdiff1d(flagged, te["edge_index"][1]).size
        items.append(("WARN" if missed else "PASS", "tool edges cover flagged contact",
                      f"r_c = {r_c*1e3:.1f} mm; {missed} node-row(s) flagged near a jaw "
                      f"without a tool edge; peak {max(s['tool_edges'] for s in stats)} "
                      "tool edges"))
    return items


def draw_row(view, ref, t, r_spatial, r_c, path, max_edges=4000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.collections import LineCollection

    tr = view.tr
    x = view.x(t)
    sp = gb.spatial_edges(x, r_spatial)
    ref_set = set(map(tuple, ref.T.tolist()))
    sp_only = np.array([e for e in sp.T.tolist() if tuple(e) not in ref_set],
                       np.int64).reshape(-1, 2).T
    rng = np.random.default_rng(seed)

    def sub(e):            # drawing only: statistics use the full graph
        e = e[:, e[0] < e[1]]                       # one line per undirected pair
        if e.shape[1] > max_edges:
            e = e[:, rng.choice(e.shape[1], max_edges, replace=False)]
        return e

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for e, col, lw in ((sub(ref), "tab:blue", 0.3), (sub(sp_only), "tab:orange", 0.6)):
        if e.size:
            ax.add_collection3d(Line3DCollection(np.stack([x[e[0]], x[e[1]]], 1),
                                                 colors=col, linewidths=lw, alpha=0.5))
    ax.scatter(*x.T, s=1, c="k", alpha=0.4)
    boxes = []
    if tr.has_jaws:
        for k in range(tr.jaw_names.size):
            corners = _box_corners(tr.jaw_box_pose[t, k], tr.jaw_half_extents[k])
            boxes.append(corners)
            for a, b in _BOX_EDGES:
                ax.plot(*np.stack([corners[a], corners[b]]).T, c="k", lw=1.2)
        if r_c is not None:
            te = gb.tool_edges(x, view.v(t), tr.jaw_box_pose[t], tr.jaw_box_vel[t],
                               tr.jaw_half_extents, r_c)
            for k, i in te["edge_index"].T:
                c = tr.jaw_box_pose[t, k, :3]
                ax.plot(*np.stack([c, x[i]]).T, c="tab:red", lw=0.6)
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")
    ax.set_box_aspect(np.ptp(np.concatenate([x] + boxes), axis=0) + 1e-3)

    # x-z slice through the grasp point (or the centre), one node pitch thick.
    ax2 = fig.add_subplot(1, 2, 2)
    y0 = float(tr.jaw_box_pose[t, :, 1].mean()) if tr.has_jaws else float(np.median(x[:, 1]))
    h = gd.median_nn_distance(view.X)
    inslab = np.abs(x[:, 1] - y0) < h
    for e, col, lw in ((ref, "tab:blue", 0.4), (sp_only, "tab:orange", 0.8)):
        if e.size:
            keep = inslab[e[0]] & inslab[e[1]] & (e[0] < e[1])
            segs = np.stack([x[e[0, keep]][:, [0, 2]], x[e[1, keep]][:, [0, 2]]], 1)
            ax2.add_collection(LineCollection(segs, colors=col, linewidths=lw, alpha=0.7))
    ax2.scatter(x[inslab, 0], x[inslab, 2], s=4, c="k")
    for corners in boxes:
        for a, b in _BOX_EDGES:
            ax2.plot(*np.stack([corners[a], corners[b]])[:, [0, 2]].T, c="k", lw=1)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]"), ax2.set_ylabel("z [m]")
    ax2.set_title(f"x-z slice |y - {y0:.3f}| < {h*1e3:.1f} mm")
    phase = cc.PHASES[int(tr.phase[t])] if tr.phase.size else ""
    fig.suptitle(f"{os.path.basename(tr.path)}  row {t}  t = {t*float(tr.dt):.3f} s  "
                 f"{phase}   ref {ref.shape[1]}  spatial {sp.shape[1]} "
                 f"(blue: reference, orange: spatial-only, red: tool)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


_BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
              (4, 5), (4, 6), (5, 7), (6, 7)]


def _box_corners(pose, half):
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    R = tg.rotmat_from_quat(np.asarray(pose[3:], np.float64))
    return (signs * half) @ R.T + pose[:3]


def plot_stats(stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [s["time_s"] for s in stats]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axs[0, 0].plot(t, [s["spatial_edges"] for s in stats], label="spatial")
    axs[0, 0].plot(t, [s["ref_edges"] for s in stats], label="reference")
    axs[0, 0].set_ylabel("directed edges"), axs[0, 0].legend()
    axs[0, 1].plot(t, [s["spatial_gained"] for s in stats], label="spatial gained")
    axs[0, 1].plot(t, [s["spatial_lost"] for s in stats], label="spatial lost")
    axs[0, 1].plot(t, [s["tool_edges"] for s in stats], label="tool edges")
    axs[0, 1].legend()
    axs[1, 0].plot(t, [s["ref_stretch_max"] for s in stats], label="max")
    axs[1, 0].plot(t, [s["ref_stretch_min"] for s in stats], label="min")
    axs[1, 0].set_ylabel("reference edge length / rest"), axs[1, 0].legend()
    axs[1, 1].plot(t, [s["combined_deg_max"] for s in stats], label="max combined degree")
    axs[1, 1].plot(t, [s["spatial_components"] for s in stats], label="spatial components")
    axs[1, 1].plot(t, [s["combined_components"] for s in stats], label="combined components")
    axs[1, 1].legend()
    for a in axs[1]:
        a.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--ids", default=None, help=".npy solver particle ids (full-record episodes)")
    ap.add_argument("--sampler", choices=("random", "fps", "voxel"), default=None)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--r-dx", type=float, default=1.5,
                    help="r_ref = r_spatial = this x grid_dx (to-do starting values 1.5/2/2.5)")
    ap.add_argument("--r-ref", type=float, default=None, help="override, metres")
    ap.add_argument("--r-spatial", type=float, default=None, help="override, metres")
    ap.add_argument("--r-c", type=float, default=0.02, help="tool edge radius, metres")
    ap.add_argument("--stride", type=int, default=1, help="rows between statistics")
    ap.add_argument("--max-edges", type=int, default=4000, help="drawn per type")
    ap.add_argument("--video", action="store_true", help="also write graph.gif")
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "graph_viz"))
    args = ap.parse_args(argv)

    tr = load_trajectory(args.episode)
    if tr.rest_pos.size == 0:
        raise SystemExit(f"{args.episode} has no rest_pos; collect a schema-3.0 episode")
    view = NodeView(tr, np.load(args.ids) if args.ids else None, args.sampler, args.n)
    dx = float(tr.grid_dx) or 1 / 64
    r_ref = args.r_ref or args.r_dx * dx
    r_sp = args.r_spatial or args.r_dx * dx
    r_c = args.r_c if tr.has_jaws else None
    os.makedirs(args.out, exist_ok=True)

    ref = gb.reference_edges(view.X, r_ref)
    stats = per_row_stats(view, ref, r_sp, r_ref, r_c, args.stride)
    with open(os.path.join(args.out, "stats_over_time.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    plot_stats(stats, os.path.join(args.out, "stats_over_time.png"))
    rows = key_rows(tr)
    for label, t in rows.items():
        draw_row(view, ref, t, r_sp, r_c, os.path.join(args.out, f"rows_{label}.png"),
                 args.max_edges)

    items = checklist(view, ref, stats, r_sp, r_ref, r_c)
    md = [f"# Graph checklist: {os.path.basename(args.episode)}", "",
          f"{view.n} nodes; r_ref = {r_ref*1e3:.1f} mm, r_spatial = {r_sp*1e3:.1f} mm "
          f"({args.r_dx:g} x grid_dx), r_c = "
          f"{'n/a' if r_c is None else f'{r_c*1e3:.1f} mm'}; rows every {args.stride}.",
          f"Key rows: {rows}", "", "| status | check | detail |", "|---|---|---|"]
    md += [f"| {s} | {c} | {d} |" for s, c, d in items]
    with open(os.path.join(args.out, "checklist.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    for s, c, d in items:
        print(f"{s:4s}  {c}: {d}")

    if args.video:
        _gif(view, ref, r_sp, os.path.join(args.out, "graph.gif"), args.stride)
    print(f"\nwrote images, stats and checklist to {args.out}/")
    return 1 if any(s == "FAIL" for s, _, _ in items) else 0


def _gif(view, ref, r_sp, path, stride):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    tr = view.tr
    fig, ax = plt.subplots(figsize=(8, 3))
    y0 = float(np.median(view.X[:, 1]))
    h = gd.median_nn_distance(view.X)

    def frame(i):
        t = i * stride
        ax.clear()
        x = view.x(t)
        m = np.abs(x[:, 1] - y0) < h
        ax.scatter(x[m, 0], x[m, 2], s=3, c="k")
        if tr.has_jaws:
            for k in range(tr.jaw_names.size):
                c = _box_corners(tr.jaw_box_pose[t, k], tr.jaw_half_extents[k])
                for a, b in _BOX_EDGES:
                    ax.plot(*np.stack([c[a], c[b]])[:, [0, 2]].T, c="tab:red", lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t*float(tr.dt):.3f} s")

    anim = FuncAnimation(fig, frame, frames=range(len(tr) // stride))
    anim.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
