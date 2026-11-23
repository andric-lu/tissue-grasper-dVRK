#!/usr/bin/env python3
"""
grasp_pilot_report.py -- does a grasp episode retract tissue, or tear it?

    python host/grasp_pilot_report.py data_pilot/grasp/mpm_0000.npz [more.npz ...]

numpy only. Written after the first schema-3.0 grasp pilot pulled a 12-particle
strand 14 cm out of the slab while 32 particles moved more than 20 mm
(DECISION_LOG 14.14). These are the numbers that distinguished "retraction"
from "numerical tearing" there, computed the same way for any grasp episode so
pilots on different arms and scales can be put side by side (14.15):

    held            solver particles attached (grasp_full_ids)
    lift            how far the held particles rose during retraction
    followed        particles that moved > 25 / 50 / 75% of that lift: how much
                    tissue came WITH the grasp. A strand has ~ the held set only.
    stretch         max reference-edge stretch on the node graph (x rest length)
    pieces          material pieces with edges > 3x rest treated as torn
                    (host/visualize_graph.py's corrected fragmentation check)
    peak speed      and when (a release recoil shows up here)
    after release   max stretch and pieces on the last row: did it rejoin?
    crush           (added 19 Nov, DECISION_LOG 14.18) how much the jaws
                    compress the tissue BEFORE any pull: particles with
                    |det F - 1| > 0.1 at the end of the approach and at the end
                    of the close, the minimum det F there, and the max
                    reference-edge stretch at the end of the close. On the
                    first Classic x5 pilot these were 485 / 987 particles,
                    min J 0.22, stretch 2.5x: the bite, not the retraction.

Works on full-resolution pilots (--n-record all) and on subset episodes
(statistics then use the recorded nodes only, and say so).
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import collection_config as cc                            # noqa: E402
import graph_builder as gb                                # noqa: E402
import graph_design as gd                                 # noqa: E402
from trajectory_io import load_trajectory                 # noqa: E402

TEAR_FACTOR = 3.0


def report(path, ids_path=None, r_dx=1.5):
    tr = load_trajectory(path)
    x = tr.tissue_pos.astype(np.float64)
    ph = tr.phase
    active = np.flatnonzero(tr.grasp_active)
    out = {"file": os.path.basename(path), "robot": tr.robot_model or "dvrk_si_psm",
           "scale": float(tr.robot_scale[0]) if tr.robot_scale.size else 1.0,
           "rows": len(tr)}
    if active.size == 0:
        out["held"] = 0
        return out
    g0, g1 = int(active[0]), int(active[-1])
    held = tr.grasp_full_ids(g0)
    out["held"] = int(held.size)
    full = tr.particle_ids.size == int(tr.n_particles_simulated)
    # On a subset episode, use the held particles that WERE recorded (typically
    # 3-10 of 25-60). An empty column set made lift = 0 and every "followed"
    # count 0 on subset episodes -- found on the 20 Nov debug set (14.19).
    rec = held if full else held[np.isin(held, tr.particle_ids)]
    col = np.searchsorted(tr.particle_ids, rec)
    out["held_recorded"] = int(rec.size)
    retract = np.flatnonzero(ph == cc.PHASE_ID["retract"])
    t_end = int(retract[-1]) if retract.size else g1
    lift_vec = (x[t_end, col] - x[g0, col]).mean(axis=0) if col.size else np.zeros(3)
    lift = float(np.linalg.norm(lift_vec))
    out["lift_mm"] = lift * 1e3
    # Displacement ALONG the lift direction, so sideways jiggle is not counted.
    u = lift_vec / max(lift, 1e-12)
    along = (x[t_end] - x[g0]) @ u
    for f in (0.25, 0.5, 0.75):
        out[f"followed_{int(f*100)}pct"] = int((along > f * lift).sum())
    out["subset_note"] = "" if full else "subset episode: held/lift from grasp ids only"

    # Node graph: the fixed node set if given, else FPS 3000 from the reference.
    rest = tr.rest_pos_simulated if tr.rest_pos_simulated.size else tr.rest_pos
    if ids_path:
        ids = np.load(ids_path)
    elif full:
        ids = gd.farthest_point_sample(rest.astype(np.float64), 3000, 0)
    else:
        ids = tr.particle_ids
    cols = np.searchsorted(tr.particle_ids, ids) if tr.particle_ids.size else ids
    X = rest[ids].astype(np.float64)
    ref = gb.reference_edges(X, r_dx * float(tr.grid_dx))
    L0 = np.linalg.norm(X[ref[1]] - X[ref[0]], axis=1)

    def stretch_pieces(t):
        xt = x[t, cols]
        s = np.linalg.norm(xt[ref[1]] - xt[ref[0]], axis=1) / L0
        return float(s.max()), gd.connected_components(ref[:, s <= TEAR_FACTOR], X.shape[0])[0]

    # Crush: measured at the last approach row and the last close row, i.e.
    # before retraction can contribute. J = det F over every recorded particle.
    # WHY |J - 1| > 0.1: the settled bulk sits at median |J - 1| ~ 0.003
    # (14.15), so 0.1 is ~30x the background and only counts real compression.
    def last_row(name):
        rows = np.flatnonzero(ph == cc.PHASE_ID[name])
        return int(rows[-1]) if rows.size else None
    for name in ("approach", "close"):
        t = last_row(name)
        if t is None:
            continue
        J = np.linalg.det(tr.tissue_F[t].astype(np.float64))
        out[f"crushed_{name}"] = int((np.abs(J - 1.0) > 0.1).sum())
        out[f"min_J_{name}"] = float(J.min())
    t_close = last_row("close")
    if t_close is not None:
        out["stretch_close"] = stretch_pieces(t_close)[0]

    worst = max((stretch_pieces(t) + (t,) for t in range(0, len(tr), 5)), key=lambda a: a[0])
    out["max_stretch"], out["max_pieces"], out["worst_row"] = worst
    out["end_stretch"], out["end_pieces"] = stretch_pieces(len(tr) - 1)
    if full:
        out.update(_detachment(tr, held, g0, last_row("retract") or g1))
    sp = np.linalg.norm(tr.tissue_vel, axis=2).max(axis=1)
    t_pk = int(np.argmax(sp))
    out["peak_speed_m_s"] = float(sp[t_pk])
    out["peak_speed_phase"] = cc.PHASES[int(ph[t_pk])] if ph.size else ""
    return out


def _detachment(tr, held, g0, t_end, shell_r=0.008, fit_r=0.012, n_top=30):
    """Does the tissue come with the grasp, and does F still describe it?

    DECISION_LOG 14.21, full-resolution episodes only. `shell` = non-held
    particles within `shell_r` of a held one at rest.
        gap_mm            median distance from each held particle to the nearest
                          shell particle, at t_end (rest spacing ~3 mm)
        shell_follow      shell rise / held rise from g0 to t_end (1 = rigid)
        top_stretch_*     median max principal stretch of the n_top most-
                          stretched particles at t_end: from stored F, and
                          from F fitted to the motion of their rest
                          neighbours within fit_r (least squares)
        top_J_*           the same for det F. Stored and fitted agreeing is
                          what "F describes the motion" means; a fitted J far
                          above 1 means the neighbourhood came apart.
    """
    from scipy.spatial import cKDTree
    import tissue_metrics as tm
    X = tr.rest_pos_simulated.astype(np.float64)
    x0, x = tr.tissue_pos[g0].astype(np.float64), tr.tissue_pos[t_end].astype(np.float64)
    tree = cKDTree(X)
    is_held = np.zeros(len(X), bool)
    is_held[held] = True
    shell = np.unique(np.concatenate(tree.query_ball_point(X[held], shell_r)))
    shell = shell[~is_held[shell]]
    gap = cKDTree(x[shell]).query(x[held])[0]
    rise_h = float((x[held, 2] - x0[held, 2]).mean())
    rise_s = float((x[shell, 2] - x0[shell, 2]).mean())
    F = tr.tissue_F[t_end].astype(np.float64)
    s = tm.max_principal_stretch(F)
    top = np.argsort(s)[::-1][:n_top]
    s_fit, J_fit = [], []
    for i in top:
        nb = np.array(tree.query_ball_point(X[i], fit_r))
        nb = nb[nb != i]
        Ft, *_ = np.linalg.lstsq(X[nb] - X[i], x[nb] - x[i], rcond=None)
        s_fit.append(float(tm.max_principal_stretch(Ft.T)))
        J_fit.append(float(np.linalg.det(Ft.T)))
    return {"gap_mm": float(np.median(gap)) * 1e3,
            "shell_follow": rise_s / rise_h if abs(rise_h) > 1e-9 else float("nan"),
            "top_stretch_stored": float(np.median(s[top])),
            "top_stretch_fit": float(np.median(s_fit)),
            "top_J_stored": float(np.median(np.linalg.det(F[top]))),
            "top_J_fit": float(np.median(J_fit))}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--ids", default=None, help="fixed node set (.npy) for the graph numbers")
    args = ap.parse_args(argv)
    rows = [report(p, args.ids) for p in args.episodes]
    keys = ["file", "robot", "scale", "rows", "held", "held_recorded", "lift_mm", "followed_25pct",
            "followed_50pct", "followed_75pct", "max_stretch", "max_pieces",
            "end_stretch", "end_pieces", "peak_speed_m_s", "peak_speed_phase",
            "crushed_approach", "crushed_close", "min_J_close", "stretch_close",
            "gap_mm", "shell_follow", "top_stretch_stored", "top_stretch_fit",
            "top_J_stored", "top_J_fit"]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(
            (f"{r[k]:.2f}" if isinstance(r.get(k), float) else str(r.get(k, "")))
            for k in keys) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
