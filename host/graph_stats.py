#!/usr/bin/env python3
"""
graph_stats.py -- what graph does a node set make, and is it one a GNN can use?

    conda activate tissue-host
    python host/coarsening_study.py --data data_mpm_full/        # writes the node sets
    python host/mpm_locality_study.py                            # writes influence radii
    python host/graph_stats.py --n 450 600 900                   # ~1 min

DECISION_LOG.md section 13.4 (brief study 0.3). numpy + matplotlib, no taichi.

WHY THIS EXISTS
---------------
A radius graph over a node set is characterised by three numbers a model's
success depends on, and each has a specific failure:

  degree       too few neighbours and a node's update is underdetermined; GNS
               targets 10-20. On a 50 mm slab only a few node-spacings thick,
               SURFACE nodes lose half their neighbourhood sphere to empty space,
               so surface and interior degree are reported separately.
  components   a node in a component with no tool node never receives a message
               about the action -- its prediction can only be "keep doing what
               you were doing", and at training time that reads as "the model
               ignores the action". Must be exactly 1.
  diameter     hops to cross the object. If one recorded step couples the whole
               object (section 13.2 says so for undamped material), the
               receptive field M*R has to span it: diameter <= M.

WHAT IT MEASURES
----------------
For node sets saved by coarsening_study.py (rest-space, chosen once), the pitch
h is MEASURED as the median nearest-neighbour distance, never assumed, and
R = ratio * h is swept over RATIOS. For each (node set, R): mean/min/median
degree overall, on the surface and in the interior; component count; exact hop
diameter.

THE M CRITERION USES THE MEASURED INFLUENCE, WHEN THERE IS ONE
--------------------------------------------------------------
If runs/locality/ exists, the largest 1% velocity-influence radius over
materials, probes and kick axes -- for the chosen config, damping level and
recording interval -- sets the required receptive field:

    M_needed = ceil(r_1% / R)        (graph_design.required_message_passing_steps)

A `saturated` radius is a LOWER bound (influence reached the far edge of the
object), so for it the diameter criterion also applies. Without locality data
only degree and connectivity are judged, and the output says so.

ACCEPTANCE, per (node set, R)
    components == 1, 10 <= mean degree <= 20, min degree >= 3,
    M_needed <= M (when influence data exists), and diameter <= M when that
    influence is saturated.
The smallest acceptable R (fewest edges) per node set is reported.

A BROKEN-INPUT DEMONSTRATION IS BUILT IN. Every run also builds each node set's
graph at R = 0.5 h -- below the pitch -- and requires it to report more than one
component. If it does not, the connectivity numbers above are not trustworthy
and the script exits non-zero.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import graph_design as GD                                # noqa: E402

RATIOS = tuple(np.round(np.arange(1.2, 2.51, 0.1), 2))
MEAN_DEG = (10.0, 20.0)
MIN_DEG = 3
BROKEN_RATIO = 0.5

SERIES = {"interior": "#2a78d6", "surface": "#eb6834"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7", "#fcfcfb")


def influence_radius(locality_dir, config, level, frames):
    """Worst-case 1% velocity influence over materials/probes/axes, or None.

    SATURATION IS STICKY. A saturated cell is a LOWER bound -- the influence
    reached the far edge of the object from that probe -- so it can hide a
    radius larger than any resolved cell's. The first version took the largest
    radius and kept that cell's status: a resolved 245 mm edge-probe cell then
    outranked saturated >=230 mm centre-probe cells, the diameter criterion was
    never applied, and node sets with 13-18-hop diameters were passed at M = 10.
    Now: the radius is the largest over resolved AND saturated cells, and the
    status is SATURATED if ANY cell saturated.
    """
    rows = sorted(glob.glob(os.path.join(locality_dir, f"{config}_*_{level}.json")))
    if not rows:
        return None
    worst, n_sat, n_res, n_unres = None, 0, 0, 0
    for path in rows:
        row = json.load(open(path))
        for c in row["cells"]:
            if c["quantity"] != "v" or c["frames"] != frames:
                continue
            rad = c["radii"]["0.01"]
            if rad["status"] not in (GD.RESOLVED, GD.SATURATED):
                n_unres += 1
                continue
            n_sat += rad["status"] == GD.SATURATED
            n_res += rad["status"] == GD.RESOLVED
            if worst is None or rad["radius"] > worst["radius"]:
                worst = {"radius": rad["radius"],
                         "material": os.path.basename(path)[len(config) + 1:].split("_")[0],
                         "probe": c["probe"], "axis": c["axis"], "cell_status": rad["status"]}
    if worst is None:
        return None
    worst["status"] = GD.SATURATED if n_sat else GD.RESOLVED
    worst.update(n_saturated=n_sat, n_resolved=n_res, n_unresolved=n_unres)
    return worst


def analyse(pos, surface, R, M, infl):
    n = pos.shape[0]
    ei = GD.radius_graph(pos, R)
    deg = GD.degrees(ei, n)
    k, _ = GD.connected_components(ei, n)
    diam = GD.hop_diameter(ei, n) if k == 1 else {"diameter": None, "connected": False}
    out = {"R": R, "E": int(ei.shape[1]), "components": k,
           "all": GD.degree_stats(deg),
           "surface": GD.degree_stats(deg[surface]) if surface.any() else None,
           "interior": GD.degree_stats(deg[~surface]) if (~surface).any() else None,
           "diameter": diam["diameter"]}
    m_needed = GD.required_message_passing_steps(infl["radius"], R) if infl else None
    out["M_needed"] = m_needed
    ok = (k == 1 and MEAN_DEG[0] <= out["all"]["mean"] <= MEAN_DEG[1]
          and out["all"]["min"] >= MIN_DEG)
    if infl is not None:
        ok = ok and m_needed <= M
        if infl["status"] == GD.SATURATED:
            ok = ok and diam["diameter"] is not None and diam["diameter"] <= M
    out["acceptable"] = bool(ok)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", default=os.path.join(REPO, "runs", "coarsening", "nodesets.npz"))
    ap.add_argument("--samplers", nargs="+", default=["fps", "lattice", "surface_quota"])
    ap.add_argument("--n", type=int, nargs="+", default=[450, 600, 900])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=10, help="intended message-passing steps")
    ap.add_argument("--locality", default=os.path.join(REPO, "runs", "locality"))
    ap.add_argument("--config", default="as_collected")
    ap.add_argument("--level", default="none")
    ap.add_argument("--frames", type=int, default=3,
                    help="recording interval in 1/240 s frames (3 = 1/80 s, the current dataset)")
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "graph_stats"))
    ap.add_argument("--figures", default=os.path.join(REPO, "figures", "phase0"))
    args = ap.parse_args(argv)
    t0 = time.time()

    if not os.path.isfile(args.nodes):
        raise SystemExit(f"{args.nodes} not found -- run host/coarsening_study.py first")
    ns = np.load(args.nodes)
    rest, surface_all = ns["rest"].astype(np.float64), ns["surface"]
    infl = influence_radius(args.locality, args.config, args.level, args.frames)

    print("=" * 96)
    print("GRAPH STATISTICS -- DECISION_LOG.md section 13.4".center(96))
    print("=" * 96)
    if infl is None:
        print(f"no locality data for config={args.config} level={args.level} "
              f"-> judging degree and connectivity ONLY (M criterion not applied)")
    else:
        print(f"influence (1%, velocity, {args.config}, damping={args.level}, "
              f"1/{round(240 / args.frames)} s): largest radius {infl['radius']*1e3:.0f} mm "
              f"({infl['material']} {infl['probe']} probe, kick along {infl['axis']}, that cell "
              f"{infl['cell_status']}); over all cells {infl['n_saturated']} saturated, "
              f"{infl['n_resolved']} resolved, {infl['n_unresolved']} unresolvable "
              f"-> status {infl['status']}; intended M = {args.M}")

    results, broken_ok = [], True
    for sampler in args.samplers:
        for n in args.n:
            key = f"{sampler}_{n}_{args.seed if sampler != 'lattice' else 0}"
            if key not in ns:
                print(f"  {key}: not in {os.path.basename(args.nodes)}, skipped")
                continue
            ids = ns[key]
            pos, surface = rest[ids], surface_all[ids]
            h = GD.median_nn_distance(pos)

            # The built-in broken-input demonstration.
            k_broken, _ = GD.connected_components(GD.radius_graph(pos, BROKEN_RATIO * h), len(ids))
            broken_ok &= k_broken > 1

            rows = [analyse(pos, surface, float(r) * h, args.M, infl) for r in RATIOS]
            good = [r for r in rows if r["acceptable"]]
            chosen = min(good, key=lambda r: r["R"]) if good else None
            results.append({"set": key, "n": int(len(ids)), "h": h,
                            "surface_share": float(surface.mean()),
                            "broken_components": int(k_broken), "rows": rows,
                            "chosen_R": chosen["R"] if chosen else None,
                            "chosen_ratio": chosen["R"] / h if chosen else None})
            print(f"\n{key}: n={len(ids)}, measured pitch h={h*1e3:.1f} mm, surface "
                  f"{surface.mean():.0%}; R=0.5h gives {k_broken} components "
                  f"({'ok, shatters as it must' if k_broken > 1 else 'FAIL: did not shatter'})")
            print(f"  {'R/h':>5} {'R mm':>6} {'E':>7} {'comp':>5} {'deg mean':>8} {'min':>4} "
                  f"{'surf mean':>9} {'surf min':>8} {'int mean':>8} {'diam':>5} {'M_need':>6} ok")
            for r in rows:
                s, it = r["surface"], r["interior"]
                print(f"  {r['R']/h:>5.2f} {r['R']*1e3:>6.1f} {r['E']:>7} {r['components']:>5} "
                      f"{r['all']['mean']:>8.1f} {r['all']['min']:>4} "
                      f"{(s['mean'] if s else float('nan')):>9.1f} {(s['min'] if s else -1):>8} "
                      f"{(it['mean'] if it else float('nan')):>8.1f} "
                      f"{('-' if r['diameter'] is None else r['diameter']):>5} "
                      f"{('-' if r['M_needed'] is None else r['M_needed']):>6} "
                      f"{'yes' if r['acceptable'] else ''}")
            print(f"  -> smallest acceptable R: "
                  + (f"{chosen['R']*1e3:.1f} mm ({chosen['R']/h:.2f} h)" if chosen else "NONE"))

    os.makedirs(args.out, exist_ok=True)
    json.dump({"influence": infl, "M": args.M, "config": args.config, "level": args.level,
               "frames": args.frames, "ratios": list(map(float, RATIOS)),
               "criteria": {"mean_degree": MEAN_DEG, "min_degree": MIN_DEG},
               "broken_input_demonstration_passed": bool(broken_ok), "results": results},
              open(os.path.join(args.out, "results.json"), "w"), indent=2, default=float)
    fig = plot(results, os.path.join(args.figures, "graph_stats.png")) if results else None
    print(f"\nbroken-input demonstration (R = {BROKEN_RATIO} h shatters every set): "
          f"{'PASS' if broken_ok else 'FAIL'}")
    print(f"Results: {os.path.join(args.out, 'results.json')}" + (f"\nFigure: {fig}" if fig else ""))
    print(f"Wall time: {time.time() - t0:.0f}s")
    return 0 if broken_ok else 1


def plot(results, path):
    """Small multiples: one row per sampler; columns are interior mean degree,
    surface mean degree and hop diameter, each against R / h. One line per node
    count on the ordinal blue ramp (an ordered quantity gets one hue, light to
    dark). The 10-20 degree target's edges are muted hairlines. Surface and
    interior are separate panels, never a second axis. The first layout -- two
    rows of panels per node set in a single strip -- was ~3,500 px wide and
    unreadable, which is why it is a grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ramp5 = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
    samplers = list(dict.fromkeys(r["set"].rsplit("_", 2)[0] for r in results))
    ns = sorted({int(r["set"].rsplit("_", 2)[1]) for r in results})
    idx = np.round(np.linspace(1, 4, len(ns))).astype(int) if len(ns) > 1 else [2]
    colour = {n: ramp5[i] for n, i in zip(ns, idx)}

    dpi = 150
    px = 72.0 / dpi
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
        "axes.linewidth": 1 * px, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    cols = [("interior", "interior mean degree"), ("surface", "surface mean degree"),
            ("diameter", "hop diameter")]
    fig, axes = plt.subplots(len(samplers), 3, figsize=(9.5, 2.3 * len(samplers) + 0.9),
                             dpi=dpi, sharex=True, squeeze=False)
    for i, s in enumerate(samplers):
        for j, (key, label) in enumerate(cols):
            ax = axes[i][j]
            ax.grid(True, axis="y", color=GRID, linewidth=1 * px)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if key != "diameter":
                for edge in MEAN_DEG:
                    ax.axhline(edge, color=MUTED, linewidth=1 * px, zorder=1)
            for res in results:
                sampler, n_nom, _ = res["set"].rsplit("_", 2)
                if sampler != s:
                    continue
                x = [r["R"] / res["h"] for r in res["rows"]]
                if key == "diameter":
                    y = [r["diameter"] if r["diameter"] is not None else np.nan for r in res["rows"]]
                else:
                    y = [r[key]["mean"] if r[key] else np.nan for r in res["rows"]]
                ax.plot(x, y, color=colour[int(n_nom)], linewidth=2 * px,
                        solid_capstyle="round", zorder=3, label=f"n = {res['n']:,}")
            if i == 0:
                ax.set_title(label, fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(s.replace("_", " "), fontsize=8)
            if i == len(samplers) - 1:
                ax.set_xlabel("R / measured pitch h", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, [f"n = {n:,} (nominal)" for n in ns], loc="upper left", ncol=len(ns),
               frameon=False, fontsize=8, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("Radius-graph degree and diameter over rest-space node sets",
                 x=0.99, ha="right", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
