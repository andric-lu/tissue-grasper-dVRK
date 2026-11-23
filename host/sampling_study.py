#!/usr/bin/env python3
"""
sampling_study.py -- to-do Item 3: which particles become nodes, and what radii.

    conda activate tissue-host
    # on the full-resolution pilot (recorded with --n-record all):
    python host/sampling_study.py --episode data_pilot/settle/mpm_0000.npz \\
        --out runs/sampling --write-ids configs/particle_ids_3000.npy
    # before any pilot exists, on a statistically identical stand-in cloud:
    python host/sampling_study.py --proxy-cloud --out runs/sampling_proxy

numpy only in the measurement (src/graph_design.py, src/graph_builder.py); this
file only orchestrates and writes tables, so it also runs in the container.

WHAT IT COMPARES (the to-do's list, in its words)
-------------------------------------------------
    samplers   random, farthest point (FPS), voxel/grid -- at ~the same count
    counts     3,000 (the to-do's start), plus 1,000 ("reduce toward 1,000
               only if needed": reported so the cost of needing it is known)
    radii      1.5, 2.0, 2.5 x grid_dx, each also expressed in MEASURED node
               pitch (median nearest-neighbour distance), because what a radius
               means for a graph depends on node spacing, not on the solver grid
    graphs     reference (rest X), spatial (current x), and their union,
               REPORTED SEPARATELY

and for each: spatial coverage, isolated-node count, connected components,
degree and edge-length distributions, edge count, runtime.

THE TWO KINDS OF INPUT
----------------------
--episode   a v3 episode. Nodes are always chosen from rest_pos_simulated (the
            to-do: "Select from reference positions"). If the episode recorded
            EVERY particle (pilot with --n-record all), the spatial graph is
            also measured at its most-deformed rows; otherwise only at rest,
            and the table says so.
--proxy-cloud
            24,000 points drawn uniformly in init_cube's box (300 x 300 x 50 mm
            at (0.1, 0.2, 0.05)), which is the distribution Taichi's
            init_cube() draws from. Statistically the same rest cloud, not the
            same points: good for the reference graph and the sampler
            comparison before a pilot exists, says nothing about deformation.
            Every output row is labelled `proxy`.

SELF-CHECK ON BROKEN INPUT (kept from Phase 0's graph_stats.py)
---------------------------------------------------------------
Every run also builds each node set's reference graph at R = 0.5 x pitch,
below the spacing, and requires it to come out fragmented. If it does not, the
component counter is not to be trusted and the script exits non-zero.

THE RECOMMENDATION IS A RULE, NOT A VERDICT
-------------------------------------------
--recommend applies: at n = 3,000, among configurations whose reference AND
combined graphs have zero isolated nodes and one component at every measured
row, the one with the fewest total edges (the compute proxy). It prints the
rule and the runner-up; the choice and its rationale go in DECISION_LOG.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import graph_builder as gb                                # noqa: E402
import graph_design as gd                                 # noqa: E402
from trajectory_io import load_trajectory                 # noqa: E402

GRID_DX = 1.0 / 64.0          # used only for the proxy cloud; episodes record theirs
RADII_DX = (1.5, 2.0, 2.5)
BROKEN_RATIO = 0.5


def proxy_cloud(seed: int = 0, n: int = 24000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)) * [0.3, 0.3, 0.05] + [0.1, 0.2, 0.05]


def choose_nodes(sampler: str, rest: np.ndarray, n: int, seed: int) -> np.ndarray:
    if sampler == "random":
        return gd.random_sample(rest.shape[0], n, seed)
    if sampler == "fps":
        return gd.farthest_point_sample(rest, n, seed)
    if sampler == "voxel":
        return gd.voxel_sample(rest, n)
    raise ValueError(sampler)


def deformed_rows(tr, k: int = 2):
    """The rows where the tissue is furthest from its reference, plus the
    last row: where proximity edges change most. Empty if the episode does
    not hold every particle."""
    if tr.particle_ids.size and tr.particle_ids.size < int(tr.n_particles_simulated):
        return []
    disp = np.linalg.norm(tr.tissue_pos - tr.rest_pos[None], axis=2).mean(axis=1)
    rows = list(np.argsort(disp)[::-1][:k]) + [len(tr) - 1]
    return sorted(set(int(r) for r in rows))


def study(rest_full, dx, samplers, counts, radii_dx, seed, tr=None, label=""):
    rows_out, problems = [], []
    frames = deformed_rows(tr) if tr is not None else []
    for n in counts:
        for sampler in samplers:
            t0 = time.perf_counter()
            ids = choose_nodes(sampler, rest_full, n, seed)
            t_sample = time.perf_counter() - t0
            X = rest_full[ids]
            pitch = gd.median_nn_distance(X)
            surface = gd.surface_mask_from_bbox(X, pitch)
            cov = gd.coverage_stats(rest_full, X)

            # Self-check: below the pitch the graph MUST fall apart.
            broken = gb.reference_edges(X, BROKEN_RATIO * pitch)
            n_comp_broken, _ = gd.connected_components(broken, X.shape[0])
            if n_comp_broken <= 1:
                problems.append(f"{sampler} n={n}: R=0.5*pitch gave ONE component; "
                                "the component counter cannot be trusted")

            for rdx in radii_dx:
                R = rdx * dx
                t0 = time.perf_counter()
                ref = gb.reference_edges(X, R)
                t_ref = time.perf_counter() - t0
                base = dict(source=label, sampler=sampler, n_target=n,
                            n_nodes=int(ids.size), radius_dx=rdx, radius_m=R,
                            radius_pitch=R / pitch, pitch_m=pitch,
                            coverage_max_m=cov["max"], coverage_p95_m=cov["p95"],
                            sample_s=t_sample)
                rec = dict(base, graph="reference", row="rest", build_s=t_ref,
                           **gd.graph_summary(X, ref, surface))
                rows_out.append(rec)
                # At rest x = X, so the spatial graph equals the reference one;
                # measure it anyway (the timing is the per-row rebuild cost).
                t0 = time.perf_counter()
                sp = gb.spatial_edges(X, R)
                t_sp = time.perf_counter() - t0
                rows_out.append(dict(base, graph="spatial", row="rest", build_s=t_sp,
                                     **gd.graph_summary(X, sp, surface)))
                rows_out.append(dict(base, graph="combined", row="rest", build_s=t_ref + t_sp,
                                     **gd.graph_summary(X, gb.combined_pairs(ref, sp), surface)))
                for f in frames:
                    x = tr.tissue_pos[f].astype(np.float64)[ids]
                    t0 = time.perf_counter()
                    sp = gb.spatial_edges(x, R)
                    t_sp = time.perf_counter() - t0
                    rows_out.append(dict(base, graph="reference", row=f, build_s=0.0,
                                         **gd.graph_summary(x, ref, surface)))
                    rows_out.append(dict(base, graph="spatial", row=f, build_s=t_sp,
                                         **gd.graph_summary(x, sp, surface)))
                    rows_out.append(dict(base, graph="combined", row=f, build_s=t_sp,
                                         **gd.graph_summary(x, gb.combined_pairs(ref, sp),
                                                            surface)))
    return rows_out, problems, frames


def recommend(rows, n_target=3000):
    """The --recommend rule (module docstring). Returns (best, runner_up, why)."""
    keyed = {}
    for r in rows:
        if r["n_target"] != n_target:
            continue
        keyed.setdefault((r["sampler"], r["radius_dx"]), []).append(r)
    ok = []
    for key, rs in keyed.items():
        must = [r for r in rs if r["graph"] in ("reference", "combined")]
        if all(r["isolated"] == 0 and r["components"] == 1 for r in must):
            cost = sum(r["n_edges"] for r in rs
                       if r["row"] == "rest" and r["graph"] in ("reference", "spatial"))
            ok.append((cost, key, rs))
    ok.sort(key=lambda c: c[0])
    if not ok:
        return None, None, "no configuration at n=%d is free of isolated nodes" % n_target
    why = ("fewest edges (reference + spatial at rest) among configurations whose "
           "reference and combined graphs have no isolated nodes and one component "
           "at every measured row")
    return ok[0], (ok[1] if len(ok) > 1 else None), why


def write_tables(rows, out_dir, frames, label):
    os.makedirs(out_dir, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out_dir, "sampling_study.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "sampling_study.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)

    lines = [f"# Item 3 sampling/radius comparison ({label})", "",
             "Radii in grid_dx and in measured node pitch h. Coverage = distance "
             "from every simulated particle to its nearest node. Degree mean "
             "overall / surface / interior. Rows measured: rest"
             + (f" and deformed rows {frames}" if frames else
                " only (no full-resolution trajectory)") + ".", "",
             "| sampler | n | R (dx) | R (h) | graph | row | edges | isolated | comps | "
             "deg mean (surf/int) | deg min | len p05–p95 mm | coverage max/p95 mm | build ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['sampler']} | {r['n_nodes']} | {r['radius_dx']:.1f} | "
            f"{r['radius_pitch']:.2f} | {r['graph']} | {r['row']} | {r['n_edges']} | "
            f"{r['isolated']} | {r['components']} | {r['deg_mean']:.1f} "
            f"({r.get('deg_mean_surface', float('nan')):.1f}/"
            f"{r.get('deg_mean_interior', float('nan')):.1f}) | {r['deg_min']} | "
            f"{r['len_p05']*1e3:.1f}–{r['len_p95']*1e3:.1f} | "
            f"{r['coverage_max_m']*1e3:.1f}/{r['coverage_p95_m']*1e3:.1f} | "
            f"{r['build_s']*1e3:.0f} |")
    with open(os.path.join(out_dir, "sampling_study.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--episode", help="v3 episode with rest_pos_simulated")
    src.add_argument("--proxy-cloud", action="store_true",
                     help="statistically identical stand-in for init_cube's cloud")
    ap.add_argument("--samplers", nargs="+", default=["random", "fps", "voxel"])
    ap.add_argument("--n", type=int, nargs="+", default=[3000, 1000])
    ap.add_argument("--radii-dx", type=float, nargs="+", default=list(RADII_DX))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "sampling"))
    ap.add_argument("--recommend", action="store_true")
    ap.add_argument("--write-ids", default=None,
                    help="save the recommended config's particle ids (.npy) for "
                         "mpm_adapter.py --particle-ids")
    args = ap.parse_args(argv)

    tr = None
    if args.episode:
        tr = load_trajectory(args.episode)
        if tr.rest_pos_simulated.size == 0:
            raise SystemExit(f"{args.episode} has no rest_pos_simulated (schema "
                             f"{tr.schema_version}); collect a schema-3.0 pilot")
        rest, dx, label = tr.rest_pos_simulated.astype(np.float64), float(tr.grid_dx), \
            os.path.basename(args.episode)
    else:
        rest, dx, label = proxy_cloud(args.seed), GRID_DX, "proxy"

    rows, problems, frames = study(rest, dx, args.samplers, args.n, args.radii_dx,
                                   args.seed, tr, label)
    write_tables(rows, args.out, frames, label)
    print(f"wrote {len(rows)} rows to {args.out}/sampling_study.{{csv,json,md}}")
    if not frames:
        print("spatial graph measured at rest only (episode does not hold every "
              "particle, or proxy cloud)")
    for p in problems:
        print("SELF-CHECK FAILED:", p)

    if args.recommend or args.write_ids:
        best, second, why = recommend(rows)
        print(f"\nrule: {why}")
        if best is None:
            print("  no configuration qualifies")
        else:
            (cost, (sampler, rdx), _) = best
            print(f"  recommended: {sampler}, n=3000, R = {rdx} x grid_dx "
                  f"({rdx*dx*1e3:.1f} mm), {cost} edges at rest")
            if second:
                print(f"  runner-up:   {second[1][0]}, R = {second[1][1]} x grid_dx, "
                      f"{second[0]} edges")
            if args.write_ids:
                ids = choose_nodes(sampler, rest, 3000, args.seed).astype(np.int32)
                np.save(args.write_ids, ids)
                with open(os.path.splitext(args.write_ids)[0] + ".json", "w") as fh:
                    json.dump({"sampler": sampler, "n": int(ids.size), "seed": args.seed,
                               "radius_dx": rdx, "grid_dx_m": dx, "source": label,
                               "rule": why}, fh, indent=2)
                print(f"  wrote {args.write_ids} ({ids.size} ids)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
