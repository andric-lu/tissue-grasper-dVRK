#!/usr/bin/env python3
"""
coarsening_study.py -- how few nodes can carry the safety and exposure metrics?

    conda activate tissue-host
    python host/mpm_adapter.py --task grasp --n-record 24000 --out data_mpm_full
    python host/validate_dataset.py --data data_mpm_full/
    python host/coarsening_study.py --data data_mpm_full/          # ~3-5 min

DECISION_LOG.md section 13.3 (brief study 0.2). numpy + matplotlib, no taichi:
this reads a recorded full-resolution episode and never touches the solver.

WHY THIS EXISTS
---------------
A graph network runs on n nodes chosen from the 24,000 MPM particles. The
controller constrains on `safety_strain` and optimises `exposure`, and both are
functions of the particle set they are computed over. If the coarse set
systematically under-reports peak stretch, the planner drives tissue past its
tear threshold while the model reports everything is fine -- the worst failure
available to this project, and invisible in a position-RMSE number. So before a
node count is chosen, the metrics are computed from each candidate node set and
compared with the full set.

WHY A GRASP EPISODE. A settling slab stretches ~1.0 everywhere, so any subset
agrees with the full set and the study would prove nothing. A grasp concentrates
peak stretch near the 1-3 attached particles (CLAUDE.md, "The PSM"), which is
the case coarsening is most likely to lose.

WHAT IS COMPARED
----------------
For each sampler (FPS, lattice, surface-quota FPS, uniform random -- random is
what mpm_adapter's 3,000-particle subset does today), each n, and each seed,
with the node set chosen ONCE from the frame-0 rest cloud:

  safety_strain  soft     |soft_c - soft_f|, worst frame
                 max      the subset max can only be <= the full max, so the
                          number reported is the EXCESS RATIO at the full set's
                          peak frame: (max_c - 1) / (max_f - 1). 1.0 = the coarse
                          set sees the whole excess stretch, 0.5 = half of it.
                 frac     |fraction above threshold, coarse - full|, worst frame.
                          A COUNT is not comparable across n; a fraction is.
  exposure       raw      coverage from n particles as-is
                 weighted each node weighted N / n (tissue_metrics `weights`)

WHY TWO EXPOSURES. Coverage is a SUM of unit Gaussians against an absolute
threshold, so the same geometry sampled by fewer particles reads as less covered
(section 13.1's finding, 27 -> 0.17 at 150 nodes). Raw exposure is expected to
collapse to "clear" under full cover; weighted exposure is the fix. Both are
reported so the collapse is shown, not asserted.

WHY A SYNTHETIC SHIFT. This episode's target lies under the slab in every frame
(logged exposure 0.0 throughout), which only tests the fully covered end. So the
rest cloud is also translated laterally across the target in 11 steps, sweeping
true exposure from 0 to 1. That is geometry only -- no dynamics -- and it is
reported separately and labelled as such.

ACCEPTANCE, FIXED BEFORE LOOKING AT RESULTS
-------------------------------------------
A (sampler, n) is acceptable when, in EVERY seed:
    peak excess ratio            >= 0.90
    worst soft-stretch error     <= 0.01
    worst weighted exposure err  <= 0.05  (hard exposure, episode + synthetic)
The smallest acceptable n per sampler is reported. If none is acceptable, that
is the result, and it is stated as such.

WHERE THE PEAK LIVES. For the full set's peak frame, the fraction of
over-threshold particles within one grid cell of a grasp-constrained particle is
reported. If it is high, the "safety peak" is partly the grasp constraint's own
artefact rather than tissue response, and a coarsening that misses it is
missing that artefact.

This checks the metrics' SAMPLING, not their physics: materials.py's ranges are
still placeholders and nothing here says the stretch values resemble tissue.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import graph_design as GD                                # noqa: E402
import tissue_metrics as TM                              # noqa: E402
import trajectory_io as TIO                              # noqa: E402

N_LIST = (150, 300, 450, 600, 900, 1800, 3000)
SEEDS = (0, 1, 2)
SAMPLERS = ("fps", "lattice", "surface_quota", "random")
SURFACE_FRAC = 0.5
N_SHIFTS = 11

PEAK_EXCESS_MIN = 0.90
SOFT_ERR_MAX = 0.01
EXPOSURE_ERR_MAX = 0.05

# Plot palette: dataviz reference categorical slots 1-4, light surface, adjacent
# order (line chart). Yellow is below 3:1 contrast, so identity rides the legend
# and every value is in table.md. Direct end labels were tried and COLLIDED
# (fps/lattice, random/surface quota end within a few pixels), and colliding
# labels are removed rather than nudged apart (dataviz marks-and-anatomy).
SERIES = {"fps": "#2a78d6", "lattice": "#eb6834", "surface_quota": "#1baf7a",
          "random": "#eda100"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7", "#fcfcfb")


def _load(path: str):
    if os.path.isdir(path):
        files = TIO.list_trajectories(path)
        if not files:
            raise SystemExit(f"no trajectories in {path}")
        path = files[0]
    tr = TIO.load_trajectory(path)
    if tr.is_subset:
        raise SystemExit(f"{path} records {tr.n_nodes} of {tr.n_particles} particles. "
                         "This study needs the FULL set: re-record with --n-record 24000.")
    if not tr.has_F:
        raise SystemExit(f"{path} has no deformation gradient; safety_strain cannot be computed")
    return path, tr


def _node_set(sampler, rest, n, seed, surface):
    if sampler == "fps":
        return GD.farthest_point_sample(rest, n, seed=seed)
    if sampler == "lattice":
        return GD.lattice_sample(rest, n)
    if sampler == "surface_quota":
        return GD.surface_quota_fps(rest, n, surface, SURFACE_FRAC, seed=seed)
    if sampler == "random":
        return np.sort(np.random.default_rng(seed).choice(rest.shape[0], n, replace=False))
    raise ValueError(sampler)


def _exposures(pos, target, weights=None):
    o, nrm, ext = target
    return (TM.compute_exposure(pos, o, nrm, ext, weights=weights),
            TM.hard_exposure(pos, o, nrm, ext, weights=weights))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(REPO, "data_mpm_full"))
    ap.add_argument("--n", type=int, nargs="+", default=list(N_LIST))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--samplers", nargs="+", default=list(SAMPLERS), choices=SAMPLERS)
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "coarsening"))
    ap.add_argument("--figures", default=os.path.join(REPO, "figures", "phase0"))
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure from --out/results.json without recomputing")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.replot:
        res = json.load(open(os.path.join(args.out, "results.json")))
        samplers = [s for s in SAMPLERS if any(r["sampler"] == s for r in res["rows"])]
        print("Figure:", plot(res["rows"], samplers, os.path.join(args.figures, "coarsening.png")))
        return 0

    path, tr = _load(args.data)
    pos = tr.tissue_pos.astype(np.float64)                  # (T, N, 3)
    F = tr.tissue_F.astype(np.float64)                      # (T, N, 3, 3)
    T, N = pos.shape[:2]
    rest = pos[0]

    # WHY: node sets are chosen in REST space. That is only true if frame 0 IS
    # the rest state; an episode captured after settling would silently choose
    # nodes in a deformed configuration. Measured, not assumed.
    rest_err = float(np.abs(F[0] - np.eye(3)).max())
    if rest_err > 1e-6:
        raise SystemExit(f"frame 0 is not at rest: max|F - I| = {rest_err:.2e}")

    ext = rest.max(axis=0) - rest.min(axis=0)
    spacing = float((np.prod(ext) / N) ** (1.0 / 3.0))
    surface = GD.surface_mask_from_bbox(rest, spacing)
    dx = float(tr.grid_dx) if float(tr.grid_dx) > 0 else None
    target = (np.asarray(tr.target_origin, np.float64), np.asarray(tr.target_normal, np.float64),
              np.asarray(tr.target_extent, np.float64))

    print("=" * 96)
    print("COARSENING STUDY -- DECISION_LOG.md section 13.3".center(96))
    print("=" * 96)
    print(f"{os.path.basename(path)}: T={T}, N={N}, rest extent {ext[0]*1e3:.0f} x "
          f"{ext[1]*1e3:.0f} x {ext[2]*1e3:.0f} mm, mean spacing {spacing*1e3:.2f} mm, "
          f"frame-0 max|F-I| = {rest_err:.1e}")
    print(f"surface (within one spacing of the rest bbox): {int(surface.sum())} particles "
          f"({surface.mean():.0%})")

    # ---- full-set reference ------------------------------------------------
    stretch = TM.max_principal_stretch(F)                   # (T, N)
    thr = TM.DEFAULT_STRAIN_THRESHOLD
    soft_f = TM.soft_max(stretch, p=8.0, axis=-1)
    max_f = stretch.max(axis=-1)
    frac_f = (stretch > thr).mean(axis=-1)
    t_peak = int(np.argmax(max_f))

    shifts = np.linspace(0.10, 0.20, N_SHIFTS)
    synth = np.stack([rest + np.array([s, 0.0, 0.0]) for s in shifts])   # (S, N, 3)
    ex_soft_f, ex_hard_f = _exposures(pos, target)
    sy_soft_f, sy_hard_f = _exposures(synth, target)

    peak_on_grasp = float("nan")
    if dx is not None:
        g_ids = tr.grasp_ids(t_peak)
        over = np.flatnonzero(stretch[t_peak] > thr)
        if g_ids.size and over.size:
            gp = pos[t_peak, g_ids]
            d = np.min(np.linalg.norm(pos[t_peak, over][:, None, :] - gp[None], axis=-1), axis=1)
            peak_on_grasp = float(np.mean(d <= dx))
    print(f"full set: peak max stretch {max_f[t_peak]:.4f} at t={t_peak}, "
          f"{int((stretch[t_peak] > thr).sum())} particles > {thr}; soft peak {soft_f.max():.4f}")
    print(f"          fraction of over-threshold particles within one cell of a grasped "
          f"particle at the peak: {peak_on_grasp:.1%}")
    print(f"          episode exposure (hard) range [{ex_hard_f.min():.3f}, {ex_hard_f.max():.3f}]; "
          f"synthetic shift sweep [{sy_hard_f.min():.3f}, {sy_hard_f.max():.3f}]")

    # ---- every (sampler, n, seed) -----------------------------------------
    os.makedirs(args.out, exist_ok=True)
    nodesets = {"rest": rest.astype(np.float32), "surface": surface,
                "spacing": np.float64(spacing)}
    rows = []
    for sampler in args.samplers:
        seeds = (0,) if sampler == "lattice" else args.seeds     # lattice is deterministic
        for n in args.n:
            per_seed = []
            for seed in seeds:
                ids = _node_set(sampler, rest, n, seed, surface)
                nodesets[f"{sampler}_{n}_{seed}"] = ids.astype(np.int32)
                s = stretch[:, ids]
                soft_c = TM.soft_max(s, p=8.0, axis=-1)
                max_c = s.max(axis=-1)
                frac_c = (s > thr).mean(axis=-1)
                w = np.full(ids.size, N / ids.size)
                ex_raw = _exposures(pos[:, ids], target)
                ex_w = _exposures(pos[:, ids], target, w)
                sy_raw = _exposures(synth[:, ids], target)
                sy_w = _exposures(synth[:, ids], target, w)
                excess_f = max_f[t_peak] - 1.0
                per_seed.append({
                    "n_actual": int(ids.size),
                    "surface_share": float(surface[ids].mean()),
                    "soft_err": float(np.abs(soft_c - soft_f).max()),
                    "peak_excess_ratio": float((max_c[t_peak] - 1.0) / excess_f) if excess_f > 0 else float("nan"),
                    "max_underreport": float((max_f - max_c).max()),
                    "frac_err": float(np.abs(frac_c - frac_f).max()),
                    "exp_raw_err": float(max(np.abs(ex_raw[1] - ex_hard_f).max(),
                                             np.abs(sy_raw[1] - sy_hard_f).max())),
                    "exp_w_err": float(max(np.abs(ex_w[1] - ex_hard_f).max(),
                                           np.abs(sy_w[1] - sy_hard_f).max())),
                    "exp_w_soft_err": float(max(np.abs(ex_w[0] - ex_soft_f).max(),
                                                np.abs(sy_w[0] - sy_soft_f).max())),
                    "exp_raw_under_cover": float(ex_raw[1].mean()),
                })
            agg = {k: [p[k] for p in per_seed] for k in per_seed[0]}
            ok = all(p["peak_excess_ratio"] >= PEAK_EXCESS_MIN and p["soft_err"] <= SOFT_ERR_MAX
                     and p["exp_w_err"] <= EXPOSURE_ERR_MAX for p in per_seed)
            row = {"sampler": sampler, "n": n, "seeds": list(seeds), "acceptable": ok,
                   "n_actual": int(np.mean(agg["n_actual"])),
                   "surface_share": float(np.mean(agg["surface_share"])),
                   "soft_err_worst": max(agg["soft_err"]),
                   "peak_excess_ratio_worst": min(agg["peak_excess_ratio"]),
                   "peak_excess_ratio_mean": float(np.mean(agg["peak_excess_ratio"])),
                   "max_underreport_worst": max(agg["max_underreport"]),
                   "frac_err_worst": max(agg["frac_err"]),
                   "exp_raw_err_worst": max(agg["exp_raw_err"]),
                   "exp_w_err_worst": max(agg["exp_w_err"]),
                   "exp_w_soft_err_worst": max(agg["exp_w_soft_err"]),
                   "exp_raw_under_cover_mean": float(np.mean(agg["exp_raw_under_cover"]))}
            rows.append(row)
            print(f"  {sampler:>13} n={n:>5} (actual {row['n_actual']:>5}, surface "
                  f"{row['surface_share']:.0%}): peak excess {row['peak_excess_ratio_worst']:.2f} "
                  f"soft err {row['soft_err_worst']:.4f}  frac err {row['frac_err_worst']:.4f}  "
                  f"exposure raw {row['exp_raw_err_worst']:.2f} / weighted "
                  f"{row['exp_w_err_worst']:.3f}  {'OK' if ok else '--'}")

    # ---- report -------------------------------------------------------------
    chosen = {}
    for sampler in args.samplers:
        good = [r for r in rows if r["sampler"] == sampler and r["acceptable"]]
        chosen[sampler] = min(good, key=lambda r: r["n"])["n"] if good else None

    lines = ["# Coarsening study -- raw table (DECISION_LOG.md section 13.3)", "",
             f"Episode `{os.path.basename(path)}`, N={N}, T={T}. Full-set peak max stretch "
             f"{max_f[t_peak]:.4f} at t={t_peak}; {int((stretch[t_peak] > thr).sum())} particles "
             f"> {thr}; {peak_on_grasp:.1%} of them within one cell of a grasped particle.", "",
             f"Acceptance (every seed): peak excess ratio >= {PEAK_EXCESS_MIN}, soft error <= "
             f"{SOFT_ERR_MAX}, weighted hard-exposure error <= {EXPOSURE_ERR_MAX}. Exposure errors "
             "are the worst over episode frames AND the synthetic lateral-shift sweep.", "",
             "| sampler | n | actual | surface | peak excess (worst / mean) | max under-report "
             "| soft err | frac>thr err | exposure raw err | exposure weighted err (hard / soft) "
             "| raw exposure under cover | OK |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['sampler']} | {r['n']} | {r['n_actual']} | {r['surface_share']:.0%} | "
            f"{r['peak_excess_ratio_worst']:.2f} / {r['peak_excess_ratio_mean']:.2f} | "
            f"{r['max_underreport_worst']:.3f} | {r['soft_err_worst']:.4f} | "
            f"{r['frac_err_worst']:.4f} | {r['exp_raw_err_worst']:.2f} | "
            f"{r['exp_w_err_worst']:.3f} / {r['exp_w_soft_err_worst']:.3f} | "
            f"{r['exp_raw_under_cover_mean']:.2f} | {'yes' if r['acceptable'] else 'no'} |")
    lines += ["", "Smallest acceptable n per sampler: "
              + ", ".join(f"{s} {chosen[s] if chosen[s] else 'NONE'}" for s in args.samplers), ""]
    with open(os.path.join(args.out, "table.md"), "w") as fh:
        fh.write("\n".join(lines))
    json.dump({"episode": path, "N": N, "T": T, "t_peak": t_peak,
               "peak_max_stretch": float(max_f[t_peak]),
               "n_over_threshold_at_peak": int((stretch[t_peak] > thr).sum()),
               "peak_on_grasp_fraction": peak_on_grasp, "spacing": spacing,
               "surface_count": int(surface.sum()), "rows": rows, "chosen": chosen,
               "criteria": {"peak_excess_min": PEAK_EXCESS_MIN, "soft_err_max": SOFT_ERR_MAX,
                            "exposure_err_max": EXPOSURE_ERR_MAX}},
              open(os.path.join(args.out, "results.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(args.out, "nodesets.npz"), **nodesets)

    fig_path = plot(rows, args.samplers, os.path.join(args.figures, "coarsening.png"))
    print("\nSmallest acceptable n per sampler: "
          + ", ".join(f"{s}={chosen[s] if chosen[s] else 'NONE'}" for s in args.samplers))
    print(f"Table: {os.path.join(args.out, 'table.md')}\nNode sets: "
          f"{os.path.join(args.out, 'nodesets.npz')}\nFigure: {fig_path}")
    print(f"Wall time: {time.time() - t0:.0f}s")
    return 0


def plot(rows, samplers, path):
    """Three panels sharing the n axis: peak excess ratio, weighted exposure
    error, raw exposure error. One line per sampler, direct-labelled at its end;
    acceptance thresholds as muted hairlines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 150
    px = 72.0 / dpi
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
        "axes.linewidth": 1 * px, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    panels = [("peak_excess_ratio_worst", "peak excess stretch seen\n(coarse / full, worst seed)",
               PEAK_EXCESS_MIN, (0, 1.05)),
              ("exp_w_err_worst", "hard exposure error,\nweighted N/n (worst)", EXPOSURE_ERR_MAX, (0, None)),
              ("exp_raw_err_worst", "hard exposure error,\nunweighted (worst)", EXPOSURE_ERR_MAX, (0, 1.05))]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), dpi=dpi)
    for ax, (key, label, line, ylim) in zip(axes, panels):
        ax.set_xscale("log")
        ax.grid(True, axis="y", color=GRID, linewidth=1 * px)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.axhline(line, color=MUTED, linewidth=1 * px, zorder=1)
        ax.annotate("acceptance", xy=(0.0, line), xycoords=("axes fraction", "data"),
                    xytext=(2, 3), textcoords="offset points", fontsize=7, color=MUTED)
        for sampler in samplers:
            rs = sorted((r for r in rows if r["sampler"] == sampler), key=lambda r: r["n"])
            x = [r["n_actual"] for r in rs]
            y = [r[key] for r in rs]
            ax.plot(x, y, color=SERIES[sampler], linewidth=2 * px, solid_capstyle="round",
                    solid_joinstyle="round", zorder=3, label=sampler.replace("_", " "))
        ax.set_ylim(*ylim)
        # Explicit plain-number ticks: matplotlib's default log minor labels
        # ("2x10^2 3x10^2 ...") overlapped into an unreadable run.
        ticks = [150, 300, 600, 900, 1800, 3000]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks])
        ax.minorticks_off()
        ax.set_ylabel(label, fontsize=8)
        ax.set_xlabel("graph nodes n", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), frameon=False,
               fontsize=8, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("Can n graph nodes carry the safety and exposure metrics? (grasp episode, full N = 24,000)",
                 x=0.99, ha="right", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.9))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
