#!/usr/bin/env python3
"""
collect_debug_set.py -- to-do Item 6's debug dataset, end to end.

    conda activate tissue-sim      # NOT tissue-host: the default config drives
                                   # SurRoL's Classic arm (DECISION_LOG 14.15)
    python host/collect_debug_set.py --out data_debug \\
        --particle-ids configs/particle_ids_3000.npy \\
        --n-settle 4 --n-grasp 8 --variants 2x2

Runs, in order, and stops at the first failure:

  1. settle episodes      mpm_adapter --task settle, one child each
  2. grasp episodes       mpm_adapter --task grasp with configs/grasp_debug_set.json:
                          every episode draws its own grasp point, retraction
                          direction and distance, and material, from its seed
  3. matched variants     Item 12: ONE grasp seed re-run with a grid of
                          retraction directions x distances. Same seed -> same
                          material, same (Taichi-fixed) cloud, same pre-roll,
                          same approach and grasp: identical up to retraction
  4. validation           validate_dataset.py on each directory; a FAIL stops
                          the run before the manifest is written
  5. split manifest       held-out episodes chosen and hashed BEFORE any
                          training window exists (src/split_manifest.py)

Every episode lands next to its exact config (mpm_XXXX.config.json) and each
directory gets collection_log.txt -- the to-do's "save the collection
command/configuration". The robot+tissue video is a separate step:

    python host/visualize_grasp.py data_debug/grasp/mpm_0000.npz --save grasp_0000.gif

Cost, roughly: ~20 s per episode at 80 Hz on the M-series Mac before schema 3.0;
at 240 Hz expect ~3x the rows, and the logged exposure metric (computed over all
24,000 particles every row) is the largest per-row cost -- see DECISION_LOG
section 14.
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
sys.path.insert(0, os.path.join(REPO, "src"))

import collection_config as cc                            # noqa: E402
import split_manifest as sm                               # noqa: E402

ADAPTER = os.path.join(REPO, "host", "mpm_adapter.py")
VALIDATOR = os.path.join(REPO, "host", "validate_dataset.py")


_LOG = None


def _run(cmd):
    print("$", " ".join(cmd), flush=True)
    if _LOG:
        with open(_LOG, "a") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_debug")
    ap.add_argument("--particle-ids", required=True,
                    help="the fixed node set from host/sampling_study.py --write-ids")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "grasp_debug_set.json"))
    ap.add_argument("--n-settle", type=int, default=4)
    ap.add_argument("--n-grasp", type=int, default=8)
    ap.add_argument("--variants", default="2x2",
                    help="DIRECTIONSxDISTANCES for Item 12's matched variants, or 0")
    ap.add_argument("--settle-steps", type=int, default=240, help="rows (1 s at 240 Hz)")
    ap.add_argument("--seed", type=int, default=1000,
                    help="first seed; settle, grasp and variants use disjoint ranges")
    ap.add_argument("--dry-run", action="store_true", help="write configs, run nothing")
    ap.add_argument("--validate-only", action="store_true",
                    help="skip simulation; validate the episodes already in --out and "
                         "write the split manifest (resume after a validator stop)")
    args = ap.parse_args(argv)

    common = ["--particle-ids", args.particle_ids, "--f-encoding", "float32"]
    d_settle, d_grasp, d_var = (os.path.join(args.out, s) for s in
                                ("settle", "grasp", "variants"))
    for d in (d_settle, d_grasp, d_var):
        os.makedirs(d, exist_ok=True)
    global _LOG
    _LOG = os.path.join(args.out, "collection_log.txt")
    with open(_LOG, "a") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "python host/collect_debug_set.py "
                 + " ".join(sys.argv[1:] if argv is None else argv) + "\n")

    simulate = not (args.dry_run or args.validate_only)
    if simulate:
        _run([sys.executable, ADAPTER, "--task", "settle", "--out", d_settle,
              "--episodes", str(args.n_settle), "--steps", str(args.settle_steps),
              "--seed", str(args.seed)] + common)
        _run([sys.executable, ADAPTER, "--task", "grasp", "--config", args.config,
              "--out", d_grasp, "--episodes", str(args.n_grasp),
              "--seed", str(args.seed + 100)] + common)

    if args.variants and args.variants != "0":
        n_dir, n_dist = (int(v) for v in args.variants.lower().split("x"))
        base = cc.load_config(args.config)
        episode = cc.sample_episode_config(base, args.seed + 200)
        nominal = cc.load_config(args.config)["retract"]["direction"]
        # The nominal direction plus (n_dir - 1) others inside the same 20 degree
        # cone the randomised episodes use, drawn with fixed seeds.
        cone = np.deg2rad(base["randomize"].get("retract_cone_deg", 20.0))
        dirs = [nominal] + [cc._random_unit_in_cone(np.random.default_rng(i),
                                                    np.asarray(nominal, float), cone).tolist()
                            for i in range(n_dir - 1)]
        lo, hi = base["randomize"].get("retract_distance_m", [0.01, 0.03])
        dists = [lo + (hi - lo) * i / max(n_dist - 1, 1) for i in range(n_dist)]
        variants = cc.matched_variants(episode, dirs, dists)
        for i, v in enumerate(variants):
            path = os.path.join(d_var, f"mpm_{i:04d}.config.json")
            with open(path, "w") as fh:
                fh.write(cc.config_to_json(v))
            if simulate:
                _run([sys.executable, ADAPTER, "--task", "grasp", "--out", d_var,
                      "--index", str(i), "--episode-config", path] + common)

    if args.dry_run:
        print("dry run: configs written, nothing simulated")
        return 0

    for d in (d_settle, d_grasp, d_var):
        if any(f.endswith(".npz") for f in os.listdir(d)):
            _run([sys.executable, VALIDATOR, "--data", d])

    eps = []
    for d, group_prefix in ((d_settle, None), (d_grasp, None), (d_var, "variants")):
        for e in sm.episodes_in_dir(d):
            if group_prefix:
                # One group: every variant shares the seed, so all go to the
                # same side of the split together.
                e = dict(e, group=f"variants:seed{e['seed']}", task=e["task"] + "_variant")
            eps.append(e)
    manifest = sm.make_manifest(eps, note="Item 6 debug set; held out before windowing")
    path = sm.write_manifest(manifest, os.path.join(args.out, "split_manifest.json"))
    print(f"\nwrote {path}")
    for w in manifest["warnings"]:
        print("NOTE:", w)
    print(json.dumps({e["path"]: e["split"] for e in manifest["episodes"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
