"""
test_split_manifest.py -- held-out episodes are chosen by rule, before windows,
and matched variants never straddle the split.
"""

import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "host"))

import split_manifest as sm  # noqa: E402


def _eps(tmp_path):
    out = []
    for i in range(4):
        p = tmp_path / f"s{i}.npz"
        p.write_bytes(bytes([i]))
        out.append({"path": str(p), "task": "tissue_settle", "seed": i})
    for i in range(3):                                       # 3 variants, one seed
        p = tmp_path / f"v{i}.npz"
        p.write_bytes(bytes([10 + i]))
        out.append({"path": str(p), "task": "variant", "seed": 7, "group": "variants:7"})
    p = tmp_path / "lonely.npz"
    p.write_bytes(b"x")
    out.append({"path": str(p), "task": "tissue_grasp", "seed": 99})
    return out


def test_one_heldout_group_per_task_and_groups_stay_together(tmp_path):
    m = sm.make_manifest(_eps(tmp_path))
    settle = [e for e in m["episodes"] if e["task"] == "tissue_settle"]
    assert sum(e["split"] == "heldout" for e in settle) == 1
    variants = {e["split"] for e in m["episodes"] if e["task"] == "variant"}
    assert len(variants) == 1                              # all on one side
    assert any("tissue_grasp" in w for w in m["warnings"])  # single episode: none held out


def test_deterministic(tmp_path):
    eps = _eps(tmp_path)
    a = [e["split"] for e in sm.make_manifest(eps, seed=3)["episodes"]]
    b = [e["split"] for e in sm.make_manifest(eps, seed=3)["episodes"]]
    assert a == b


def test_a_file_changed_after_the_split_is_refused(tmp_path):
    m = sm.make_manifest(_eps(tmp_path))
    path = sm.write_manifest(m, str(tmp_path / "m.json"))
    held = sm.load_split(path, "heldout")
    with open(held[0], "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ValueError, match="changed after"):
        sm.load_split(path, "heldout")


def test_collect_debug_set_dry_run_writes_matched_variant_configs(tmp_path):
    import collect_debug_set as cds
    import collection_config as cc
    assert cds.main(["--out", str(tmp_path), "--particle-ids", "x.npy",
                     "--variants", "2x2", "--dry-run"]) == 0
    d = tmp_path / "variants"
    cfgs = [json.loads((d / f).read_text()) for f in sorted(os.listdir(d))]
    assert len(cfgs) == 4 and len({c["seed"] for c in cfgs}) == 1
    pre = [cc.build_waypoints(c) for c in cfgs]
    k = 1 + pre[0]["frames"]["approach"] + pre[0]["frames"]["close"] + pre[0]["frames"]["hold"]
    for w in pre[1:]:
        assert np.array_equal(w["poses"][:k], pre[0]["poses"][:k])
    assert len({tuple(np.round(c["retract"]["direction"], 6)) for c in cfgs}) == 2
