"""
test_visualize_graph.py -- Item 4's checklist must be able to FAIL.

matplotlib is a host dependency, so the drawing test skips where it is absent
(the container); the checklist logic is exercised either way.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "host"))

import graph_builder as gb  # noqa: E402
import visualize_graph as vg  # noqa: E402
from test_schema_v3 import _episode  # noqa: E402


def _items(tr, r):
    view = vg.NodeView(tr)
    ref = gb.reference_edges(view.X, r)
    stats = vg.per_row_stats(view, ref, r, r, 0.02)
    return {c: s for s, c, _ in vg.checklist(view, ref, stats, r, r, 0.02)}


def test_checklist_passes_fragmentation_on_a_connected_graph(tmp_path):
    tr = _episode(str(tmp_path / "e.npz"))
    assert _items(tr, 0.03)["no material fragmentation"] == "PASS"


def test_checklist_fails_fragmentation_when_the_radius_is_too_small(tmp_path):
    """Broken input: a radius below the node spacing fragments the material,
    and the checklist must say FAIL, not PASS."""
    tr = _episode(str(tmp_path / "e.npz"))
    assert _items(tr, 0.002)["no material fragmentation"] == "FAIL"


def test_main_writes_every_deliverable(tmp_path):
    pytest.importorskip("matplotlib")
    tr_path = str(tmp_path / "e.npz")
    _episode(tr_path)
    out = tmp_path / "viz"
    vg.main([tr_path, "--r-ref", "0.03", "--r-spatial", "0.03", "--out", str(out)])
    names = set(os.listdir(out))
    assert {"checklist.md", "stats_over_time.csv", "stats_over_time.png",
            "rows_rest.png"} <= names


def test_checklist_fails_fragmentation_on_a_torn_strand(tmp_path):
    """The first grasp pilot pulled a 12-particle strand 14 cm out of the slab
    and the old check PASSed it, because it tested the combined graph -- which
    contains the fixed reference edges and so can never come apart. Pull a
    few nodes far away and the check must FAIL."""
    import numpy as np_
    from test_schema_v3 import _episode
    from trajectory_io import load_trajectory
    _episode(str(tmp_path / "a.npz"))
    with np_.load(tmp_path / "a.npz") as z:
        d = {k: z[k] for k in z.files}
    pos = d["tissue_pos"].copy()
    pos[-3:, :3, 2] += 0.5                 # three nodes yanked 50 cm up
    d["tissue_pos"] = pos
    np_.savez(tmp_path / "b.npz", **d)
    tr = load_trajectory(str(tmp_path / "b.npz"))
    assert _items(tr, 0.03)["no material fragmentation"] == "FAIL"
