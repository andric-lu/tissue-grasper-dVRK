"""
test_sampling_study.py -- the Item 3 samplers, statistics and study script.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "host"))

import graph_builder as gb  # noqa: E402
import graph_design as gd  # noqa: E402
import sampling_study as ss  # noqa: E402

CLOUD = ss.proxy_cloud(0, 4000)


class TestSamplers:
    @pytest.mark.parametrize("name", ["random", "fps", "voxel"])
    def test_unique_sorted_in_range_and_near_the_target(self, name):
        ids = ss.choose_nodes(name, CLOUD, 500, 0)
        assert np.array_equal(ids, np.unique(ids))
        assert ids.min() >= 0 and ids.max() < CLOUD.shape[0]
        assert abs(ids.size - 500) <= 25

    def test_voxel_is_more_even_than_random(self):
        """The point of grid/FPS sampling: fewer gaps. Coverage max is the
        worst-represented spot in the material."""
        rnd = gd.coverage_stats(CLOUD, CLOUD[ss.choose_nodes("random", CLOUD, 500, 0)])
        vox = gd.coverage_stats(CLOUD, CLOUD[ss.choose_nodes("voxel", CLOUD, 500, 0)])
        assert vox["max"] < rnd["max"]


class TestStatistics:
    def test_summary_counts_isolated_nodes_and_components(self):
        pos = np.array([[0, 0, 0], [0.01, 0, 0], [1, 1, 1]], float)
        s = gd.graph_summary(pos, gb.radius_pairs(pos, 0.02))
        assert s["isolated"] == 1 and s["components"] == 2 and s["n_edges"] == 2
        assert s["len_median"] == pytest.approx(0.01)

    def test_coverage_is_zero_when_every_particle_is_a_node(self):
        assert gd.coverage_stats(CLOUD[:300], CLOUD[:300])["max"] == pytest.approx(0, abs=1e-7)


class TestStudyScript:
    def test_runs_self_checks_and_recommends(self, tmp_path):
        rows, problems, frames = ss.study(CLOUD, 1 / 64, ["fps", "voxel"], [600],
                                          [1.5, 2.5], 0)
        assert not problems and frames == []
        graphs = {(r["sampler"], r["radius_dx"], r["graph"]) for r in rows}
        assert ("fps", 1.5, "reference") in graphs and ("voxel", 2.5, "combined") in graphs
        best, _, why = ss.recommend(rows, n_target=600)
        assert best is not None and "isolated" in why
        ss.write_tables(rows, str(tmp_path), frames, "test")
        assert (tmp_path / "sampling_study.md").read_text().count("\n") > 5

    def test_the_self_check_fires_when_components_cannot_be_counted(self, monkeypatch):
        """Broken-input demonstration: a component counter that always says 1
        must make the study report a problem."""
        monkeypatch.setattr(ss.gd, "connected_components",
                            lambda e, n: (1, np.zeros(n, int)))
        _, problems, _ = ss.study(CLOUD, 1 / 64, ["fps"], [300], [2.0], 0)
        assert problems and "cannot be trusted" in problems[0]
