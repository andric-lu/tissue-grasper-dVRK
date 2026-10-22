"""test_graph_design.py -- the Phase 0 arithmetic, checked against known answers.

src/graph_design.py is numpy only, so these run in the main environment and never
import taichi or torch. Every class includes at least one test on BROKEN or
degenerate input (CLAUDE.md, validation discipline): an influence radius that
has only been shown returning numbers has not been shown to refuse one.
"""

import os
import sys
from collections import deque

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from graph_design import (  # noqa: E402
    BELOW_NOISE,
    NO_SIGNAL,
    RESOLVED,
    SATURATED,
    connected_components,
    degree_stats,
    degrees,
    farthest_point_sample,
    hop_diameter,
    influence_profile,
    influence_radius,
    lattice_sample,
    median_nn_distance,
    radius_graph,
    required_message_passing_steps,
    surface_mask_from_bbox,
    surface_quota_fps,
)


def _slab(rng, n=4000):
    """Random cloud with init_cube's aspect ratio (0.3 x 0.3 x 0.05)."""
    return rng.random((n, 3)) * np.array([0.3, 0.3, 0.05])


def _cubic_lattice(k, pitch=1.0):
    ax = np.arange(k) * pitch
    return np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)


def _bfs_diameter(edge_index, n):
    adj = [[] for _ in range(n)]
    for s, r in edge_index.T:
        adj[s].append(r)
    best = 0
    for src in range(n):
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        best = max(best, max(dist.values()))
    return best


# --------------------------------------------------------------------------

class TestInfluenceRadius:
    def _field(self, rng, decay, n=20000, extent=0.3):
        d = rng.random(n) * extent
        return d, np.exp(-d / decay)

    def test_exponential_decay_gives_the_analytic_radius(self):
        # WHY: catches an off-by-one in binning or a first-below/farthest mixup.
        # exp(-d/L) falls to 1% at d = L ln 100 exactly; the reported radius is a
        # bin upper edge, so it must land within one bin above that.
        rng = np.random.default_rng(0)
        L, bw = 0.02, 0.002
        d, y = self._field(rng, L)
        res = influence_radius(influence_profile(d, y, bw), ref=1.0, frac=0.01)
        assert res["status"] == RESOLVED
        truth = L * np.log(100.0)
        assert truth <= res["radius"] <= truth + bw + 1e-12

    def test_noise_only_field_is_refused_not_measured(self):
        # WHY: the failure this whole status mechanism exists for. A kicked patch
        # that barely responds, measured against a run-to-run noise floor larger
        # than the 1% threshold, would otherwise produce a confident "radius"
        # that is really the spatial extent of GPU nondeterminism.
        rng = np.random.default_rng(1)
        d = rng.random(5000) * 0.3
        noise = 1e-4 * rng.random(5000)
        res = influence_radius(influence_profile(d, noise, 0.01),
                               ref=1e-3, frac=0.01, noise_floor=1e-4)
        assert res["status"] == BELOW_NOISE
        assert np.isnan(res["radius"])

    def test_signal_that_never_decays_is_a_lower_bound(self):
        # WHY: at 1/30 s the stiff material couples the whole slab. Reporting
        # "300 mm" as if influence stopped at the edge would understate it.
        rng = np.random.default_rng(2)
        d = rng.random(5000) * 0.3
        res = influence_radius(influence_profile(d, np.ones_like(d), 0.01),
                               ref=1.0, frac=0.01)
        assert res["status"] == SATURATED
        assert res["radius"] == pytest.approx(0.3, abs=0.01)

    def test_nothing_above_threshold_is_no_signal(self):
        rng = np.random.default_rng(3)
        d = rng.random(1000) * 0.3
        res = influence_radius(influence_profile(d, np.full_like(d, 1e-6), 0.01),
                               ref=1.0, frac=0.01)
        assert res["status"] == NO_SIGNAL and np.isnan(res["radius"])

    def test_far_reflection_counts(self):
        # WHY: a free-surface reflection can re-exceed the threshold far out. That
        # node IS influenced; a first-below-threshold rule would miss it.
        d = np.linspace(0, 0.3, 3001)
        y = np.where(d < 0.02, 1.0, 1e-4)
        y[(d > 0.25) & (d < 0.26)] = 0.05
        res = influence_radius(influence_profile(d, y, 0.005, r_max=0.3),
                               ref=1.0, frac=0.01)
        assert res["status"] == RESOLVED
        assert res["radius"] == pytest.approx(0.26, abs=0.005)

    def test_empty_bins_are_nan_not_zero(self):
        # WHY: empty-array means "not recorded" in this repo; the same rule for a
        # bin with no particles. Zero would claim "measured, nothing arrived".
        prof = influence_profile(np.array([0.001, 0.25]), np.array([1.0, 1.0]), 0.05)
        assert prof["count"][1] == 0 and np.isnan(prof["max"][1])

    def test_required_steps(self):
        assert required_message_passing_steps(0.10, 0.02) == 5
        assert required_message_passing_steps(0.101, 0.02) == 6
        with pytest.raises(ValueError):
            required_message_passing_steps(float("nan"), 0.02)


class TestSampling:
    def test_fps_is_unique_sorted_and_seeded(self):
        rng = np.random.default_rng(4)
        p = _slab(rng)
        a = farthest_point_sample(p, 300, seed=0)
        assert a.size == 300 and np.unique(a).size == 300
        assert np.all(np.diff(a) > 0)
        assert np.array_equal(a, farthest_point_sample(p, 300, seed=0))
        assert not np.array_equal(a, farthest_point_sample(p, 300, seed=1))

    def test_fps_covers_better_than_random(self):
        # WHY: catches an FPS that has silently degenerated (e.g. never updating
        # its distance array), which returns valid unique ids with random-quality
        # coverage and nothing else looks wrong.
        rng = np.random.default_rng(5)
        p = _slab(rng)

        def covering_radius(ids):
            d2 = ((p[:, None, :] - p[ids][None, :, :]) ** 2).sum(-1)
            return np.sqrt(d2.min(axis=1).max())

        fps = covering_radius(farthest_point_sample(p, 200, seed=0))
        rnd = covering_radius(rng.choice(p.shape[0], 200, replace=False))
        assert fps < 0.75 * rnd

    def test_n_larger_than_set_is_rejected(self):
        p = np.zeros((10, 3))
        with pytest.raises(ValueError):
            farthest_point_sample(p, 11)
        with pytest.raises(ValueError):
            lattice_sample(p, 11)

    def test_lattice_on_a_lattice_returns_every_point(self):
        p = _cubic_lattice(5)
        assert lattice_sample(p, 125).size == 125

    def test_lattice_count_is_near_request_on_the_slab(self):
        rng = np.random.default_rng(6)
        ids = lattice_sample(_slab(rng, 24000), 600)
        assert 0.6 * 600 <= ids.size <= 1.4 * 600

    def test_surface_mask_on_a_cube_is_the_outer_shell(self):
        p = _cubic_lattice(6)
        m = surface_mask_from_bbox(p, 0.5)
        assert m.sum() == 6 ** 3 - 4 ** 3

    def test_surface_quota_is_honoured(self):
        rng = np.random.default_rng(7)
        p = _slab(rng)
        m = surface_mask_from_bbox(p, 0.006)
        ids = surface_quota_fps(p, 400, m, surface_frac=0.5, seed=0)
        assert ids.size == 400 and np.unique(ids).size == 400
        assert m[ids].sum() == 200

    def test_surface_quota_rejects_a_wrong_mask(self):
        with pytest.raises(ValueError):
            surface_quota_fps(np.zeros((10, 3)), 5, np.zeros(9, bool), 0.5)


class TestGraph:
    def test_cubic_lattice_degrees(self):
        # WHY: catches self-loops, one-directional edges, or <= vs < at the
        # radius. Interior nodes of a unit lattice have 6 neighbours inside
        # r = 1.01 and 18 inside r = 1.5 (faces + edges), exactly.
        p = _cubic_lattice(5)
        interior = np.all((p > 0) & (p < 4), axis=1)
        e6 = radius_graph(p, 1.01)
        assert np.all(degrees(e6, len(p))[interior] == 6)
        e18 = radius_graph(p, 1.5)
        assert np.all(degrees(e18, len(p))[interior] == 18)
        assert not np.any(e18[0] == e18[1])
        fwd = set(map(tuple, e18.T))
        assert all((r, s) in fwd for s, r in fwd)

    def test_degree_stats(self):
        s = degree_stats(np.array([2, 2, 3]))
        assert s["min"] == 2 and s["hist"] == [0, 0, 2, 1]

    def test_two_clusters_are_two_components(self):
        p = np.vstack([_cubic_lattice(3), _cubic_lattice(3) + 10.0])
        k, labels = connected_components(radius_graph(p, 1.01), len(p))
        assert k == 2 and len(set(labels[:27])) == 1

    def test_radius_below_spacing_shatters_the_graph(self):
        # WHY: the broken-input case for the connectivity acceptance criterion --
        # a radius mistuned below the node pitch must read as n components, not 1.
        p = _cubic_lattice(4)
        k, _ = connected_components(radius_graph(p, 0.9), len(p))
        assert k == len(p)

    def test_path_and_cycle_diameters(self):
        n = 12
        path = np.array([[i, i + 1] for i in range(n - 1)] +
                        [[i + 1, i] for i in range(n - 1)]).T
        assert hop_diameter(path, n) == {"diameter": n - 1, "connected": True}
        cyc = np.array([[i, (i + 1) % n] for i in range(n)] +
                       [[(i + 1) % n, i] for i in range(n)]).T
        assert hop_diameter(cyc, n)["diameter"] == n // 2

    def test_disconnected_diameter_is_flagged(self):
        e = np.array([[0, 1], [1, 0]]).T
        res = hop_diameter(e, 3)
        assert res["connected"] is False

    def test_diameter_matches_python_bfs(self):
        rng = np.random.default_rng(8)
        p = rng.random((150, 3))
        e = radius_graph(p, 0.3)
        assert hop_diameter(e, 150)["diameter"] == _bfs_diameter(e, 150)

    def test_median_nn_distance_on_a_lattice(self):
        assert median_nn_distance(_cubic_lattice(4, pitch=0.02)) == pytest.approx(0.02)
