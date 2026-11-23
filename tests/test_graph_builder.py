"""
test_graph_builder.py -- the hybrid graph's construction rules (to-do Items 1,
2, 6), each shown failing on broken input before it is trusted on good input.

The property checks are written as small functions (`_assert_*`) so that each
can be pointed at a deliberately broken graph or feature function first: a
test that has only ever passed has not been shown to test anything (CLAUDE.md).
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

import graph_builder as gb  # noqa: E402
import tool_geometry as tg  # noqa: E402

RNG = np.random.default_rng(0)
N = 400
X = RNG.uniform(0, 0.1, (N, 3)) * [1, 1, 0.2]                 # a thin slab, metres
DISP = 0.002 * np.sin(X * 60)                                 # a smooth deformation
x = X + DISP
v = RNG.normal(0, 0.01, (N, 3))
F = np.tile(np.eye(3), (N, 1, 1)) + RNG.normal(0, 1e-3, (N, 3, 3))
MAT = np.tile([np.log(3e3), np.log(3e5), 1050.0], (N, 1))
R = 0.012


def _edge_set(ei):
    return set(map(tuple, np.asarray(ei).T.tolist()))


# --- property checks, and their demonstrations on broken input ---------------

def _assert_well_formed(ei, n):
    problems = gb.check_edge_index(ei, n)
    assert not problems, problems


class TestChecksCatchBrokenGraphs:
    def test_one_directional_edges_are_caught(self):
        ei = gb.radius_pairs(X, R)
        one_way = ei[:, ei[0] < ei[1]]
        assert any("reverse" in p for p in gb.check_edge_index(one_way, N))

    def test_self_loops_are_caught(self):
        ei = np.concatenate([gb.radius_pairs(X, R), [[3], [3]]], axis=1)
        assert any("self-loop" in p for p in gb.check_edge_index(ei, N))

    def test_out_of_range_indices_are_caught(self):
        assert gb.check_edge_index(np.array([[0, N], [N, 0]]), N)

    def test_an_absolute_position_feature_fails_the_translation_check(self):
        """The invariance test below must be able to fail: a feature leaking
        absolute position is exactly what it is for."""
        def leaky(ei, X_, x_):
            return np.concatenate([gb.reference_edge_features(ei, X_, x_), x_[ei[0]]], 1)
        ei = gb.reference_edges(X, R)
        shift = np.array([0.3, -0.2, 0.1])
        assert not np.allclose(leaky(ei, X, x), leaky(ei, X + shift, x + shift))


# --- the builder against those checks -----------------------------------------

class TestRadiusPairs:
    def test_well_formed_and_exactly_the_brute_force_set(self):
        ei = gb.radius_pairs(x, R)
        _assert_well_formed(ei, N)
        d = np.linalg.norm(x[:, None] - x[None], axis=-1)
        want = {(i, j) for i, j in zip(*np.nonzero(d < R)) if i != j}
        assert _edge_set(ei) == want

    def test_symmetric_even_for_pairs_at_the_radius(self):
        """Pairs placed within rounding of R: the decision must be the same in
        both directions (the expanded |a|^2-2ab+|b|^2 form alone can differ)."""
        rng = np.random.default_rng(5)
        a = rng.uniform(0.2, 0.4, (300, 3))
        u = rng.normal(size=(300, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        b = a + u * R * (1 + rng.uniform(-1e-12, 1e-12, (300, 1)))
        ei = gb.radius_pairs(np.concatenate([a, b]), R)
        _assert_well_formed(ei, 600)


class TestTissueEdges:
    def test_reference_connectivity_depends_only_on_X(self):
        e1 = gb.reference_edges(X, R)
        e2 = gb.reference_edges(X, R)       # x is not an input at all
        assert np.array_equal(e1, e2)
        f = gb.reference_edge_features(e1, X, x)
        assert f.shape == (e1.shape[1], 8)
        s, r = e1
        assert np.allclose(f[:, :3], X[r] - X[s])
        assert np.allclose(f[:, 3:6], x[r] - x[s])
        assert np.allclose(f[:, 7], np.linalg.norm(x[r] - x[s], axis=1))

    def test_spatial_features_shape_and_direction(self):
        e = gb.spatial_edges(x, R)
        f = gb.spatial_edge_features(e, x, v)
        assert f.shape == (e.shape[1], 7)
        s, r = e
        assert np.allclose(f[:, 4:], v[r] - v[s])           # receiver minus sender
        # The reverse edge carries the negated relative vectors.
        idx = {pair: k for k, pair in enumerate(map(tuple, e.T.tolist()))}
        k0 = 0
        k1 = idx[(int(e[1, k0]), int(e[0, k0]))]
        assert np.allclose(f[k0, :3], -f[k1, :3]) and np.isclose(f[k0, 3], f[k1, 3])

    def test_translation_leaves_edge_features_unchanged(self):
        shift = np.array([0.3, -0.2, 0.1])
        er, es = gb.reference_edges(X, R), gb.spatial_edges(x, R)
        er2, es2 = gb.reference_edges(X + shift, R), gb.spatial_edges(x + shift, R)
        assert np.array_equal(er, er2) and np.array_equal(es, es2)
        assert np.allclose(gb.reference_edge_features(er, X, x),
                           gb.reference_edge_features(er2, X + shift, x + shift))
        assert np.allclose(gb.spatial_edge_features(es, x, v),
                           gb.spatial_edge_features(es2, x + shift, v))
        # Node features see x - X, so a rigid shift of BOTH leaves them unchanged.
        assert np.allclose(gb.node_features(x, v, F, X, MAT),
                           gb.node_features(x + shift, v, F, X + shift, MAT))

    def test_permuting_nodes_permutes_edges_consistently(self):
        perm = np.random.default_rng(3).permutation(N)
        inv = np.argsort(perm)
        e = gb.spatial_edges(x, R)
        ep = gb.spatial_edges(x[perm], R)
        mapped = {(int(perm[i]), int(perm[j])) for i, j in ep.T}
        assert mapped == _edge_set(e)
        # Features follow their edge, not their row number.
        f = dict(zip(map(tuple, e.T.tolist()), gb.spatial_edge_features(e, x, v)))
        fp = gb.spatial_edge_features(ep, x[perm], v[perm])
        for k, (i, j) in enumerate(ep.T):
            assert np.allclose(fp[k], f[(int(perm[i]), int(perm[j]))])
        nf = gb.node_features(x, v, F, X, MAT)
        assert np.allclose(gb.node_features(x[perm], v[perm], F[perm], X[perm],
                                            MAT[perm])[inv], nf)

    def test_a_pair_inside_both_radii_appears_in_both_types(self):
        er = _edge_set(gb.reference_edges(X, R))
        es = _edge_set(gb.spatial_edges(x, R))
        both = er & es
        assert both, "fixture must contain pairs in both sets"
        # And neither set is a subset of the other here (they mean different
        # things), so a builder that merged them would be visible.
        assert er - es and es - er

    def test_reference_edges_survive_stretching_spatial_ones_do_not(self):
        er = gb.reference_edges(X, R)
        stretched = X * [2.0, 1.0, 1.0]                     # 100% stretch along x
        assert np.array_equal(gb.reference_edges(X, R), er)  # fixed by construction
        es = gb.spatial_edges(stretched, R)
        assert es.shape[1] < er.shape[1]                    # proximity edges vanish


class TestNodeFeatures:
    def test_widths_and_column_meaning(self):
        nf = gb.node_features(x, v, F, X, MAT, g_contact=np.ones(N), g_grasp=np.zeros(N))
        assert nf.shape == (N, 21) and len(gb.NODE_FEATURES) == 21
        assert np.allclose(nf[:, 0:3], x - X)
        assert np.allclose(nf[:, 6:15], (F - np.eye(3)).reshape(N, 9))
        assert np.allclose(nf[5, 6:15].reshape(3, 3), F[5] - np.eye(3))  # row-major
        assert np.allclose(nf[:, 15:18], MAT)
        assert (nf[:, 19] == 1).all() and (nf[:, 20] == 0).all()

    def test_omitting_g_drops_columns_rather_than_zeroing(self):
        assert gb.node_features(x, v, F, X, MAT, include_g=False).shape == (N, 19)
        assert gb.feature_widths(False)["node"] == 19


class TestToolEdges:
    BOX = np.array([[0.05, 0.05, 0.03, 0, 0, 0, 1.0], [0.05, 0.03, 0.03, 0, 0, 0, 1.0]])
    HALF = np.array([[0.0025, 0.00575, 0.00158]] * 2)
    VEL = np.array([[0.1, 0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, 0.0]])

    def test_edges_are_exactly_the_nodes_within_r_c(self):
        te = gb.tool_edges(x, v, self.BOX, self.VEL, self.HALF, 0.015)
        for k in range(2):
            d = tg.unsigned_box_distance(x, self.BOX[k], self.HALF[k])
            got = set(te["edge_index"][1, te["edge_index"][0] == k].tolist())
            assert got == set(np.flatnonzero(d < 0.015).tolist())
        assert te["features"].shape == (te["edge_index"].shape[1], len(gb.TOOL_EDGE_FEATURES))

    def test_feature_contents(self):
        te = gb.tool_edges(x, v, self.BOX, self.VEL, self.HALF, 0.015,
                           g_grasp=np.ones(N))
        k, i = te["edge_index"][:, 0]
        f = te["features"][0]
        near = tg.box_nearest_point(x[i:i + 1], self.BOX[k], self.HALF[k])
        assert np.allclose(f[:3], x[i] - near["nearest"][0])
        assert np.isclose(f[3], near["signed_distance"][0])
        v_surf = self.VEL[k, :3] + np.cross(self.VEL[k, 3:], near["nearest"][0] - self.BOX[k, :3])
        assert np.allclose(f[7:10], v[i] - v_surf)
        assert f[10 + k] == 1 and f[13] == 1

    def test_no_edges_when_the_jaw_is_far(self):
        far = self.BOX.copy()
        far[:, 2] += 1.0
        te = gb.tool_edges(x, v, far, self.VEL, self.HALF, 0.015)
        assert te["edge_index"].shape == (2, 0)

    def test_tool_node_features(self):
        a = np.arange(7.0)
        tn = gb.tool_node_features(self.BOX, self.VEL, np.array([0.5, -0.5]), a, self.HALF)
        assert tn.shape == (2, len(gb.TOOL_NODE_FEATURES))
        assert np.allclose(tn[:, 15:22], a)                 # commanded action, both jaws
        assert np.allclose(tn[:, 9:15], [1, 0, 0, 0, 1, 0])  # identity orientation, 6D


class TestEpisodeGraphBuilder:
    def test_builds_every_row_of_a_v3_episode(self, tmp_path):
        from test_schema_v3 import _episode
        from trajectory_io import load_trajectory  # noqa: F401
        tr = _episode(str(tmp_path / "e.npz"))
        b = gb.EpisodeGraphBuilder(tr, r_ref=0.02, r_spatial=0.02, r_c=0.02)
        for t in range(len(tr)):
            g = b.graph(t)
            n = tr.n_nodes
            assert g["node_features"].shape == (n, 21)
            _assert_well_formed(g["ref_edge_index"], n)
            _assert_well_formed(g["spatial_edge_index"], n)
            assert g["tool_node_features"].shape == (2, len(gb.TOOL_NODE_FEATURES))
            assert np.isfinite(g["tool_edge_features"]).all()
            # The node's g columns are the stored node-level flags.
            assert np.array_equal(g["node_features"][:, 20], tr.node_grasp[t])
        # Reference connectivity is the same object every row: fixed.
        assert b.graph(0)["ref_edge_index"] is b.graph(len(tr) - 1)["ref_edge_index"]

    def test_refuses_a_file_without_a_reference_configuration(self, tmp_path):
        import numpy as np_
        from test_schema_v3 import _episode
        from trajectory_io import load_trajectory
        _episode(str(tmp_path / "a.npz"))
        with np_.load(tmp_path / "a.npz") as z:
            d = {k: z[k] for k in z.files}
        d["rest_pos"] = np_.zeros((0, 3), np_.float32)
        np_.savez(tmp_path / "b.npz", **d)
        with pytest.raises(ValueError, match="frame 0"):
            gb.EpisodeGraphBuilder(load_trajectory(str(tmp_path / "b.npz")),
                                   r_ref=0.02, r_spatial=0.02, r_c=0.02)
