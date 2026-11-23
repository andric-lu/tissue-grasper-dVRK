"""
test_collection_config.py -- the scripted grasp as data (to-do Item 6).

The one test that matters most here is the equivalence with the section 10
script: the old hardcoded waypoints are what smoke_test_grasp.py verified on
the Mac, and the config path must reproduce them, or "moved into a config" has
quietly become "changed the experiment".
"""

import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

import collection_config as cc  # noqa: E402


def _section10_waypoints(n_approach, n_close, n_retract, n_release):
    """mpm_adapter._scripted_waypoints as it was before schema 3.0, verbatim
    in its arithmetic (copied here because the function was replaced)."""
    quat = np.array([0.7048, -0.7048, -0.0576, 0.0576])
    hover = np.array([0.274, 0.35, 0.226])
    grasp_pos = np.array([0.252, 0.35, 0.088])
    poses, jaws = [np.concatenate([hover, quat])], [1.0]
    for i in range(1, n_approach + 1):
        t = i / n_approach
        poses.append(np.concatenate([hover * (1 - t) + grasp_pos * t, quat]))
        jaws.append(1.0)
    for i in range(1, n_close + 1):
        t = i / n_close
        poses.append(np.concatenate([grasp_pos, quat]))
        jaws.append(1.0 * (1 - t))
    for i in range(1, n_retract + 1):
        t = i / n_retract
        poses.append(np.concatenate([grasp_pos * (1 - t) + hover * t, quat]))
        jaws.append(0.0)
    for i in range(1, n_release + 1):
        t = i / n_release
        poses.append(np.concatenate([hover, quat]))
        jaws.append(1.0 * t)
    return np.array(poses), np.array(jaws)


class TestSection10Equivalence:
    @pytest.mark.parametrize("n_release", [0, 10])
    def test_default_geometry_at_80hz_reproduces_the_verified_script(self, n_release):
        cfg = cc.load_config(None)
        cfg["frame_dt_s"] = 1 / 80
        cfg["preroll_s"] = 0.0
        cfg["release"]["duration_s"] = n_release / 80
        wp = cc.build_waypoints(cfg)
        old_p, old_j = _section10_waypoints(20, 10, 20, n_release)
        assert wp["poses"].shape == old_p.shape
        # Positions: identical to float rounding (hover is reconstructed as
        # grasp + distance * unit direction).
        assert np.abs(wp["poses"][:, :3] - old_p[:, :3]).max() < 1e-12
        # Orientation: the config normalises the quaternion; the old literal had
        # norm 1.000058. Same rotation, 6e-5 apart in components.
        assert np.abs(wp["poses"][:, 3:] - old_p[:, 3:]).max() < 1e-4
        assert np.allclose(wp["jaw"], old_j)

    def test_the_same_physical_motion_at_240hz_has_three_times_the_rows(self):
        cfg = cc.load_config(None)
        cfg["preroll_s"] = 0.0
        wp240 = cc.build_waypoints(cfg)
        cfg["frame_dt_s"] = 1 / 80
        wp80 = cc.build_waypoints(cfg)
        assert len(wp240["poses"]) - 1 == 3 * (len(wp80["poses"]) - 1)
        assert wp240["speeds"]["approach_m_s"] == pytest.approx(wp80["speeds"]["approach_m_s"])


class TestConfigFile:
    def test_unknown_keys_are_refused(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"approach": {"duraton_s": 0.1}}))
        with pytest.raises(KeyError, match="duraton_s"):
            cc.load_config(str(p))

    def test_non_unit_direction_is_refused(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"retract": {"direction": [0, 0, 2]}}))
        with pytest.raises(ValueError, match="unit"):
            cc.load_config(str(p))

    def test_shipped_configs_load(self):
        for name in ("grasp_pilot.json", "grasp_pilot_classic.json", "grasp_debug_set.json"):
            path = os.path.join(_ROOT, "configs", name)
            cfg = cc.load_config(path)
            wp = cc.build_waypoints(cfg)
            assert np.isfinite(wp["poses"]).all()


class TestRandomisation:
    def test_draws_are_deterministic_and_vary_with_seed(self):
        base = cc.load_config(None)
        base["randomize"] = dict(cc.DEBUG_SET_RANDOMIZE)
        a, b = cc.sample_episode_config(base, 1), cc.sample_episode_config(base, 1)
        c = cc.sample_episode_config(base, 2)
        assert a == b and a != c

    def test_draws_stay_inside_the_declared_ranges(self):
        base = cc.load_config(None)
        base["randomize"] = dict(cc.DEBUG_SET_RANDOMIZE)
        g0 = np.array(base["grasp_position_m"])
        d0 = np.array(base["retract"]["direction"])
        for seed in range(200):
            c = cc.sample_episode_config(base, seed)
            off = np.array(c["grasp_position_m"]) - g0
            assert np.all(np.abs(off[:2]) <= 0.02 + 1e-12) and off[2] == 0
            ang = np.degrees(np.arccos(np.clip(np.dot(c["retract"]["direction"], d0), -1, 1)))
            assert ang <= 20.0 + 1e-9
            assert 0.01 <= c["retract"]["distance_m"] <= 0.03

    def test_retract_speed_is_held_not_duration(self):
        base = cc.load_config(None)
        base["randomize"] = {"retract_distance_m": [0.06, 0.06]}
        c = cc.sample_episode_config(base, 0)
        v0 = base["retract"]["distance_m"] / base["retract"]["duration_s"]
        assert c["retract"]["distance_m"] / c["retract"]["duration_s"] == pytest.approx(v0)


class TestMatchedVariants:
    def test_only_the_retraction_changes(self):
        base = cc.sample_episode_config(
            dict(cc.load_config(None), randomize=dict(cc.DEBUG_SET_RANDOMIZE)), 4)
        vs = cc.matched_variants(base, [[0, 0, 1], [1, 0, 1]], [0.05, 0.1])
        assert len(vs) == 4
        for v in vs:
            assert v["seed"] == base["seed"]
            for k in ("grasp_position_m", "preroll_s", "approach", "close", "hold",
                      "orientation_xyzw", "frame_dt_s"):
                assert v[k] == base[k], k
        # The state up to the start of retraction is therefore identical.
        w = [cc.build_waypoints(v) for v in vs]
        k_pre = 1 + w[0]["frames"]["approach"] + w[0]["frames"]["close"] + w[0]["frames"]["hold"]
        for x in w[1:]:
            assert np.array_equal(x["poses"][:k_pre], w[0]["poses"][:k_pre])
            assert np.array_equal(x["jaw"][:k_pre], w[0]["jaw"][:k_pre])
