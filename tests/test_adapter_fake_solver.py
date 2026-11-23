"""
test_adapter_fake_solver.py -- host/mpm_adapter.py's bookkeeping, without Taichi.

WHY THIS FILE EXISTS
--------------------
mpm_adapter.py can only really run on the Mac (Taichi + Metal + PyBullet). The
schema-3.0 work changed a lot of its PLUMBING -- rest positions, pre-roll,
fixed node sets, per-jaw arrays, node flags, crash handling, the config-driven
grasp script -- none of which depends on the physics being right. So this file
swaps in numpy stand-ins for the solver (`MPM.mpm3d`), `taichi`, the arm
(`psm`) and the grasp constraint, runs the REAL adapter code paths end to end,
and puts every episode through the REAL validator.

What it can NOT tell you: anything about the physics, the IK, the SDF contact,
or whether the real GraspConstraint fires. Those stay with
host/smoke_test_mpm.py, smoke_test_psm.py and smoke_test_grasp.py on the Mac.
A PASS here means "the adapter writes what the schema says it writes".
"""

import os
import sys
import types

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "host"))

import actions  # noqa: E402
import collection_config  # noqa: E402
import tool_geometry  # noqa: E402
from trajectory_io import load_trajectory  # noqa: E402

N_SIM = 24000   # the real cloud size, so subset-density checks judge it fairly


# --------------------------------------------------------------------------
# Stand-ins
# --------------------------------------------------------------------------

class _Field:
    def __init__(self, arr):
        self.a = np.asarray(arr)

    def to_numpy(self):
        return self.a.copy()

    def fill(self, v):
        self.a[...] = v

    def from_numpy(self, arr):
        self.a[...] = arr

    def __getitem__(self, k):
        return self.a[()] if k is None else self.a[k]

    def __setitem__(self, k, v):
        if k is None:
            self.a[()] = v
        else:
            self.a[k] = v


def _make_mpm3d(fail_at_substep=None):
    """A fake solver: particles fall under gravity onto a per-particle floor
    3 mm below where they spawned, and F drifts slightly so it is not I."""
    m = types.ModuleType("MPM.mpm3d")
    m.n_particles = N_SIM
    m.dx = 1 / 64
    m.p_vol = (m.dx * 0.5) ** 3
    m.p_rho, m.p_mass, m.gravity = 1000.0, m.p_vol * 1000.0, 9.8
    m.dt, m.steps, m.timestep = 5e-4, 25, 0.0125
    for name in ("mu", "la", "E", "nu"):
        setattr(m, name, _Field(np.zeros(())))
    m.F_x = _Field(np.zeros((N_SIM, 3), np.float32))
    m.F_v = _Field(np.zeros((N_SIM, 3), np.float32))
    m.F = _Field(np.zeros((N_SIM, 3, 3), np.float32))
    m.F_C = _Field(np.zeros((N_SIM, 3, 3), np.float32))
    m.F_grid_m = _Field(np.zeros(1))
    m.F_grid_v = _Field(np.zeros((1, 3)))
    m.sdf = types.SimpleNamespace(tmp_sdf=_Field(np.zeros(1)), co_mask=_Field(np.zeros(1)))
    m.set_base_position = lambda p: None
    m.init_collision_field = lambda: None
    m.init_pos = lambda: None
    m._count = {"substeps": 0}

    def init_cube():
        rng = np.random.default_rng(1234)          # "Taichi's fixed RNG"
        x = rng.random((N_SIM, 3)) * [0.3, 0.3, 0.05] + [0.1, 0.2, 0.05]
        m.F_x.a[:] = x
        m._floor = (x[:, 2] - 0.003).astype(np.float32)

    def init_F():
        m.F.a[:] = np.eye(3, dtype=np.float32)

    def substep(scale, threshold):
        m._count["substeps"] += 1
        if fail_at_substep is not None and m._count["substeps"] >= fail_at_substep:
            raise RuntimeError("fake solver diverged")
        dt = m.dt
        m.F_v.a[:, 2] -= dt * m.gravity
        m.F_x.a[:] += dt * m.F_v.a
        hit = m.F_x.a[:, 2] < m._floor
        m.F_x.a[hit, 2] = m._floor[hit]
        m.F_v.a[hit] = 0.0
        m.F.a[:, 2, 2] -= 1e-4 * dt

    m.init_cube, m.init_deformation_gradient, m.substep = init_cube, init_F, substep
    return m


def _make_taichi():
    ti = types.ModuleType("taichi")
    ti.sync = lambda: None
    cfg = types.SimpleNamespace(arch="metal")
    ti.lang = types.SimpleNamespace(impl=types.SimpleNamespace(current_cfg=lambda: cfg))
    return ti


def _make_psm():
    psm = types.ModuleType("psm")
    psm.JAW_LINKS = ("jaw_a", "jaw_b")
    psm.DEFAULT_BASE_POSITION = (0, 0, 0)
    psm.DEFAULT_BASE_ORIENTATION = (0, 0, 0, 1)
    psm.DRIVEN_JOINTS = tuple(f"j{i}" for i in range(7))
    psm.rotmat_from_quat = tool_geometry.rotmat_from_quat
    half = np.array([0.0025, 0.00575, 0.00158])

    class PSM:
        jaw_links = psm.JAW_LINKS
        model_name = "dvrk_si_psm"
        scaling = 1.0

        def __init__(self, m, sdf, base_position, base_orientation):
            self.pose, self.jaw = np.array([0, 0, 1, 0, 0, 0, 1.0]), 1.0

        def jaw_joint_values(self, jaw):
            return np.array([jaw, -jaw])

        def set_ee_pose(self, pose, jaw):
            self.pose, self.jaw = np.asarray(pose, float), float(jaw)

        def jaw_box_poses(self):
            R = tool_geometry.rotmat_from_quat(self.pose[3:])
            sep = 0.004 + 0.03 * self.jaw     # open jaws are far apart: no capture
            return {n: (self.pose[:3] + R @ np.array([0, s * sep, 0]), self.pose[3:].copy())
                    for n, s in zip(psm.JAW_LINKS, (1, -1))}

        def jaw_half_extents(self, name):
            return half

        def jaw_box_offset(self, name):
            return np.array([0, 0, 0, 0, 0, 0, 1.0])

        def joint_positions(self):
            return np.zeros(7, np.float32)

        def update_collision_geometry(self, m, sdf):
            pass

        def update_collision_velocity(self, m, frame_dt):
            pass

        def close(self):
            pass

    psm.PSM = PSM
    return psm


def _make_grasp_constraint():
    gc = types.ModuleType("grasp_constraint")
    gc.GRASP_CAPTURE_RADIUS_M = tool_geometry.GRASP_CAPTURE_RADIUS_M
    gc.JAW_RELEASE_TRIGGER_RAD = tool_geometry.JAW_RELEASE_TRIGGER_RAD

    class GraspConstraint:
        """The real class's selection/latch logic in numpy, with rigid follow."""

        def __init__(self, m, ti):
            self.m, self._active, self._full = m, False, np.zeros(0, np.int32)

        def is_active(self):
            return self._active

        def full_ids(self):
            return self._full.copy()

        def maybe_freeze(self, rec, robot, pose_t, jaw_angle):
            if self._active:
                return
            pos = rec.state()[0].astype(np.float64)
            boxes = robot.jaw_box_poses()
            bp = np.stack([np.concatenate(boxes[n]) for n in robot_links()])
            half = np.stack([robot.jaw_half_extents(n) for n in robot_links()])
            sel = tool_geometry.grasp_capture_candidates(pos, bp, half)
            ids = np.flatnonzero(sel).astype(np.int32)
            if ids.size == 0:
                return
            R0 = tool_geometry.rotmat_from_quat(pose_t[3:7])
            self._r = (pos[ids] - pose_t[:3]) @ R0
            self._full, self._active = ids, True

        def maybe_release(self, jaw_angle):
            if self._active and jaw_angle >= gc.JAW_RELEASE_TRIGGER_RAD:
                self._full, self._active = np.zeros(0, np.int32), False

        def apply(self, pose_start, pose_end, frame_dt, alpha):
            if not self._active:
                return
            d = actions.pose_delta(pose_start, pose_end)
            p = actions.apply_pose_delta(pose_start, alpha * d)
            R = tool_geometry.rotmat_from_quat(p[3:7])
            self.m.F_x.a[self._full] = p[:3] + self._r @ R.T
            self.m.F_v.a[self._full] = d[:3] / frame_dt

        def recorded_node_ids(self, particle_ids):
            if self._full.size == 0:
                return np.zeros(0, np.int32)
            idx = np.clip(np.searchsorted(particle_ids, self._full), 0, len(particle_ids) - 1)
            return idx[particle_ids[idx] == self._full].astype(np.int32)

    def robot_links():
        return sys.modules["psm"].JAW_LINKS

    gc.GraspConstraint = GraspConstraint
    return gc


@pytest.fixture
def adapter(monkeypatch):
    """Import host/mpm_adapter.py against the stand-ins; restore sys.modules after."""
    def install(fail_at_substep=None):
        mpm3d = _make_mpm3d(fail_at_substep)
        pkg = types.ModuleType("MPM")
        pkg.mpm3d = mpm3d
        monkeypatch.setitem(sys.modules, "MPM", pkg)
        monkeypatch.setitem(sys.modules, "MPM.mpm3d", mpm3d)
        monkeypatch.setitem(sys.modules, "taichi", _make_taichi())
        monkeypatch.setitem(sys.modules, "psm", _make_psm())
        monkeypatch.setitem(sys.modules, "grasp_constraint", _make_grasp_constraint())
        monkeypatch.delitem(sys.modules, "mpm_adapter", raising=False)
        import mpm_adapter
        # The logged exposure metric costs ~0.5 s per row over 24,000
        # particles. Its correctness is tested in test_tissue_metrics.py, not
        # here, so the adapter logs 0.0 ("fully occluded"), a value the
        # validator's subset inequality accepts only if the stored 3,000-node
        # subset really does cover the target -- so the check still runs.
        monkeypatch.setattr(mpm_adapter.tissue_metrics, "compute_exposure",
                            lambda *a, **k: 0.0)
        return mpm_adapter
    yield install
    sys.modules.pop("mpm_adapter", None)


def _short_config(**over):
    cfg = collection_config.load_config(None)
    cfg["preroll_s"] = 0.05
    cfg["approach"]["duration_s"] = 0.05
    cfg["close"]["duration_s"] = 0.025
    cfg["hold"]["duration_s"] = 0.0125
    cfg["retract"]["duration_s"] = 0.05
    cfg["release"]["duration_s"] = 0.025
    for k, v in over.items():
        cfg[k] = v
    return cfg


def _validate(path):
    import validate_dataset as vd
    tr = load_trajectory(path)
    return tr, {fn.__name__: fn(tr) for fn in vd.PER_EPISODE}


IDS = np.sort(np.random.default_rng(7).choice(N_SIM, 3000, replace=False))


# --------------------------------------------------------------------------

class TestSettleEpisode:
    def test_settle_writes_a_v3_episode_that_validates(self, adapter, tmp_path):
        ad = adapter()
        out = ad.record_episode(str(tmp_path / "s.npz"), n_steps=12, seed=1,
                                frame_dt=1 / 240, f_encoding="float32",
                                particle_ids=IDS, quiet=True)
        tr, res = _validate(out)
        fails = {k: r.message for k, r in res.items() if r.status == "FAIL"}
        assert not fails, fails
        assert tr.schema_version == "3.0" and len(tr) == 12
        assert np.array_equal(tr.particle_ids, IDS)
        # Row 0 IS the reference: no pre-roll, x = X, F = I.
        assert float(tr.preroll_s[0]) == 0.0
        assert np.array_equal(tr.tissue_pos[0], tr.rest_pos)
        assert res["check_reference_configuration"].status == "PASS"
        assert tr.rest_pos_simulated.shape == (N_SIM, 3)
        assert np.allclose(tr.time, np.arange(12) / 240)
        assert tr.material_field.shape == (3000, 3)

    def test_the_same_ids_give_the_same_nodes_in_every_episode(self, adapter, tmp_path):
        a = load_trajectory(adapter().record_episode(
            str(tmp_path / "a.npz"), n_steps=3, seed=1, particle_ids=IDS, quiet=True))
        b = load_trajectory(adapter().record_episode(
            str(tmp_path / "b.npz"), n_steps=3, seed=2, particle_ids=IDS, quiet=True))
        assert np.array_equal(a.particle_ids, b.particle_ids)
        assert np.array_equal(a.rest_pos, b.rest_pos)
        assert not np.array_equal(a.material_params, b.material_params)

    def test_a_crash_is_kept_as_partial_and_never_as_npz(self, adapter, tmp_path):
        ad = adapter(fail_at_substep=40)
        target = tmp_path / "crash.npz"
        with pytest.raises(RuntimeError, match="diverged"):
            ad.record_episode(str(target), n_steps=50, seed=1, frame_dt=1 / 240,
                              particle_ids=IDS, quiet=True)
        assert not target.exists()
        partial = tmp_path / "crash.npz.partial"
        assert partial.exists()
        with np.load(partial) as z:
            assert "diverged" in str(z["partial_reason"])
        # And the loader's glob does not see it.
        from trajectory_io import list_trajectories
        assert list_trajectories(str(tmp_path)) == []


class TestGraspEpisode:
    # One episode serves both tests below: the stand-in is cheap, but the
    # adapter's logged exposure metric over 24,000 particles costs ~0.5 s per
    # row (the same cost the real collector pays).
    _cache = {}

    def _episode(self, adapter, tmp_path):
        if "path" not in self._cache:
            out = adapter().record_grasp_episode(
                str(tmp_path / "g.npz"), config=_short_config(), seed=3,
                f_encoding="float32", particle_ids=IDS, quiet=True)
            self._cache["path"] = out
        return self._cache["path"]

    def test_config_driven_grasp_validates_with_no_fail(self, adapter, tmp_path):
        cfg = _short_config()
        out = self._episode(adapter, tmp_path)
        tr, res = _validate(out)
        fails = {k: r.message for k, r in res.items() if r.status == "FAIL"}
        assert not fails, fails
        wp = collection_config.build_waypoints(cfg)
        assert len(tr) == len(wp["poses"])
        assert np.array_equal(tr.phase, wp["phase"])
        # The pre-roll ran and is recorded; row 0 is therefore NOT the reference.
        assert abs(float(tr.preroll_s[0]) - 0.05) < 1e-12
        assert not np.array_equal(tr.tissue_pos[0], tr.rest_pos)
        assert tr.has_jaws and tr.jaw_box_pose.shape == (len(tr), 2, 7)
        for name in ("check_jaw_geometry", "check_node_indicators",
                     "check_action_state_indexing", "check_full_resolution_grasp",
                     "check_positive_time_intervals"):
            assert res[name].status == "PASS", (name, res[name].message)

    def test_the_grasp_actually_happens_in_the_stand_in(self, adapter, tmp_path):
        """Not a physics claim -- a check that the test above exercised the
        grasp branches rather than passing vacuously on an empty grasp."""
        tr = load_trajectory(self._episode(adapter, tmp_path))
        assert tr.grasp_active.any()
        assert tr.node_grasp.any() and tr.node_contact.any()
        assert sum(tr.grasp_full_ids(t).size for t in range(len(tr))) > 0
        # Released by the end (release phase reopens the jaw).
        assert not tr.grasp_active[-1] and not tr.node_grasp[-1].any()

    def test_legacy_frame_count_call_still_works(self, adapter, tmp_path):
        """The smoke tests' call form: 80 Hz frame counts, no pre-roll."""
        out = adapter().record_grasp_episode(
            str(tmp_path / "l.npz"), n_steps=12, seed=3, n_approach=4, n_close=3,
            n_retract=3, n_release=2, quiet=True, n_record=3000)
        tr = load_trajectory(out)
        assert len(tr) == 13 and abs(float(tr.dt) - 0.0125) < 1e-9
        assert float(tr.preroll_s[0]) == 0.0

    def test_config_and_legacy_counts_together_are_refused(self, adapter, tmp_path):
        with pytest.raises(ValueError, match="either config"):
            adapter().record_grasp_episode(str(tmp_path / "x.npz"),
                                           config=_short_config(), n_approach=5)


class TestCommandLine:
    def test_cli_grasp_writes_episode_config_and_log(self, adapter, tmp_path):
        import json
        ad = adapter()
        cfg_path = tmp_path / "c.json"
        cfg_path.write_text(json.dumps({
            "preroll_s": 0.0, "approach": {"duration_s": 3 / 240},
            "close": {"duration_s": 2 / 240}, "retract": {"duration_s": 2 / 240},
            "release": {"duration_s": 0.0}}))
        ids_path = tmp_path / "ids.npy"
        np.save(ids_path, IDS)
        out = tmp_path / "data"
        assert ad.main(["--task", "grasp", "--config", str(cfg_path), "--out", str(out),
                        "--particle-ids", str(ids_path), "--seed", "5"]) == 0
        assert (out / "mpm_0000.npz").exists()
        saved = json.loads((out / "mpm_0000.config.json").read_text())
        assert saved["seed"] == 5 and saved["frame_dt_s"] == pytest.approx(1 / 240)
        assert "--particle-ids" in (out / "collection_log.txt").read_text()

    def test_cli_refuses_a_frame_dt_that_contradicts_the_config(self, adapter, tmp_path):
        ad = adapter()
        with pytest.raises(SystemExit):
            ad.main(["--task", "grasp", "--out", str(tmp_path), "--frame-dt", "0.0125"])
