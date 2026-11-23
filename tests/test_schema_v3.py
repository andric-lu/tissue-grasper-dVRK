"""
test_schema_v3.py -- schema 3.0: the writer's integrity fixes and the new
validator checks, each demonstrated against data broken on purpose.

CLAUDE.md: "A check only shown passing has not been shown to work." Every new
check in host/validate_dataset.py appears here twice: once PASSing on a
correct episode, once FAILing on the specific corruption its `# WHY:` comment
names.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "host"))

import actions  # noqa: E402
import tool_geometry as tg  # noqa: E402
import validate_dataset as vd  # noqa: E402
from trajectory_io import (  # noqa: E402
    TrajectoryWriter,
    list_trajectories,
    load_trajectory,
)

N, N_SIM, DT = 30, 90, 1 / 240
HALF = np.array([[0.0025, 0.00575, 0.00158]] * 2)
MAT = np.array([np.log(3000.0), np.log(3e5), 1050.0], np.float32)


def _episode(path, *, T=10, preroll=0.0, corrupt=None, **writer_kw):
    """A small, fully consistent schema-3.0 grasp-like episode.

    `corrupt` names one thing to break, AFTER the episode is built
    consistently, so each validator test changes exactly one property.
    """
    rng = np.random.default_rng(0)
    full_rest = rng.uniform(0, 0.05, (N_SIM, 3)).astype(np.float32)
    ids = np.sort(rng.choice(N_SIM, N, replace=False)).astype(np.int32)
    X = full_rest[ids]
    # EE moves down 1 mm per row toward the cloud's centre; jaws close then open.
    quat = np.array([0, 0, 0, 1.0])
    ee = np.array([np.concatenate([[0.025, 0.025, 0.06 - 0.001 * t], quat])
                   for t in range(T)])
    jaw = np.clip(np.linspace(1.0, -0.2, T), 0, 1)
    jaw[-2:] = 0.9
    actions_ = np.zeros((T, 7))
    for t in range(T - 1):
        actions_[t] = actions.encode_action(ee[t], ee[t + 1], jaw[t + 1])
    boxes = np.stack([np.stack([np.concatenate([ee[t, :3] + [0, s * (0.004 + 0.01 * jaw[t]), 0],
                                                quat]) for s in (1, -1)])
                      for t in range(T)]).astype(np.float32)
    vel = tg.box_velocity_backward(boxes, DT)
    pos = np.stack([X + (0 if (t == 0 and preroll == 0) else 0.0005 * (t + 1))
                    for t in range(T)]).astype(np.float32)
    F = np.tile(np.eye(3, dtype=np.float32), (T, N, 1, 1))
    F[1:, :, 0, 0] += 0.001
    if preroll > 0:
        F[0, :, 0, 0] += 0.001
    half32 = HALF.astype(np.float32).astype(np.float64)
    contact = tg.derive_node_contact(pos.astype(np.float64), boxes.astype(np.float64), half32)
    grasp = tg.derive_node_grasp(pos.astype(np.float64), boxes.astype(np.float64), half32, jaw)
    active = grasp.any(axis=1)
    full_held = [ids[np.flatnonzero(grasp[t])] if active[t] else np.zeros(0, np.int32)
                 for t in range(T)]
    node_ids = [np.flatnonzero(grasp[t]) for t in range(T)]

    if corrupt == "action_off_by_one":
        actions_[1:-1] = actions_[:-2].copy()
    if corrupt == "forward_velocity":
        vel = np.concatenate([vel[1:], vel[-1:]])
    if corrupt == "flags":
        contact = contact.copy()
        contact[3, 0] = ~contact[3, 0]

    kw = dict(material_params=MAT, particle_ids=ids, n_particles_simulated=N_SIM,
              rest_pos=X, rest_pos_simulated=full_rest,
              material_field=np.tile(MAT, (N, 1)), control_dt=DT, preroll_s=preroll,
              jaw_names=["a", "b"], jaw_half_extents=HALF,
              jaw_box_offset=np.tile([0, 0, 0, 0, 0, 0, 1.0], (2, 1)),
              grasp_capture_radius=tg.GRASP_CAPTURE_RADIUS_M,
              jaw_release_rad=tg.JAW_RELEASE_TRIGGER_RAD,
              contact_radius=tg.CONTACT_RADIUS_M, action_spec="delta_pose_jaw",
              f_encoding="float32")
    kw.update(writer_kw)
    with TrajectoryWriter(path, "test_mpm", "tissue_grasp", DT, **kw) as w:
        for t in range(T):
            w.append(tissue_pos=pos[t], tissue_vel=np.full((N, 3), 0.0005 / DT),
                     ee_pose=ee[t], action=actions_[t], jaw=jaw[t], tissue_F=F[t],
                     time=t * DT, jaw_box_pose=boxes[t], jaw_box_vel=vel[t],
                     jaw_joint=[jaw[t], -jaw[t]], node_contact=contact[t],
                     node_grasp=grasp[t], grasp_active=bool(active[t]),
                     grasp_node_ids=node_ids[t], grasp_full_ids=full_held[t])
    return load_trajectory(path)


def _status(check, tr):
    return check(tr).status


# --------------------------------------------------------------------------
# Writer integrity (Phase 0 bugs)
# --------------------------------------------------------------------------

class TestWriterIntegrity:
    def test_a_rejected_row_leaves_no_trace(self, tmp_path):
        """append() used to store tissue_pos before validating the rest."""
        w = TrajectoryWriter(str(tmp_path / "e.npz"), "s", "t", DT)
        w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7), action=np.zeros(7))
        with pytest.raises(ValueError):
            # Bad: grasp id out of range. Validated after pos in the old order.
            w.append(tissue_pos=np.ones((N, 3)), ee_pose=np.zeros(7),
                     action=np.zeros(7), grasp_node_ids=[N + 5])
        w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7), action=np.zeros(7))
        tr = load_trajectory(w.close())
        assert len(tr) == 2 and tr.ee_pose.shape == (2, 7)
        assert not (tr.tissue_pos == 1).any()

    def test_a_rejected_first_row_does_not_lock_optional_fields(self, tmp_path):
        w = TrajectoryWriter(str(tmp_path / "e.npz"), "s", "t", DT)
        with pytest.raises(ValueError):
            w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                     action=np.zeros(7), exposure=0.5, contact_mode=99)
        # The failed row must not have recorded "exposure is present".
        w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7), action=np.zeros(7))
        assert load_trajectory(w.close()).exposure.size == 0

    def test_an_exception_writes_partial_not_npz(self, tmp_path):
        path = tmp_path / "e.npz"
        with pytest.raises(RuntimeError):
            with TrajectoryWriter(str(path), "s", "t", DT) as w:
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7))
                raise RuntimeError("solver blew up")
        assert not path.exists()
        assert (tmp_path / "e.npz.partial").exists()
        assert list_trajectories(str(tmp_path)) == []
        with np.load(tmp_path / "e.npz.partial") as z:
            assert "solver blew up" in str(z["partial_reason"])

    def test_close_leaves_no_temp_file(self, tmp_path):
        _episode(str(tmp_path / "e.npz"))
        assert sorted(os.listdir(tmp_path)) == ["e.npz"]

    def test_set_static_is_refused_after_the_first_row(self, tmp_path):
        w = TrajectoryWriter(str(tmp_path / "e.npz"), "s", "t", DT)
        w.set_static(preroll_s=0.5)
        w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7), action=np.zeros(7))
        with pytest.raises(RuntimeError):
            w.set_static(preroll_s=0.0)

    def test_rest_pos_must_be_the_recorded_particles(self, tmp_path):
        full = np.random.default_rng(0).random((N_SIM, 3)).astype(np.float32)
        ids = np.arange(N, dtype=np.int32)
        with pytest.raises(ValueError, match="rest_pos"):
            TrajectoryWriter(str(tmp_path / "e.npz"), "s", "t", DT, particle_ids=ids,
                             n_particles_simulated=N_SIM, rest_pos=full[ids + 1],
                             rest_pos_simulated=full)


class TestV21FilesStillLoad:
    def test_v3_fields_read_as_empty_on_an_old_file(self, tmp_path):
        tr = _episode(str(tmp_path / "e.npz"))
        # Simulate a v2.1 file: drop every v3 key and relabel.
        with np.load(tmp_path / "e.npz") as z:
            d = {k: z[k] for k in z.files}
        for k in ("rest_pos", "rest_pos_simulated", "material_field", "control_dt",
                  "preroll_s", "jaw_names", "jaw_half_extents", "jaw_box_offset",
                  "grasp_capture_radius", "jaw_release_rad", "contact_radius", "time",
                  "jaw_box_pose", "jaw_box_vel", "jaw_joint", "node_contact",
                  "node_grasp", "grasp_full_ids_flat", "grasp_full_ids_offset", "phase"):
            d.pop(k)
        d["schema_version"] = np.array("2.1")
        np.savez(tmp_path / "old.npz", **d)
        old = load_trajectory(str(tmp_path / "old.npz"))
        assert old.rest_pos.size == 0 and not old.has_jaws and old.time.size == 0
        res = {fn.__name__: fn(old).status for fn in vd.PER_EPISODE}
        for name in ("check_reference_configuration", "check_material_field",
                     "check_jaw_geometry", "check_node_indicators",
                     "check_full_resolution_grasp"):
            assert res[name] == "SKIP", name
        assert len(tr) == len(old)


# --------------------------------------------------------------------------
# New validator checks: PASS on good data, FAIL on the corruption they target
# --------------------------------------------------------------------------

class TestNewChecks:
    def test_the_consistent_episode_passes_every_new_check(self, tmp_path):
        tr = _episode(str(tmp_path / "e.npz"))
        for fn in (vd.check_positive_time_intervals, vd.check_reference_configuration,
                   vd.check_material_field, vd.check_action_state_indexing,
                   vd.check_jaw_geometry, vd.check_node_indicators,
                   vd.check_full_resolution_grasp):
            r = fn(tr)
            assert r.status == "PASS", (fn.__name__, r.message)
        assert tr.node_grasp.any(), "the fixture must exercise the grasp branch"

    def test_action_off_by_one_is_caught(self, tmp_path):
        tr = _episode(str(tmp_path / "e.npz"), corrupt="action_off_by_one")
        assert _status(vd.check_action_state_indexing, tr) == "FAIL"

    def test_forward_difference_labelled_backward_is_caught(self, tmp_path):
        tr = _episode(str(tmp_path / "e.npz"), corrupt="forward_velocity")
        assert _status(vd.check_jaw_geometry, tr) == "FAIL"

    def test_flags_not_from_the_rule_are_caught(self, tmp_path):
        tr = _episode(str(tmp_path / "e.npz"), corrupt="flags")
        assert _status(vd.check_node_indicators, tr) == "FAIL"

    def test_frame0_claimed_as_reference_after_a_preroll_is_caught(self, tmp_path):
        """The to-do: do not substitute the first recorded position for X
        unless it is the reference. A file whose row 0 has moved but which
        claims preroll_s = 0 is exactly that substitution."""
        good = _episode(str(tmp_path / "a.npz"), preroll=0.5)
        assert _status(vd.check_reference_configuration, good) == "PASS"
        # Same data, but the pre-roll is not admitted.
        with np.load(tmp_path / "a.npz") as z:
            d = {k: z[k] for k in z.files}
        d["preroll_s"] = np.array([0.0])
        np.savez(tmp_path / "b.npz", **d)
        bad = load_trajectory(str(tmp_path / "b.npz"))
        assert _status(vd.check_reference_configuration, bad) == "FAIL"

    def test_material_field_in_pascals_is_caught(self, tmp_path):
        tr = _episode(str(tmp_path / "e.npz"),
                      material_field=np.tile([3000.0, 3e5, 1050.0], (N, 1)))
        assert _status(vd.check_material_field, tr) == "FAIL"

    def test_material_field_from_another_episode_is_caught(self, tmp_path):
        other = MAT + np.array([0.5, 0.0, 0.0], np.float32)
        tr = _episode(str(tmp_path / "e.npz"), material_field=np.tile(other, (N, 1)))
        assert _status(vd.check_material_field, tr) == "FAIL"

    def test_uneven_timestamps_are_caught(self, tmp_path):
        _episode(str(tmp_path / "a.npz"))
        with np.load(tmp_path / "a.npz") as z:
            d = {k: z[k] for k in z.files}
        d["time"] = d["time"].copy()
        d["time"][5:] += DT          # a dropped frame
        np.savez(tmp_path / "b.npz", **d)
        assert _status(vd.check_positive_time_intervals,
                       load_trajectory(str(tmp_path / "b.npz"))) == "FAIL"

    def test_non_positive_dt_is_caught_even_without_timestamps(self, tmp_path):
        _episode(str(tmp_path / "a.npz"))
        with np.load(tmp_path / "a.npz") as z:
            d = {k: z[k] for k in z.files}
        d["dt"] = np.float32(0.0)
        np.savez(tmp_path / "b.npz", **d)
        assert _status(vd.check_positive_time_intervals,
                       load_trajectory(str(tmp_path / "b.npz"))) == "FAIL"

    def test_full_resolution_grasp_disagreeing_with_active_flag_is_caught(self, tmp_path):
        _episode(str(tmp_path / "a.npz"))
        with np.load(tmp_path / "a.npz") as z:
            d = {k: z[k] for k in z.files}
        assert d["grasp_active"].any()
        d["grasp_active"] = np.zeros_like(d["grasp_active"])
        np.savez(tmp_path / "b.npz", **d)
        assert _status(vd.check_full_resolution_grasp,
                       load_trajectory(str(tmp_path / "b.npz"))) == "FAIL"

    def test_integration_residual_reports_and_warns_when_large(self, tmp_path):
        tr = _episode(str(tmp_path / "e.npz"))
        r = vd.check_integration_residual(tr)
        assert r.status in ("PASS", "WARN") and "residual RMS" in r.message
        # Velocities that disagree with the motion by 10x must WARN.
        with np.load(tmp_path / "e.npz") as z:
            d = {k: z[k] for k in z.files}
        d["tissue_vel"] = d["tissue_vel"] * 10
        np.savez(tmp_path / "b.npz", **d)
        assert _status(vd.check_integration_residual,
                       load_trajectory(str(tmp_path / "b.npz"))) == "WARN"


class TestDatasetNodeIdentity:
    def test_shared_ids_pass_and_differing_ids_warn(self, tmp_path):
        a = _episode(str(tmp_path / "a.npz"))
        b = _episode(str(tmp_path / "b.npz"))
        assert vd.check_node_identity_is_shared([a, b]).status == "PASS"
        with np.load(tmp_path / "b.npz") as z:
            d = {k: z[k] for k in z.files}
        d["particle_ids"] = d["particle_ids"] + 1
        np.savez(tmp_path / "c.npz", **d)
        c = load_trajectory(str(tmp_path / "c.npz"))
        assert vd.check_node_identity_is_shared([a, c]).status == "WARN"


class TestPartialReport:
    def test_run_reports_partial_files(self, tmp_path, capsys):
        _episode(str(tmp_path / "a.npz"))
        (tmp_path / "b.npz.partial").write_bytes(b"x")
        vd.run(str(tmp_path), colour=False)
        assert "b.npz.partial" in capsys.readouterr().out


class TestRotationCheckConditioning:
    def test_float32_stored_qw0_orientation_is_not_a_misalignment(self, tmp_path):
        """The Classic tool-down quaternion has qw = 0; stored as float32 its
        norm is 0.99999994. The first action-indexing check read that as a
        3.7e-4 rad misalignment (dot-product angle formula). Must PASS."""
        q = np.array([-0.7071067811865476, -0.7071067811865476, 0.0, 0.0])
        T = 6
        ee = np.array([np.concatenate([[0.25, 0.35, 0.16 - 0.002 * t], q]) for t in range(T)])
        act = np.zeros((T, 7))
        for t in range(T - 1):
            act[t] = actions.encode_action(ee[t], ee[t + 1], 0.0)
        with TrajectoryWriter(str(tmp_path / "e.npz"), "s", "t", DT,
                              action_spec="delta_pose_jaw") as w:
            for t in range(T):
                w.append(tissue_pos=np.zeros((4, 3)), ee_pose=ee[t], action=act[t])
        assert vd.check_action_state_indexing(
            load_trajectory(str(tmp_path / "e.npz"))).status == "PASS"

    def test_a_real_rotation_misalignment_is_still_caught(self, tmp_path):
        q = np.array([-0.7071067811865476, -0.7071067811865476, 0.0, 0.0])
        T = 6
        ee = np.array([np.concatenate([[0.25, 0.35, 0.16], q]) for _ in range(T)])
        act = np.zeros((T, 7))
        act[2, 3] = 0.01                          # a 10 mrad rotation that did not happen
        with TrajectoryWriter(str(tmp_path / "e.npz"), "s", "t", DT,
                              action_spec="delta_pose_jaw") as w:
            for t in range(T):
                w.append(tissue_pos=np.zeros((4, 3)), ee_pose=ee[t], action=act[t])
        assert vd.check_action_state_indexing(
            load_trajectory(str(tmp_path / "e.npz"))).status == "FAIL"


# --------------------------------------------------------------------------
# safety_strain_id / safety_strain_pos (DECISION_LOG 14.20)
# --------------------------------------------------------------------------

from tissue_metrics import compute_safety_strain  # noqa: E402


def _peak_episode(path, *, T=6, corrupt=None, peak_recorded=True):
    """N of N_SIM particles recorded; one particle is stretched more each row.

    With `peak_recorded` the stretched particle is a recorded node, so the
    check can compare the logged peak against the stored state exactly.
    """
    rng = np.random.default_rng(1)
    ids = np.sort(rng.choice(N_SIM, N, replace=False)).astype(np.int32)
    hot = int(ids[4]) if peak_recorded else int(np.setdiff1d(np.arange(N_SIM), ids)[0])
    x0 = rng.uniform(0, 0.05, (N_SIM, 3))
    with TrajectoryWriter(path, "test_mpm", "tissue_grasp", DT, particle_ids=ids,
                          n_particles_simulated=N_SIM, f_encoding="float32") as w:
        for t in range(T):
            F = np.tile(np.eye(3), (N_SIM, 1, 1))
            F[:, 0, 0] += 0.001 * rng.random(N_SIM)
            F[hot, 2, 2] = 1.2 + 0.1 * t                     # the peak, growing
            x = x0 + 0.0005 * t
            s = compute_safety_strain(F)
            sid, spos = int(s["argmax"]), x[int(s["argmax"])]
            if corrupt == "subset_index":
                # The bug this field invites: an index into the RECORDED
                # nodes instead of the solver array.
                sid = int(np.searchsorted(ids, sid))
            if corrupt == "stale_pos" and t:
                spos = x0[sid] + 0.0005 * (t - 1)            # one row late
            w.append(tissue_pos=x[ids], ee_pose=[0, 0, 0, 0, 0, 0, 1], action=np.zeros(7),
                     tissue_F=F[ids], safety_strain=float(s["max"]),
                     safety_strain_id=sid, safety_strain_pos=spos)
    return load_trajectory(path)


class TestSafetyPeak:
    def test_consistent_peak_passes(self, tmp_path):
        r = vd.check_safety_peak_is_consistent(_peak_episode(str(tmp_path / "a.npz")))
        assert r.status == vd.PASS, r.message
        assert "6 of 6 rows" in r.message

    def test_peak_off_the_node_set_is_skip_not_pass(self, tmp_path):
        tr = _peak_episode(str(tmp_path / "a.npz"), peak_recorded=False)
        assert _status(vd.check_safety_peak_is_consistent, tr) == vd.SKIP

    def test_subset_index_instead_of_solver_index_is_caught(self, tmp_path):
        tr = _peak_episode(str(tmp_path / "a.npz"), corrupt="subset_index")
        assert _status(vd.check_safety_peak_is_consistent, tr) == vd.FAIL

    def test_stale_position_is_caught(self, tmp_path):
        tr = _peak_episode(str(tmp_path / "a.npz"), corrupt="stale_pos")
        assert _status(vd.check_safety_peak_is_consistent, tr) == vd.FAIL

    def test_old_files_skip(self, tmp_path):
        tr = _episode(str(tmp_path / "a.npz"))
        assert _status(vd.check_safety_peak_is_consistent, tr) == vd.SKIP

    def test_writer_refuses_id_without_pos_or_out_of_range(self, tmp_path):
        w = TrajectoryWriter(str(tmp_path / "a.npz"), "t", "x", DT, n_particles_simulated=N_SIM)
        row = dict(tissue_pos=np.zeros((N, 3)), ee_pose=[0, 0, 0, 0, 0, 0, 1],
                   action=np.zeros(7), safety_strain=1.0)
        with pytest.raises(ValueError):
            w.append(**row, safety_strain_id=3)
        with pytest.raises(ValueError):
            w.append(**row, safety_strain_id=N_SIM, safety_strain_pos=[0, 0, 0])
        with pytest.raises(ValueError):
            w.append(tissue_pos=np.zeros((N, 3)), ee_pose=[0, 0, 0, 0, 0, 0, 1],
                     action=np.zeros(7), safety_strain_id=3, safety_strain_pos=[0, 0, 0])


def _hidden_peak_episode(path, *, with_id, T=4):
    """The peak (3x) sits on a particle the subset does NOT record, so the
    subset's max (1.0) falls 100% short of the logged value: the old margin
    check's trigger. With `with_id` the file also says which particle it was."""
    rng = np.random.default_rng(2)
    ids = np.sort(rng.choice(N_SIM, N, replace=False)).astype(np.int32)
    hot = int(np.setdiff1d(np.arange(N_SIM), ids)[0])
    x = rng.uniform(0, 0.05, (N_SIM, 3))
    F = np.tile(np.eye(3), (N_SIM, 1, 1))
    F[hot, 2, 2] = 3.0
    with TrajectoryWriter(path, "test_mpm", "tissue_grasp", DT, particle_ids=ids,
                          n_particles_simulated=N_SIM, f_encoding="float32",
                          target_origin=[0.0, 0.0, 0.2], target_normal=[0, 0, 1.0],
                          target_extent=[0.01, 0.01]) as w:
        for t in range(T):
            kw = dict(safety_strain_id=hot, safety_strain_pos=x[hot]) if with_id else {}
            w.append(tissue_pos=x[ids], ee_pose=[0, 0, 0, 0, 0, 0, 1], action=np.zeros(7),
                     tissue_F=F[ids], safety_strain=3.0, **kw)
    return load_trajectory(path)


class TestMarginDowngrade:
    """DECISION_LOG 14.24: the subset-margin guess is a WARN where the exact
    peak-particle check applies, and still a FAIL where it cannot."""

    def test_without_peak_id_the_margin_still_fails(self, tmp_path):
        tr = _hidden_peak_episode(str(tmp_path / "a.npz"), with_id=False)
        r = vd.check_logged_metrics_match_recomputation(tr)
        assert r.status == vd.FAIL and "underestimates" in r.message

    def test_with_peak_id_the_margin_warns(self, tmp_path):
        tr = _hidden_peak_episode(str(tmp_path / "a.npz"), with_id=True)
        r = vd.check_logged_metrics_match_recomputation(tr)
        assert r.status == vd.WARN and "downgraded" in r.message
        # ...and the exact check is what now carries the question.
        assert _status(vd.check_safety_peak_is_consistent, tr) == vd.SKIP

    def test_a_logged_value_above_the_full_set_still_fails_with_peak_id(self, tmp_path):
        """The downgrade must not soften the impossible direction: a subset
        maximum above the logged full-set maximum is still a hard FAIL."""
        tr = _hidden_peak_episode(str(tmp_path / "a.npz"), with_id=True)
        tr._d["safety_strain"] = np.full(len(tr), 0.5, np.float32)
        r = vd.check_logged_metrics_match_recomputation(tr)
        assert r.status == vd.FAIL and "EXCEEDS" in r.message
