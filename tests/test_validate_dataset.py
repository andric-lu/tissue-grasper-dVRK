"""
test_validate_dataset.py -- do the checks actually fire?

A validator that has only ever been run on good data is not a validator; it is
a green tick with no evidence behind it. Every check added in this phase is
exercised here against an episode broken on purpose, so a PASS on the real
dataset means the property was tested rather than that the test was vacuous.

The substep check is the one that most needs this. On synthetic episodes it
reports a ratio of exactly 1.00, because synthetic_traj.py sets substep_dt from
the same `suggested_substep_dt` the validator compares against. It cannot fail
there no matter how wrong the physics is. It is the MPM collector this check
exists for, and until that lands, this file is the only thing demonstrating it
works.

Also pins down the requirement that F, material, contact and metric checks SKIP
rather than FAIL on v1 files -- pre-v2 episodes are legitimate data, and a
validator that fails them is a validator people stop running.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "host"))

import validate_dataset as vd  # noqa: E402
from materials import suggested_substep_dt  # noqa: E402
from trajectory_io import (  # noqa: E402
    CONTACT_GRASP,
    CONTACT_NONE,
    CONTACT_TOUCH,
    SCHEMA_VERSION,
    TrajectoryWriter,
    load_trajectory,
)

N_PARTICLES, N_STEPS = 25, 12
TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], np.float32)
TARGET_NORMAL = np.array([0.0, 0.0, 1.0], np.float32)
TARGET_EXTENT = np.array([0.01, 0.01], np.float32)
# mu = 20 kPa, the stiff end of the default range, where the advisory substep is
# smallest and an over-large one is easiest to state unambiguously.
STIFF_MATERIAL = np.array([np.log(20000.0), np.log(2.0e5), 1050.0], np.float32)


def write_episode(path, *, F=None, contact_modes=None, exposure=None,
                  safety_strain=None, material=STIFF_MATERIAL, substep_dt=None,
                  positions=None, with_target=True):
    """A minimal, valid v2 episode, with any one property replaceable."""
    if substep_dt is None:
        substep_dt = float(suggested_substep_dt(20000.0, 1050.0, vd.ASSUMED_MPM_DX))
    a = np.linspace(-0.02, 0.02, 5)
    xx, yy = np.meshgrid(a, a, indexing="ij")
    base = np.stack([xx.ravel(), yy.ravel(), np.full(25, 0.005)], axis=-1)

    with TrajectoryWriter(
            path, "test", "tissue_retraction", 0.010,
            material_params=material, substep_dt=substep_dt, n_substeps=40,
            action_spec="delta_pose_jaw",
            target_origin=TARGET_ORIGIN if with_target else None,
            target_normal=TARGET_NORMAL if with_target else None,
            target_extent=TARGET_EXTENT if with_target else None) as w:
        for t in range(N_STEPS):
            pos = base + np.array([0.004 * t, 0.0, 0.0]) if positions is None \
                else positions[t]
            step_F = np.tile(np.eye(3), (N_PARTICLES, 1, 1)) if F is None else F[t]
            kwargs = {}
            if contact_modes is not None:
                kwargs["contact_mode"] = int(contact_modes[t])
            if exposure is not None:
                kwargs["exposure"] = float(exposure[t])
            if safety_strain is not None:
                kwargs["safety_strain"] = float(safety_strain[t])
            w.append(tissue_pos=pos, ee_pose=np.array([0, 0, 0.1, 0, 0, 0, 1.0]),
                     action=np.zeros(7), tissue_F=step_F, **kwargs)
    return load_trajectory(path)


def write_v1_episode(path):
    """A file in the v1 format, written by hand.

    The v2 writer cannot produce one, and the point is to prove the reader and
    the checks handle a file that predates every field they look at.
    """
    T, N = 6, 4
    np.savez_compressed(
        path, schema_version="1.0", simulator="pybullet", task="tissue_retraction",
        dt=np.float32(0.008), notes="hand-written v1 fixture",
        tissue_faces=np.zeros((0, 3), np.int32), tissue_tets=np.zeros((0, 4), np.int32),
        tissue_pos=np.random.default_rng(0).normal(scale=0.01, size=(T, N, 3)).astype(np.float32),
        tissue_vel=np.zeros((T, N, 3), np.float32),
        ee_pose=np.tile(np.array([0, 0, 0.1, 0, 0, 0, 1], np.float32), (T, 1)),
        ee_vel=np.zeros((T, 6), np.float32), jaw=np.zeros(T, np.float32),
        joint_pos=np.zeros((T, 0), np.float32), action=np.zeros((T, 4), np.float32),
        grasp_active=np.ones(T, bool), contact_force=np.zeros((T, 3), np.float32),
        grasp_ids_flat=np.tile(np.array([0, 1], np.int32), T),
        grasp_ids_offset=np.arange(0, 2 * T + 1, 2, dtype=np.int32))
    return load_trajectory(path)


def by_name(name):
    """Look a check up by function name, so a rename breaks the test loudly."""
    for fn in vd.PER_EPISODE + vd.PER_DATASET:
        if fn.__name__ == name:
            return fn
    raise AssertionError(f"no check named {name}")


# --------------------------------------------------------------------------
# Check 1 -- det(F) > 0
# --------------------------------------------------------------------------

class TestFAdmissible:
    def test_passes_on_identity(self, tmp_path):
        tr = write_episode(str(tmp_path / "ok.npz"))
        assert by_name("check_F_admissible")(tr).status == vd.PASS

    def test_fails_on_an_inverted_element(self, tmp_path):
        F = np.tile(np.eye(3), (N_STEPS, N_PARTICLES, 1, 1))
        F[7, 3] = np.diag([1.0, 1.0, -1.0])          # one particle, one step
        tr = write_episode(str(tmp_path / "inv.npz"), F=F)
        r = by_name("check_F_admissible")(tr)
        assert r.status == vd.FAIL
        assert "inverted" in r.message and "step 7" in r.message

    def test_fails_on_a_flattened_element(self, tmp_path):
        """det(F) == 0 exactly: no volume left, and ln(J) undefined."""
        F = np.tile(np.eye(3), (N_STEPS, N_PARTICLES, 1, 1))
        F[2, :] = np.diag([1.0, 1.0, 0.0])
        tr = write_episode(str(tmp_path / "flat.npz"), F=F)
        assert by_name("check_F_admissible")(tr).status == vd.FAIL


# --------------------------------------------------------------------------
# Check 2 -- near-incompressibility (soft)
# --------------------------------------------------------------------------

class TestFIncompressible:
    def test_passes_on_isochoric_deformation(self, tmp_path):
        s = 1.3
        F = np.tile(np.diag([s, 1 / np.sqrt(s), 1 / np.sqrt(s)]),
                    (N_STEPS, N_PARTICLES, 1, 1))
        tr = write_episode(str(tmp_path / "iso.npz"), F=F)
        assert by_name("check_F_incompressible")(tr).status == vd.PASS

    def test_warns_but_does_not_fail_on_large_volume_change(self, tmp_path):
        """SOFT check: a compressible phantom is legitimate, so this must warn
        rather than reject the episode."""
        F = np.tile(np.diag([1.2, 1.2, 1.2]), (N_STEPS, N_PARTICLES, 1, 1))
        tr = write_episode(str(tmp_path / "comp.npz"), F=F)
        r = by_name("check_F_incompressible")(tr)
        assert r.status == vd.WARN
        assert "volume change" in r.message


# --------------------------------------------------------------------------
# Check 3 -- material varies across episodes
# --------------------------------------------------------------------------

class TestMaterialDiversity:
    def test_warns_when_every_episode_shares_one_material(self, tmp_path):
        trs = [write_episode(str(tmp_path / f"same{i}.npz")) for i in range(4)]
        r = by_name("check_material_is_diverse")(trs)
        assert r.status == vd.WARN
        assert "barely varies" in r.message

    def test_passes_when_material_is_randomised(self, tmp_path):
        rng = np.random.default_rng(0)
        trs = []
        for i in range(4):
            m = np.array([rng.uniform(np.log(200), np.log(20000)),
                          np.log(2e5), 1050.0], np.float32)
            trs.append(write_episode(str(tmp_path / f"var{i}.npz"), material=m,
                                     substep_dt=1e-6))
        assert by_name("check_material_is_diverse")(trs).status == vd.PASS

    def test_skips_when_no_episode_records_material(self, tmp_path):
        tr = write_v1_episode(str(tmp_path / "v1.npz"))
        assert by_name("check_material_is_diverse")([tr, tr]).status == vd.SKIP


# --------------------------------------------------------------------------
# Check 4 -- substep is stable for the sampled stiffness
# --------------------------------------------------------------------------

class TestSubstepStability:
    def test_passes_at_the_advisory_value(self, tmp_path):
        tr = write_episode(str(tmp_path / "ok.npz"))
        r = by_name("check_substep_is_stable_for_stiffness")(tr)
        assert r.status == vd.PASS

    def test_fails_when_the_substep_is_too_large_for_the_stiffness(self, tmp_path):
        """§2.6's failure mode: one fixed substep, tuned on soft tissue, used
        for a stiff episode. 200 us is ~5x the advisory at mu = 20 kPa."""
        tr = write_episode(str(tmp_path / "big.npz"), substep_dt=200e-6)
        r = by_name("check_substep_is_stable_for_stiffness")(tr)
        assert r.status == vd.FAIL
        assert "advisory" in r.message

    def test_the_same_substep_is_fine_for_soft_tissue(self, tmp_path):
        """The point of the check: the SAME substep that fails above passes
        here. A stiffness-independent threshold could not tell these apart."""
        soft = np.array([np.log(200.0), np.log(2.0e5), 1050.0], np.float32)
        tr = write_episode(str(tmp_path / "soft.npz"), material=soft,
                           substep_dt=200e-6)
        assert by_name("check_substep_is_stable_for_stiffness")(tr).status == vd.PASS

    def test_skips_when_substep_not_recorded(self, tmp_path):
        tr = write_episode(str(tmp_path / "nosub.npz"), substep_dt=0.0)
        assert by_name("check_substep_is_stable_for_stiffness")(tr).status == vd.SKIP


# --------------------------------------------------------------------------
# Check 5 -- contact mode transitions
# --------------------------------------------------------------------------

class TestContactModeTransitions:
    def test_passes_on_none_touch_grasp(self, tmp_path):
        modes = [CONTACT_NONE] * 3 + [CONTACT_TOUCH] * 3 + [CONTACT_GRASP] * 6
        tr = write_episode(str(tmp_path / "ok.npz"), contact_modes=modes)
        assert by_name("check_contact_mode_transitions")(tr).status == vd.PASS

    def test_fails_on_a_direct_none_to_grasp(self, tmp_path):
        """The jaws cannot close on tissue they were not touching."""
        modes = [CONTACT_NONE] * 6 + [CONTACT_GRASP] * 6
        tr = write_episode(str(tmp_path / "jump.npz"), contact_modes=modes)
        r = by_name("check_contact_mode_transitions")(tr)
        assert r.status == vd.FAIL
        assert "none->grasp" in r.message and "step 6" in r.message

    def test_warns_on_chattering(self, tmp_path):
        """A mode alternating every step is a threshold oscillating around its
        cutoff, not contact."""
        modes = [CONTACT_NONE if t % 2 else CONTACT_TOUCH for t in range(N_STEPS)]
        tr = write_episode(str(tmp_path / "chat.npz"), contact_modes=modes)
        r = by_name("check_contact_mode_transitions")(tr)
        assert r.status == vd.WARN
        assert "chattering" in r.message

    def test_skips_when_not_recorded(self, tmp_path):
        tr = write_episode(str(tmp_path / "none.npz"))
        assert by_name("check_contact_mode_transitions")(tr).status == vd.SKIP


# --------------------------------------------------------------------------
# Check 6 -- logged metrics match recomputation
# --------------------------------------------------------------------------

class TestLoggedMetricsMatch:
    def _honest(self, tmp_path, name):
        from tissue_metrics import compute_exposure, compute_safety_strain
        a = np.linspace(-0.02, 0.02, 5)
        xx, yy = np.meshgrid(a, a, indexing="ij")
        base = np.stack([xx.ravel(), yy.ravel(), np.full(25, 0.005)], axis=-1)
        pos = np.stack([base + np.array([0.004 * t, 0, 0]) for t in range(N_STEPS)])
        F = np.tile(np.eye(3), (N_STEPS, N_PARTICLES, 1, 1))
        exp = [float(compute_exposure(p, TARGET_ORIGIN, TARGET_NORMAL, TARGET_EXTENT))
               for p in pos]
        saf = [float(compute_safety_strain(f)["max"]) for f in F]
        return pos, F, exp, saf

    def test_passes_when_logged_values_are_honest(self, tmp_path):
        pos, F, exp, saf = self._honest(tmp_path, "ok")
        tr = write_episode(str(tmp_path / "ok.npz"), positions=pos, F=F,
                           exposure=exp, safety_strain=saf)
        assert by_name("check_logged_metrics_match_recomputation")(tr).status == vd.PASS

    def test_fails_when_exposure_drifted(self, tmp_path):
        """The drift this catches: a collector pinned to an old sigma, or a
        metric whose default changed after collection."""
        pos, F, exp, saf = self._honest(tmp_path, "drift")
        drifted = [e + 0.05 for e in exp]
        tr = write_episode(str(tmp_path / "drift.npz"), positions=pos, F=F,
                           exposure=drifted, safety_strain=saf)
        r = by_name("check_logged_metrics_match_recomputation")(tr)
        assert r.status == vd.FAIL and "exposure differs" in r.message

    def test_fails_when_safety_strain_drifted(self, tmp_path):
        pos, F, exp, saf = self._honest(tmp_path, "drift2")
        tr = write_episode(str(tmp_path / "drift2.npz"), positions=pos, F=F,
                           exposure=exp, safety_strain=[s + 0.2 for s in saf])
        r = by_name("check_logged_metrics_match_recomputation")(tr)
        assert r.status == vd.FAIL and "safety_strain differs" in r.message

    def test_skips_without_a_target_region(self, tmp_path):
        tr = write_episode(str(tmp_path / "notgt.npz"), with_target=False)
        assert by_name("check_logged_metrics_match_recomputation")(tr).status == vd.SKIP


# --------------------------------------------------------------------------
# v1 files must SKIP, never FAIL
# --------------------------------------------------------------------------

class TestLegacyFilesSkipCleanly:
    """Pre-v2 episodes are legitimate data collected before the schema grew. A
    validator that fails them is a validator people stop running."""

    @pytest.mark.parametrize("name", [
        "check_F_admissible",
        "check_F_incompressible",
        "check_substep_is_stable_for_stiffness",
        "check_logged_metrics_match_recomputation",
        "check_contact_mode_transitions",
    ])
    def test_v2_only_checks_skip(self, tmp_path, name):
        tr = write_v1_episode(str(tmp_path / "v1.npz"))
        assert by_name(name)(tr).status == vd.SKIP

    def test_no_check_fails_on_a_v1_file(self, tmp_path):
        tr = write_v1_episode(str(tmp_path / "v1.npz"))
        for fn in vd.PER_EPISODE:
            assert fn(tr).status != vd.FAIL, f"{fn.__name__} failed a legacy file"

    def test_the_skip_message_says_why(self, tmp_path):
        tr = write_v1_episode(str(tmp_path / "v1.npz"))
        r = by_name("check_F_admissible")(tr)
        assert "1.0" in r.message and "pybullet" in r.message


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------

class TestRunner:
    def test_exit_zero_on_a_clean_directory(self, tmp_path, capsys):
        for i in range(2):
            write_episode(str(tmp_path / f"ep{i}.npz"),
                          material=np.array([np.log(200.0 * (i + 40)), np.log(2e5),
                                             1050.0], np.float32),
                          substep_dt=1e-6)
        assert vd.run(str(tmp_path), colour=False) == 0

    def test_exit_one_on_a_failure(self, tmp_path):
        F = np.tile(np.eye(3), (N_STEPS, N_PARTICLES, 1, 1))
        F[0, 0] = np.diag([-1.0, 1.0, 1.0])
        write_episode(str(tmp_path / "bad.npz"), F=F)
        assert vd.run(str(tmp_path), colour=False) == 1

    def test_exit_one_on_an_empty_directory(self, tmp_path):
        assert vd.run(str(tmp_path), colour=False) == 1

    def test_a_corrupt_file_does_not_hide_the_others(self, tmp_path, capsys):
        """One unreadable episode must not stop the report on the other 199."""
        write_episode(str(tmp_path / "aaa_good.npz"), substep_dt=1e-6)
        (tmp_path / "zzz_corrupt.npz").write_bytes(b"not an npz at all")
        code = vd.run(str(tmp_path), colour=False)
        out = capsys.readouterr().out
        assert code == 1
        assert "aaa_good.npz" in out and "zzz_corrupt.npz" in out

    def test_every_check_is_registered_once(self):
        names = [fn.__name__ for fn in vd.PER_EPISODE + vd.PER_DATASET]
        assert len(names) == len(set(names))
        assert len(names) >= 12

    def test_schema_version_constant_is_current(self):
        """Bumping the schema must be deliberate, and must not orphan old data.

        This test is a tripwire: it fails the moment SCHEMA_VERSION changes, so
        the bump cannot happen as a side effect of editing something else. When
        it fails, the question to answer is not "what is the new string" but
        "can the reader still open every file already on disk" -- which is why
        the older versions are asserted here too rather than just the new one.
        """
        from trajectory_io import KNOWN_SCHEMA_VERSIONS
        assert SCHEMA_VERSION == "2.1"
        assert SCHEMA_VERSION in KNOWN_SCHEMA_VERSIONS
        # data/ is v1.0 and the synthetic episodes are v2.0. Dropping either
        # from the reader would strand data that is still on disk.
        for older in ("1.0", "2.0"):
            assert older in KNOWN_SCHEMA_VERSIONS
