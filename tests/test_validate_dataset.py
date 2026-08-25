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
from materials import suggested_substep_dt, unpack_material  # noqa: E402
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
# The soft counterpart. lambda drops with mu -- see
# TestSubstepStability.test_the_same_substep_is_fine_for_soft_tissue for why
# softening mu alone does almost nothing to a P-wave bound.
SOFT_MATERIAL = np.array([np.log(200.0), np.log(2.0e4), 1050.0], np.float32)
# A substep that is too large for STIFF_MATERIAL (advisory 19.8 us, ratio 2.52)
# and comfortably fine for SOFT_MATERIAL (advisory 68.1 us, ratio 0.73), with
# SUBSTEP_SLACK = 1.5 sitting between the two. That straddle is what makes the
# pair of tests below a demonstration rather than a coincidence, and
# test_the_shared_substep_really_does_straddle_the_two_advisories pins it.
SHARED_SUBSTEP = 50e-6
# Seconds per recorded frame in the fixture. 10 ms divides SHARED_SUBSTEP a
# whole 200 times, so the snap below leaves that value untouched and the
# substep-stability tests get the substep they asked for.
FIXTURE_DT = 0.010


def write_episode(path, *, F=None, contact_modes=None, exposure=None,
                  safety_strain=None, material=STIFF_MATERIAL, substep_dt=None,
                  positions=None, with_target=True,
                  particle_ids=None, n_particles_simulated=0):
    """A minimal, valid v2 episode, with any one property replaceable."""
    if substep_dt is None:
        # The advisory must be computed the way the CHECK computes it, which
        # since the P-wave fix means passing lambda. Omitting it here made the
        # fixture pick a bar-wave step ~16x too large and the check correctly
        # FAIL an episode the fixture called valid -- the check was right and
        # the fixture was stale. Derived from `material` rather than hardcoded
        # so a test that varies the material varies its advisory with it.
        f_mu, f_lam, f_rho = unpack_material(material)
        substep_dt = float(suggested_substep_dt(f_mu, f_rho, vd.ASSUMED_MPM_DX,
                                                lam=f_lam))
    # Close the timebase the way a real collector must: whole substeps per
    # frame, then the substep snapped so n_substeps * substep_dt == dt exactly.
    # This used to be a hardcoded n_substeps=40 bearing no relation to dt or to
    # substep_dt, which the writer now refuses -- correctly, since a fixture
    # that models an impossible episode cannot demonstrate anything about real
    # ones. `substep_dt=0.0` still means "not recorded" and skips the snap.
    if substep_dt > 0.0:
        n_substeps = max(1, int(np.ceil(FIXTURE_DT / substep_dt)))
        substep_dt = FIXTURE_DT / n_substeps
    else:
        n_substeps = 0

    a = np.linspace(-0.02, 0.02, 5)
    xx, yy = np.meshgrid(a, a, indexing="ij")
    base = np.stack([xx.ravel(), yy.ravel(), np.full(25, 0.005)], axis=-1)

    with TrajectoryWriter(
            path, "test", "tissue_retraction", FIXTURE_DT,
            material_params=material, substep_dt=substep_dt, n_substeps=n_substeps,
            action_spec="delta_pose_jaw",
            particle_ids=particle_ids,
            n_particles_simulated=n_particles_simulated,
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

class TestTimebaseIsConsistent:
    """dt == substep_dt * n_substeps, demonstrated against a file that lies.

    The MPM adapter shipped this bug: it picked n_substeps from the material's
    P-wave bound, wrote that to disk, and integrated with the vendored module
    default of 25. The recorded frame claimed 12.5 ms and delivered 2.95 ms.
    Every check in the validator passed. These tests are the reason it cannot
    happen twice, so they must be shown firing, not merely shown green.
    """

    def test_passes_when_the_timebase_closes(self, tmp_path):
        tr = write_episode(str(tmp_path / "ok.npz"))
        r = by_name("check_timebase_is_consistent")(tr)
        assert r.status == vd.PASS

    def test_the_writer_refuses_to_create_an_inconsistent_file(self, tmp_path):
        """First line of defence: the caller with the bad numbers is still on
        the stack, so it is the one that gets the error."""
        with pytest.raises(ValueError, match="timebase does not close"):
            TrajectoryWriter(str(tmp_path / "bad.npz"), "mpm", "t", 0.0125,
                             substep_dt=1.179e-4, n_substeps=25)

    def test_the_check_fires_on_a_file_that_already_exists(self, tmp_path):
        """Second line: the writer's guard cannot help data already on disk, so
        the check is demonstrated against a file built to be wrong -- written
        past the writer by editing the npz directly, which is exactly the
        position we were in with the two stale adapter episodes."""
        p = str(tmp_path / "lying.npz")
        tr = write_episode(p)
        d = dict(np.load(p, allow_pickle=True))
        # The real bug, reproduced: n_substeps says one thing, dt another.
        # 0-d arrays, which is how the writer stores scalars -- a 1-element 1-d
        # array is not the same thing and numpy 2 refuses to scalarise it.
        d["n_substeps"] = np.int32(25)
        d["substep_dt"] = np.float32(1.179e-4)
        d["dt"] = np.float32(0.0125)
        np.savez_compressed(p, **d)
        r = by_name("check_timebase_is_consistent")(load_trajectory(p))
        assert r.status == vd.FAIL
        assert "advances a different amount of time" in r.message

    def test_skips_on_a_file_that_records_neither_field(self, tmp_path):
        """v1 files and the PyBullet collector record no substep information.
        SKIP and say why -- never a quiet PASS. See the SKIP discipline note."""
        tr = write_episode(str(tmp_path / "v1ish.npz"), substep_dt=0.0)
        r = by_name("check_timebase_is_consistent")(tr)
        assert r.status == vd.SKIP
        assert "not both recorded" in r.message

    def test_an_off_by_one_substep_count_is_caught(self, tmp_path):
        """The tolerance has to be tight enough to catch the plausible error,
        not just the gross one. One substep out of 100 is a 1% timebase error
        and must fail."""
        with pytest.raises(ValueError, match="timebase does not close"):
            TrajectoryWriter(str(tmp_path / "offbyone.npz"), "mpm", "t", 0.0125,
                             substep_dt=0.0125 / 100, n_substeps=99)


class TestSubstepStability:
    def test_passes_at_the_advisory_value(self, tmp_path):
        tr = write_episode(str(tmp_path / "ok.npz"))
        r = by_name("check_substep_is_stable_for_stiffness")(tr)
        assert r.status == vd.PASS

    def test_fails_when_the_substep_is_too_large_for_the_stiffness(self, tmp_path):
        """§2.6's failure mode: one fixed substep, tuned on soft tissue, used
        for a stiff episode. At mu = 20 kPa, lambda = 200 kPa the P-wave
        advisory is 19.8 us, so SHARED_SUBSTEP is 2.5x over -- past the 1.5x
        slack, and a FAIL."""
        tr = write_episode(str(tmp_path / "big.npz"), substep_dt=SHARED_SUBSTEP)
        r = by_name("check_substep_is_stable_for_stiffness")(tr)
        assert r.status == vd.FAIL
        assert "advisory" in r.message

    def test_the_same_substep_is_fine_for_soft_tissue(self, tmp_path):
        """The point of the check: the SAME substep that fails above passes
        here. A stiffness-independent threshold could not tell these apart.

        SOFT_MATERIAL softens LAMBDA, not just mu, and that is the whole
        lesson of the P-wave bound. The previous version of this test dropped
        mu tenfold (20 kPa -> 200 Pa) but left lambda at 200 kPa; since the
        P-wave speed goes as sqrt((lambda + 2mu)/rho) and lambda dominates by
        750:1 in tissue, that moved the advisory by 9% -- from 19.8 to 21.7 us
        -- and the "soft" episode failed at the same substep as the stiff one.
        Under this bound, soft means small lambda.
        """
        tr = write_episode(str(tmp_path / "soft.npz"), material=SOFT_MATERIAL,
                           substep_dt=SHARED_SUBSTEP)
        assert by_name("check_substep_is_stable_for_stiffness")(tr).status == vd.PASS

    def test_the_shared_substep_really_does_straddle_the_two_advisories(self):
        """The two tests above are only meaningful if SHARED_SUBSTEP lies
        between the materials' advisories. Pin that, so a later edit to either
        material cannot quietly make both tests trivially agree."""
        s_mu, s_lam, s_rho = unpack_material(STIFF_MATERIAL)
        f_mu, f_lam, f_rho = unpack_material(SOFT_MATERIAL)
        stiff_adv = float(suggested_substep_dt(s_mu, s_rho, vd.ASSUMED_MPM_DX, lam=s_lam))
        soft_adv = float(suggested_substep_dt(f_mu, f_rho, vd.ASSUMED_MPM_DX, lam=f_lam))
        assert stiff_adv < SHARED_SUBSTEP < soft_adv
        # And each is clear of the slack boundary, not balanced on it.
        assert SHARED_SUBSTEP / stiff_adv > vd.SUBSTEP_SLACK
        assert SHARED_SUBSTEP / soft_adv < vd.SUBSTEP_SLACK

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


class TestSubsetMetricsUseBoundsNotEquality:
    """A subset record cannot reproduce a full-set metric, and must not be
    asked to. Both metrics are bounded rather than equated, in the direction
    physics allows -- and the impossible direction is a hard failure.

    This is not hypothetical tidying. The equality bound was rejecting correct
    data: the first fresh MPM episode logged exposure 0.0 over 24,000 particles
    and recomputed 1.44e-3 over the stored 3,000, which the 1e-3 equality bound
    called drift. The two episodes before it passed the same bound at 7.4e-4 --
    the check was already wrong and had simply not been pushed hard enough.
    """

    def _subset_episode(self, tmp_path, name, *, logged_exposure):
        """A subset record whose stored particles are a strict subset.

        The stored positions occlude the target LESS than the (unrecorded)
        full set did, which is the only physically reachable arrangement.
        `logged_exposure` is what the file claims the full set measured.
        """
        a = np.linspace(-0.005, 0.005, 5)
        xx, yy = np.meshgrid(a, a, indexing="ij")
        # Directly over the target, so these particles genuinely occlude it.
        pos = np.stack([np.stack([xx.ravel(), yy.ravel(),
                                  np.full(25, 0.004)], axis=-1)] * N_STEPS)
        F = np.tile(np.eye(3), (N_STEPS, N_PARTICLES, 1, 1))
        return write_episode(
            str(tmp_path / name), positions=pos, F=F,
            exposure=[logged_exposure] * N_STEPS,
            safety_strain=[1.0] * N_STEPS,
            particle_ids=np.arange(N_PARTICLES, dtype=np.int32),
            n_particles_simulated=10 * N_PARTICLES)

    def test_the_valid_direction_passes(self, tmp_path):
        """Logged full-set exposure BELOW what the subset recomputes: the
        dropped particles were occluding, so the full set saw less of the
        target. Expected, and must not be flagged."""
        tr = self._subset_episode(tmp_path, "valid.npz", logged_exposure=0.0)
        r = by_name("check_logged_metrics_match_recomputation")(tr)
        assert tr.is_subset
        assert r.status == vd.PASS
        assert "consistent with full-set bounds" in r.message
        assert "reproduce" not in r.message

    def test_the_impossible_direction_fails(self, tmp_path):
        """Logged full-set exposure ABOVE what the subset recomputes would mean
        removing particles ADDED occlusion. It cannot. Something computed the
        logged metric over the subset, or particle_ids and tissue_pos disagree
        about which particles these are."""
        tr = self._subset_episode(tmp_path, "impossible.npz", logged_exposure=0.9)
        r = by_name("check_logged_metrics_match_recomputation")(tr)
        assert r.status == vd.FAIL
        assert "falls BELOW the logged full-set value" in r.message
        assert "cannot add occlusion" in r.message

    def test_a_full_record_still_demands_equality(self, tmp_path):
        """The inequality is for SUBSETS only. When the file stores every
        particle it simulated, the logged metric must still reproduce exactly
        -- otherwise the loosening would silently cover real drift."""
        pos, F, exp, saf = TestLoggedMetricsMatch()._honest(tmp_path, "full")
        tr = write_episode(str(tmp_path / "full.npz"), positions=pos, F=F,
                           exposure=[e + 0.05 for e in exp], safety_strain=saf)
        assert not tr.is_subset
        r = by_name("check_logged_metrics_match_recomputation")(tr)
        assert r.status == vd.FAIL and "exposure differs" in r.message


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
