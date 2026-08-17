"""
test_actions.py -- the rotation code, where the bugs hide.

Two failure modes get most of the attention here, because both produce data
that looks fine and trains badly:

  - NaN at small angles, from `2*acos(qw)` with qw a few ulps above 1. Asserted
    on explicitly with np.isfinite rather than left to a comparison, because
    `NaN == NaN` is False and a round-trip test can "fail loudly" for the wrong
    reason -- or worse, an approx() comparison can be satisfied by luck.
  - The quaternion double cover. q and -q are the same rotation, so every
    quaternion comparison here goes through `quat_close`, which accepts either.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from actions import (  # noqa: E402
    absolute_to_delta_actions,
    apply_pose_delta,
    decode_action,
    encode_action,
    pose_delta,
    quat_conjugate,
    quat_multiply,
    quat_to_rotvec,
    rotvec_to_quat,
)

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def quat_close(a, b, tol=1e-9):
    """True if a and b are the same ROTATION, allowing for the double cover.

    q and -q describe an identical rotation, so a raw element-wise comparison
    reports a false failure roughly half the time.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return min(np.abs(a - b).max(), np.abs(a + b).max()) < tol


def random_poses(n, seed=0, scale=0.1):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(scale=scale, size=(n, 3))
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return np.concatenate([xyz, q], axis=-1)


def axis(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=3)
    return a / np.linalg.norm(a)


# --------------------------------------------------------------------------
# Rotation representations
# --------------------------------------------------------------------------

class TestQuatRotvecRoundTrip:
    def test_identity(self):
        assert np.allclose(quat_to_rotvec(IDENTITY_QUAT), 0.0)
        assert quat_close(rotvec_to_quat(np.zeros(3)), IDENTITY_QUAT)

    @pytest.mark.parametrize("theta", [1e-12, 1e-10, 1e-8, 1e-6, 1e-3, 0.1, 1.0,
                                       2.0, 3.0, np.pi - 1e-6])
    def test_round_trip_over_the_full_angle_range(self, theta):
        r = axis(1) * theta
        back = quat_to_rotvec(rotvec_to_quat(r))
        assert np.all(np.isfinite(back))
        assert np.allclose(back, r, atol=1e-12, rtol=1e-9)

    def test_quat_to_rotvec_to_quat(self):
        rng = np.random.default_rng(3)
        q = rng.normal(size=(200, 4))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        back = rotvec_to_quat(quat_to_rotvec(q))
        for i in range(len(q)):
            assert quat_close(q[i], back[i])

    def test_batched_shapes(self):
        r = np.zeros((5, 6, 3))
        assert rotvec_to_quat(r).shape == (5, 6, 4)
        assert quat_to_rotvec(rotvec_to_quat(r)).shape == (5, 6, 3)


class TestSmallAngleStability:
    """§2.5: `2*acos(qw)` returns NaN when float error pushes qw above 1.
    `2*atan2(|qv|, qw)` cannot."""

    @pytest.mark.parametrize("theta", [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 0.0])
    def test_no_nan_at_tiny_angles(self, theta):
        r = axis(2) * theta
        q = rotvec_to_quat(r)
        assert np.all(np.isfinite(q)), f"rotvec_to_quat produced non-finite at {theta}"
        back = quat_to_rotvec(q)
        assert np.all(np.isfinite(back)), f"quat_to_rotvec produced non-finite at {theta}"
        assert np.allclose(back, r, atol=1e-15)

    def test_the_acos_formulation_would_have_failed(self):
        """Guards the guard: show the naive formula really does break, so the
        tests above mean the implementation is right rather than that the input
        was benign.

        ONE ULP is all it takes. `nextafter(1.0, 2.0)` is the smallest float64
        above 1.0 -- exactly what normalising an almost-identity quaternion
        can land on -- and acos returns NaN there while atan2 does not.
        """
        q = rotvec_to_quat(axis(2) * 1e-9)
        qw = np.nextafter(1.0, 2.0)        # a normalisation nudging qw over 1
        assert qw > 1.0
        with np.errstate(invalid="ignore"):
            assert np.isnan(2.0 * np.arccos(qw))
        assert np.isfinite(2.0 * np.arctan2(np.linalg.norm(q[:3]), qw))

    def test_exactly_unit_qw_is_finite(self):
        assert np.all(np.isfinite(quat_to_rotvec(np.array([0.0, 0.0, 0.0, 1.0]))))
        assert np.all(np.isfinite(quat_to_rotvec(np.array([0.0, 0.0, 0.0, -1.0]))))


class TestLargeAngle:
    def test_near_pi_round_trips(self):
        r = axis(4) * (np.pi - 1e-6)
        back = quat_to_rotvec(rotvec_to_quat(r))
        assert np.all(np.isfinite(back))
        assert np.allclose(back, r, atol=1e-12)

    def test_beyond_pi_comes_back_the_short_way(self):
        """A 359-degree rotation is a -1-degree rotation. The delta a sampler
        perturbs must be the small one."""
        r = axis(5) * (2 * np.pi - np.deg2rad(1.0))
        back = quat_to_rotvec(rotvec_to_quat(r))
        assert np.linalg.norm(back) == pytest.approx(np.deg2rad(1.0), abs=1e-9)
        assert np.dot(back, r) < 0        # opposite direction, as it should be

    def test_angle_never_exceeds_pi(self):
        rng = np.random.default_rng(6)
        q = rng.normal(size=(500, 4))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        assert np.linalg.norm(quat_to_rotvec(q), axis=-1).max() <= np.pi + 1e-12


class TestQuatAlgebra:
    def test_conjugate_undoes(self):
        rng = np.random.default_rng(7)
        q = rng.normal(size=(50, 4))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        prod = quat_multiply(q, quat_conjugate(q))
        for p in prod:
            assert quat_close(p, IDENTITY_QUAT)

    def test_multiply_matches_rotation_matrix_composition(self):
        """Cross-check the Hamilton product against acting on a vector, so a
        transposed or scalar-first convention cannot slip through."""
        rng = np.random.default_rng(8)
        for _ in range(20):
            a, b = rng.normal(size=(2, 4))
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            v = rng.normal(size=3)
            ab = quat_multiply(a, b)
            assert np.allclose(_rotate(ab, v), _rotate(a, _rotate(b, v)))


def _rotate(q, v):
    """Rotate v by scalar-last quaternion q, via the quaternion sandwich."""
    qv, qw = q[:3], q[3]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


# --------------------------------------------------------------------------
# Poses
# --------------------------------------------------------------------------

class TestPoseDeltaRoundTrip:
    def test_apply_of_delta_recovers_the_target(self):
        a, b = random_poses(64, seed=10), random_poses(64, seed=11)
        got = apply_pose_delta(a, pose_delta(a, b))
        assert np.allclose(got[:, :3], b[:, :3], atol=1e-12)
        for i in range(len(b)):
            assert quat_close(got[i, 3:], b[i, 3:], tol=1e-9)

    def test_identical_poses_give_a_zero_delta(self):
        a = random_poses(16, seed=12)
        assert np.allclose(pose_delta(a, a), 0.0, atol=1e-12)

    def test_zero_delta_is_a_no_op(self):
        a = random_poses(16, seed=13)
        got = apply_pose_delta(a, np.zeros((16, 6)))
        assert np.allclose(got[:, :3], a[:, :3])
        for i in range(len(a)):
            assert quat_close(got[i, 3:], a[i, 3:])

    def test_translation_is_world_frame(self):
        """The translation half must not be rotated by the pose's orientation:
        the two halves of the delta have to live in the same frame."""
        a = np.array([0.0, 0.0, 0.0, *rotvec_to_quat(np.array([0.0, 0.0, np.pi / 2]))])
        got = apply_pose_delta(a, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert np.allclose(got[:3], [1.0, 0.0, 0.0])

    def test_tiny_pose_difference(self):
        a = random_poses(8, seed=14)
        b = a.copy()
        b[:, :3] += 1e-12
        b[:, 3:] = quat_multiply(rotvec_to_quat(np.full((8, 3), 1e-12)), a[:, 3:])
        d = pose_delta(a, b)
        assert np.all(np.isfinite(d))
        assert np.allclose(apply_pose_delta(a, d)[:, :3], b[:, :3], atol=1e-15)


class TestComposition:
    def test_two_deltas_equal_the_single_delta(self):
        """Deltas compose. Note the composition is by APPLYING them in turn --
        rotation vectors do not add, their quaternions multiply."""
        a, b, c = (random_poses(32, seed=s) for s in (20, 21, 22))
        stepwise = apply_pose_delta(apply_pose_delta(a, pose_delta(a, b)),
                                    pose_delta(b, c))
        direct = apply_pose_delta(a, pose_delta(a, c))
        assert np.allclose(stepwise[:, :3], direct[:, :3], atol=1e-12)
        for i in range(len(a)):
            assert quat_close(stepwise[i, 3:], direct[i, 3:], tol=1e-9)

    def test_translation_deltas_add(self):
        a, b, c = (random_poses(32, seed=s) for s in (23, 24, 25))
        assert np.allclose(pose_delta(a, b)[:, :3] + pose_delta(b, c)[:, :3],
                           pose_delta(a, c)[:, :3], atol=1e-12)

    def test_delta_is_antisymmetric(self):
        a, b = random_poses(32, seed=26), random_poses(32, seed=27)
        assert np.allclose(pose_delta(a, b)[:, :3], -pose_delta(b, a)[:, :3])
        # The rotation half negates too, because both are the short way round.
        assert np.allclose(pose_delta(a, b)[:, 3:], -pose_delta(b, a)[:, 3:],
                           atol=1e-9)

    def test_long_chain_stays_on_the_unit_sphere(self):
        """A rollout composes hundreds of deltas. Without renormalisation the
        quaternion drifts off the unit sphere and the rotation quietly gains a
        scale factor."""
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        step = np.concatenate([np.full(3, 1e-3), axis(28) * 0.01])
        for _ in range(2000):
            pose = apply_pose_delta(pose, step)
        assert np.linalg.norm(pose[3:]) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

class TestActionEncoding:
    def test_encode_decode_round_trip(self):
        a, b = random_poses(64, seed=30), random_poses(64, seed=31)
        rng = np.random.default_rng(32)
        jaw = rng.uniform(0, 1.0, size=64)
        act = encode_action(a, b, jaw)
        assert act.shape == (64, 7)
        pose_next, jaw_back = decode_action(a, act)
        assert np.allclose(pose_next[:, :3], b[:, :3], atol=1e-12)
        assert np.allclose(jaw_back, jaw)
        for i in range(len(b)):
            assert quat_close(pose_next[i, 3:], b[i, 3:], tol=1e-9)

    def test_single_pose_shapes(self):
        a, b = random_poses(1, seed=33)[0], random_poses(1, seed=34)[0]
        act = encode_action(a, b, 0.4)
        assert act.shape == (7,)
        pose_next, jaw = decode_action(a, act)
        assert pose_next.shape == (7,) and float(jaw) == pytest.approx(0.4)

    def test_actions_are_small_and_zero_centred(self):
        """The property the whole design exists for: a delta action space is
        bounded and centred at zero, which is what MPPI needs to sample in."""
        poses = random_poses(200, seed=35, scale=0.02)
        act = encode_action(poses[:-1], poses[1:], 0.0)
        assert np.abs(act[:, :3]).max() < 0.5
        assert np.abs(np.mean(act[:, :3])) < 0.02

    def test_wrong_width_raises(self):
        a = random_poses(1, seed=36)[0]
        with pytest.raises(ValueError, match=r"\(\.\.\., 7\)"):
            decode_action(a, np.zeros(6))


class TestAbsoluteToDeltaActions:
    def test_shape_and_reconstruction(self):
        poses = random_poses(50, seed=40)
        jaw = np.linspace(0, 1, 50)
        act = absolute_to_delta_actions(poses, jaw)
        assert act.shape == (50, 7)
        # Every action but the pad reconstructs the next pose exactly.
        got = apply_pose_delta(poses[:-1], act[:-1, :6])
        assert np.allclose(got[:, :3], poses[1:, :3], atol=1e-12)
        for i in range(len(got)):
            assert quat_close(got[i, 3:], poses[i + 1, 3:], tol=1e-9)

    def test_jaw_is_the_commanded_end_of_step_value(self):
        poses = random_poses(6, seed=41)
        jaw = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        act = absolute_to_delta_actions(poses, jaw)
        assert np.allclose(act[:-1, 6], jaw[1:])
        assert act[-1, 6] == pytest.approx(jaw[-1])

    def test_final_row_is_a_zero_pad(self):
        """There is no pose T, so the last delta is unknowable. It must be an
        obvious zero rather than a plausible-looking guess."""
        act = absolute_to_delta_actions(random_poses(20, seed=42), np.zeros(20))
        assert np.allclose(act[-1, :6], 0.0)

    def test_single_step_episode(self):
        act = absolute_to_delta_actions(random_poses(1, seed=43), np.array([0.7]))
        assert act.shape == (1, 7)
        assert np.allclose(act[0, :6], 0.0) and act[0, 6] == pytest.approx(0.7)

    def test_stationary_trajectory_gives_zero_actions(self):
        poses = np.tile(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]), (30, 1))
        act = absolute_to_delta_actions(poses, np.zeros(30))
        assert np.allclose(act[:, :6], 0.0, atol=1e-15)

    def test_mismatched_jaw_length_raises(self):
        with pytest.raises(ValueError, match="jaw has"):
            absolute_to_delta_actions(random_poses(10, seed=44), np.zeros(9))


class TestOnRealV1Data:
    """The conversion has to survive an actual recorded episode, not just
    synthetic poses. v1 episodes hold a constant identity orientation, which is
    exactly the degenerate case the atan2 rule exists for."""

    def test_converts_a_recorded_episode(self):
        import glob

        from trajectory_io import load_trajectory
        files = sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "*.npz")))
        if not files:
            pytest.skip("no recorded episodes in data/")
        tr = load_trajectory(files[0])
        act = absolute_to_delta_actions(tr.ee_pose, tr.jaw)
        assert act.shape == (len(tr), 7)
        assert np.all(np.isfinite(act)), "identity-quaternion episode produced NaN"
        # The gripper never rotates in these episodes, so every rotation delta
        # must be exactly zero -- not merely small.
        assert np.abs(act[:, 3:6]).max() == 0.0
        got = apply_pose_delta(tr.ee_pose[:-1].astype(np.float64), act[:-1, :6])
        assert np.allclose(got[:, :3], tr.ee_pose[1:, :3], atol=1e-6)
