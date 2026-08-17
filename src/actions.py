"""
actions.py -- what the controller commands, in a form a sampler can perturb.

WHY THIS FILE EXISTS
--------------------
The action space is chosen for the planner, not for the simulator. MPPI and CEM
work by drawing thousands of perturbations around a nominal action sequence, and
that sampling is only well behaved when the action space is small, bounded and
centred at zero. Two consequences follow, and this file exists to enforce them
in one place rather than in every collector.

1. DELTAS, NOT ABSOLUTE POSES.
   An absolute pose is a point in a workspace whose centre is arbitrary; a
   Gaussian around it is not a natural distribution over anything. A delta is
   zero-centred by construction.

2. AXIS-ANGLE FOR ROTATION DELTAS, QUATERNION FOR ABSOLUTE POSES.
   Quaternions double-cover the rotation group -- q and -q are the same
   rotation -- so a network regressing a quaternion sees two correct answers
   for every input and is penalised for picking either. Small rotations written
   as axis-angle are smooth through zero and have no such ambiguity. Absolute
   poses stay as quaternions because that is what the schema and every
   simulator use, and because axis-angle is singular at pi.

THE atan2 RULE
--------------
Extract a rotation angle with

    theta = 2 * atan2(|q_v|, q_w)      NOT    theta = 2 * acos(q_w)

Normalising a quaternion in floating point can leave q_w a few ulps above 1.0,
and acos(1.0000000000000002) is NaN. The NaN then propagates through a whole
trajectory of composed rotations and is discovered much later, as a dataset
full of holes. atan2 is total: it is defined for every pair of finite inputs
and needs no clamping. This has bitten every codebase that has implemented this
pattern.

CONVENTIONS, STATED ONCE
------------------------
- Quaternion layout is [qx, qy, qz, qw] -- SCALAR LAST. This is not a
  preference; it is what `ee_pose` already stores in the trajectory schema, and
  a file that mixes the two conventions is unrecoverable after the fact.
- A pose is (7,): [x, y, z, qx, qy, qz, qw].
- A pose delta is (6,): [dx, dy, dz, rx, ry, rz], rotation vector, magnitude in
  radians.
- An action is (7,): the pose delta plus the jaw angle.
- Rotation deltas are expressed in the WORLD frame, so that

      q_b = delta_rotation * q_a           (delta applied on the left)

  Both halves of the delta are then in the same frame, which is what makes
  `pose_delta` compose: translation deltas add, and the corresponding rotations
  multiply, with no per-step change of basis to get wrong.

numpy only, and it must import under both numpy 1.23 (the container pin) and
numpy 2.x (the host env).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Below this quaternion vector norm, the axis of rotation is not determined
# (there is no rotation to have an axis) and the series expansion is used
# instead. Chosen well above float64 epsilon so the expansion is entered only
# where the direct formula would actually lose precision.
_TINY = 1e-12


# --------------------------------------------------------------------------
# Rotation representations
# --------------------------------------------------------------------------

def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Quaternion [qx,qy,qz,qw] -> rotation vector (axis * angle in radians).

    Accepts (..., 4) and returns (..., 3).

    The double cover is resolved by flipping any quaternion with qw < 0, which
    picks the rotation of angle <= pi -- the short way round. Without this a
    delta of 1 degree and one of 359 degrees are both representable, and the
    sampler that is supposed to be exploring small perturbations occasionally
    draws a near-full revolution.
    """
    q = _as_quat(q)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    # Short way round. `<` not `<=`: at qw == 0 the rotation is exactly pi and
    # both signs are equally valid, so leaving it alone keeps the map stable.
    q = np.where(q[..., 3:4] < 0.0, -q, q)

    qv, qw = q[..., :3], q[..., 3]
    n = np.linalg.norm(qv, axis=-1)

    # theta = 2*atan2(n, qw), and the rotation vector is qv * (theta / n).
    # As n -> 0 that ratio tends to 2/qw, which is finite. Both branches are
    # evaluated by np.where, so both denominators are made safe first --
    # otherwise the unused branch still emits a divide-by-zero warning and,
    # worse, an inf that np.where then has to discard.
    n_safe = np.where(n > _TINY, n, 1.0)
    qw_safe = np.where(np.abs(qw) > _TINY, qw, 1.0)
    scale = np.where(n > _TINY, 2.0 * np.arctan2(n, qw) / n_safe, 2.0 / qw_safe)
    return qv * scale[..., None]


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation vector -> quaternion [qx,qy,qz,qw]. Accepts (...,3) -> (...,4).

    The exact inverse of `quat_to_rotvec` up to the sign convention, which is
    fixed by always returning qw >= 0.
    """
    r = np.asarray(r, np.float64)
    if r.shape[-1] != 3:
        raise ValueError(f"expected rotation vectors of shape (..., 3), got {r.shape}")
    theta = np.linalg.norm(r, axis=-1)
    half = 0.5 * theta
    # qv = r * sin(half)/theta. The ratio tends to 1/2 as theta -> 0, which is
    # the value a first-order expansion gives and is exact to float64 there.
    theta_safe = np.where(theta > _TINY, theta, 1.0)
    scale = np.where(theta > _TINY, np.sin(half) / theta_safe, 0.5)
    return np.concatenate([r * scale[..., None], np.cos(half)[..., None]], axis=-1)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b for scalar-last quaternions. (...,4) x (...,4)."""
    a, b = _as_quat(a), _as_quat(b)
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = b[..., :3], b[..., 3:4]
    return np.concatenate([aw * bv + bw * av + np.cross(av, bv),
                           aw * bw - np.sum(av * bv, axis=-1, keepdims=True)], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Inverse rotation, for unit quaternions."""
    q = _as_quat(q)
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


# --------------------------------------------------------------------------
# Poses
# --------------------------------------------------------------------------

def pose_delta(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """The world-frame motion from pose_a to pose_b. (...,7),(...,7) -> (...,6).

    Returns [dx, dy, dz, rx, ry, rz]: a translation in world coordinates and a
    rotation vector such that `apply_pose_delta(pose_a, delta)` reproduces
    pose_b exactly.
    """
    a, b = _as_pose(pose_a), _as_pose(pose_b)
    dp = b[..., :3] - a[..., :3]
    # q_b = q_rel * q_a  =>  q_rel = q_b * q_a^-1. Left-multiplication is what
    # makes this a WORLD-frame rotation, matching the world-frame translation
    # above. Right-multiplication (q_a^-1 * q_b) would be the tool-frame delta:
    # also defensible, but then the two halves live in different frames and
    # composing deltas silently stops working.
    q_rel = quat_multiply(b[..., 3:], quat_conjugate(a[..., 3:]))
    return np.concatenate([dp, quat_to_rotvec(q_rel)], axis=-1)


def apply_pose_delta(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a world-frame delta to a pose. (...,7),(...,6) -> (...,7)."""
    p = _as_pose(pose)
    d = np.asarray(delta, np.float64)
    if d.shape[-1] != 6:
        raise ValueError(f"expected pose deltas of shape (..., 6), got {d.shape}")
    xyz = p[..., :3] + d[..., :3]
    q = quat_multiply(rotvec_to_quat(d[..., 3:]), p[..., 3:])
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)   # renormalise: composing
                                                        # many deltas otherwise
                                                        # drifts off the unit
                                                        # sphere over a rollout
    return np.concatenate([xyz, q], axis=-1)


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

def encode_action(pose_t: np.ndarray, pose_next: np.ndarray,
                  jaw: float) -> np.ndarray:
    """(pose_t, pose_next, jaw) -> the (7,) action [dx,dy,dz,rx,ry,rz,jaw].

    `jaw` is the COMMANDED jaw angle -- where the gripper should be at the end
    of the step, not where it was at the start. See `absolute_to_delta_actions`.
    """
    d = pose_delta(pose_t, pose_next)
    jaw_arr = np.asarray(jaw, np.float64)
    return np.concatenate([d, np.broadcast_to(jaw_arr[..., None], d.shape[:-1] + (1,))],
                          axis=-1)


def decode_action(pose_t: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The inverse of `encode_action`: (...,7),(...,7) -> (pose_next, jaw)."""
    a = np.asarray(action, np.float64)
    if a.shape[-1] != 7:
        raise ValueError(
            f"expected actions of shape (..., 7) = [dx,dy,dz,rx,ry,rz,jaw], "
            f"got {a.shape}")
    return apply_pose_delta(pose_t, a[..., :6]), a[..., 6]


def absolute_to_delta_actions(ee_pose: np.ndarray, jaw: np.ndarray) -> np.ndarray:
    """Convert a recorded absolute-pose trajectory into delta actions. (T,7).

    For re-reading v1 episodes, whose `action` field is an absolute target and
    therefore unusable by a sampler that perturbs around zero.

    TWO THINGS TO KNOW ABOUT ROW T-1. There is no pose T, so the final delta is
    unknowable and is written as zeros. It is a pad, not a measurement -- drop
    it when building one-step training pairs, exactly as `make_transition_pairs`
    already drops the last state.

    The jaw component of row t is jaw[t+1], not jaw[t]: an action is what
    carries the system from t to t+1, so its jaw entry is the angle in effect at
    the END of the step. Pairing jaw[t] with the delta out of t would teach a
    model that the jaw closes one step after it actually does.
    """
    poses = _as_pose(ee_pose)
    jaw = np.asarray(jaw, np.float64).reshape(-1)
    if poses.ndim != 2:
        raise ValueError(f"expected ee_pose of shape (T, 7), got {poses.shape}")
    if jaw.shape[0] != poses.shape[0]:
        raise ValueError(f"jaw has {jaw.shape[0]} entries, ee_pose has {poses.shape[0]}")

    T = poses.shape[0]
    out = np.zeros((T, 7), np.float64)
    if T > 1:
        out[:-1, :6] = pose_delta(poses[:-1], poses[1:])
        out[:-1, 6] = jaw[1:]
    # Row T-1 keeps its zero delta; its jaw is the last known angle, which is
    # the only honest value available.
    out[-1, 6] = jaw[-1]
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _as_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    if q.shape[-1] != 4:
        raise ValueError(
            f"expected quaternions of shape (..., 4) laid out [qx,qy,qz,qw] "
            f"(scalar LAST, as ee_pose stores them), got {q.shape}")
    return q


def _as_pose(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float64)
    if p.shape[-1] != 7:
        raise ValueError(
            f"expected poses of shape (..., 7) = [x,y,z,qx,qy,qz,qw], got {p.shape}")
    return p
