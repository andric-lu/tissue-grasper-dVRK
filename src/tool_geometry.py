"""tool_geometry.py -- distances from tissue nodes to the jaw boxes, and the
node-level contact / grasp rules built on them.

numpy only (CLAUDE.md `src/` rule): the collector on the host, the validator,
the graph builder and -- later -- inference on predicted tissue all import this
one file, so the rule that labels training data and the rule a model must apply
to its own predictions cannot drift apart.

WHY THIS FILE EXISTS (to-do Items 1, 6, 10; decision D3)
-------------------------------------------------------
The solver's grasp (host/grasp_constraint.py) latches onto FULL-RESOLUTION
particles that sit within GRASP_CAPTURE_RADIUS_M of BOTH jaw boxes when the rule
is evaluated. That is typically 1-3 of 24,000 particles, and a 3,000-node
recorded subset usually contains none of them. So "is node i grasped" cannot
simply be read off the solver: it needs a definition at node level. D3 chose:

    g_grasp: the solver's own two-jaw rule, evaluated on the NODES, with the
             solver's own latch/release timing. Needs no future information,
             so a model can apply it to its own predicted positions at
             inference (Item 10: "no silent ground truth").
    g_contact: node within `contact_radius` of either jaw box surface. Item 1
             asks for contact and grasp as SEPARATE indicators; Item 6 says
             proximity is not proof of contact, so this is named a proximity
             flag everywhere, never "in contact".

Both are stand-ins. g_grasp is a node-level proxy for 1-3 full-resolution
particles, and the episode file keeps the solver's true full-resolution set
(grasp_full_ids) so Item 11 can be checked against truth.

CONVENTIONS
-----------
- Box poses are world-frame [x, y, z, qx, qy, qz, qw], scalar-last.
- Half-extents are in the box's own frame, metres.
- SIGNED distance: positive outside the box, NEGATIVE inside (minus the depth
  to the nearest face), zero on the surface. The solver's rule uses the
  UNSIGNED distance max(d, 0), i.e. zero anywhere inside -- see
  `unsigned_box_distance`, which reproduces grasp_constraint.py exactly.
- Normal: unit vector pointing from the nearest surface point OUT of the box,
  toward the particle when it is outside. For a particle inside, the outward
  normal of the nearest face. For a particle exactly on an edge/corner from
  outside, the direction to the particle (the gradient of the distance).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# The rule's defaults, copied from where the solver defines them. They are
# recorded into every v3 episode (grasp_capture_radius, jaw_release_rad,
# contact_radius) and readers should use the RECORDED values; these constants
# only seed the collector. host/grasp_constraint.py imports the two solver
# values from here so the collector and the rule cannot disagree.
GRASP_CAPTURE_RADIUS_M = 0.02      # grasp_constraint.GRASP_CAPTURE_RADIUS_M
JAW_RELEASE_TRIGGER_RAD = 0.3      # grasp_constraint.JAW_RELEASE_TRIGGER_RAD
# Proximity flag radius. Set equal to the capture radius so that "close enough
# to be grasped" and "flagged as near a jaw" are one scale; the tool-edge
# radius r_c (Item 6) is a separate, configurable graph parameter.
CONTACT_RADIUS_M = 0.02


# How each arm's recorded `jaw` angle maps to its two jaw JOINT angles
# (jaw_joint = jaw * factor). Si: jaw is per-jaw, jaw_joint_2 mimics -jaw_joint_1.
# Classic (SurRoL Psm.move_jaw): jaw is the TOTAL opening, joints at -jaw/2, +jaw/2.
# Episodes record `robot_model`; a v3 file without it predates the Classic
# arm and is Si.
JAW_JOINT_FACTORS = {
    "dvrk_si_psm": (1.0, -1.0),
    "dvrk_classic_psm_surrol": (-0.5, 0.5),
}


def rotmat_from_quat(q: np.ndarray) -> np.ndarray:
    """[qx,qy,qz,qw] (...,4) -> (...,3,3). Normalises first.

    A numpy twin of host/psm.rotmat_from_quat (which wraps PyBullet), so this
    file stays importable without PyBullet. Tested against it on the host by
    tests/test_tool_geometry.py when PyBullet is present.
    """
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> scalar-last [qx,qy,qz,qw], qw >= 0 (Shepperd's method:
    branch on the largest of trace and the diagonal, so no branch divides by
    a small number). Tested as the inverse of rotmat_from_quat in tests/test_tool_geometry.py."""
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    d = np.diag(R)
    k = int(np.argmax([t, d[0], d[1], d[2]]))
    if k == 0:
        s = np.sqrt(1.0 + t) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    elif k == 1:
        s = np.sqrt(1.0 + d[0] - d[1] - d[2]) * 2
        q = [s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif k == 2:
        s = np.sqrt(1.0 + d[1] - d[0] - d[2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + d[2] - d[0] - d[1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4, (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def compose_box_pose(link_pose: np.ndarray, box_offset: np.ndarray) -> np.ndarray:
    """T_world_box = T_world_link . T_link_box, for (...,7) poses.

    How a jaw box is reconstructed from a jaw-link pose and the episode's
    recorded `jaw_box_offset`. Same composition host/psm.py does by hand.
    """
    lp = np.asarray(link_pose, np.float64)
    off = np.asarray(box_offset, np.float64)
    R = rotmat_from_quat(lp[..., 3:7])
    pos = lp[..., :3] + np.einsum("...ij,...j->...i", R, off[..., :3])
    # Hamilton product q_link * q_off, scalar-last.
    a, b = lp[..., 3:7], off[..., 3:7]
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = b[..., :3], b[..., 3:4]
    q = np.concatenate([aw * bv + bw * av + np.cross(av, bv),
                        aw * bw - np.sum(av * bv, axis=-1, keepdims=True)], axis=-1)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.concatenate([pos, q], axis=-1)


def box_local(points: np.ndarray, box_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Points expressed in the box frame. Returns (local (N,3), R (3,3))."""
    p = np.asarray(points, np.float64)
    bp = np.asarray(box_pose, np.float64)
    R = rotmat_from_quat(bp[3:7])
    # Row-vector form of R^T (x - c), as host/grasp_constraint.py writes it.
    return (p - bp[:3]) @ R, R


def unsigned_box_distance(points: np.ndarray, box_pose: np.ndarray,
                          half_extents: np.ndarray) -> np.ndarray:
    """(N,) Euclidean distance to the box, 0 inside. Bit-for-bit the formula
    host/grasp_constraint.py and host/mpm_adapter._contact_mode use."""
    local, _ = box_local(points, box_pose)
    h = np.asarray(half_extents, np.float64)
    return np.linalg.norm(np.maximum(np.abs(local) - h, 0.0), axis=-1)


def box_nearest_point(points: np.ndarray, box_pose: np.ndarray,
                      half_extents: np.ndarray) -> Dict[str, np.ndarray]:
    """Nearest point on the box SURFACE to each point, with signed distance and
    outward normal. All outputs world frame.

    Returns {"nearest" (N,3), "signed_distance" (N,), "normal" (N,3)}.
    See the module docstring for the sign and normal conventions.
    """
    local, R = box_local(points, box_pose)
    h = np.asarray(half_extents, np.float64)
    if np.any(h <= 0):
        raise ValueError(f"half_extents must be positive, got {h}")

    clamped = np.clip(local, -h, h)
    diff = local - clamped
    d_out = np.linalg.norm(diff, axis=-1)
    outside = d_out > 0.0

    nearest = clamped.copy()
    normal = np.zeros_like(local)
    signed = d_out.copy()

    # Outside: the clamp IS the nearest surface point; normal is the direction
    # from it to the particle.
    if outside.any():
        normal[outside] = diff[outside] / d_out[outside, None]

    # Inside (or exactly on the surface): push out through the nearest face.
    inside = ~outside
    if inside.any():
        li = local[inside]
        depth = h - np.abs(li)                      # (n,3) distance to each face pair
        axis = np.argmin(depth, axis=-1)            # nearest face's axis
        rows = np.arange(li.shape[0])
        sign = np.where(li[rows, axis] >= 0.0, 1.0, -1.0)
        ni = li.copy()
        ni[rows, axis] = sign * h[axis]
        nearest[inside] = ni
        nrm = np.zeros_like(li)
        nrm[rows, axis] = sign
        normal[inside] = nrm
        signed[inside] = -depth[rows, axis]

    bp = np.asarray(box_pose, np.float64)
    return {"nearest": nearest @ R.T + bp[:3],
            "signed_distance": signed,
            "normal": normal @ R.T}


# --------------------------------------------------------------------------
# Node-level indicators (decision D3)
# --------------------------------------------------------------------------

def jaw_distances(points: np.ndarray, box_poses: np.ndarray,
                  half_extents: np.ndarray) -> np.ndarray:
    """(K, N) unsigned distance from every point to every jaw box."""
    bp = np.asarray(box_poses, np.float64)
    he = np.asarray(half_extents, np.float64)
    if bp.shape[0] != he.shape[0]:
        raise ValueError(f"{bp.shape[0]} box poses but {he.shape[0]} half-extent rows")
    return np.stack([unsigned_box_distance(points, bp[k], he[k])
                     for k in range(bp.shape[0])])


def node_contact_flags(points: np.ndarray, box_poses: np.ndarray,
                       half_extents: np.ndarray,
                       contact_radius: float = CONTACT_RADIUS_M) -> np.ndarray:
    """(N,) bool: within `contact_radius` of ANY jaw box surface.

    A proximity flag. It says a node is near enough to a jaw that contact is
    plausible -- the solver's sticky boundary acts through a halo of the same
    size -- not that a force is being transmitted.
    """
    return (jaw_distances(points, box_poses, half_extents) < contact_radius).any(axis=0)


def grasp_capture_candidates(points: np.ndarray, box_poses: np.ndarray,
                             half_extents: np.ndarray,
                             capture_radius: float = GRASP_CAPTURE_RADIUS_M) -> np.ndarray:
    """(N,) bool: the solver's freeze predicate -- within `capture_radius` of
    BOTH jaw boxes (host/grasp_constraint.GraspConstraint.maybe_freeze)."""
    d = jaw_distances(points, box_poses, half_extents)
    if d.shape[0] != 2:
        raise ValueError(f"the two-jaw rule needs exactly 2 jaws, got {d.shape[0]}")
    return (d[0] < capture_radius) & (d[1] < capture_radius)


def grasp_latch_step(latched: np.ndarray, points: np.ndarray, box_poses: np.ndarray,
                     half_extents: np.ndarray, jaw_angle: float, *,
                     capture_radius: float = GRASP_CAPTURE_RADIUS_M,
                     release_rad: float = JAW_RELEASE_TRIGGER_RAD) -> np.ndarray:
    """One row of the grasp latch, in the solver's own order: freeze, then release.

    Mirrors record_grasp_episode()'s `_capture_row`, which calls
    GraspConstraint.maybe_freeze() and then maybe_release() with the row's jaw
    angle:

      1. If nothing is latched, latch every node satisfying the two-jaw
         predicate (possibly none). Membership is chosen ONCE per grasp and
         never grows or shrinks while held -- exactly like the solver, which
         never re-evaluates the predicate while active.
      2. If anything is latched and jaw_angle >= release_rad, clear it.

    `latched` is not modified; the new (N,) bool state is returned. Pure, so a
    rollout can carry the state alongside its predicted positions.
    """
    held = np.asarray(latched, bool)
    if not held.any():
        held = grasp_capture_candidates(points, box_poses, half_extents, capture_radius)
    if held.any() and float(jaw_angle) >= release_rad:
        held = np.zeros_like(held)
    return held


def derive_node_grasp(tissue_pos: np.ndarray, jaw_box_pose: np.ndarray,
                      half_extents: np.ndarray, jaw: np.ndarray, *,
                      capture_radius: float = GRASP_CAPTURE_RADIUS_M,
                      release_rad: float = JAW_RELEASE_TRIGGER_RAD) -> np.ndarray:
    """(T, N) bool: run `grasp_latch_step` over a whole recorded episode.

    The collector writes `node_grasp` with this; the validator re-runs it from
    the stored positions and jaw poses and demands the same answer.
    """
    T, N = tissue_pos.shape[:2]
    out = np.zeros((T, N), bool)
    state = np.zeros(N, bool)
    for t in range(T):
        state = grasp_latch_step(state, tissue_pos[t], jaw_box_pose[t], half_extents,
                                 float(jaw[t]), capture_radius=capture_radius,
                                 release_rad=release_rad)
        out[t] = state
    return out


def derive_node_contact(tissue_pos: np.ndarray, jaw_box_pose: np.ndarray,
                        half_extents: np.ndarray,
                        contact_radius: float = CONTACT_RADIUS_M) -> np.ndarray:
    """(T, N) bool: `node_contact_flags` at every row."""
    return np.stack([node_contact_flags(tissue_pos[t], jaw_box_pose[t], half_extents,
                                        contact_radius)
                     for t in range(tissue_pos.shape[0])])


def box_velocity_backward(box_pose: np.ndarray, dt: float) -> np.ndarray:
    """(T,K,7) world box poses -> (T,K,6) [v, omega] by backward difference.

    Row 0 is zero: the arm starts at rest, the same first-frame convention
    host/psm.PSM.update_collision_velocity uses. Row t is the motion over
    (t-1 -> t); the motion the solver applies over (t -> t+1) is row t+1.
    Angular velocity is the rotation vector of q_t * q_{t-1}^-1 over dt,
    extracted with 2*atan2(|q_v|, q_w) (CLAUDE.md), never acos.
    """
    bp = np.asarray(box_pose, np.float64)
    out = np.zeros(bp.shape[:-1] + (6,))
    if bp.shape[0] < 2:
        return out
    out[1:, ..., :3] = (bp[1:, ..., :3] - bp[:-1, ..., :3]) / dt
    a, b = bp[1:, ..., 3:7], bp[:-1, ..., 3:7]
    bconj = np.concatenate([-b[..., :3], b[..., 3:]], axis=-1)
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = bconj[..., :3], bconj[..., 3:4]
    q = np.concatenate([aw * bv + bw * av + np.cross(av, bv),
                        aw * bw - np.sum(av * bv, axis=-1, keepdims=True)], axis=-1)
    q = np.where(q[..., 3:4] < 0.0, -q, q)          # short way round
    qv, qw = q[..., :3], q[..., 3]
    n = np.linalg.norm(qv, axis=-1)
    n_safe = np.where(n > 1e-12, n, 1.0)
    qw_safe = np.where(np.abs(qw) > 1e-12, qw, 1.0)
    scale = np.where(n > 1e-12, 2.0 * np.arctan2(n, qw) / n_safe, 2.0 / qw_safe)
    out[1:, ..., 3:] = qv * scale[..., None] / dt
    return out
