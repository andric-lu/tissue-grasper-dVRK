#!/usr/bin/env python3
"""
psm.py -- drive the dVRK Si PSM (psm_Si_model/) as a rigid collider for the
Taichi MPM solver, on host (macOS).

    conda activate tissue-host
    (imported by host/mpm_adapter.py; not run standalone -- see
     host/smoke_test_psm.py for a driver)

WHAT THIS OWNS. All PyBullet state: the one connection this process ever
opens, the URDF, inverse kinematics, the two jaw collision proxies, and
pushing collider geometry/velocity into the vendored solver's `co_obj`/`co_v`/
`co_w`/`centroid`/`SDF` fields (`third_party/MPM/{mpm3d,sdf}.py`). It never
imports `taichi` or `MPM.mpm3d` itself -- both are handed in already imported,
because importing `MPM.mpm3d` runs `ti.init()` (DECISION_LOG.md 9.4), and
`host/mpm_adapter.py`'s `MPMRecorder.__init__` is the sole place that is
allowed to trigger that. A `PSM` can therefore only be constructed AFTER
`MPMRecorder.__init__` has already run -- see `record_grasp_episode()` in
mpm_adapter.py for the exact ordering.

FOUR THINGS THIS CLASS EXISTS SPECIFICALLY TO GET RIGHT, because the vendored
collision path (`third_party/MPM/sdf.py`) has sharp, undocumented edges. Each
is checked by host/smoke_test_psm.py, not just asserted here.

1. `co_obj[i][1] == -1` (PyBullet's "base" convention) makes `sdf.py` read a
   256^3 precomputed SDF field (`static_sdf`) that NOTHING in this repository
   ever initialises -- it sits at Taichi's zero default, i.e. a phantom
   zero-distance surface across the whole grid the instant that slot is
   queried. Every `co_obj` slot here is therefore a real LINK (index >= 0) of
   a one-link proxy body, never a body's base.
2. `sdf.py`'s box path reads ONLY `p.getCollisionShapeData(...)[0][3]` (the
   shape's extents) -- it never reads a collision shape's local frame offset
   (`collisionFramePosition`). A box's assumed world pose is entirely
   whatever `i_rot_list`/`i_pos_list` this module supplies; nothing about
   where the shape "really" sits inside its own body is consulted. The two
   jaw proxies are therefore built as ZERO-OFFSET boxes at their own link
   origin, and the AABB-derived offset from the real jaw mesh
   (`local_center`) is tracked here, in numpy, and composed by hand every
   frame (`_jaw_box_pose`) -- never delegated to PyBullet's own local-frame
   machinery, which `sdf.py` cannot see.
3. `mpm3d.init_pos()` populates `sdf.py`'s `position` grid field
   (`sdf.init_switch_base`); nothing else does. Owned here, once, in
   `__init__` -- `MPMRecorder._init_solver()` never calls it, robot or not.
4. `sdf.py`'s PyBullet calls (`p.getCollisionShapeData`, ...) never pass
   `physicsClientId` -- they implicitly target client 0. This class is the
   only thing in a collection process allowed to call `p.connect()`, and
   asserts it got client 0.

PYBULLET AS AN FK/IK LIBRARY, NOT A PHYSICS ENGINE. Every joint is driven with
`p.resetJointState` (teleport), never `setJointMotorControl2` -- most of this
URDF's joints declare `<limit effort="0" velocity="0">` (kinematic-display
authoring, not torque control), and `container/collect_retraction.py`'s own
gripper-block proxy uses the same teleport idiom for a different reason
(zero-mass kinematic actuator). `p.stepSimulation()` is never called anywhere
in this file. Every `getLinkState` call passes `computeForwardKinematics=1`
explicitly -- PyBullet does not guarantee fresh forward kinematics after
`resetJointState` without it, and this file has no `stepSimulation()` call to
refresh the cache implicitly.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pybullet as p

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import actions  # noqa: E402 -- quat_multiply/quat_conjugate/quat_to_rotvec reused
                # directly rather than re-deriving axis-angle math; also the
                # scalar-last [qx,qy,qz,qw] convention this file must match.

URDF_PATH = os.path.join(REPO, "psm_Si_model", "psm_si_surrol.urdf")

# The 7 joints IK actually drives, in the order joint_pos records them. j2/j3
# are excluded: reading the URDF's own <parent>/<child> tags, link_1 has TWO
# children -- j4 (real tool chain) and j2->link_2->j3->link_3 (the dVRK's
# decorative parallelogram linkage). Neither link_2 nor link_3 is an ancestor
# of tool_gripper_center, so driving j2/j3 has no effect on tool pose or
# collision; they are pinned instead (INERT_JOINTS, below).
DRIVEN_JOINTS = ("j1", "j4", "outer_insertion",
                  "tool_roll_joint", "tool_pitch_joint", "tool_yaw_joint",
                  "jaw_joint_1")
EE_LINK = "tool_gripper_center"          # the asset's own EE reference point
JAW_LINKS = ("tool_gripper_link_1", "tool_gripper_link_2")
INERT_JOINTS = ("j2", "j3")              # pinned to 0.0, never driven -- see above
MIMIC_JAW_JOINT = "jaw_joint_2"          # = -jaw_joint_1; PyBullet has no <mimic>

# IK-residual gate: an unreachable target must fail loudly in set_ee_pose(),
# not drift silently through the rest of an episode.
IK_POSITION_TOLERANCE_M = 0.002
IK_ORIENTATION_TOLERANCE_RAD = 0.02

# Per-frame collider sweep gate, checked in update_collision_velocity(). The
# translation bound matches the cadence justification for refreshing
# collision once per recorded frame rather than once per substep: more than
# half a grid cell of undersampled motion per frame risks missing contact.
# The rotation bound is chosen so jaw_joint_1's full close sweep (0 to
# 1.5717 rad, its own URDF <limit>) stays inside it as long as the closing
# phase spans at least ~8 frames -- checked against the real scripted
# trajectory in host/smoke_test_psm.py, not just asserted here.
MAX_TRANSLATION_FRACTION_OF_DX = 0.5
MAX_ROTATION_RAD_PER_FRAME = 0.2

# Placed once, far outside the unit-cube domain, never moved again -- the
# inert third co_obj slot (MAX_COLLISION_OBJECTS=3, mpm3d.py:26). Still a
# real link (never co_obj[i][1]==-1, see module docstring point 1).
DUMMY_POSITION = np.array([100.0, 100.0, 100.0])

# Picked by inspection (workspace_aabb(), not derived analytically -- the
# chain has too many compounded rotated offsets, e.g. j1's own origin
# translates 0.834 m along a rotated axis before any wrist joint applies).
# At base_position=[0,0,0], identity orientation, tool_gripper_center's
# reachable region over the full joint ranges sits around x~0.79-0.84,
# y~+-0.22, z~-0.26..0.03 relative to link_0. Tissue occupies
# x in [0.1,0.4], y in [0.2,0.5], z in [0.05,0.10] domain-metres
# (init_cube(), mpm3d.py:369-377). This placement's sampled workspace bbox
# (500 samples, workspace_aabb()) is x[0.006,0.516] y[0.117,0.564]
# z[-0.058,0.445] -- comfortably contains the tissue region on every side,
# no base rotation needed.
DEFAULT_BASE_POSITION = (-0.565, 0.35, 0.195)
DEFAULT_BASE_ORIENTATION = (0.0, 0.0, 0.0, 1.0)


def rotmat_from_quat(quat: np.ndarray) -> np.ndarray:
    return np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)


class PSM:
    """One dVRK Si PSM, driven kinematically, colliding with the MPM tissue
    through two jaw-shaped box proxies.

    One instance per process (see the module docstring, point 4): the
    PyBullet connection it opens is the only one `third_party/MPM/sdf.py`'s
    unparameterised calls can possibly resolve to correctly.
    """

    def __init__(self, mpm3d_module, sdf_module, *,
                 base_position: Tuple[float, float, float],
                 base_orientation: Tuple[float, float, float, float],
                 urdf_path: str = URDF_PATH):
        self.client = p.connect(p.DIRECT)
        # Load-bearing: sdf.py's own p.getCollisionShapeData(...) call carries
        # no physicsClientId and therefore implicitly targets client 0. If
        # this assertion ever fires, something else in this process already
        # held a PyBullet connection before this constructor ran.
        if self.client != 0:
            raise RuntimeError(
                f"p.connect() returned client {self.client}, not 0 -- "
                "another PyBullet connection already exists in this process. "
                "third_party/MPM/sdf.py's calls have no physicsClientId and "
                "would silently target the wrong client. One PSM per process.")

        self.body = p.loadURDF(
            urdf_path, basePosition=list(base_position),
            baseOrientation=list(base_orientation), useFixedBase=True)
        # useFixedBase=True is required: link_0's parent joint is commented
        # out in the URDF, so without this PyBullet treats the whole 13-link
        # arm as a free-floating 6-DOF root.

        self._joint_index: Dict[str, int] = {}
        self._link_index: Dict[str, int] = {}
        self._joint_limits: Dict[str, Tuple[float, float]] = {}
        n = p.getNumJoints(self.body)
        for i in range(n):
            info = p.getJointInfo(self.body, i)
            self._joint_index[info[1].decode("utf-8")] = i
            self._link_index[info[12].decode("utf-8")] = i
            self._joint_limits[info[1].decode("utf-8")] = (info[8], info[9])

        required_joints = set(DRIVEN_JOINTS) | set(INERT_JOINTS) | {MIMIC_JAW_JOINT}
        missing_joints = required_joints - self._joint_index.keys()
        required_links = set(JAW_LINKS) | {EE_LINK}
        missing_links = required_links - self._link_index.keys()
        if missing_joints or missing_links:
            raise RuntimeError(
                f"psm_si_surrol.urdf is missing required "
                f"joints={sorted(missing_joints)} links={sorted(missing_links)}; "
                "the asset changed shape since this file was written.")

        # calculateInverseKinematics() returns one value per MOVABLE joint --
        # fixed joints are skipped entirely -- in increasing raw-joint-index
        # order, not per raw joint index. Resolved once here.
        self._movable = [i for i in range(n)
                          if p.getJointInfo(self.body, i)[2] != p.JOINT_FIXED]
        self._ik_slot = {name: self._movable.index(self._joint_index[name])
                          for name in DRIVEN_JOINTS}

        for name in INERT_JOINTS:
            p.resetJointState(self.body, self._joint_index[name], 0.0)

        # Owned here, and only here -- see module docstring point 3.
        mpm3d_module.init_pos()

        self._proxy: Dict[str, dict] = {}
        for link_name in JAW_LINKS:
            self._proxy[link_name] = self._build_jaw_proxy(link_name)
        self._latest_box_pose: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._prev_box_pose: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = \
            {name: None for name in JAW_LINKS}

        dummy_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.001] * 3)
        self._dummy_id = _create_proxy_body(dummy_shape)
        p.resetBasePositionAndOrientation(
            self._dummy_id, DUMMY_POSITION.tolist(), [0.0, 0.0, 0.0, 1.0])
        # float32: Taichi's Metal (SPIRV) backend has no f64 primitive type,
        # and reverse_rotation_matrix/reverse_offset_vector.from_numpy() goes
        # through a compiled kernel that requires the numpy array's dtype to
        # match the field's -- unlike plain field[i]=... assignment, this one
        # errors outright ("Type f64 not supported") rather than silently
        # casting. Every array fed to switch_reference_frame_and_update_sdf
        # must be float32.
        self._dummy_inv_rot = np.eye(3, dtype=np.float32)
        self._dummy_inv_pos = -DUMMY_POSITION.astype(np.float32)

        mpm3d_module.co_obj[0] = [self._proxy[JAW_LINKS[0]]["id"], 0]
        mpm3d_module.co_obj[1] = [self._proxy[JAW_LINKS[1]]["id"], 0]
        mpm3d_module.co_obj[2] = [self._dummy_id, 0]
        # Every slot is a real link (index 0 of a one-link proxy body), never
        # a body's base -- see module docstring point 1.
        assert (mpm3d_module.co_obj[:, 1] != -1).all()

    # -- construction-time geometry ----------------------------------------

    def _build_jaw_proxy(self, link_name: str) -> dict:
        """One-time: measure the jaw mesh's AABB at the URDF's default pose
        (all driven joints at 0, per PyBullet's own default after loadURDF)
        and build a zero-offset box proxy sized to it. See module docstring
        point 2 for why the offset is tracked here rather than passed to
        PyBullet's own collision-frame parameters."""
        link_idx = self._link_index[link_name]
        aabb_min, aabb_max = p.getAABB(self.body, link_idx)
        jaw_pos, jaw_quat = p.getLinkState(
            self.body, link_idx, computeForwardKinematics=1)[4:6]
        aabb_min, aabb_max = np.array(aabb_min), np.array(aabb_max)
        jaw_pos, jaw_quat = np.array(jaw_pos), np.array(jaw_quat)

        half_extents = (aabb_max - aabb_min) / 2.0
        world_center = (aabb_max + aabb_min) / 2.0
        # world -> jaw-local. The box's rotation relative to the jaw frame is
        # taken as identity ("axis-aligned at its proxy link origin"); this
        # is a possibly-loose box if the jaw's local axes aren't well aligned
        # with its own mesh extent at this reference pose -- a safe direction
        # to be wrong in for a sticky-contact collider, and checked
        # quantitatively (against measured half-extents) in the smoke test
        # rather than assumed tight here.
        local_center = rotmat_from_quat(jaw_quat).T @ (world_center - jaw_pos)

        box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents.tolist())
        proxy_id = _create_proxy_body(box_shape)
        return {"id": proxy_id, "half_extents": half_extents, "local_center": local_center}

    # -- per-frame collision geometry ---------------------------------------

    def _jaw_box_pose(self, link_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """T_world_box = T_world_jaw . T_jaw_box, composed by hand (module
        docstring point 2). T_jaw_box's rotation is identity, so the box's
        world orientation equals the jaw's live orientation exactly -- no
        matrix->quaternion conversion is ever needed, only rotating the fixed
        local_center offset by the jaw's current orientation."""
        link_idx = self._link_index[link_name]
        jaw_pos, jaw_quat = p.getLinkState(
            self.body, link_idx, computeForwardKinematics=1)[4:6]
        jaw_pos = np.array(jaw_pos, dtype=np.float64)
        jaw_quat = np.array(jaw_quat, dtype=np.float64)
        box_pos = jaw_pos + rotmat_from_quat(jaw_quat) @ self._proxy[link_name]["local_center"]
        return box_pos, jaw_quat

    def update_collision_geometry(self, mpm3d_module, sdf_module) -> None:
        """Once per recorded frame (mpm_adapter.py's advance()), not once per
        substep -- matches the vendored solver's own step()'s cadence, which
        refreshes collision once and holds it for the whole frame's substep
        loop. Call this AFTER driving the PSM to the frame's target pose (see
        mpm_adapter.py's record_grasp_episode() for the exact ordering): this
        reads the END-of-frame jaw transforms."""
        i_rot_list, i_pos_list = [], []
        for link_name in JAW_LINKS:
            box_pos, box_quat = self._jaw_box_pose(link_name)
            p.resetBasePositionAndOrientation(
                self._proxy[link_name]["id"], box_pos.tolist(), box_quat.tolist())
            R = rotmat_from_quat(box_quat)
            R_inv = R.T
            t_inv = -R_inv @ box_pos
            # float32 -- see the __init__ note by _dummy_inv_rot; Metal has
            # no f64 primitive and from_numpy() requires a dtype match.
            i_rot_list.append(R_inv.astype(np.float32))
            i_pos_list.append(t_inv.astype(np.float32))
            self._latest_box_pose[link_name] = (box_pos, box_quat)
        i_rot_list.append(self._dummy_inv_rot)
        i_pos_list.append(self._dummy_inv_pos)
        sdf_module.switch_reference_frame_and_update_sdf(
            i_rot_list=i_rot_list, i_pos_list=i_pos_list,
            co_obj=mpm3d_module.co_obj)

    def update_collision_velocity(self, mpm3d_module, frame_dt: float) -> None:
        """Per-proxy finite difference of consecutive T_world_box across
        recorded frames -- NOT derived from the commanded EE-pose delta,
        because jaw closure moves the two jaws differently from each other
        and from the EE frame. Call this immediately after
        update_collision_geometry() in the same frame.

        First-frame convention: zero velocity (documented, not accidental --
        self._prev_box_pose starts at None and there is no prior frame to
        difference against; consistent with starting from a momentarily
        static hover pose)."""
        for slot, link_name in enumerate(JAW_LINKS):
            box_pos, box_quat = self._latest_box_pose[link_name]
            prev = self._prev_box_pose[link_name]
            if prev is None:
                v, w = np.zeros(3), np.zeros(3)
            else:
                prev_pos, prev_quat = prev
                v = (box_pos - prev_pos) / frame_dt
                q_rel = actions.quat_multiply(box_quat, actions.quat_conjugate(prev_quat))
                rotvec = actions.quat_to_rotvec(q_rel)
                w = rotvec / frame_dt

                dpos = float(np.linalg.norm(box_pos - prev_pos))
                dx_limit = MAX_TRANSLATION_FRACTION_OF_DX * float(mpm3d_module.dx)
                if dpos > dx_limit:
                    raise RuntimeError(
                        f"{link_name} proxy moved {dpos * 1000:.2f} mm in one "
                        f"recorded frame, exceeding {dx_limit * 1000:.2f} mm "
                        f"({MAX_TRANSLATION_FRACTION_OF_DX}*dx) -- collision "
                        "geometry is under-resolved at this cadence; slow the "
                        "trajectory")
                dtheta = float(np.linalg.norm(rotvec))
                if dtheta > MAX_ROTATION_RAD_PER_FRAME:
                    raise RuntimeError(
                        f"{link_name} proxy rotated {dtheta:.4f} rad in one "
                        f"recorded frame, exceeding the "
                        f"{MAX_ROTATION_RAD_PER_FRAME} rad/frame sweep limit")
            mpm3d_module.co_v[slot] = v.tolist()
            mpm3d_module.co_w[slot] = w.tolist()
            mpm3d_module.centroid[slot] = box_pos.tolist()
            self._prev_box_pose[link_name] = (box_pos, box_quat)

        # The dummy slot never moves -- zero velocity unconditionally.
        mpm3d_module.co_v[2] = [0.0, 0.0, 0.0]
        mpm3d_module.co_w[2] = [0.0, 0.0, 0.0]
        mpm3d_module.centroid[2] = DUMMY_POSITION.tolist()

    # -- kinematics -----------------------------------------------------------

    def set_ee_pose(self, pose_xyzquat: np.ndarray, jaw_angle: float) -> None:
        """Drive the arm so tool_gripper_center reaches `pose_xyzquat`
        (world [x,y,z,qx,qy,qz,qw]) and the jaw opens to `jaw_angle` radians.
        Raises RuntimeError on an unreachable target, a non-finite or
        out-of-limit IK solution, or an IK residual beyond tolerance -- this
        must fail loudly rather than let a bad solve drift silently through
        an episode."""
        pose = np.asarray(pose_xyzquat, dtype=np.float64)
        ik = p.calculateInverseKinematics(
            self.body, self._link_index[EE_LINK],
            targetPosition=pose[:3].tolist(), targetOrientation=pose[3:7].tolist(),
            maxNumIterations=100, residualThreshold=1e-5)

        for name in DRIVEN_JOINTS:
            value = float(ik[self._ik_slot[name]])
            if not np.isfinite(value):
                raise RuntimeError(f"IK returned non-finite value for {name}: {value}")
            lo, hi = self._joint_limits[name]
            if not (lo - 1e-6 <= value <= hi + 1e-6):
                raise RuntimeError(
                    f"IK solution for {name} = {value:.4f} violates URDF "
                    f"limits [{lo:.4f}, {hi:.4f}] -- target likely unreachable")
            p.resetJointState(self.body, self._joint_index[name], value)

        p.resetJointState(self.body, self._joint_index["jaw_joint_1"], float(jaw_angle))
        p.resetJointState(self.body, self._joint_index[MIMIC_JAW_JOINT], float(-jaw_angle))

        achieved = self.ee_pose()
        pos_err = float(np.linalg.norm(achieved[:3] - pose[:3]))
        rot_err = float(np.linalg.norm(actions.quat_to_rotvec(
            actions.quat_multiply(achieved[3:], actions.quat_conjugate(pose[3:])))))
        if pos_err > IK_POSITION_TOLERANCE_M:
            raise RuntimeError(
                f"IK residual too large: position error {pos_err * 1000:.3f} mm "
                f"exceeds {IK_POSITION_TOLERANCE_M * 1000:.1f} mm tolerance -- "
                "target likely unreachable")
        if rot_err > IK_ORIENTATION_TOLERANCE_RAD:
            raise RuntimeError(
                f"IK residual too large: orientation error {rot_err:.4f} rad "
                f"exceeds {IK_ORIENTATION_TOLERANCE_RAD} rad tolerance")

    def ee_pose(self) -> np.ndarray:
        """(7,) [x,y,z,qx,qy,qz,qw], the world-LINK-frame pose (index 4/5 of
        getLinkState, not the COM-frame pose at 0/1 -- tool_gripper_center
        has no declared <inertial>, so these likely coincide for this link,
        but the correct index is used regardless)."""
        pos, quat = p.getLinkState(
            self.body, self._link_index[EE_LINK], computeForwardKinematics=1)[4:6]
        return np.array(list(pos) + list(quat), dtype=np.float64)

    def joint_positions(self) -> np.ndarray:
        """(7,) float32, DRIVEN_JOINTS order -- the ACHIEVED joint state
        (post-IK), for the schema's joint_pos field."""
        return np.array(
            [p.getJointState(self.body, self._joint_index[name])[0]
             for name in DRIVEN_JOINTS], dtype=np.float32)

    def jaw_box_poses(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Current (position, quaternion) of each jaw's collision proxy in
        world frame, computed fresh via FK -- does NOT touch the solver's
        SDF or the proxy bodies' own PyBullet pose (unlike
        update_collision_geometry()). For a caller that wants jaw geometry
        for diagnostics (e.g. a contact_mode estimate) without that method's
        side effects on the solver."""
        return {name: self._jaw_box_pose(name) for name in JAW_LINKS}

    def jaw_half_extents(self, link_name: str) -> np.ndarray:
        return self._proxy[link_name]["half_extents"]

    def workspace_aabb(self, n_samples: int = 200,
                        rng: Optional[np.random.Generator] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Random-sample DRIVEN_JOINTS within their URDF <limit> bounds, FK
        tool_gripper_center each time. Used only for the one-off base-
        placement calibration and the smoke test -- never during collection.
        Leaves the arm in a random final joint configuration; call
        set_ee_pose() again before driving a real episode."""
        rng = rng if rng is not None else np.random.default_rng()
        pts = []
        for _ in range(n_samples):
            for name in DRIVEN_JOINTS:
                lo, hi = self._joint_limits[name]
                p.resetJointState(self.body, self._joint_index[name],
                                   float(rng.uniform(lo, hi)))
            pos = np.array(p.getLinkState(
                self.body, self._link_index[EE_LINK], computeForwardKinematics=1)[4])
            pts.append(pos)
        pts = np.array(pts)
        return pts.min(axis=0), pts.max(axis=0)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        p.disconnect(self.client)

    def __enter__(self) -> "PSM":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def _create_proxy_body(collision_shape_id: int) -> int:
    """A one-link, zero-mass, fixed-joint multibody whose single link (index
    0, never the base -- module docstring point 1) carries the given
    collision shape at zero local offset. `resetBasePositionAndOrientation`
    on the returned id then sets that link's world pose directly."""
    return p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
        linkMasses=[0], linkCollisionShapeIndices=[collision_shape_id],
        linkVisualShapeIndices=[-1], linkPositions=[[0, 0, 0]],
        linkOrientations=[[0, 0, 0, 1]], linkInertialFramePositions=[[0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1]], linkParentIndices=[0],
        linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 0]])
