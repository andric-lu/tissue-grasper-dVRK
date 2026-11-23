#!/usr/bin/env python3
"""
psm_classic.py -- the dVRK CLASSIC PSM, through SurRoL's robot interface,
as the MPM tissue's collider. Decision D2 as amended by the to-do list
(DECISION_LOG section 14.2) and the grasp-scale decision of section 14.15.

    conda activate tissue-sim          # SurRoL main needs numpy < 1.24
    (imported by host/mpm_adapter.py when the config says "arm": {"model": "classic"})

WHY THIS EXISTS. Item 6: "Use the SurRoL robot control interface coupled to
the MPM tissue simulation." The lab's arms are Classic; host/psm.py drives the
Si model with our own free-space IK. This class drives SurRoL's `Psm1` -- the
Classic PSM (dVRK Classic modified DH, large needle driver 400006), at
SurRoL `main` a903fa6 -- and reuses everything in host/psm.py that concerns
the SOLVER (box proxies, co_obj registration, per-jaw collision velocity,
sweep limits, the single PyBullet client), which is arm-independent.

WHAT "THROUGH SURROL'S INTERFACE" MEANS HERE, PRECISELY.
  - Poses are commanded the dVRK way: SurRoL's `pose_world2rcm` turns the
    commanded tool-tip pose into the RCM-frame pose `Psm.move()` takes, and
    SurRoL's own `inverse_kinematics` solves it, with the RCM-preserving
    kinematic structure of the Classic arm (yaw, pitch and insertion axes meet
    at the remote centre). This is exactly what `move()` does internally.
  - `move()` itself is NOT called: it only sets PyBullet MOTOR TARGETS and
    needs `p.stepSimulation()` to move anything, i.e. a dynamic arm. This
    project drives every arm kinematically (host/psm.py, "PyBullet as an FK/IK
    library, not a physics engine"), so the solved joints are applied with
    SurRoL's `reset_joint`, which also applies its parallel-linkage and jaw
    joint mapping (`_get_joint_positions_all`).
  - The jaw uses SurRoL's convention: `jaw` is the TOTAL opening angle,
    joints 6/7 at -jaw/2, +jaw/2 (`Psm.move_jaw`), 0..80 deg. The Si wrapper's
    `jaw` is the PER-JAW angle -- the episode records `robot_model`, and
    jaw_joint_values() makes the per-jaw angles explicit in every file.

SCALE. `scaling=5`, as SurRoL's own tasks use (needle_pick, peg_transfer,
gauze_retrieve, ...). Measured: the Classic jaw's collision box is
1.4 x 10.2 x 1.8 mm at scale 1 -- smaller than the Si jaw and far under one
MPM grid cell (15.6 mm), which is why the first grasp pilot held 12 particles
and tore them out as a strand (section 14.14). At x5 the box is
7 x 51 x 9 mm. The MPM world is unchanged (1 m domain, 300 x 300 x 50 mm
slab, real gravity); read physically, the arm is 5x life-size, i.e. the slab
stands for ~60 x 60 x 10 mm of tissue. Mapping material, density and time
onto real tissue needs a similarity scaling that is NOT applied here
(section 14.15).

PLACEMENT. The RCM is put at `rcm_position` (world metres), measured, not
derived: the base is loaded at the origin, the RCM's world position is read
off FK, and the base is shifted by the difference. At zero joint angles the
Classic tool points straight down (-z) from the RCM, so an RCM directly above
the tissue gives a vertical tool, and retraction is a pure insertion-axis pull
-- out through the port, as on the real arm.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pybullet as p

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "host"))
import psm as psm_si                                       # noqa: E402
from tool_geometry import quat_from_rotmat                 # noqa: E402

MODEL_NAME = "dvrk_classic_psm_surrol"
JAW_LINKS = ("psm_tool_gripper1_link", "psm_tool_gripper2_link")
EEF_LINK_INDEX = 5          # psm_tool_yaw_link: SurRoL's EEF_LINK_INDEX
DEFAULT_SCALING = 5.0
# RCM above the slab centre. 0.6 m puts the tip ~0.5 m from the RCM at the
# grasp height: insertion ~0.1 m of the 0.24 m (x scaling) range, mid-stroke.
DEFAULT_RCM_POSITION = (0.25, 0.35, 0.60)
JAW_OPEN_RAD = np.deg2rad(80.0)       # SurRoL Psm.open_jaw()
JAW_CLOSED_RAD = 0.0                  # SurRoL Psm.close_jaw()


def _mat_from_pose(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = psm_si.rotmat_from_quat(np.asarray(quat, np.float64))
    T[:3, 3] = pos
    return T


def _pose_from_mat(T) -> np.ndarray:
    return np.concatenate([T[:3, 3], quat_from_rotmat(T[:3, :3])])


class ClassicPSM(psm_si.PSM):
    """Same interface as host/psm.PSM (set_ee_pose, ee_pose, joint_positions,
    jaw_box_poses, jaw_half_extents, jaw_box_offset, jaw_joint_values,
    update_collision_geometry/velocity, close), backed by SurRoL's Psm1."""

    def __init__(self, mpm3d_module, sdf_module, *,
                 rcm_position: Tuple[float, float, float] = DEFAULT_RCM_POSITION,
                 scaling: float = DEFAULT_SCALING):
        # NOT psm_si.PSM.__init__: that loads the Si URDF. Everything after
        # the arm itself exists is shared via _register_proxies().
        from surrol.robots.psm import Psm1          # SurRoL main, a903fa6
        self.client = p.connect(p.DIRECT)
        if self.client != 0:
            raise RuntimeError(f"p.connect() returned client {self.client}, not 0; "
                               "third_party/MPM/sdf.py targets client 0 implicitly")
        self.scaling = float(scaling)
        self.model_name = MODEL_NAME
        self.jaw_links = JAW_LINKS

        # Place the RCM by measurement: load at the origin, read the RCM off
        # FK, then shift. Psm1 multiplies `pos` by `scaling`, hence the divide.
        probe = Psm1(pos=(0, 0, 0), orn=(0, 0, 0, 1), scaling=self.scaling)
        rcm0 = probe.wTr[:3, 3].copy()
        p.removeBody(probe.body)
        base = (np.asarray(rcm_position, np.float64) - rcm0) / self.scaling
        self.arm = Psm1(pos=tuple(base), orn=(0, 0, 0, 1), scaling=self.scaling)
        self.body = self.arm.body
        got = self.arm.wTr[:3, 3]
        if np.linalg.norm(got - np.asarray(rcm_position)) > 1e-6:
            raise RuntimeError(f"RCM landed at {got}, asked for {rcm_position}")
        self.rcm_position = got

        n = p.getNumJoints(self.body)
        self._link_index = {p.getJointInfo(self.body, i)[12].decode(): i for i in range(n)}
        missing = set(JAW_LINKS) - self._link_index.keys()
        if missing:
            raise RuntimeError(f"SurRoL's psm.urdf has no links {sorted(missing)}")
        self._register_proxies(mpm3d_module)

    # -- geometry ----------------------------------------------------------

    def _build_jaw_proxy(self, link_name: str) -> dict:
        """The Classic jaw's collision shape IS a box primitive, so it is read
        from PyBullet (already scaled by globalScaling), not fitted to a mesh.
        PyBullet reports the shape's frame relative to the link's INERTIAL
        frame; composed with the inertial offset here to get the box centre in
        the link frame, which is what _jaw_box_pose() composes with FK."""
        li = self._link_index[link_name]
        shapes = p.getCollisionShapeData(self.body, li)
        if len(shapes) != 1 or shapes[0][2] != p.GEOM_BOX:
            raise RuntimeError(f"{link_name}: expected one box collision shape, got "
                               f"{[(s[2], s[3]) for s in shapes]}")
        dims = np.asarray(shapes[0][3], np.float64)           # FULL extents
        local_in_inertial = np.asarray(shapes[0][5], np.float64)
        if np.abs(np.asarray(shapes[0][6]) - [0, 0, 0, 1]).max() > 1e-9:
            raise RuntimeError(f"{link_name}: rotated collision frame not supported")
        dyn = p.getDynamicsInfo(self.body, li)
        inertial_pos = np.asarray(dyn[3], np.float64)
        inertial_R = psm_si.rotmat_from_quat(np.asarray(dyn[4], np.float64))
        local_center = inertial_pos + inertial_R @ local_in_inertial
        half_extents = dims / 2.0
        box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents.tolist())
        proxy_id = psm_si._create_proxy_body(box_shape)
        return {"id": proxy_id, "half_extents": half_extents, "local_center": local_center}

    # -- kinematics ----------------------------------------------------------

    def set_ee_pose(self, pose_xyzquat: np.ndarray, jaw_angle: float) -> None:
        """Drive the SurRoL tool-tip frame to `pose_xyzquat` (world
        [x,y,z,qx,qy,qz,qw]) with total jaw opening `jaw_angle` rad.

        The tool-tip frame is SurRoL's: the EEF link (psm_tool_yaw_link,
        the jaw pivot) composed with Psm.tool_T_tip, a pure rotation -- so its
        position is the jaw pivot. Same residual gate as the Si arm.
        """
        a = self.arm
        T_tip = _mat_from_pose(pose_xyzquat[:3], pose_xyzquat[3:7])
        T_eef = T_tip @ np.linalg.inv(a.tool_T_tip)          # world_T_eef
        pose_rcm = a.pose_world2rcm(T_eef, "matrix")         # what Psm.move() takes
        q = a.inverse_kinematics(a.pose_rcm2world(pose_rcm, "tuple"), a.EEF_LINK_INDEX)
        if not np.all(np.isfinite(q)):
            raise RuntimeError(f"SurRoL IK returned non-finite joints {q}")
        if not a._check_joint_limits(q):
            raise RuntimeError(f"SurRoL IK solution {np.round(q, 4)} violates joint "
                               "limits -- target likely unreachable")
        a.move_jaw(float(jaw_angle))          # sets _jaw_angle used by reset_joint
        a.reset_joint(q)

        achieved = self.ee_pose()
        pos_err = float(np.linalg.norm(achieved[:3] - pose_xyzquat[:3]))
        import actions
        rot_err = float(np.linalg.norm(actions.quat_to_rotvec(
            actions.quat_multiply(achieved[3:], actions.quat_conjugate(pose_xyzquat[3:])))))
        if pos_err > psm_si.IK_POSITION_TOLERANCE_M:
            raise RuntimeError(f"IK residual {pos_err*1e3:.3f} mm exceeds "
                               f"{psm_si.IK_POSITION_TOLERANCE_M*1e3:.1f} mm")
        if rot_err > psm_si.IK_ORIENTATION_TOLERANCE_RAD:
            raise RuntimeError(f"IK orientation residual {rot_err:.4f} rad exceeds "
                               f"{psm_si.IK_ORIENTATION_TOLERANCE_RAD}")

    def ee_pose(self) -> np.ndarray:
        """(7,) world pose of SurRoL's tool-tip frame (see set_ee_pose)."""
        ls = p.getLinkState(self.body, EEF_LINK_INDEX, computeForwardKinematics=1)
        return _pose_from_mat(_mat_from_pose(ls[4], ls[5]) @ self.arm.tool_T_tip)

    def joint_positions(self) -> np.ndarray:
        """(7,) float32: the 6 arm joints as SurRoL reports them (insertion
        UNSCALED, i.e. in real-arm metres) and the total jaw opening."""
        q = np.asarray(self.arm.get_current_joint_position(), np.float64)
        return np.concatenate([q, [self.arm.get_current_jaw_position()]]).astype(np.float32)

    def jaw_joint_values(self, jaw: float) -> np.ndarray:
        """SurRoL convention: joint 6 = -jaw/2, joint 7 = +jaw/2."""
        return np.array([-jaw / 2.0, jaw / 2.0], np.float64)

    def tip_down_quat(self) -> np.ndarray:
        """Tool-tip orientation with all arm joints at zero (tool vertical,
        pointing down). The natural orientation for a top-down grasp."""
        q_save = np.asarray(self.arm.get_current_joint_position(), np.float64)
        self.arm.reset_joint(np.zeros(6))
        quat = self.ee_pose()[3:].copy()
        self.arm.reset_joint(q_save)
        return quat

    def workspace_aabb(self, n_samples=200, rng=None):
        raise NotImplementedError("Classic placement is by RCM position, not by "
                                  "sampling; see DEFAULT_RCM_POSITION")
