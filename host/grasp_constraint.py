#!/usr/bin/env python3
"""
grasp_constraint.py -- a host-owned, persistent rigid attachment between the
PSM's jaws and a frozen set of MPM particles.

WHY THIS EXISTS. `Boundary()`'s sticky-BC collision (third_party/MPM/mpm3d.py)
is not a grasp: it is a zero-slip velocity constraint applied wherever
SDF[I] < threshold, with no persistence -- a particle that drifts past the
threshold is released immediately, no bookkeeping holds it. The schema
(src/trajectory_io.py) defines CONTACT_GRASP as "jaws closed, tissue
kinematically attached to the tool"; nothing before this file ever earns
that label honestly. This file does: once triggered, it freezes a specific
particle set relative to the gripper and forces its position/velocity every
MPM substep, matching

    x_p(t) = T_gripper(t) . r_p
    v_p    = v_gripper + omega_gripper x (x_p - c)

WHY A POST-SUBSTEP CORRECTION, NOT A KERNEL EDIT. Read directly:
third_party/MPM/mpm3d.py's `substep(scale, threshold)` (lines 309-320) is ONE
Taichi kernel -- clear grid, then P2G(), Boundary(), G2P(), all three
`@ti.func` and inlined. There is no interception point between them from
Python. So enforcement has to be a correction applied strictly AFTER a
`substep()` call returns: let the vendored kernel run normally for every
particle, attached ones included (they get pulled by ordinary
P2G/Boundary/G2P like anything else), then immediately overwrite F_x/F_v for
just the attached indices. This touches no vendored code.

WHY THE GRIPPER TRANSFORM LIVES IN PERSISTENT 0-D FIELDS, NOT KERNEL
ARGUMENTS. host/mpm_adapter.py, host/psm.py and this file all use
`from __future__ import annotations`, which turns kernel argument
annotations into strings -- Taichi reads argument annotations as types, so an
annotated scalar argument silently fails to resolve. host/mpm_energy.py hit
exactly this and documents the fix: define kernels with NO annotated
arguments, and feed them through module-level or field state instead. Here
that means small persistent 0-d Taichi fields for the gripper's rotation,
translation, linear and angular velocity, updated via plain Python
assignment before each `apply()` call -- the same idiom
`third_party/MPM/mpm3d.py` already uses for `co_v`/`co_w`/`centroid`.

F_C[p] IS ZEROED FOR ATTACHED PARTICLES, AND THIS WAS MEASURED, NOT ASSUMED.
The first version of this file left F_C[p] untouched, reasoning that zeroing
it would falsely claim the attached particle experiences no local straining.
That reasoning turned out to be wrong in practice: leaving F_C[p] alone
creates a feedback loop. Every substep, G2P() recomputes F_C[p] from the
grid velocity GRADIENT sampled at the particle's CURRENT position -- but that
position was just teleported to the rigid target by this file's own
correction, into a grid neighbourhood the particle's momentum was never
actually part of. The resulting sampled gradient is a numerical artifact of
the teleport, not a real strain-rate, and P2G()'s `F[p] = (I + dt*F_C[p]) @
F[p]` then compounds it into F[p] every subsequent substep. Measured directly
(host/smoke_test_grasp.py, and a scripted 60-row episode during development):
with F_C[p] left alone, logged safety_strain (max principal stretch) spiked
to 26.6x and 15.9x at several frames -- physically absurd for tissue, and
`host/validate_dataset.py` correctly FAILed on it. Zeroing F_C[p] for
attached particles removes the feedback loop entirely: the same episode's
peak safety_strain dropped to 2.25x, in line with the surrounding frames.
Zeroing F_C[p] does still mean an attached particle's LOCAL strain-RATE
reads as zero while grasped -- its accumulated deformation gradient F[p]
stops accruing NEW strain from the grid, though F[p] itself is never reset
and keeps whatever value it already carried. This is a real, named
approximation (a grasped point's local strain state is treated as frozen at
whatever it was when the runaway-feedback problem was fixed, not as
continuing to evolve), but it is bounded and physically far more plausible
than the alternative's unbounded numerical blowup. F[p] is never written
directly by this file.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "host"))
import actions                                            # noqa: E402
import trajectory_io                                       # noqa: E402
import psm                                                  # noqa: E402

# Reuse mpm_adapter.py's CONTACT_STICK_RADIUS_M value (kept as an
# independently-defined constant, not an import, to avoid a circular import:
# mpm_adapter.py imports this module, so this module cannot import
# mpm_adapter.py back). If that constant changes, change this one to match --
# it is already calibrated to roughly (jaw half-extent + one grid cell),
# which is the same physical scale relevant to "close enough to be captured
# by a rigid grasp."
GRASP_CAPTURE_RADIUS_M = 0.02

# [judgment call] Release must be driven by an explicit jaw-angle command,
# never by geometric drift -- the whole point of persistence is that a
# forced particle ending up outside the (now irrelevant) proximity radius
# must NOT auto-release. 0.3 rad is clearly above 0 (so it isn't triggered by
# float jitter or a single near-zero waypoint) and well under full open
# (JAW_OPEN_RAD=1.0 in mpm_adapter.py) -- there is no hardware reference to
# calibrate this more precisely; verify empirically that tissue separates
# cleanly once forcing stops at this angle (host/smoke_test_grasp.py).
JAW_RELEASE_TRIGGER_RAD = 0.3


class GraspConstraint:
    """One instance per episode. Constructed after MPMRecorder (needs the
    already-imported, ti.init()'d mpm3d module -- same ordering constraint as
    host/psm.py's PSM)."""

    def __init__(self, mpm3d_module, ti_module):
        self.m, self.ti = mpm3d_module, ti_module
        n = int(self.m.n_particles)
        self._n_particles = n

        # Persistent Taichi state. i32/f32 throughout -- Metal has no f64
        # primitive (see host/psm.py's own note on this), and matches
        # mpm3d.py's own field dtypes (`float` there resolves to ti.f32,
        # since ti.init() here is never given default_fp=ti.f64).
        self._mask = self.ti.field(self.ti.i32, shape=n)
        self._r_local = self.ti.Vector.field(3, self.ti.f32, shape=n)
        self._gripper_c = self.ti.Vector.field(3, self.ti.f32, shape=())
        self._gripper_R = self.ti.Matrix.field(3, 3, self.ti.f32, shape=())
        self._gripper_v = self.ti.Vector.field(3, self.ti.f32, shape=())
        self._gripper_w = self.ti.Vector.field(3, self.ti.f32, shape=())
        self._mask.fill(0)
        self._kernel = None

        # Python-side source of truth for membership -- the Taichi mask is
        # kept in sync with this, never the other way around.
        self._active = False
        self._full_ids = np.zeros(0, np.int32)

    # -- kernel construction, lazy -------------------------------------------

    def _build_kernel(self):
        """Same pattern as host/mpm_energy.py's _build_kernels(): closes over
        the real field objects (via the captured module, exactly like that
        file's `k_clear`/`k_p2g` close over `m.F_grid_v`/`m.P2G`), takes NO
        annotated arguments -- see the module docstring for why."""
        if self._kernel is not None:
            return self._kernel
        ti, m = self.ti, self.m
        mask, r_local = self._mask, self._r_local
        gc, gR, gv, gw = (self._gripper_c, self._gripper_R,
                          self._gripper_v, self._gripper_w)

        @ti.kernel
        def k_apply():
            for p in m.F_x:
                if mask[p] == 1:
                    xp = gc[None] + gR[None] @ r_local[p]
                    m.F_x[p] = xp
                    m.F_v[p] = gv[None] + gw[None].cross(xp - gc[None])
                    # Zeroed, not left alone -- see the module docstring's
                    # "F_C[p] IS ZEROED" section. Leaving it produces a
                    # feedback loop (G2P resamples a velocity gradient at the
                    # teleported position that P2G then compounds into F[p]
                    # every subsequent substep), measured to blow up
                    # safety_strain to >26x in a real episode.
                    m.F_C[p] = ti.Matrix.zero(ti.f32, 3, 3)

        self._kernel = k_apply
        return self._kernel

    # -- state -----------------------------------------------------------

    def is_active(self) -> bool:
        return self._active

    # -- freeze / release --------------------------------------------------

    def maybe_freeze(self, rec, robot: "psm.PSM", pose_t: np.ndarray,
                      jaw_angle: float) -> None:
        """No-op if already active. Else evaluate the "captured between both
        inner jaw surfaces" predicate over ALL simulated particles (never the
        recorded subset -- same Rule 2 reasoning as elsewhere in this
        project) and freeze if it selects a nonempty set.

        No jaw-angle gate is needed: the predicate is already vacuous while
        the jaws are open (the two boxes are geometrically far apart, so
        nothing can be within GRASP_CAPTURE_RADIUS_M of both at once)."""
        if self._active:
            return
        pos_full, _, _ = rec.state()
        pos_full = pos_full.astype(np.float64)

        poses = robot.jaw_box_poses()
        d = {}
        for link_name in psm.JAW_LINKS:
            box_pos, box_quat = poses[link_name]
            half_extents = robot.jaw_half_extents(link_name)
            local = (pos_full - box_pos) @ psm.rotmat_from_quat(box_quat)
            d[link_name] = np.linalg.norm(
                np.maximum(np.abs(local) - half_extents, 0.0), axis=-1)
        name1, name2 = psm.JAW_LINKS
        selected = (d[name1] < GRASP_CAPTURE_RADIUS_M) & (d[name2] < GRASP_CAPTURE_RADIUS_M)
        full_ids = np.flatnonzero(selected).astype(np.int32)
        if full_ids.size == 0:
            return

        # r_p relative to the COMMANDED pose (poses[t]), not robot.ee_pose()
        # (achieved) -- matches record_grasp_episode()'s own documented
        # state/action timing contract: commanded values are what's
        # recorded, and mixing commanded/achieved would introduce a silent
        # offset bounded by IK tolerance instead of an explicit, testable one
        # (host/smoke_test_grasp.py test 5).
        c0 = pose_t[:3].astype(np.float64)
        R0 = psm.rotmat_from_quat(pose_t[3:7].astype(np.float64))
        r_local = (pos_full[full_ids] - c0) @ R0   # == R0.T @ (x - c0), row-vector form

        self._full_ids = full_ids
        mask_np = np.zeros(self._n_particles, np.int32)
        mask_np[full_ids] = 1
        self._mask.from_numpy(mask_np)
        r_local_full = np.zeros((self._n_particles, 3), np.float32)
        r_local_full[full_ids] = r_local.astype(np.float32)
        self._r_local.from_numpy(r_local_full)
        self._active = True

    def maybe_release(self, jaw_angle: float) -> None:
        """No-op if not active. Else, if the jaw has been commanded open past
        JAW_RELEASE_TRIGGER_RAD, clear membership."""
        if not self._active:
            return
        if jaw_angle >= JAW_RELEASE_TRIGGER_RAD:
            self._mask.fill(0)
            self._full_ids = np.zeros(0, np.int32)
            self._active = False

    # -- per-substep enforcement ---------------------------------------------

    def apply(self, pose_start: np.ndarray, pose_end: np.ndarray,
              frame_dt: float, alpha: float) -> None:
        """No-op if not active. Called once per SUBSTEP (not once per frame
        like host/psm.py's collision-geometry hook -- the grasp constraint is
        meant to be exact, not just approximately sufficient).

        `alpha` in (0, 1]: the fraction of the frame elapsed AFTER the
        substep that just ran (correction is strictly post-substep).
        v_gripper/omega_gripper are FRAME-CONSTANT (derived once from the
        frame's total pose_delta / frame_dt), matching how
        psm.PSM.update_collision_velocity() already treats jaw-box collision
        velocity as a frame-constant backward difference -- and exactly
        correct here, not just consistent in style: apply_pose_delta()
        composes a single rotation vector scaled linearly by alpha, i.e.
        constant angular velocity over the frame, so differentiating the
        interpolated position reproduces the same v_p this formula states."""
        if not self._active:
            return
        delta6 = actions.pose_delta(pose_start, pose_end)          # (6,) world-frame
        pose_i = actions.apply_pose_delta(pose_start, alpha * delta6)
        pos_i, quat_i = pose_i[:3], pose_i[3:7]
        R_i = psm.rotmat_from_quat(quat_i)

        v_gripper = delta6[:3] / frame_dt
        omega_gripper = delta6[3:] / frame_dt

        self._gripper_c[None] = pos_i.astype(np.float32).tolist()
        self._gripper_R[None] = R_i.astype(np.float32).tolist()
        self._gripper_v[None] = v_gripper.astype(np.float32).tolist()
        self._gripper_w[None] = omega_gripper.astype(np.float32).tolist()

        kernel = self._build_kernel()
        kernel()

    # -- schema integration ---------------------------------------------

    def recorded_node_ids(self, particle_ids: np.ndarray) -> np.ndarray:
        """Full-solver attached ids, remapped to RECORDED-subset indices
        (0..N-1) -- grasp_node_ids must index the recorded subset because
        downstream consumers (the visualizer) use them directly against
        tissue_pos, which is already subsetted.

        np.searchsorted ALONE is not enough: it returns some insertion index
        even for a full-solver id that isn't in particle_ids at all, which
        would silently alias a not-recorded attached particle to whichever
        recorded particle happens to sort nearest. The equality check after
        it is what makes this correct -- drop ids not present, don't alias
        them."""
        if self._full_ids.size == 0:
            return np.zeros(0, np.int32)
        idx = np.searchsorted(particle_ids, self._full_ids)
        idx = np.clip(idx, 0, len(particle_ids) - 1)
        found = particle_ids[idx] == self._full_ids
        return idx[found].astype(np.int32)

    def contact_mode_override(self) -> Optional[int]:
        """CONTACT_GRASP while (and only while) the persistent full-particle
        mask is nonempty; None otherwise, letting the caller fall back to the
        proximity-based NONE/TOUCH/STICK classification."""
        return trajectory_io.CONTACT_GRASP if self._active else None
