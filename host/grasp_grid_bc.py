#!/usr/bin/env python3
"""
grasp_grid_bc.py -- a persistent grasp applied as a GRID boundary condition,
not as a particle teleport (DECISION_LOG 14.21, option A; 14.22).

WHY THIS EXISTS. host/grasp_constraint.py's `GraspConstraint` holds a frozen
particle set by overwriting its positions and velocities after every
substep and zeroing its F_C. 14.21 measured what that does on the Classic x5
debug set: the held cluster (25-64 particles, ~1-3 grid cells) is lifted
out of the surrounding tissue's interpolation reach, the material at its edge
comes apart (neighbourhood volume up 7.5x while the particles' own F reads
0.65), and the logged peak stretch of 5-17x is that torn edge. Three causes:
the teleport drags the tissue only through the cluster's small mass in shared
grid nodes; the frozen F means the cluster develops no tension; and a
one-cell cluster is below what the grid can hold on to.

WHAT THIS DOES INSTEAD. The standard MPM way to impose a moving body: a
velocity boundary condition on grid nodes. Every substep, in ONE kernel of
this file's own:

    clear grid -> P2G() -> Boundary() -> grasp BC -> G2P()

where "grasp BC" sets every grid node in the held particles' 3x3x3 B-spline
stencil (that carries mass) to the gripper's rigid velocity
v_g + omega_g x (x_node - c). Nothing is overwritten afterwards: G2P() moves
EVERY particle, held ones included, with the grid velocity, and F_C comes
from the same field. So F stays consistent with motion by construction.
Quadratic B-splines reproduce linear fields exactly, so a held particle whose
whole stencil is rigid moves exactly rigidly -- no drift to correct.

WHY NO VENDORED EDIT. `mpm3d.substep()` is a single @ti.kernel, but P2G,
Boundary and G2P are @ti.func, callable from any kernel. This file's kernel
calls them in the vendored order and adds one step; third_party/ is
untouched. With no grasp active, the kernel's grasp step does nothing, and
`host/smoke_test_grid_grasp.py` checks that it is then bitwise the vendored
substep.

WHAT IT IS NOT. The grip footprint is the held particles' stencil: about 1.5
grid cells (~23 mm) around them, larger than the jaw. That is the finest grip
a 15.6 mm grid can impose, stated rather than hidden. Membership (who is
held, when it is released) is exactly GraspConstraint's: same capture rule,
same release rule, same `full_ids()`, same CONTACT_GRASP labelling.
"""

# NO `from __future__ import annotations` here, unlike its neighbours: the
# substep kernel takes two annotated scalar arguments, and that import turns
# annotations into strings Taichi cannot resolve (grasp_constraint.py's
# docstring records the failure).

import numpy as np

import actions
import psm
from grasp_constraint import GraspConstraint


class GridGraspConstraint(GraspConstraint):
    """GraspConstraint's membership, imposed through the grid.

    The adapter drives it through `substep()` instead of `mpm3d.substep()`
    followed by `apply()`; `apply()` is a no-op so a caller that still calls
    it cannot reintroduce the teleport.
    """

    model_name = "grid_bc"

    def __init__(self, mpm3d_module, ti_module):
        super().__init__(mpm3d_module, ti_module)
        ti, m = self.ti, self.m
        self._node_mask = ti.field(ti.i32, shape=m.grid_shape)
        self._substep_kernel = None

    def apply(self, *args, **kwargs) -> None:
        """No-op: the constraint lives inside substep()."""
        return None

    def _build_substep(self):
        if self._substep_kernel is not None:
            return self._substep_kernel
        ti, m = self.ti, self.m
        mask, node_mask = self._mask, self._node_mask
        gc, gR, gv, gw = (self._gripper_c, self._gripper_R,
                          self._gripper_v, self._gripper_w)
        active = self._active_field = ti.field(ti.i32, shape=())
        active[None] = 0

        @ti.kernel
        def k_substep(scale: ti.f32, threshold: ti.f32):
            # Identical to mpm3d.substep() up to and including Boundary().
            for I in ti.grouped(m.F_grid_m):
                m.F_grid_v[I] = ti.zero(m.F_grid_v[I])
                m.F_grid_m[I] = 0
            m.P2G()
            m.Boundary(scale, threshold)
            # Every loop stays at the kernel's top level: Taichi parallelises
            # only those, and one nested under an `if` would run serially.
            # The `active` test is therefore inside each loop body. With
            # active == 0 none of the three loops writes a solver field.
            for I in ti.grouped(node_mask):
                node_mask[I] = 0
            # The stencil P2G used for each held particle (same base).
            for p in m.F_x:
                if active[None] == 1 and mask[p] == 1:
                    base = int(m.F_x[p] / m.dx - 0.5)
                    for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                        node_mask[base + offset] = 1
            for I in ti.grouped(node_mask):
                if node_mask[I] == 1 and m.F_grid_m[I] > 0:
                    x_node = I * m.dx * scale + m.soft_body_base_position[None]
                    m.F_grid_v[I] = gv[None] + gw[None].cross(x_node - gc[None])
            m.G2P()

        self._substep_kernel = k_substep
        return k_substep

    def substep(self, scale: float, threshold: float, pose_start: np.ndarray,
                pose_end: np.ndarray, frame_dt: float, alpha_start: float) -> None:
        """One solver substep with the grasp BC. `alpha_start` is the fraction
        of the frame elapsed BEFORE this substep: the rigid field is the
        gripper's at the start of the substep, moving with the frame-constant
        velocity (the same interpolation GraspConstraint.apply() uses)."""
        kernel = self._build_substep()
        if self._active:
            delta6 = actions.pose_delta(pose_start, pose_end)
            pose_i = actions.apply_pose_delta(pose_start, alpha_start * delta6)
            self._gripper_c[None] = pose_i[:3].astype(np.float32).tolist()
            self._gripper_R[None] = psm.rotmat_from_quat(pose_i[3:7]).astype(np.float32).tolist()
            self._gripper_v[None] = (delta6[:3] / frame_dt).astype(np.float32).tolist()
            self._gripper_w[None] = (delta6[3:] / frame_dt).astype(np.float32).tolist()
        self._active_field[None] = 1 if self._active else 0
        kernel(scale, threshold)
