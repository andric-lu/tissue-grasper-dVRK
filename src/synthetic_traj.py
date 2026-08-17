#!/usr/bin/env python3
"""
synthetic_traj.py -- episodes whose correct answer is known in closed form.

Run:

    python src/synthetic_traj.py --out data_synth/ --kinds all
    python src/synthetic_traj.py --out data_synth/ --kinds rotation --n-particles 2000

WHY THIS FILE EXISTS
--------------------
The MPM simulator is not set up yet, so there is no episode anywhere that
carries a deformation gradient. Everything built to consume one -- the schema,
the metrics, the validator, and eventually the model -- is therefore untestable
against real data, and "untestable until the simulator lands" means the bugs
all surface at once, tangled together with the simulator's own.

These four episodes break that deadlock. Each is a deformation written down
analytically, so F is not estimated from the motion: the motion is generated
FROM F. That makes the expected value of every metric exact rather than
approximate, and it means a discrepancy points at the code under test rather
than at discretisation error.

    kind        deformation                       what it pins down
    --------    ------------------------------    ------------------------------
    rest        nothing moves, F = I              zero strain, constant exposure
    uniaxial    isochoric stretch 1.0 -> 1.4      J = 1 exactly at every step,
                                                  max stretch = the ramp value
    rotation    rigid-body rotation, F = R(t)     ZERO strain despite large motion
    retract     clamped slab sheared off target   exposure rises monotonically

THE ROTATION CASE IS THE ONE THAT MATTERS
-----------------------------------------
It is the end-to-end version of the pure-rotation test in
tests/test_tissue_metrics.py. That test proves the metric is frame-indifferent
in isolation; this proves nothing between here and the metric reintroduces the
error -- not the writer, not the float32 cast, not the reader, not the
validator. A stack that reports strain for a rigidly rotating body is measuring
its own coordinate system.

WHAT THIS IS NOT
----------------
Not a simulator. Nothing here solves an equation of motion, and none of these
deformations is a response to the forces logged alongside it. The end-effector
pose and contact modes are plausible decoration so that consumers expecting a
complete v2 file get one. Do not train a dynamics model on these episodes and
conclude anything: the dynamics are prescribed, so a model would be learning
the ramp, not the physics.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from actions import encode_action
from materials import sample_material, suggested_substep_dt, unpack_material
from tissue_metrics import (
    DEFAULT_GRID,
    DEFAULT_SIGMA,
    DEFAULT_THRESHOLD,
    compute_exposure,
    compute_safety_strain,
)
from trajectory_io import (
    CONTACT_GRASP,
    CONTACT_NONE,
    CONTACT_TOUCH,
    TrajectoryWriter,
)

KINDS = ("rest", "uniaxial", "rotation", "retract")

# WHICH KINDS ACTUALLY CLAMP ANYTHING.
#
# Only `retract` holds particles fixed. That is not an oversight in the other
# three, it is forced by what each one is for:
#
#   rotation  is a RIGID rotation. Clamp any particle and it stops being rigid
#             -- you introduce real strain, and this episode's entire purpose is
#             to carry large motion with ZERO strain.
#   uniaxial  is a UNIFORM stretch. Clamp an edge and the stretch is no longer
#             uniform, F stops equalling the analytic value, and J = 1 exactly
#             stops holding.
#   rest      applies the identity map, which constrains nothing. Its particles
#             do not move because the map does not move them, not because
#             anything holds them. Nothing here solves an equation of motion, so
#             there is no gravity for a clamp to resist.
#
# The other three therefore get an ALL-FALSE mask, not an empty one. Empty means
# "this simulator did not record it"; all-False means "recorded, and nothing is
# clamped". We know the answer, so claiming ignorance would be its own small
# lie -- the same distinction trajectory_io draws for every other v2 field.
#
# Fixed on 17 Aug: all four kinds previously shipped the same 22-particle mask,
# so rotation.npz asserted 22 clamped particles while moving them 60 mm. The
# deformations were right; the metadata was wrong. Nothing caught it because no
# check in validate_dataset read boundary_mask -- check_boundary_is_held now
# does.
_CLAMPS = {"rest": False, "uniaxial": False, "rotation": False, "retract": True}

# Model step, per §2.6. Fixed: a variable-dt model is a different and harder
# model, and nothing is served by making the first dataset heterogeneous in it.
DT = 0.010

# Geometry, in metres. A 6 cm square sheet 5 mm above a 2 cm square target is
# surgical scale, and it matches the scale collect_retraction.py works at.
SHEET_HALF = 0.03
SHEET_Z = 0.005
TARGET_ORIGIN = np.array([0.0, -0.015, 0.0], np.float32)
TARGET_NORMAL = np.array([0.0, 0.0, 1.0], np.float32)
TARGET_EXTENT = np.array([0.01, 0.01], np.float32)

# Grid spacing an MPM collector would plausibly use at this scale. Only feeds
# suggested_substep_dt, so that the substep recorded here is consistent with
# the sampled stiffness and validate_dataset's stability check has something
# real to check.
MPM_DX = 1.0e-3


# --------------------------------------------------------------------------
# Particle layouts
# --------------------------------------------------------------------------

def _sheet(n_particles: int) -> np.ndarray:
    """A flat square sheet of particles at z = SHEET_Z, as (N, 3).

    Laid out on a regular grid rather than sampled randomly: exposure sums
    Gaussian contributions, so a random layout has clumps and voids that make
    coverage vary from run to run for reasons that have nothing to do with the
    deformation being tested.
    """
    k = max(2, int(round(np.sqrt(n_particles))))
    a = np.linspace(-SHEET_HALF, SHEET_HALF, k)
    xx, yy = np.meshgrid(a, a, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), np.full(k * k, SHEET_Z)],
                    axis=-1).astype(np.float64)


def _clamped_edge(X: np.ndarray) -> np.ndarray:
    """Boundary mask: the far edge of the sheet, held fixed."""
    return X[:, 1] >= SHEET_HALF - 1e-9


# --------------------------------------------------------------------------
# Deformation maps -- (positions, F) as explicit functions of time
# --------------------------------------------------------------------------

def _rest(X: np.ndarray, t: int, T: int):
    return X.copy(), np.tile(np.eye(3), (len(X), 1, 1))


def _uniaxial(X: np.ndarray, t: int, T: int):
    """Isochoric uniaxial stretch along x, ramping 1.0 -> 1.4.

    F = diag(s, 1/sqrt(s), 1/sqrt(s)), so det(F) = s * (1/s) = 1 exactly at
    every step and the largest principal stretch is exactly s.
    """
    s = 1.0 + 0.4 * (t / max(T - 1, 1))
    A = np.diag([s, 1.0 / np.sqrt(s), 1.0 / np.sqrt(s)])
    # Deform about the sheet centroid so the cloud stretches in place instead
    # of also translating, which would confound the exposure signal.
    c = X.mean(axis=0)
    return (X - c) @ A.T + c, np.tile(A, (len(X), 1, 1))


def _rotation(X: np.ndarray, t: int, T: int):
    """Rigid-body rotation about the sheet centroid, up to 90 degrees.

    F = R(t), a proper rotation, so every strain measure must read exactly zero
    however far it has turned. The axis is deliberately not a coordinate axis:
    a rotation about +z leaves the flat sheet in its own plane and would not
    exercise the out-of-plane terms of F at all.
    """
    ang = 0.5 * np.pi * (t / max(T - 1, 1))
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    R = _rodrigues(axis, ang)
    c = X.mean(axis=0)
    return (X - c) @ R.T + c, np.tile(R, (len(X), 1, 1))


def _retract(X: np.ndarray, t: int, T: int):
    """A slab clamped along its far edge, sheared laterally off the target.

    The map is    x' = x + d(t) * u(y),    u(y) = (y_far - y) / (2 * SHEET_HALF)

    so u is 0 at the clamped edge and 1 at the free edge. Its gradient gives

        F = [[1, -d/(2*SHEET_HALF), 0], [0, 1, 0], [0, 0, 1]]

    a simple shear: det(F) = 1 exactly, which keeps the episode admissible
    (det > 0) and consistent with near-incompressibility, so validate_dataset's
    F checks have a case that should cleanly pass. The target sits toward the
    free edge, where u is large, so the tissue covering it sweeps clear and
    exposure rises monotonically.
    """
    d = 0.08 * (t / max(T - 1, 1))
    L = 2.0 * SHEET_HALF
    u = (SHEET_HALF - X[:, 1]) / L
    pos = X.copy()
    pos[:, 0] = X[:, 0] + d * u
    F = np.tile(np.eye(3), (len(X), 1, 1))
    F[:, 0, 1] = -d / L      # dx'/dy; negative because u decreases with y
    return pos, F


_MAPS = {"rest": _rest, "uniaxial": _uniaxial,
         "rotation": _rotation, "retract": _retract}


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from an axis and an angle, via Rodrigues' formula."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# --------------------------------------------------------------------------
# Episode construction
# --------------------------------------------------------------------------

def make_synthetic_episode(path: str, kind: str, *, n_particles: int = 500,
                           n_steps: int = 100, seed: int = 0,
                           store_F_as_float16: bool = False) -> str:
    """Write one synthetic v2 episode and return its path.

    `n_particles` is rounded to a perfect square so the sheet is a regular grid.
    """
    if kind not in _MAPS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be at least 2, got {n_steps}")

    rng = np.random.default_rng(seed)
    X = _sheet(n_particles)
    N = len(X)
    # Only the kinds whose deformation map actually holds particles still --
    # see _CLAMPS. The mask must describe what the motion does, not what a
    # sheet-shaped thing usually has.
    boundary = _clamped_edge(X) if _CLAMPS[kind] else np.zeros(N, bool)

    material = sample_material(rng)
    mu, _, rho = unpack_material(material)
    # Recorded per episode precisely because it depends on the sampled
    # stiffness -- §2.6. A fixed substep would be wrong at one end of the range.
    substep_dt = float(suggested_substep_dt(mu, rho, MPM_DX))
    n_substeps = max(1, int(np.ceil(DT / substep_dt)))

    deform = _MAPS[kind]
    grasped = _grasped(X)
    prev_pos = X.copy()
    prev_pose: Optional[np.ndarray] = None

    with TrajectoryWriter(
            path, simulator="synthetic", task="tissue_retraction", dt=DT,
            notes=(f"synthetic {kind}; analytic F; n_particles={N}; "
                   f"n_steps={n_steps}; seed={seed}; "
                   "NOT a simulation -- the deformation is prescribed"),
            material_params=material, substep_dt=substep_dt,
            n_substeps=n_substeps, boundary_mask=boundary,
            action_spec="delta_pose_jaw",
            target_origin=TARGET_ORIGIN, target_normal=TARGET_NORMAL,
            target_extent=TARGET_EXTENT,
            store_F_as_float16=store_F_as_float16) as w:
        for t in range(n_steps):
            pos, F = deform(X, t, n_steps)

            # Metrics are computed with the same functions and the same module
            # defaults the validator will use to recompute them. If these two
            # ever disagree it is a real drift, not a parameter mismatch.
            exposure = float(compute_exposure(
                pos, TARGET_ORIGIN, TARGET_NORMAL, TARGET_EXTENT,
                sigma=DEFAULT_SIGMA, threshold=DEFAULT_THRESHOLD, grid=DEFAULT_GRID))
            safety = compute_safety_strain(F)

            # The tool rides the free corner of the sheet. Decoration, but it
            # has to be self-consistent: ee_vel and action are derived from the
            # pose sequence rather than invented independently.
            pose = _tool_pose(pos, grasped)
            if prev_pose is None:
                prev_pose = pose
            jaw = 0.0 if t >= _GRASP_STEP else 0.5
            action = encode_action(prev_pose, pose, jaw)
            ee_vel = np.concatenate([(pose[:3] - prev_pose[:3]) / DT, np.zeros(3)])

            w.append(
                tissue_pos=pos,
                tissue_vel=(pos - prev_pos) / DT,
                ee_pose=pose,
                ee_vel=ee_vel,
                action=action,
                jaw=jaw,
                grasp_active=t >= _GRASP_STEP,
                grasp_node_ids=np.flatnonzero(grasped) if t >= _GRASP_STEP else None,
                contact_force=np.zeros(3),
                tissue_F=F,
                contact_mode=_contact_mode(t),
                exposure=exposure,
                safety_strain=float(safety["max"]),
            )
            prev_pos, prev_pose = pos, pose

    return path


# The tool touches before it grasps, and holds the grasp thereafter. Written as
# a schedule rather than per-step randomness so validate_dataset's transition
# check (no NONE -> GRASP in one step, no mode flapping) has a clean case.
_TOUCH_STEP = 3
_GRASP_STEP = 6


def _contact_mode(t: int) -> int:
    if t < _TOUCH_STEP:
        return CONTACT_NONE
    if t < _GRASP_STEP:
        return CONTACT_TOUCH
    return CONTACT_GRASP


def _grasped(X: np.ndarray) -> np.ndarray:
    """The handful of particles nearest the free corner the tool holds."""
    corner = np.array([SHEET_HALF, -SHEET_HALF, SHEET_Z])
    d = np.linalg.norm(X - corner, axis=1)
    keep = np.zeros(len(X), bool)
    keep[np.argsort(d)[:4]] = True
    return keep


def _tool_pose(pos: np.ndarray, grasped: np.ndarray) -> np.ndarray:
    """End-effector pose: at the centroid of the GRASPED particles, no rotation.

    Grasped particles, not the whole sheet. Three of the four deformations here
    are about the sheet centroid, so a tool placed there never moves --
    and visualize_trajectory.py reads tissue moving while the gripper does not
    as a divergence signature and prints an instability warning. The episode is
    fine; the tool was in the wrong place.

    Identity orientation throughout. That is not laziness -- it is the
    degenerate case for quaternion-to-rotation-vector conversion (qw = 1
    exactly), so every episode written here exercises the atan2 path in
    actions.py that exists to survive it.
    """
    return np.concatenate([pos[grasped].mean(axis=0),
                           [0.0, 0.0, 0.0, 1.0]]).astype(np.float32)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_synth",
                    help="directory to write episodes into")
    ap.add_argument("--kinds", nargs="+", default=["all"],
                    help=f"any of {KINDS}, or 'all'")
    ap.add_argument("--n-particles", type=int, default=500,
                    help="rounded to a perfect square")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--float16-F", action="store_true",
                    help="halve file size by storing F as float16. Costs "
                         "precision: float16 spacing near F = I is ~1e-3, so "
                         "sub-0.1%% strain becomes unrepresentable.")
    args = ap.parse_args()

    kinds = list(KINDS) if "all" in args.kinds else args.kinds
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise SystemExit(f"unknown kind(s) {bad}; choose from {KINDS} or 'all'")

    os.makedirs(args.out, exist_ok=True)
    print(f"writing {len(kinds)} synthetic episode(s) to {args.out}/  "
          f"({args.n_steps} steps, dt={DT}s)")
    for i, kind in enumerate(kinds):
        path = os.path.join(args.out, f"{kind}.npz")
        # Distinct seeds per kind, so material_params actually varies across the
        # set -- validate_dataset checks for exactly that, and a shared seed
        # would give four episodes with identical stiffness.
        make_synthetic_episode(path, kind, n_particles=args.n_particles,
                               n_steps=args.n_steps, seed=args.seed + i,
                               store_F_as_float16=args.float16_F)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {kind:9s} -> {os.path.basename(path):16s} {size_mb:6.2f} MB")

    print("\nCheck them with:\n"
          f"  python host/validate_dataset.py --data {args.out}\n"
          f"  python host/visualize_trajectory.py {args.out}/uniaxial.npz")


if __name__ == "__main__":
    main()
