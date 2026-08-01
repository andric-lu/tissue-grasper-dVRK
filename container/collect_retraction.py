#!/usr/bin/env python3
"""
collect_retraction.py -- the first real experiment: scripted tissue retraction.

Run INSIDE the container:

    docker compose run --rm surrol python container/collect_retraction.py --episodes 5

WHAT THIS DOES
--------------
A deformable sheet (standing in for tissue) rests on a table. A small rigid
block (standing in for the gripper jaws) descends, "grasps" a patch of the sheet
by anchoring nearby nodes to itself, lifts, retracts laterally, holds, and
releases. Every step is logged in the shared trajectory format.

WHY A BLOCK AND NOT THE dVRK PSM
--------------------------------
Deliberately: this script isolates the SOFT BODY physics, which is the part
that is genuinely uncertain. Adding a 7-DOF robot at the same time means that
when something misbehaves you cannot tell whether it is the tissue model, the
inverse kinematics, or the contact handling. Get the tissue behaving sensibly
first, then swap the block for `surrol.robots.psm.Psm1` and drive it with
`psm.move()` -- the logging code below does not change at all.

WHAT THIS IS AND IS NOT
-----------------------
This is a mass-spring cloth, not a constitutive model of tissue. Stiffness here
is a tuning knob with no physical units, and the sheet has no volume, so it
cannot capture the incompressibility that dominates real tissue response. Treat
the resulting dataset as a pipeline test, not as ground truth for a dynamics
model. The MPM/Neo-Hookean work is where physically meaningful data starts.
"""

import argparse
import os
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, "/work/src")
from trajectory_io import TrajectoryWriter  # noqa: E402

# --------------------------------------------------------------------------
# Parameters. Units: metres, seconds, kilograms.
# --------------------------------------------------------------------------
DT = 1.0 / 240.0          # physics timestep; 240 Hz is PyBullet's default and
                          # deformables become unstable much below it
SHEET_SCALE = 0.30        # ~30 cm square of "tissue"
SHEET_MASS = 0.10         # kg
GRIPPER_HALF = 0.012      # 12 mm half-extent -> 24 mm block
GRASP_RADIUS = 0.030      # nodes within 3 cm of the jaw get anchored

PHASES = [                # (name, duration in seconds)
    ("approach", 1.0),    # descend toward the sheet
    ("grasp",    0.3),    # close: create anchors
    ("lift",     1.2),    # raise vertically
    ("retract",  1.5),    # pull laterally -- the deformation of interest
    ("hold",     0.8),    # let oscillations settle
    ("release",  0.7),    # remove anchors, observe recoil
]


def build_scene(seed: int):
    """Create the world and return (gripper_id, sheet_id, rest_positions)."""
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    # ^ Deformables live in a different world type. Without this flag,
    #   loadSoftBody fails with "not supported in this physics engine".

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    # Sparse SDF voxels accelerate deformable-vs-rigid collision queries.
    # Without it, soft/rigid contact is very slow.
    p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)

    p.loadURDF("plane.urdf", [0, 0, 0])

    rng = np.random.default_rng(seed)
    # Randomise per episode so the dataset covers a range of conditions rather
    # than 500 copies of one trajectory. A dynamics model trained on a single
    # stiffness learns that stiffness, not the underlying dynamics.
    stiffness = float(rng.uniform(25.0, 60.0))
    damping = float(rng.uniform(0.05, 0.30))

    sheet = p.loadSoftBody(
        "cloth_z_up.obj",
        basePosition=[0, 0, 0.05],
        scale=SHEET_SCALE,
        mass=SHEET_MASS,
        useNeoHookean=0,        # mass-spring, not FEM: less realistic, far more
        useMassSpring=1,        # stable for a first pipeline test
        useBendingSprings=1,
        springElasticStiffness=stiffness,
        springDampingStiffness=damping,
        springDampingAllDirections=1,
        useSelfCollision=0,     # expensive; enable later if the sheet folds
        frictionCoeff=0.5,
        useFaceContact=1,
    )

    # A zero-mass multibody is kinematic: gravity and contacts do not move it,
    # so we can position it by hand each step and it acts as an infinitely
    # stiff actuator. Real robots are not like this, but it removes controller
    # dynamics from the picture while we are studying tissue response.
    gripper = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[GRIPPER_HALF] * 3),
        basePosition=[0, 0, 0.35],
    )

    n, verts = p.getMeshData(sheet, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    faces = load_obj_faces("cloth_z_up.obj", expect_verts=n)
    return gripper, sheet, faces, stiffness, damping


def load_obj_faces(obj_name: str, expect_verts: int):
    """Recover the mesh triangle connectivity from the source .obj file.

    KNOWN GAP: PyBullet's getMeshData returns node POSITIONS but not the
    topology connecting them. A graph-network dynamics model needs that
    topology -- it is the whole point of the architecture -- so it has to come
    from somewhere, and the only available source is the original .obj.

    This is fragile: it assumes PyBullet's simulation-mesh node ordering matches
    the .obj vertex ordering. That holds for the stock cloth assets but is not
    guaranteed in general. The vertex-count check below catches the obvious
    mismatch; a subtler reordering would pass it silently. If you build a graph
    model and its predictions are nonsense while an MLP works, suspect this
    function first.

    Returns an (F,3) int32 array, or an empty array if recovery fails.
    """
    path = os.path.join(pybullet_data.getDataPath(), obj_name)
    verts, faces = 0, []
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("v "):
                    verts += 1
                elif line.startswith("f "):
                    # .obj faces are 1-indexed and may be "v/vt/vn"; take the
                    # vertex index and convert to 0-indexed.
                    idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                    # Fan-triangulate anything with more than 3 corners.
                    for k in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[k], idx[k + 1]))
    except OSError:
        print(f"  warning: could not read {path}; trajectories will have no topology")
        return np.zeros((0, 3), np.int32)

    if verts != expect_verts:
        print(f"  warning: {obj_name} has {verts} vertices but the simulation "
              f"mesh has {expect_verts}. Node ordering cannot be trusted, so "
              "topology is being dropped rather than stored incorrectly.")
        return np.zeros((0, 3), np.int32)
    return np.asarray(faces, np.int32)


def node_positions(sheet_id):
    """Current simulation-mesh node positions as an (N,3) float32 array."""
    # MESH_DATA_SIMULATION_MESH gives the nodes the solver actually integrates.
    # Without the flag you get the render mesh, which has different (usually
    # more) vertices and is not what your dynamics model should predict.
    _, verts = p.getMeshData(sheet_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    return np.asarray(verts, np.float32)


def gripper_trajectory(t, phase, phase_t, grasp_xy, lift_dir):
    """Scripted end-effector position for the current phase.

    Returns the target (x, y, z). Kept as an explicit function of time so it is
    easy to replace with a policy later -- the rest of the script does not care
    where the target comes from.
    """
    gx, gy = grasp_xy
    approach_z, grasp_z = 0.35, 0.055
    if phase == "approach":
        s = phase_t  # 0 -> 1 over the phase
        return gx, gy, approach_z + s * (grasp_z - approach_z)
    if phase == "grasp":
        return gx, gy, grasp_z
    if phase == "lift":
        return gx, gy, grasp_z + 0.10 * phase_t
    if phase == "retract":
        return (gx + lift_dir[0] * 0.12 * phase_t,
                gy + lift_dir[1] * 0.12 * phase_t,
                grasp_z + 0.10)
    # hold and release: stay put and let the tissue settle / recoil
    return (gx + lift_dir[0] * 0.12, gy + lift_dir[1] * 0.12, grasp_z + 0.10)


def run_episode(index: int, out_dir: str, record_every: int, seed: int) -> str:
    cid = p.connect(p.DIRECT)  # headless: no window, no GPU, maximum speed
    try:
        gripper, sheet, faces, stiffness, damping = build_scene(seed)
        rng = np.random.default_rng(seed)

        # Pick a grasp point somewhere on the sheet, and a retraction direction.
        grasp_xy = rng.uniform(-0.08, 0.08, size=2)
        theta = rng.uniform(0, 2 * np.pi)
        lift_dir = np.array([np.cos(theta), np.sin(theta)], np.float32)

        anchors, grasped_ids = [], np.zeros(0, np.int32)
        prev_pos = node_positions(sheet)
        prev_ee = np.array([grasp_xy[0], grasp_xy[1], 0.35], np.float32)

        path = os.path.join(out_dir, f"retraction_{index:04d}.npz")
        notes = (f"pybullet mass-spring cloth; stiffness={stiffness:.2f} "
                 f"damping={damping:.3f}; grasp_xy={grasp_xy.round(4).tolist()}; "
                 f"retract_dir={lift_dir.round(3).tolist()}; seed={seed}")

        with TrajectoryWriter(path, simulator="pybullet", task="tissue_retraction",
                              dt=DT * record_every, tissue_faces=faces,
                              notes=notes) as w:
            step = 0
            for phase, duration in PHASES:
                n_steps = int(duration / DT)
                for k in range(n_steps):
                    phase_t = (k + 1) / n_steps

                    target = np.array(
                        gripper_trajectory(step * DT, phase, phase_t, grasp_xy, lift_dir),
                        np.float32)
                    p.resetBasePositionAndOrientation(gripper, target.tolist(), [0, 0, 0, 1])

                    # --- grasp / release -----------------------------------
                    if phase == "grasp" and not anchors:
                        pos = node_positions(sheet)
                        d = np.linalg.norm(pos - target, axis=1)
                        grasped_ids = np.where(d < GRASP_RADIUS)[0].astype(np.int32)
                        # createSoftBodyAnchor welds a soft-body node to a rigid
                        # body. This is how PyBullet fakes grasping: its contact
                        # model cannot hold a deformable by friction alone.
                        for nid in grasped_ids:
                            anchors.append(
                                p.createSoftBodyAnchor(sheet, int(nid), gripper, -1))
                        if len(grasped_ids) == 0:
                            print(f"  ep{index}: warning -- grasp caught 0 nodes; "
                                  "increase GRASP_RADIUS or lower grasp_z")
                    if phase == "release" and anchors:
                        for a in anchors:
                            p.removeConstraint(a)
                        anchors, grasped_ids = [], np.zeros(0, np.int32)

                    p.stepSimulation()
                    step += 1

                    # --- logging -------------------------------------------
                    if step % record_every:
                        continue
                    pos = node_positions(sheet)
                    # PyBullet does not expose soft-body node velocities, so
                    # they are finite-differenced. This is noisier than true
                    # velocities; if your model struggles, that noise is a
                    # prime suspect.
                    dt_rec = DT * record_every
                    vel = (pos - prev_pos) / dt_rec
                    ee_vel = np.concatenate([(target - prev_ee) / dt_rec, np.zeros(3)])
                    prev_pos, prev_ee = pos, target

                    w.append(
                        tissue_pos=pos,
                        tissue_vel=vel,
                        ee_pose=np.concatenate([target, [0, 0, 0, 1]]),
                        ee_vel=ee_vel,
                        action=np.concatenate([target, [1.0 if anchors else 0.0]]),
                        jaw=0.0 if anchors else 0.5,
                        grasp_active=bool(anchors),
                        grasp_node_ids=grasped_ids,
                    )
        return path
    finally:
        # `finally` guarantees the physics server is released even if the
        # episode raises. Leaked servers eventually exhaust PyBullet's slots.
        p.disconnect(cid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="/work/data")
    ap.add_argument("--record-every", type=int, default=8,
                    help="log every Nth physics step (8 -> 30 Hz at 240 Hz sim). "
                         "Logging every step makes huge files with almost no "
                         "extra information.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = sum(d for _, d in PHASES)
    print(f"Collecting {args.episodes} episode(s), {total:.1f}s each, "
          f"logging at {1/(DT*args.record_every):.0f} Hz -> {args.out}")

    for i in range(args.episodes):
        t0 = time.time()
        path = run_episode(i, args.out, args.record_every, args.seed + i)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  [{i+1}/{args.episodes}] {os.path.basename(path)}  "
              f"{size_mb:.2f} MB  ({time.time()-t0:.1f}s wall, "
              f"{total/(time.time()-t0):.1f}x realtime)")

    print(f"\nDone. Files are in {args.out}, which is your Mac's "
          "~/tissue-dynamics/data folder.")


if __name__ == "__main__":
    main()
