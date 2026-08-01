#!/usr/bin/env python3
"""
collect_retraction.py -- the first real experiment: scripted tissue retraction.

Run INSIDE the container:

    docker compose run --rm surrol python container/collect_retraction.py --episodes 5
    docker compose run --rm surrol python container/collect_retraction.py --episodes 2 --video

WHAT THIS DOES
--------------
A deformable sheet (standing in for tissue) is dropped onto a table and allowed
to settle. A small rigid block (standing in for the gripper jaws) descends onto
a point on the sheet, "grasps" it by anchoring nearby nodes to itself, lifts,
retracts laterally, holds, and releases. Every step is logged in the shared
trajectory format.

WHY A BLOCK AND NOT THE dVRK PSM
--------------------------------
Deliberately: this script isolates the SOFT BODY physics, which is the part that
is genuinely uncertain. Adding a 7-DOF robot at the same time means that when
something misbehaves you cannot tell whether it is the tissue model, the inverse
kinematics, or the contact handling. Get the tissue behaving sensibly first,
then swap the block for `surrol.robots.psm.Psm1` and drive it with `psm.move()`
-- the logging code below does not change at all.

WHY THE GRASP POINT IS MEASURED, NOT HARDCODED
----------------------------------------------
An earlier version of this script picked the grasp location from hardcoded
coordinates. It caught zero nodes on every episode, for two reasons worth
remembering:

  1. The sheet falls under gravity during the approach, so wherever it starts is
     not where it ends up. Any height fixed in advance is wrong.
  2. The extent of a mesh asset is a property of the file, not something you can
     infer from `scale=0.30`. Guessing where the sheet lies in xy is guessing.

The fix, and the general rule: let the simulation settle, read the actual node
positions, and derive the grasp target from the mesh. Code that measures the
world it is operating in survives changes of asset, scale, and resolution.
Code that hardcodes coordinates breaks silently the first time any of those move.

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
DT = 1.0 / 1000.0          # physics timestep; 240 Hz is PyBullet's default and
                          # deformables become unstable much below it
DEFAULT_MESH = "/work/assets/tissue_20x20.obj"
SHEET_MASS = 0.05         # kg -- a 10 cm square of tissue-like material
SETTLE_TIME = 0.7         # seconds of settling before anything is recorded
GRIPPER_HALF = 0.004      # 4 mm half-extent -> 8 mm jaw, surgical scale
GRASP_RADIUS = 0.008      # nodes this close to the jaw get anchored
MIN_GRASP_NODES = 3       # if the radius catches fewer, take this many nearest

# Motion scaled to a 10 cm sheet. These were 10x larger when the asset was a
# 60 cm demo cloth; lengths in this file are only meaningful relative to the
# tissue's own size, so they must move together with it.
APPROACH_HEIGHT = 0.05    # how far above the grasp point the descent starts
LIFT_HEIGHT = 0.020       # vertical travel during the lift phase
RETRACT_DIST = 0.025      # lateral travel during the retract phase

# The perimeter is pinned to the world. This is the difference between
# retraction and dragging: an unattached sheet just translates under the
# gripper, recording rigid motion with a little flap and no strain field.
# Real tissue is continuous with surrounding structure, so pulling it stretches
# it. Anchoring the boundary is the cheapest honest approximation of that, and
# it is what makes the recorded deformation worth modelling at all.
ANCHOR_BOUNDARY = True

PHASES = [                # (name, duration in seconds)
    ("approach", 0.8),    # descend toward the measured grasp point
    ("grasp",    0.2),    # close: create anchors
    ("lift",     1.2),    # raise vertically
    ("retract",  1.5),    # pull laterally -- the deformation of interest
    ("hold",     0.8),    # let oscillations settle
    ("release",  0.7),    # remove anchors, observe recoil
]


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------

def build_scene(seed: int, mesh_path: str):
    """Create the world, pin the sheet's edges, settle, and measure the result.

    Returns (gripper, sheet, faces, stiffness, damping, settled_pos, boundary).
    """
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
    # than N copies of one trajectory. A dynamics model trained on a single
    # stiffness learns that stiffness, not the underlying dynamics.
    stiffness = float(rng.uniform(40.0, 120.0))
    damping = float(rng.uniform(0.05, 0.30))

    sheet = p.loadSoftBody(
        mesh_path,
        basePosition=[0, 0, 0],  # the mesh is authored at its resting height
        scale=1.0,               # authored in metres, so no rescaling
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

    # Park the gripper well clear so it cannot interfere with settling.
    gripper = p.createMultiBody(
        baseMass=0.0,           # zero mass = kinematic: gravity and contacts do
                                # not move it, so we position it by hand each
                                # step and it acts as an infinitely stiff
                                # actuator. Real robots are not like this, but
                                # it removes controller dynamics from the
                                # picture while we study tissue response.
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[GRIPPER_HALF] * 3),
        basePosition=[0, 0, 1.0],
    )

    # Pin the perimeter to the world. Passing bodyUniqueId = -1 to
    # createSoftBodyAnchor means "anchor to the static world" rather than to
    # another body, which fixes those nodes in place permanently.
    boundary = boundary_nodes(node_positions(sheet))
    if ANCHOR_BOUNDARY:
        for nid in boundary:
            p.createSoftBodyAnchor(sheet, int(nid), -1, -1)

    # Let the sheet sag into equilibrium BEFORE choosing where to grasp.
    for _ in range(int(SETTLE_TIME / DT)):
        p.stepSimulation()

    n, _ = p.getMeshData(sheet, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    faces = load_obj_faces(mesh_path, expect_verts=n)
    return gripper, sheet, faces, stiffness, damping, node_positions(sheet), boundary


def boundary_nodes(pos: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Indices of nodes on the outer edge of the sheet's xy bounding box.

    Geometric rather than topological, so it works for any sheet-like mesh
    without needing to know how it was generated.
    """
    lo, hi = pos[:, :2].min(axis=0), pos[:, :2].max(axis=0)
    on_edge = (
        (np.abs(pos[:, 0] - lo[0]) < tol) | (np.abs(pos[:, 0] - hi[0]) < tol) |
        (np.abs(pos[:, 1] - lo[1]) < tol) | (np.abs(pos[:, 1] - hi[1]) < tol)
    )
    return np.where(on_edge)[0].astype(np.int32)


def load_obj_faces(obj_name: str, expect_verts: int):
    """Recover the mesh triangle connectivity from the source .obj file.

    KNOWN GAP: PyBullet's getMeshData returns node POSITIONS but not the
    topology connecting them. A graph-network dynamics model needs that
    topology -- it is the whole point of the architecture -- so it has to come
    from somewhere, and the only available source is the original .obj.

    This is fragile: it assumes PyBullet's simulation-mesh node ordering matches
    the .obj vertex ordering. Stock cloth assets often have a finer render mesh
    than simulation mesh, in which case the counts differ and topology is
    dropped rather than stored wrongly. Expect that with the demo cloth; it
    matters when you move to a graph model, and the fix is to author your own
    tissue mesh at a chosen resolution.

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
        return np.zeros((0, 3), np.int32)

    if verts != expect_verts:
        return np.zeros((0, 3), np.int32)
    return np.asarray(faces, np.int32)


def node_positions(sheet_id):
    """Current simulation-mesh node positions as an (N,3) float32 array."""
    # MESH_DATA_SIMULATION_MESH gives the nodes the solver actually integrates.
    # Without the flag you get the render mesh, which has different (usually
    # more) vertices and is not what your dynamics model should predict.
    _, verts = p.getMeshData(sheet_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    return np.asarray(verts, np.float32)


def choose_grasp_point(pos: np.ndarray, rng, margin: float = 0.25) -> tuple:
    """Pick an interior node to grasp.

    Interior is defined geometrically: inside the central (1 - 2*margin) box of
    the sheet's xy extent. With `margin=0.25` that is the middle half in each
    direction, so a pinned edge is never grasped -- pulling a pinned node
    produces nothing, and pulling one adjacent to it produces a local tear
    rather than a retraction.

    A previous version ranked nodes by distance from the centroid and took the
    nearest half. On a coarse mesh that still admitted edge nodes, because
    "half the nodes" of a 5x5 grid reaches the boundary. Defining interior by
    geometry rather than by rank is resolution-independent.

    Returns (grasp_xyz, node_index).
    """
    lo, hi = pos[:, :2].min(axis=0), pos[:, :2].max(axis=0)
    span = hi - lo
    inner_lo, inner_hi = lo + margin * span, hi - margin * span
    inside = np.where(
        (pos[:, 0] > inner_lo[0]) & (pos[:, 0] < inner_hi[0]) &
        (pos[:, 1] > inner_lo[1]) & (pos[:, 1] < inner_hi[1])
    )[0]
    if inside.size == 0:                      # degenerate mesh: fall back to
        inside = np.array([                   # the node nearest the centroid
            int(np.argmin(np.linalg.norm(pos[:, :2] - pos[:, :2].mean(0), axis=1)))])
    node = int(rng.choice(inside))
    return pos[node].astype(np.float32), node


def select_grasp_nodes(pos: np.ndarray, jaw_xyz: np.ndarray, exclude=None):
    """Node indices to anchor: everything within GRASP_RADIUS of the jaw, with a
    nearest-N fallback so a grasp can never silently catch nothing.

    `exclude` removes nodes that are already pinned to the world. Grasping one
    would create two competing constraints on the same node, which the solver
    resolves by fighting itself.
    """
    d = np.linalg.norm(pos - jaw_xyz, axis=1)
    ok = np.ones(len(pos), bool)
    if exclude is not None and len(exclude):
        ok[np.asarray(exclude, int)] = False

    ids = np.where((d < GRASP_RADIUS) & ok)[0]
    if len(ids) < MIN_GRASP_NODES:
        order = np.argsort(d)
        order = order[ok[order]]
        ids = order[:min(MIN_GRASP_NODES, len(order))]
    return ids.astype(np.int32)


# --------------------------------------------------------------------------
# Rendering (optional)
# --------------------------------------------------------------------------

def make_camera(width: int, height: int, target):
    """Fixed three-quarter view of the workspace, as matrices PyBullet wants.

    The container has no display, so there is no window to look at. Instead we
    ask the physics server to rasterise a frame into an array, which we then
    write to a video file. This works headless precisely because it never
    touches a GPU or a window system.
    """
    view = p.computeViewMatrix(
        cameraEyePosition=[target[0] + 0.42, target[1] - 0.38, target[2] + 0.34],
        cameraTargetPosition=list(target),
        cameraUpVector=[0, 0, 1],
    )
    proj = p.computeProjectionMatrixFOV(
        fov=55, aspect=width / height, nearVal=0.01, farVal=3.0)
    return view, proj


def grab_frame(width, height, view, proj):
    """Rasterise one frame and return it as an (H,W,3) uint8 array."""
    # ER_TINY_RENDERER is PyBullet's built-in CPU software rasteriser. The
    # alternative, ER_BULLET_HARDWARE_OPENGL, needs a GPU and a window -- it
    # cannot work in this container. Tiny is slow but entirely self-contained.
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    return np.reshape(np.asarray(rgba, np.uint8), (height, width, 4))[:, :, :3]


# --------------------------------------------------------------------------
# The scripted motion
# --------------------------------------------------------------------------

def gripper_target(phase, phase_t, grasp_pt, retract_dir):
    """End-effector position for the current phase, in metres.

    `grasp_pt` is measured from the settled mesh, so every height here is
    relative to where the tissue actually is. Kept as an explicit function of
    phase so it is easy to replace with a policy later -- nothing else in this
    file cares where the target comes from.
    """
    # Jaw centre sits one half-extent above the node so the block's lower face
    # rests on the sheet rather than penetrating it.
    hold = grasp_pt + np.array([0, 0, GRIPPER_HALF], np.float32)

    if phase == "approach":
        # Descend from APPROACH_HEIGHT above the grasp point down to it.
        return hold + np.array([0, 0, APPROACH_HEIGHT * (1.0 - phase_t)], np.float32)
    if phase == "grasp":
        return hold
    if phase == "lift":
        return hold + np.array([0, 0, LIFT_HEIGHT * phase_t], np.float32)
    if phase == "retract":
        return hold + np.array([retract_dir[0] * RETRACT_DIST * phase_t,
                                retract_dir[1] * RETRACT_DIST * phase_t,
                                LIFT_HEIGHT], np.float32)
    # hold and release: stay put and let the tissue settle / recoil
    return hold + np.array([retract_dir[0] * RETRACT_DIST,
                            retract_dir[1] * RETRACT_DIST,
                            LIFT_HEIGHT], np.float32)


def run_episode(index, out_dir, record_every, seed, mesh_path=DEFAULT_MESH,
                video=False, video_size=(480, 360), verbose=False):
    cid = p.connect(p.DIRECT)  # headless: no window, no GPU, maximum speed
    writer = None
    try:
        (gripper, sheet, faces, stiffness, damping,
         settled, boundary) = build_scene(seed, mesh_path)
        rng = np.random.default_rng(seed)

        grasp_pt, grasp_node = choose_grasp_point(settled, rng)
        theta = rng.uniform(0, 2 * np.pi)
        retract_dir = np.array([np.cos(theta), np.sin(theta)], np.float32)

        if verbose:
            lo, hi = settled.min(0), settled.max(0)
            print(f"  mesh: {os.path.basename(mesh_path)}, {len(settled)} nodes, "
                  f"{len(faces)} faces, {len(boundary)} pinned to world")
            print(f"  extent: x[{lo[0]:+.3f},{hi[0]:+.3f}] "
                  f"y[{lo[1]:+.3f},{hi[1]:+.3f}] "
                  f"z[{lo[2]:+.3f},{hi[2]:+.3f}] (m), "
                  f"sag {(hi[2]-lo[2])*1000:.1f} mm")
            print(f"  grasp node {grasp_node} at {grasp_pt.round(4).tolist()}, "
                  f"retract dir {retract_dir.round(2).tolist()}")

        anchors, grasped_ids = [], np.zeros(0, np.int32)
        n_grasped = 0               # captured at grasp time -- grasped_ids is
                                    # cleared on release, so reading it at the
                                    # end of the episode always gives 0
        grasp_done = False          # so the grasp block runs exactly once
        prev_pos = settled.copy()
        prev_ee = gripper_target("approach", 0.0, grasp_pt, retract_dir)

        path = os.path.join(out_dir, f"retraction_{index:04d}.npz")
        notes = (f"pybullet mass-spring cloth; stiffness={stiffness:.2f} "
                 f"damping={damping:.3f}; grasp_node={grasp_node}; "
                 f"grasp_pt={grasp_pt.round(4).tolist()}; "
                 f"retract_dir={retract_dir.round(3).tolist()}; seed={seed}")

        if video:
            import imageio
            vw, vh = video_size
            view, proj = make_camera(vw, vh, grasp_pt)
            writer = imageio.get_writer(
                os.path.join(out_dir, f"retraction_{index:04d}.mp4"),
                fps=max(1, int(1 / (DT * record_every))))

        with TrajectoryWriter(path, simulator="pybullet", task="tissue_retraction",
                              dt=DT * record_every, tissue_faces=faces,
                              notes=notes) as w:
            step = 0
            for phase, duration in PHASES:
                n_steps = int(duration / DT)
                for k in range(n_steps):
                    phase_t = (k + 1) / n_steps

                    target = gripper_target(phase, phase_t, grasp_pt, retract_dir)
                    p.resetBasePositionAndOrientation(
                        gripper, target.tolist(), [0, 0, 0, 1])

                    # --- grasp / release -----------------------------------
                    if phase == "grasp" and not grasp_done:
                        grasp_done = True   # run once, whatever the outcome --
                                            # without this the block retries
                                            # every step of the phase
                        grasped_ids = select_grasp_nodes(
                            node_positions(sheet), target, exclude=boundary)
                        n_grasped = len(grasped_ids)
                        # createSoftBodyAnchor welds a soft-body node to a rigid
                        # body. This is how PyBullet fakes grasping: its contact
                        # model cannot hold a deformable by friction alone.
                        for nid in grasped_ids:
                            anchors.append(
                                p.createSoftBodyAnchor(sheet, int(nid), gripper, -1))
                        if verbose:
                            print(f"  grasped {n_grasped} node(s)")
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

                    if writer is not None:
                        writer.append_data(grab_frame(*video_size, view, proj))

        # Peak deformation over the whole episode, not just the final frame:
        # the sheet recoils after release, so the end state understates it.
        # Boundary nodes are excluded because they are pinned and never move.
        interior = np.setdiff1d(np.arange(len(settled)), boundary)
        traj_pos = np.load(path)["tissue_pos"]
        moved = np.linalg.norm(
            traj_pos[:, interior] - settled[interior], axis=2).max() * 1000
        return path, n_grasped, moved
    finally:
        # `finally` guarantees these are released even if the episode raises.
        # Leaked physics servers eventually exhaust PyBullet's slots, and an
        # unclosed video writer produces a truncated, unplayable file.
        if writer is not None:
            writer.close()
        p.disconnect(cid)


def main():
    # Must be declared before DT is read anywhere in this function -- Python
    # requires `global` to precede every use of the name in the scope, and the
    # --dt help text below reads it.
    global DT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="/work/data")
    ap.add_argument("--record-every", type=int, default=8,
                    help="log every Nth physics step (8 -> 30 Hz at 240 Hz sim). "
                         "Logging every step makes huge files with almost no "
                         "extra information.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mesh", default=DEFAULT_MESH,
                    help="tissue mesh .obj (generate with make_tissue_mesh.py)")
    ap.add_argument("--dt", type=float, default=None,
                    help=f"physics timestep in seconds (default {DT:.6f} = "
                         f"1/{1/DT:.0f}). Smaller is more stable and slower. "
                         "Establish the right value with timestep_study.py "
                         "rather than guessing.")
    ap.add_argument("--video", action="store_true",
                    help="also write an .mp4 per episode. Rendering uses a CPU "
                         "software rasteriser and roughly triples runtime, so "
                         "use it on a few episodes to check behaviour, not on "
                         "a full data-collection run.")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the per-episode geometry report")
    args = ap.parse_args()

    if args.dt is not None:
        # Rebind the module-level constant. Every function reads DT from module
        # globals at call time, so this takes effect everywhere. Keeping one
        # source of truth beats threading a dt argument through six functions.
        DT = float(args.dt)
        print(f"timestep overridden: dt = {DT:.6f} s (1/{1/DT:.0f})")

    if not os.path.isfile(args.mesh):
        raise SystemExit(
            f"Tissue mesh not found: {args.mesh}\nGenerate it first:\n"
            "  docker compose run --rm surrol python container/make_tissue_mesh.py")

    os.makedirs(args.out, exist_ok=True)
    total = sum(d for _, d in PHASES)
    print(f"Collecting {args.episodes} episode(s), {total:.1f}s each "
          f"(+{SETTLE_TIME:.1f}s settling), "
          f"logging at {1/(DT*args.record_every):.0f} Hz -> {args.out}")
    if args.video:
        print("video enabled (CPU rendering -- expect this to be slow)")

    n_bad = 0
    for i in range(args.episodes):
        t0 = time.time()
        # Report geometry on the first episode only: enough to confirm the
        # scene is sane, not enough to drown the log.
        path, n_grasped, moved_mm = run_episode(
            i, args.out, args.record_every, args.seed + i, mesh_path=args.mesh,
            video=args.video, verbose=(i == 0 and not args.quiet))
        dt_wall = time.time() - t0
        flag = ""
        if moved_mm < 1.0:
            flag, n_bad = "   <- tissue barely moved", n_bad + 1
        print(f"  [{i+1}/{args.episodes}] {os.path.basename(path)}  "
              f"{os.path.getsize(path)/1e6:.2f} MB  "
              f"grasped {n_grasped} node(s)  max motion {moved_mm:.1f} mm  "
              f"({dt_wall:.1f}s, {total/max(dt_wall,1e-6):.0f}x realtime){flag}")

    print(f"\nDone. Files are in {args.out}, which is your Mac's "
          "~/tissue-dynamics/data folder.")
    if n_bad:
        print(f"{n_bad} episode(s) showed almost no deformation. Inspect one:\n"
              "  python host/visualize_trajectory.py data/retraction_0000.npz")


if __name__ == "__main__":
    main()
