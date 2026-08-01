"""
trajectory_io.py -- the "data contract" between simulators and models.

WHY THIS FILE EXISTS
--------------------
You will collect data in at least three different simulators over this project:
PyBullet (now), Taichi/MPM (soon), Isaac Sim (later). If your model-training code
reads simulator objects directly, you rewrite the model every time you switch.

Instead, every simulator writes the SAME file format, and every model reads that
format. The simulator becomes a swappable component. This file defines that format
and is the only place that knows about it.

This module is intentionally dependency-light: numpy only. It is imported by code
running INSIDE the container (writers) and on the HOST (readers), so it must not
depend on anything exotic.

STORAGE FORMAT
--------------
One compressed .npz file per episode. Chosen over HDF5 because numpy is already
everywhere and there is nothing extra to install. If episodes later get large
enough that you need partial/streaming reads, swap the two functions at the bottom
for h5py equivalents -- nothing else in your codebase should have to change.

THE SCHEMA
----------
Static (recorded once per episode):
    schema_version   str    format version, so old files stay readable
    simulator        str    "pybullet" | "taichi_mpm" | "isaacsim"
    task             str    e.g. "tissue_retraction"
    dt               float  seconds of simulated time per recorded step
    tissue_faces     (F,3) int32   surface triangle topology, or empty
    tissue_tets      (E,4) int32   tetrahedral topology, or empty
    notes            str    free text: mesh resolution, material params, seed...

Per-step (T = number of steps, N = number of tissue nodes):
    tissue_pos       (T,N,3) float32   node positions, metres, world frame
    tissue_vel       (T,N,3) float32   node velocities, m/s (zeros if unavailable)
    ee_pose          (T,7)   float32   end-effector [x,y,z,qx,qy,qz,qw]
    ee_vel           (T,6)   float32   [vx,vy,vz,wx,wy,wz]
    jaw              (T,)    float32   gripper opening angle, radians
    joint_pos        (T,J)   float32   robot joint positions
    action           (T,A)   float32   the command issued at this step
    grasp_active     (T,)    bool      was the gripper holding tissue
    contact_force    (T,3)   float32   net force on the end-effector, newtons

Grasped node indices vary in length per step, so they are stored CSR-style:
    grasp_ids_flat    (sum_of_lengths,) int32
    grasp_ids_offset  (T+1,) int32      step t's ids = flat[offset[t]:offset[t+1]]

UNITS ARE NOT OPTIONAL. Metres, seconds, kilograms, newtons, radians. Every
simulator has different defaults; convert at the writer, never at the reader.
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import numpy as np

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

class TrajectoryWriter:
    """Accumulates one episode in memory, then writes it as a single .npz.

    Typical use (the `with` block guarantees the file is written even on error):

        with TrajectoryWriter("data/ep_000.npz",
                              simulator="pybullet",
                              task="tissue_retraction",
                              dt=1/240) as w:
            for step in range(500):
                ...
                w.append(tissue_pos=..., ee_pose=..., action=...)
    """

    def __init__(
        self,
        path: str,
        simulator: str,
        task: str,
        dt: float,
        tissue_faces: Optional[np.ndarray] = None,
        tissue_tets: Optional[np.ndarray] = None,
        notes: str = "",
    ):
        self.path = path
        self.simulator = simulator
        self.task = task
        self.dt = float(dt)
        self.notes = notes

        self.tissue_faces = _as(tissue_faces, np.int32, (-1, 3))
        self.tissue_tets = _as(tissue_tets, np.int32, (-1, 4))

        # Per-step buffers.
        self._pos, self._vel = [], []
        self._ee_pose, self._ee_vel = [], []
        self._jaw, self._joint, self._action = [], [], []
        self._grasp_active, self._force = [], []
        self._grasp_ids_flat, self._grasp_offset = [], [0]

        # Shapes are locked in on the first append and enforced afterwards.
        # Catching a shape change at step 200 is far cheaper than discovering
        # a corrupt dataset a week later.
        self._n_nodes = None
        self._n_joints = None
        self._n_action = None

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    # -- context manager plumbing so `with` works --------------------------
    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Write even if the episode crashed: a partial trajectory is usually
        # still worth inspecting, and losing 20 minutes of sim to an exception
        # in the last step is infuriating.
        if len(self._pos) > 0:
            self.close()
        return False  # never swallow the exception

    def append(
        self,
        *,
        tissue_pos: np.ndarray,
        ee_pose: np.ndarray,
        action: np.ndarray,
        tissue_vel: Optional[np.ndarray] = None,
        ee_vel: Optional[np.ndarray] = None,
        jaw: float = 0.0,
        joint_pos: Optional[np.ndarray] = None,
        grasp_active: bool = False,
        grasp_node_ids: Optional[Sequence[int]] = None,
        contact_force: Optional[np.ndarray] = None,
    ) -> None:
        """Record one timestep. Keyword-only on purpose: positional args here
        would be a silent-bug factory once there are ten of them."""
        pos = _as(tissue_pos, np.float32, (-1, 3))
        if self._n_nodes is None:
            self._n_nodes = pos.shape[0]
        elif pos.shape[0] != self._n_nodes:
            raise ValueError(
                f"node count changed mid-episode: {self._n_nodes} -> {pos.shape[0]}. "
                "Topology must be constant within one trajectory."
            )

        vel = _as(tissue_vel, np.float32, (-1, 3)) if tissue_vel is not None \
            else np.zeros_like(pos)
        if vel.shape != pos.shape:
            raise ValueError(f"tissue_vel {vel.shape} != tissue_pos {pos.shape}")

        pose = _as(ee_pose, np.float32, (7,))
        evel = _as(ee_vel, np.float32, (6,)) if ee_vel is not None \
            else np.zeros(6, np.float32)
        act = _as(action, np.float32, (-1,))
        jnt = _as(joint_pos, np.float32, (-1,)) if joint_pos is not None \
            else np.zeros(0, np.float32)
        frc = _as(contact_force, np.float32, (3,)) if contact_force is not None \
            else np.zeros(3, np.float32)

        for name, arr, cache in (("action", act, "_n_action"),
                                 ("joint_pos", jnt, "_n_joints")):
            have = getattr(self, cache)
            if have is None:
                setattr(self, cache, arr.shape[0])
            elif arr.shape[0] != have:
                raise ValueError(f"{name} dim changed mid-episode: {have} -> {arr.shape[0]}")

        self._pos.append(pos)
        self._vel.append(vel)
        self._ee_pose.append(pose)
        self._ee_vel.append(evel)
        self._jaw.append(np.float32(jaw))
        self._joint.append(jnt)
        self._action.append(act)
        self._grasp_active.append(bool(grasp_active))
        self._force.append(frc)

        ids = np.asarray(grasp_node_ids, np.int32).ravel() if grasp_node_ids is not None \
            else np.zeros(0, np.int32)
        self._grasp_ids_flat.append(ids)
        self._grasp_offset.append(self._grasp_offset[-1] + ids.size)

    def close(self) -> str:
        if not self._pos:
            raise RuntimeError("nothing to write: append() was never called")

        flat = np.concatenate(self._grasp_ids_flat) if self._grasp_ids_flat \
            else np.zeros(0, np.int32)

        np.savez_compressed(
            self.path,
            schema_version=SCHEMA_VERSION,
            simulator=self.simulator,
            task=self.task,
            dt=np.float32(self.dt),
            notes=self.notes,
            tissue_faces=self.tissue_faces,
            tissue_tets=self.tissue_tets,
            tissue_pos=np.stack(self._pos),
            tissue_vel=np.stack(self._vel),
            ee_pose=np.stack(self._ee_pose),
            ee_vel=np.stack(self._ee_vel),
            jaw=np.asarray(self._jaw, np.float32),
            joint_pos=np.stack(self._joint),
            action=np.stack(self._action),
            grasp_active=np.asarray(self._grasp_active, bool),
            contact_force=np.stack(self._force),
            grasp_ids_flat=flat.astype(np.int32),
            grasp_ids_offset=np.asarray(self._grasp_offset, np.int32),
        )
        self._pos.clear()  # so __exit__ does not write a second time
        return self.path


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

class Trajectory:
    """A loaded episode. Attribute access, e.g. `traj.tissue_pos`."""

    def __init__(self, data: dict, path: str = ""):
        self._d = data
        self.path = path

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(
                f"'{name}' not in trajectory. Available: {sorted(self._d)}"
            ) from None

    def __len__(self) -> int:
        return int(self._d["tissue_pos"].shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self._d["tissue_pos"].shape[1])

    def grasp_ids(self, t: int) -> np.ndarray:
        """Node indices grasped at step t (variable length)."""
        off = self._d["grasp_ids_offset"]
        return self._d["grasp_ids_flat"][off[t]:off[t + 1]]

    def __repr__(self) -> str:
        return (f"<Trajectory {os.path.basename(self.path)} "
                f"sim={self._d['simulator']} task={self._d['task']} "
                f"steps={len(self)} nodes={self.n_nodes} dt={float(self._d['dt']):.5f}>")


def load_trajectory(path: str) -> Trajectory:
    """Read one .npz episode back into memory."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    # numpy stores strings as 0-d arrays; unwrap them so they behave like str.
    for k in ("schema_version", "simulator", "task", "notes"):
        if k in d:
            d[k] = str(d[k])
    if d.get("schema_version") != SCHEMA_VERSION:
        print(f"[trajectory_io] warning: {path} is schema "
              f"{d.get('schema_version')}, this code expects {SCHEMA_VERSION}")
    return Trajectory(d, path)


def list_trajectories(directory: str) -> list:
    """All episode files in a directory, sorted by name."""
    return sorted(glob.glob(os.path.join(directory, "*.npz")))


# --------------------------------------------------------------------------
# Turning trajectories into supervised learning pairs
# --------------------------------------------------------------------------

def make_transition_pairs(traj: Trajectory, horizon: int = 1):
    """Convert an episode into (state_t, action_t) -> (state_{t+horizon}) pairs.

    This is the single most common thing you do with a dynamics dataset, so it
    lives next to the format rather than being re-implemented in every script.

    Returns three float32 arrays:
        x     (T-h, N*6 + 7)   flattened node pos+vel, then end-effector pose
        u     (T-h, A)         action
        y     (T-h, N*3)       node positions `horizon` steps later

    Predicting absolute positions is the naive choice. Predicting the DELTA
    (y - current positions) usually trains far better, because the network no
    longer has to memorise where the tissue happens to sit in world coordinates.
    Set `delta=True` in your own variant once you get to that point.
    """
    T = len(traj)
    if T <= horizon:
        raise ValueError(f"episode has {T} steps, need more than horizon={horizon}")

    pos = traj.tissue_pos.astype(np.float32)
    vel = traj.tissue_vel.astype(np.float32)
    n = pos.shape[1]

    state = np.concatenate(
        [pos.reshape(T, n * 3), vel.reshape(T, n * 3), traj.ee_pose.astype(np.float32)],
        axis=1,
    )
    x = state[:-horizon]
    u = traj.action.astype(np.float32)[:-horizon]
    y = pos[horizon:].reshape(T - horizon, n * 3)
    return x, u, y


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _as(arr, dtype, shape) -> np.ndarray:
    """Coerce to a contiguous array of `dtype`, checking `shape` (-1 = any)."""
    if arr is None:
        return np.zeros([0 if s == -1 else s for s in shape], dtype)
    a = np.ascontiguousarray(arr, dtype=dtype)
    if len(shape) == 1 and a.ndim != 1:
        a = a.ravel()
    if a.ndim != len(shape):
        raise ValueError(f"expected {len(shape)} dimensions, got shape {a.shape}")
    for got, want in zip(a.shape, shape):
        if want != -1 and got != want:
            raise ValueError(f"expected shape {shape}, got {a.shape}")
    return a


if __name__ == "__main__":
    # Running this file directly is a self-test. If it prints OK, the format
    # round-trips correctly on this machine.
    import tempfile

    rng = np.random.default_rng(0)
    N, T = 40, 25
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "selftest.npz")
        with TrajectoryWriter(p, "selftest", "tissue_retraction", 1 / 240,
                              tissue_faces=rng.integers(0, N, (60, 3))) as w:
            for t in range(T):
                w.append(
                    tissue_pos=rng.normal(size=(N, 3)),
                    tissue_vel=rng.normal(size=(N, 3)),
                    ee_pose=np.array([0, 0, 0.1 * t, 0, 0, 0, 1]),
                    action=rng.normal(size=4),
                    joint_pos=rng.normal(size=6),
                    jaw=0.3,
                    grasp_active=t > 5,
                    grasp_node_ids=[1, 2, 3] if t > 5 else None,
                    contact_force=rng.normal(size=3),
                )
        tr = load_trajectory(p)
        assert len(tr) == T and tr.n_nodes == N
        assert tr.grasp_ids(0).size == 0 and tr.grasp_ids(10).tolist() == [1, 2, 3]
        x, u, y = make_transition_pairs(tr, horizon=1)
        assert x.shape == (T - 1, N * 6 + 7) and u.shape == (T - 1, 4) and y.shape == (T - 1, N * 3)
        print(tr)
        print(f"pairs: x{x.shape} u{u.shape} y{y.shape}")
        print("OK -- trajectory_io round-trip passed")
