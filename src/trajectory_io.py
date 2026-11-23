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

WHEN TO ACTUALLY MAKE THAT SWAP: `tissue_F` is the trigger. It is nine floats per
particle per step, which dwarfs everything else in the file. Measured against
real MPM output (24,000 particles, J down to 0.55): 79.3 MB per 100 recorded
steps for the deformation gradient alone. A 300-step episode is 238 MB and 200
of them is 48 GB.

Do not expect compression to rescue this. `savez_compressed` recovers **8.3%**
on float32 physics data -- the low mantissa bits are noise and zlib cannot model
them. The size has to come out of the data, not out of the encoder.

Two knobs do that, and they compose (see `f_encoding` and `particle_ids` below):

    f_encoding="delta16"     2.0x   quantise F - I, not F
    record 3000 of 24000     8.0x   the solver's resolution is not the dataset's
    both                    15.5x   79.3 MB -> 5.1 MB per 100 steps

When 15x is no longer enough, move to h5py -- the two functions at the bottom of
this file are the only ones that would change.

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
  -- added in v2 --
    material_params  (3,) float32  [log(mu), log(lambda), rho]. Logs, not Pascals:
                                   tissue stiffness spans orders of magnitude and
                                   a network fed raw Pascals burns capacity on the
                                   exponent. Empty (0,) if the simulator has no
                                   constitutive model (PyBullet mass-spring).
    substep_dt       float32       solver substep, seconds. 0.0 = not recorded.
    n_substeps       int32         substeps per recorded step. 0 = not recorded.
    boundary_mask    (N,) bool     True where a particle is kinematically clamped.
                                   Empty (0,) if unknown.
    action_spec      str           "abs_pose_jaw" | "delta_pose_jaw" | "unknown"
    target_origin    (3,) float32  centre of the target region to expose
    target_normal    (3,) float32  unit normal of the target plane
    target_extent    (2,) float32  half-extents of the target rectangle, metres
  -- added in v2.1 --
    particle_ids     (N,) int32    for each recorded node, its index in the
                                   SOLVER's full particle array. Empty (0,) means
                                   "not recorded", which for a mesh solver is the
                                   normal case: nodes are all the nodes there are.
    n_particles_simulated int32    how many particles the solver actually stepped.
                                   0 = not recorded. When this exceeds N, the file
                                   holds a SUBSET and every per-particle array is
                                   a sample, not a census. See SUBSETS below.
    f_encoding       str           how tissue_F is encoded ON DISK; the reader
                                   always hands back float32 F. See F ENCODINGS.
    grid_dx          float32       solver grid spacing, metres. 0.0 = not
                                   recorded. Stability advisories scale linearly
                                   with it, so a validator that has to ASSUME a
                                   value is off by whatever ratio it guessed
                                   wrong -- and it will guess wrong, because dx
                                   is a property of the solver, not of the
                                   universe. Record it.

  -- added in v3.0 (the graph-model schema; field dictionary in
     docs/FIELD_DICTIONARY.md, which is the authoritative description) --
    rest_pos         (N,3) float32  reference (undeformed, F = I) position X of
                                    every recorded node, aligned to particle_ids.
                                    Empty = not recorded. NOT "frame 0" unless
                                    the file says frame 0 is the reference state
                                    (see preroll_s).
    rest_pos_simulated (n_sim,3) float32  reference position of EVERY solver
                                    particle. Lets a node set be re-chosen after
                                    collection, and holds the true rest positions
                                    of grasped particles that the recorded subset
                                    misses. 288 KB at 24,000 particles.
    material_field   (N,3) float32  per-node [log(mu), log(lambda), rho]; the
                                    episode-level material_params broadcast for
                                    homogeneous tissue.
    control_dt       (1,) float64   seconds between successive actions. Empty =
                                    not recorded.
    preroll_s        (1,) float64   simulated seconds run BEFORE row 0 (a settle
                                    pre-roll). 0.0 means row 0 is the reference
                                    configuration itself. Empty = not recorded.
    jaw_names        (K,) str       one name per jaw, fixing the order of every
                                    (.., K, ..) array below. K = 2 for the PSM.
    jaw_half_extents (K,3) float32  half-extents of each jaw's collision box, in
                                    that box's own frame, metres.
    jaw_box_offset   (K,7) float32  T_jaw_box: the box frame in the jaw-link
                                    frame [x,y,z,qx,qy,qz,qw]. Recorded so the box
                                    can be rebuilt from a jaw-link pose.
    grasp_capture_radius (1,) float64  the two-jaw capture distance, metres
    jaw_release_rad  (1,) float64   jaw angle at/above which a grasp releases
    contact_radius   (1,) float64   node-to-jaw-surface distance that sets the
                                    node-level contact flag, metres
    robot_model      str            "dvrk_si_psm" | "dvrk_classic_psm_surrol";
                                    "" = not recorded (a v3 file without it is Si)
    robot_scale      (1,) float64   the arm's geometric scaling (SurRoL
                                    globalScaling); 1 = life-size

Per-step (T = number of steps, N = number of tissue nodes/particles):
    tissue_pos       (T,N,3) float32   node positions, metres, world frame
    tissue_vel       (T,N,3) float32   node velocities, m/s (zeros if unavailable)
    ee_pose          (T,7)   float32   end-effector [x,y,z,qx,qy,qz,qw]
    ee_vel           (T,6)   float32   [vx,vy,vz,wx,wy,wz]
    jaw              (T,)    float32   gripper opening angle, radians
    joint_pos        (T,J)   float32   robot joint positions
    action           (T,A)   float32   the command issued at this step
    grasp_active     (T,)    bool      was the gripper holding tissue
    contact_force    (T,3)   float32   net force on the end-effector, newtons
  -- added in v2 --
    tissue_F         (T,N,3,3) float32 deformation gradient. Empty (0,) when the
                                       solver has none: a mass-spring cloth has no
                                       F, and writing identity there would be a
                                       fabricated measurement, not a default.
    contact_mode     (T,) int8         CONTACT_* enum below
    exposure         (T,) float32      logged success metric, see tissue_metrics
    safety_strain    (T,) float32      logged safety metric, see tissue_metrics

  -- added in v3.0 --
    time             (T,) float64      seconds since row 0. Row 0 is 0.0.
    jaw_box_pose     (T,K,7) float32   each jaw collision box, world frame
    jaw_box_vel      (T,K,6) float32   [v, omega] world frame, BACKWARD
                                       difference (pose_t - pose_{t-1}) / dt;
                                       row 0 is zero (the arm starts at rest).
                                       The motion over t -> t+1 is the NEXT row's
                                       value -- that is what the solver applied.
    jaw_joint        (T,K) float32     each jaw's own joint angle, radians
    node_contact     (T,N) bool        node within contact_radius of a jaw box
                                       surface (proximity, NOT proof of contact)
    phase            (T,) int8         scripted phase of each row, see
                                       src/collection_config.PHASES
    node_grasp       (T,N) bool        node-level grasp latch, derived by
                                       src/tool_geometry.py's rule from recorded
                                       geometry -- the same rule a model must use
                                       at inference
    grasp_full_ids_flat / grasp_full_ids_offset   CSR like grasp_ids below, but
                                       FULL-SOLVER particle indices: the ground
                                       truth of what the solver actually held
    safety_strain_id (T,) int32        FULL-SOLVER index of the particle that
                                       carried the logged safety_strain (the
                                       argmax of max principal stretch over all
                                       simulated particles). Added 20 Nov
                                       (DECISION_LOG 14.20): without it nothing
                                       in a subset file says whether the
                                       logged peak was recordable at all.
    safety_strain_pos (T,3) float32    that particle's world position, so its
                                       distance to the jaws can be derived from
                                       jaw_box_pose even when it is not a node

Grasped node indices vary in length per step, so they are stored CSR-style:
    grasp_ids_flat    (sum_of_lengths,) int32
    grasp_ids_offset  (T+1,) int32      step t's ids = flat[offset[t]:offset[t+1]]

F ENCODINGS
-----------
`f_encoding` describes the bytes on disk. Every reader gets float32 F back
regardless, so nothing downstream branches on it.

    "float32"   (T,N,3,3) float32. Lossless, the default, 79.3 MB/100 steps.
    "float16"   (T,N,3,3) float16 of F itself. Halves it.
    "delta16"   (T,N,3,3) float16 of H = F - I. Halves it, and is far more
                accurate than "float16" for the same bytes.

WHY "delta16" BEATS "float16" AT IDENTICAL SIZE: float16 spacing is not uniform.
Near 1.0 it is ~9.8e-4, so storing F itself throws away exactly the small strains
-- a 0.1% stretch is not representable at all. Near 0.0 the spacing falls to
~6e-8. The undeformed state is F = I, so subtracting I moves the whole dataset
into the precise part of the number line. Measured on real MPM output:

                     max err in J     in max principal stretch
    "float16"           1.5e-3                4.9e-4
    "delta16"           1.4e-4                3.1e-5

`validate_dataset.py` recomputes the logged metrics to a tolerance of 1e-3, so
"float16" lands at half the tolerance while "delta16" sits 32x below it. Prefer
"delta16"; "float16" is kept because v2.0 files were written with it.

CAUTION IF YOU np.load A FILE BY HAND: under "delta16" the array stored at key
`tissue_F` holds F - I, not F. `load_trajectory` reconstructs it. Reading the
raw npz and forgetting that gives you a deformation gradient centred on zero,
which is not a deformation gradient at all -- check `f_encoding` first.

SUBSETS: THE SOLVER'S RESOLUTION IS NOT THE DATASET'S
-----------------------------------------------------
An MPM needs enough particles for stable physics (~24,000 here); a graph network
needs few enough to train (MeshGraphNets runs 1.5k-5k nodes). These are different
numbers and conflating them makes the dataset infeasible for the model long
before it is inconvenient for the disk. So a writer may record a fixed SUBSET of
particles, naming them in `particle_ids` and the full count in
`n_particles_simulated`.

The subset must be FIXED for the whole episode. Node identity has to be stable
across time or consecutive frames describe different particles, which is
meaningless for a dynamics model and silently wrong rather than loud.

THE TRAP, AND THE RULE THAT AVOIDS IT: `safety_strain` is a MAXIMUM over
particles, and a maximum over a subset is biased low by construction. Measured:
recording 3,000 of 24,000 particles underestimated peak stretch by 8% of the
stretch above rest, and erratically -- one random 6,000-particle draw missed the
worst particle while a 3,000-particle draw happened to catch it. A safety number
that reads low because you stored less data is the worst failure available here.

    RULE: compute `exposure` and `safety_strain` over ALL simulated particles,
    at collection time, and log the scalars. Never over the stored subset.

This costs nothing -- the full state is already in memory when the metrics are
computed -- and it is exactly why the schema logs metrics that are otherwise
derivable. `validate_dataset.py` knows about subsets and checks the logged value
with an inequality rather than an equality.

READING v1 FILES
----------------
v1 files load without error. Fields introduced in v2 come back as EMPTY arrays,
never as zeros. That distinction is deliberate: 0.0 is a legitimate value for
`exposure` (fully occluded) and for `safety_strain`, and all-False is a
legitimate value for `boundary_mask`, so a zero default is indistinguishable from
a real measurement. `arr.size == 0` means "this simulator did not record it" and
callers can branch on it honestly. The two scalars `substep_dt`/`n_substeps` are
the exception -- zero substeps is not a physically meaningful reading, so 0
carries "not recorded" without ambiguity.

UNITS ARE NOT OPTIONAL. Metres, seconds, kilograms, newtons, radians. Every
simulator has different defaults; convert at the writer, never at the reader.
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import numpy as np

SCHEMA_VERSION = "3.0"

# Every version this reader knows how to turn into a v2-shaped Trajectory.
# A file claiming anything else cannot be interpreted safely, and guessing is
# worse than stopping: silently misreading a field is how a whole training run
# gets thrown away a week later.
KNOWN_SCHEMA_VERSIONS = ("1.0", "2.0", "2.1", "3.0")

# How tissue_F may be laid out on disk. The reader normalises all of these to
# float32 F, so this never leaks past load_trajectory(). See F ENCODINGS above.
F_ENCODINGS = ("float32", "float16", "delta16")

# How far dt may drift from substep_dt * n_substeps before the writer refuses
# the episode, as a FRACTION of dt.
#
# WHY RELATIVE AND NOT ABSOLUTE: substep_dt is stored as float32, so
# reconstructing dt from it accumulates up to n_substeps roundings. Measured on
# a real episode: dt = 12.5 ms in 24 substeps drifts 4.7e-10 s, which an
# absolute 1e-9 s bound clears by only a factor of two -- one unlucky material
# away from rejecting a perfectly good episode for arithmetic reasons. The
# relative form is scale-free, so it does not need re-tuning when frame_dt or
# the substep count changes.
#
# 1e-6 is four orders of magnitude above float32 reconstruction error and four
# below the smallest error worth catching: one substep miscounted out of 1000
# is a 1e-3 timebase error, and the bug this exists for was 3.2 (a factor of
# 4.2). There is nothing in between that is both physically meaningful and
# undetectable here.
TIMEBASE_TOLERANCE = 1e-6

# The undeformed deformation gradient. "delta16" stores F - I so that the
# quantisation error lands where float16 is precise (near 0) instead of where
# it is coarse (near 1). Defined once so the encoder and decoder cannot drift.
_IDENTITY_3 = np.eye(3, dtype=np.float32)


def _encode_F(F: Optional[np.ndarray], encoding: str) -> np.ndarray:
    """(T,N,3,3) float32 deformation gradients -> the array to put on disk."""
    if F is None or F.size == 0:
        # Empty means "this solver has no F". Kept float32 so that a file with
        # no deformation gradient never looks like a quantised one.
        return np.zeros(0, np.float32)
    F = np.asarray(F, np.float32)
    if encoding == "float32":
        return F
    if encoding == "float16":
        return F.astype(np.float16)
    if encoding == "delta16":
        return (F - _IDENTITY_3).astype(np.float16)
    raise ValueError(f"unknown f_encoding {encoding!r}")


def _decode_F(arr: np.ndarray, encoding: str) -> np.ndarray:
    """The inverse of `_encode_F`. Always returns float32, whatever went in."""
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    if encoding == "delta16":
        # Upcast BEFORE adding I: doing it in float16 would round the sum back
        # to float16 spacing near 1.0 and throw away precisely the precision
        # this encoding exists to preserve.
        return arr.astype(np.float32) + _IDENTITY_3
    return arr.astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# Contact mode
# --------------------------------------------------------------------------
# Defined here, next to the schema, because this is the ONLY place that should
# know the integer values. A collector, a validator and a model that each carry
# their own copy of this enum will disagree eventually, and the disagreement
# shows up as a model that has learned the wrong contact semantics rather than
# as an error.
CONTACT_NONE = 0     # tool is not touching tissue
CONTACT_TOUCH = 1    # touching, no tangential constraint
CONTACT_STICK = 2    # touching, tangentially held by friction
CONTACT_SLIP = 3     # touching, sliding
CONTACT_GRASP = 4    # jaws closed, tissue kinematically attached to the tool

CONTACT_MODE_NAMES = {
    CONTACT_NONE: "none",
    CONTACT_TOUCH: "touch",
    CONTACT_STICK: "stick",
    CONTACT_SLIP: "slip",
    CONTACT_GRASP: "grasp",
}

# Legal values for the static `action_spec` field. "unknown" exists for v1 files,
# whose 4-element [target_xyz, grasp_flag] action is neither of the two real
# specs; calling it "abs_pose_jaw" would misdescribe its width and its contents.
ACTION_SPECS = ("abs_pose_jaw", "delta_pose_jaw", "unknown")


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
        *,
        material_params: Optional[np.ndarray] = None,
        substep_dt: float = 0.0,
        n_substeps: int = 0,
        boundary_mask: Optional[np.ndarray] = None,
        action_spec: str = "unknown",
        target_origin: Optional[np.ndarray] = None,
        target_normal: Optional[np.ndarray] = None,
        target_extent: Optional[np.ndarray] = None,
        store_F_as_float16: bool = False,
        f_encoding: Optional[str] = None,
        particle_ids: Optional[np.ndarray] = None,
        n_particles_simulated: int = 0,
        grid_dx: float = 0.0,
        rest_pos: Optional[np.ndarray] = None,
        rest_pos_simulated: Optional[np.ndarray] = None,
        material_field: Optional[np.ndarray] = None,
        control_dt: Optional[float] = None,
        preroll_s: Optional[float] = None,
        jaw_names: Optional[Sequence[str]] = None,
        jaw_half_extents: Optional[np.ndarray] = None,
        jaw_box_offset: Optional[np.ndarray] = None,
        grasp_capture_radius: Optional[float] = None,
        jaw_release_rad: Optional[float] = None,
        contact_radius: Optional[float] = None,
        robot_model: str = "",
        robot_scale: Optional[float] = None,
    ):
        # Everything added in v2 is keyword-only. The four leading positional
        # arguments are kept positional because callers already pass them that
        # way; adding a tenth positional argument to this list would be the
        # silent-bug factory that `append` is already keyword-only to avoid.
        self.path = path
        self.simulator = simulator
        self.task = task
        self.dt = float(dt)
        self.notes = notes

        self.tissue_faces = _as(tissue_faces, np.int32, (-1, 3))
        self.tissue_tets = _as(tissue_tets, np.int32, (-1, 4))

        if action_spec not in ACTION_SPECS:
            raise ValueError(f"action_spec must be one of {ACTION_SPECS}, got {action_spec!r}")
        self.action_spec = action_spec

        # Static v2 fields. `_as(None, ...)` yields the empty array that means
        # "not recorded" -- see the READING v1 FILES note in the module docstring.
        self.material_params = _as(material_params, np.float32, (3,)) \
            if material_params is not None else np.zeros(0, np.float32)
        self.substep_dt = float(substep_dt)
        self.n_substeps = int(n_substeps)
        # THE TIMEBASE MUST CLOSE. dt is seconds per RECORDED frame; substep_dt
        # x n_substeps is how much time the solver was actually asked to
        # advance between frames. If they disagree, every velocity, every
        # finite difference and every learned dynamics model downstream is
        # scaled by the ratio, and nothing in the file looks wrong -- the three
        # numbers are individually plausible and only their product is a lie.
        #
        # WHY THE ZERO GUARD: v1 files and the PyBullet collector record
        # neither field (0.0 means "not recorded", per the module docstring),
        # and a solver that does not substep is not lying about anything. Only
        # a caller claiming BOTH is held to the claim.
        if self.substep_dt > 0.0 and self.n_substeps > 0 and self.dt > 0.0:
            implied = self.substep_dt * self.n_substeps
            rel = abs(implied - self.dt) / self.dt
            if rel > TIMEBASE_TOLERANCE:
                raise ValueError(
                    f"timebase does not close: substep_dt {self.substep_dt:.6e} s "
                    f"x n_substeps {self.n_substeps} = {implied:.6e} s, but dt "
                    f"is {self.dt:.6e} s (off by {rel:.2%}, tolerance "
                    f"{TIMEBASE_TOLERANCE:.0e}). A recorded frame must advance "
                    "the solver by exactly dt.")
        self.boundary_mask = _as(boundary_mask, bool, (-1,)) \
            if boundary_mask is not None else np.zeros(0, bool)
        self.target_origin = _as(target_origin, np.float32, (3,)) \
            if target_origin is not None else np.zeros(0, np.float32)
        self.target_normal = _as(target_normal, np.float32, (3,)) \
            if target_normal is not None else np.zeros(0, np.float32)
        self.target_extent = _as(target_extent, np.float32, (2,)) \
            if target_extent is not None else np.zeros(0, np.float32)

        # How F is laid out on disk. This is a STORAGE choice, not a compute
        # one: the reader hands back float32 whichever branch is taken here.
        #
        # `store_F_as_float16` predates `f_encoding` and is kept because v2.0
        # files and src/synthetic_traj.py both use it. The two are not allowed
        # to disagree -- silently honouring one and ignoring the other is how a
        # dataset ends up encoded differently from what its caller believes.
        if f_encoding is None:
            f_encoding = "float16" if store_F_as_float16 else "float32"
        elif store_F_as_float16 and f_encoding != "float16":
            raise ValueError(
                f"store_F_as_float16=True conflicts with f_encoding={f_encoding!r}. "
                "Pass only f_encoding; the bool is the older spelling of "
                "f_encoding='float16'.")
        if f_encoding not in F_ENCODINGS:
            raise ValueError(f"f_encoding must be one of {F_ENCODINGS}, "
                             f"got {f_encoding!r}")
        self.f_encoding = f_encoding
        self.store_F_as_float16 = (f_encoding == "float16")

        # A subset record names the particles it kept. Empty means "these are
        # all the particles there are", which is the normal case for a mesh
        # solver and for any MPM run that records everything.
        self.particle_ids = _as(particle_ids, np.int32, (-1,)) \
            if particle_ids is not None else np.zeros(0, np.int32)
        self.n_particles_simulated = int(n_particles_simulated)
        # Zero is unambiguous: a solver cannot have zero grid spacing, so 0.0
        # reads as "not recorded" without colliding with a real measurement.
        self.grid_dx = float(grid_dx)
        if self.grid_dx < 0.0:
            raise ValueError(f"grid_dx must be positive or 0 (not recorded), "
                             f"got {self.grid_dx}")

        # Caught here rather than at close(), because the caller that built a
        # bad index set is on the stack right now and will not be later.
        if self.particle_ids.size:
            if np.any(self.particle_ids < 0):
                raise ValueError("particle_ids contains negative indices")
            if np.unique(self.particle_ids).size != self.particle_ids.size:
                raise ValueError(
                    "particle_ids contains duplicates: the same solver particle "
                    "would be recorded as two independent nodes, inflating the "
                    "node count with perfectly correlated data.")
            if self.n_particles_simulated and \
                    self.particle_ids.max() >= self.n_particles_simulated:
                raise ValueError(
                    f"particle_ids max is {int(self.particle_ids.max())} but only "
                    f"{self.n_particles_simulated} particles were simulated")

        # -- v3.0 static fields ------------------------------------------
        # Each is either the measured value or an EMPTY array; never a zero
        # stand-in (module docstring, READING v1 FILES).
        self.rest_pos = _opt(rest_pos, np.float32, (-1, 3))
        self.rest_pos_simulated = _opt(rest_pos_simulated, np.float32, (-1, 3))
        self.material_field = _opt(material_field, np.float32, (-1, 3))
        self.control_dt = _opt_scalar(control_dt)
        self.preroll_s = _opt_scalar(preroll_s)
        self.grasp_capture_radius = _opt_scalar(grasp_capture_radius)
        self.jaw_release_rad = _opt_scalar(jaw_release_rad)
        self.contact_radius = _opt_scalar(contact_radius)
        self.robot_model = str(robot_model)
        self.robot_scale = _opt_scalar(robot_scale)
        self.jaw_names = np.asarray(list(jaw_names) if jaw_names is not None
                                    else [], dtype=np.str_)
        self.jaw_half_extents = _opt(jaw_half_extents, np.float32, (-1, 3))
        self.jaw_box_offset = _opt(jaw_box_offset, np.float32, (-1, 7))
        K = self.jaw_names.size
        for name in ("jaw_half_extents", "jaw_box_offset"):
            arr = getattr(self, name)
            if arr.size and arr.shape[0] != K:
                raise ValueError(f"{name} has {arr.shape[0]} rows but jaw_names "
                                 f"names {K} jaws; every per-jaw array is "
                                 "indexed by jaw_names order")
        if self.control_dt.size and not self.control_dt[0] > 0:
            raise ValueError(f"control_dt must be positive, got {self.control_dt[0]}")
        if self.preroll_s.size and self.preroll_s[0] < 0:
            raise ValueError(f"preroll_s must be >= 0, got {self.preroll_s[0]}")
        # rest_pos_simulated is the full cloud; rest_pos must be its subset
        # under particle_ids. Checked here because both arrive before append().
        # WHY: two copies of the same reference positions that disagree mean
        # the node features (x - X) are computed against the wrong particles.
        if self.rest_pos.size and self.rest_pos_simulated.size \
                and self.particle_ids.size:
            if self.particle_ids.max() >= self.rest_pos_simulated.shape[0]:
                raise ValueError("particle_ids index past rest_pos_simulated")
            if not np.array_equal(self.rest_pos,
                                  self.rest_pos_simulated[self.particle_ids]):
                raise ValueError(
                    "rest_pos != rest_pos_simulated[particle_ids]: the node "
                    "reference positions are not those of the recorded particles")

        # Per-step buffers.
        self._pos, self._vel = [], []
        self._ee_pose, self._ee_vel = [], []
        self._jaw, self._joint, self._action = [], [], []
        self._grasp_active, self._force = [], []
        self._grasp_ids_flat, self._grasp_offset = [], [0]
        self._F, self._contact_mode = [], []
        self._exposure, self._safety_strain = [], []
        self._time, self._jaw_box_pose, self._jaw_box_vel = [], [], []
        self._jaw_joint, self._node_contact, self._node_grasp = [], [], []
        self._grasp_full_flat, self._grasp_full_offset = [], [0]
        self._phase = []
        self._safety_strain_id, self._safety_strain_pos = [], []

        # Shapes are locked in on the first append and enforced afterwards.
        # Catching a shape change at step 200 is far cheaper than discovering
        # a corrupt dataset a week later.
        self._n_nodes = None
        self._n_joints = None
        self._n_action = None
        # Optional per-step fields are all-or-nothing across an episode. Half a
        # column of deformation gradients is worse than none: it produces an
        # array whose length silently disagrees with tissue_pos.
        self._optional_present = {}

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    # -- static fields known only after construction ---------------------
    def set_static(self, **fields) -> None:
        """Set v3 static fields that the caller learns after building the
        writer (tool geometry exists only once the robot is constructed; the
        pre-roll length only once it has run). Refused after the first
        append(): a static field that changes mid-episode is not static, and
        rows already accepted were validated against the old value.
        """
        if self._pos:
            raise RuntimeError("set_static() after append(): static fields "
                               "must be fixed before the first row")
        allowed = {"jaw_names", "jaw_half_extents", "jaw_box_offset",
                   "grasp_capture_radius", "jaw_release_rad", "contact_radius",
                   "preroll_s", "control_dt", "robot_model", "robot_scale"}
        unknown = set(fields) - allowed
        if unknown:
            raise TypeError(f"set_static() does not take {sorted(unknown)}")
        for k in ("grasp_capture_radius", "jaw_release_rad", "contact_radius",
                  "preroll_s", "control_dt", "robot_scale"):
            if k in fields:
                setattr(self, k, _opt_scalar(fields[k]))
        if "robot_model" in fields:
            self.robot_model = str(fields["robot_model"])
        if "jaw_names" in fields:
            self.jaw_names = np.asarray(list(fields["jaw_names"]), dtype=np.str_)
        if "jaw_half_extents" in fields:
            self.jaw_half_extents = _opt(fields["jaw_half_extents"], np.float32, (-1, 3))
        if "jaw_box_offset" in fields:
            self.jaw_box_offset = _opt(fields["jaw_box_offset"], np.float32, (-1, 7))
        K = self.jaw_names.size
        for name in ("jaw_half_extents", "jaw_box_offset"):
            arr = getattr(self, name)
            if arr.size and arr.shape[0] != K:
                raise ValueError(f"{name} has {arr.shape[0]} rows, jaw_names has {K}")

    # -- context manager plumbing so `with` works --------------------------
    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # A CRASHED EPISODE IS NEVER SAVED UNDER ITS REAL NAME. The first
        # version of this method wrote whatever rows it had to `path`, so a
        # run that died at step 40 of 200 produced a normal-looking .npz that
        # every check passed: shapes agree with each other, the timebase
        # closes, and nothing records that 160 steps are missing. Rows are
        # still worth inspecting, so they go to `<path>.partial` instead --
        # a name list_trajectories() does not match, so no validator or
        # training run can pick it up by accident.
        if exc_type is not None:
            if self._pos:
                self.abort(f"{exc_type.__name__}: {exc}")
        elif self._pos:
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
        tissue_F: Optional[np.ndarray] = None,
        contact_mode: Optional[int] = None,
        exposure: Optional[float] = None,
        safety_strain: Optional[float] = None,
        time: Optional[float] = None,
        jaw_box_pose: Optional[np.ndarray] = None,
        jaw_box_vel: Optional[np.ndarray] = None,
        jaw_joint: Optional[np.ndarray] = None,
        node_contact: Optional[np.ndarray] = None,
        node_grasp: Optional[np.ndarray] = None,
        grasp_full_ids: Optional[Sequence[int]] = None,
        phase: Optional[int] = None,
        safety_strain_id: Optional[int] = None,
        safety_strain_pos: Optional[np.ndarray] = None,
    ) -> None:
        """Record one timestep. Keyword-only on purpose: positional args here
        would be a silent-bug factory once there are ten of them.

        Every v2/v3 field is optional. A collector with no constitutive model --
        PyBullet mass-spring, which has no deformation gradient at all -- calls
        this exactly as it did under v1 and gets a valid file back.

        VALIDATE EVERYTHING, THEN STORE ANYTHING. The first version appended
        tissue_pos to its buffer and only then validated the other fields, so
        a rejected row left one buffer a row longer than the rest. A caller
        that caught the ValueError and carried on (or __exit__ saving the
        rows) then wrote arrays whose lengths disagreed by one -- the exact
        off-by-one check_shapes_agree exists to catch, manufactured by the
        writer itself. Every array below is built and checked into locals
        first; the buffers are touched only in the final block.
        """
        step = len(self._pos)
        pos = _as(tissue_pos, np.float32, (-1, 3))
        n_nodes = self._n_nodes if self._n_nodes is not None else pos.shape[0]
        if pos.shape[0] != n_nodes:
            raise ValueError(
                f"node count changed mid-episode: {n_nodes} -> {pos.shape[0]}. "
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

        widths = {}
        for name, arr, cache in (("action", act, "_n_action"),
                                 ("joint_pos", jnt, "_n_joints")):
            have = getattr(self, cache)
            if have is not None and arr.shape[0] != have:
                raise ValueError(f"{name} dim changed mid-episode: {have} -> {arr.shape[0]}")
            widths[cache] = arr.shape[0]

        ids = np.asarray(grasp_node_ids, np.int32).ravel() if grasp_node_ids is not None \
            else np.zeros(0, np.int32)
        # grasp_node_ids indexes the RECORDED N-particle subset (0..N-1),
        # never full-solver particle indices (0..n_particles_simulated) -- a
        # downstream visualizer uses them directly against tissue_pos, which
        # is already subsetted. A leaked full-solver index would silently
        # alias to the wrong recorded node whenever it happens to be < N, and
        # IndexError whenever it isn't -- catch it here, at the point the
        # caller that produced it is still on the stack.
        if ids.size and (np.any(ids < 0) or np.any(ids >= n_nodes)):
            raise ValueError(
                f"grasp_node_ids must satisfy 0 <= id < {n_nodes} (the "
                f"recorded node count N), got range [{int(ids.min())},"
                f"{int(ids.max())}] at step {step} -- looks "
                "like a full-solver particle index leaked through unmapped")

        # --- optional per-step fields: decide presence, validate, stage ----
        present = {}
        staged = {}

        def opt(name, value):
            ok = self._check_consistent(name, value is not None, step)
            present[name] = value is not None
            return ok

        if opt("tissue_F", tissue_F):
            F = _as(tissue_F, np.float32, (-1, 3, 3))
            if F.shape[0] != n_nodes:
                raise ValueError(
                    f"tissue_F has {F.shape[0]} particles, tissue_pos has {n_nodes}")
            staged["_F"] = F
        if opt("contact_mode", contact_mode):
            if int(contact_mode) not in CONTACT_MODE_NAMES:
                raise ValueError(
                    f"contact_mode {contact_mode} is not one of "
                    f"{sorted(CONTACT_MODE_NAMES)} ({CONTACT_MODE_NAMES})")
            staged["_contact_mode"] = np.int8(contact_mode)
        if opt("exposure", exposure):
            staged["_exposure"] = np.float32(exposure)
        if opt("safety_strain", safety_strain):
            staged["_safety_strain"] = np.float32(safety_strain)

        K = self.jaw_names.size
        if opt("time", time):
            t_now = float(time)
            # WHY: time is what turns row indices into seconds. Non-increasing
            # time means two rows claim the same instant, and every velocity
            # finite-differenced from them divides by zero or flips sign.
            if self._time and not t_now > self._time[-1]:
                raise ValueError(f"time must strictly increase: step {step} has "
                                 f"{t_now} after {self._time[-1]}")
            staged["_time"] = np.float64(t_now)
        for name, width in (("jaw_box_pose", 7), ("jaw_box_vel", 6)):
            value = {"jaw_box_pose": jaw_box_pose, "jaw_box_vel": jaw_box_vel}[name]
            if opt(name, value):
                arr = _as(value, np.float32, (-1, width))
                if arr.shape[0] != K or K == 0:
                    raise ValueError(f"{name} has {arr.shape[0]} jaws, jaw_names "
                                     f"names {K}; pass jaw_names to the writer")
                staged["_" + name] = arr
        if opt("jaw_joint", jaw_joint):
            arr = _as(jaw_joint, np.float32, (-1,))
            if arr.shape[0] != K or K == 0:
                raise ValueError(f"jaw_joint has {arr.shape[0]} jaws, jaw_names names {K}")
            staged["_jaw_joint"] = arr
        for name, value in (("node_contact", node_contact), ("node_grasp", node_grasp)):
            if opt(name, value):
                arr = _as(value, bool, (-1,))
                if arr.shape[0] != n_nodes:
                    raise ValueError(f"{name} has {arr.shape[0]} entries, "
                                     f"tissue_pos has {n_nodes} nodes")
                staged["_" + name] = arr
        if opt("grasp_full_ids", grasp_full_ids):
            full = np.asarray(grasp_full_ids, np.int32).ravel()
            if full.size and (full.min() < 0 or (self.n_particles_simulated and
                                                 full.max() >= self.n_particles_simulated)):
                raise ValueError("grasp_full_ids outside [0, n_particles_simulated)")
            staged["_grasp_full"] = full

        if opt("phase", phase):
            staged["_phase"] = np.int8(phase)

        # The peak's id and position travel together: one without the other
        # cannot answer "was the logged maximum on a recorded node, and where".
        if (safety_strain_id is None) != (safety_strain_pos is None):
            raise ValueError("safety_strain_id and safety_strain_pos must be "
                             "supplied together")
        if (safety_strain_id is not None) and safety_strain is None:
            raise ValueError("safety_strain_id without safety_strain: the id names "
                             "the particle carrying a value that was not logged")
        if opt("safety_strain_id", safety_strain_id):
            sid = int(safety_strain_id)
            if sid < 0 or (self.n_particles_simulated and sid >= self.n_particles_simulated):
                raise ValueError(f"safety_strain_id {sid} outside "
                                 f"[0, {self.n_particles_simulated})")
            staged["_safety_strain_id"] = np.int32(sid)
        if opt("safety_strain_pos", safety_strain_pos):
            staged["_safety_strain_pos"] = _as(safety_strain_pos, np.float32, (3,))

        # --- commit: nothing above touched a buffer ------------------------
        self._n_nodes = n_nodes
        for cache, width in widths.items():
            setattr(self, cache, width)
        for name, is_present in present.items():
            self._optional_present.setdefault(name, is_present)
        self._pos.append(pos)
        self._vel.append(vel)
        self._ee_pose.append(pose)
        self._ee_vel.append(evel)
        self._jaw.append(np.float32(jaw))
        self._joint.append(jnt)
        self._action.append(act)
        self._grasp_active.append(bool(grasp_active))
        self._force.append(frc)
        self._grasp_ids_flat.append(ids)
        self._grasp_offset.append(self._grasp_offset[-1] + ids.size)
        for buf, value in staged.items():
            if buf == "_grasp_full":
                self._grasp_full_flat.append(value)
                self._grasp_full_offset.append(self._grasp_full_offset[-1] + value.size)
            else:
                getattr(self, buf).append(value)

    def _check_consistent(self, name: str, present: bool, step: int) -> bool:
        """Is an optional field present-or-absent the same way as on row 0?

        WHY: supplying `tissue_F` for some steps and not others yields an array
        shorter than `tissue_pos`, and every downstream index into it is then
        off by an amount that depends on which steps were skipped. Catching it
        at the step it happens names the field; catching it at training time
        does not. Read-only: append() records the choice only after the whole
        row has validated.
        """
        have = self._optional_present.get(name)
        if have is None or have == present:
            return present
        was, now = ("supplied", "omitted") if have else ("omitted", "supplied")
        raise ValueError(
            f"{name} was {was} on earlier steps and is {now} at step "
            f"{step}. Optional fields are all-or-nothing "
            "within one episode.")

    def abort(self, reason: str = "") -> Optional[str]:
        """Save the rows so far to `<path>.partial` and never to `path`.

        Returns the partial path, or None if there was nothing to save. The
        file is a normal npz underneath (np.load reads it) with an extra
        `partial_reason` string, but its name deliberately does not end in
        .npz, so list_trajectories() -- and therefore the validator and every
        training script -- never sees it.
        """
        if not self._pos:
            return None
        out = self.path + ".partial"
        self._write(out, partial_reason=reason or "aborted")
        return out

    def close(self) -> str:
        """Write the episode to `path`, atomically.

        WHY ATOMIC: np.savez_compressed writes the zip member by member. A
        process killed mid-write (Ctrl-C, out of disk, the Mac sleeping) left a
        truncated file under the real name. It is written to `<path>.tmp` and
        renamed into place; a rename within one directory is all-or-nothing,
        so `path` is either the complete episode or absent.
        """
        if not self._pos:
            raise RuntimeError("nothing to write: append() was never called")
        self._check_static_against_nodes()
        tmp = self.path + ".tmp"
        self._write(tmp)
        os.replace(tmp, self.path)
        self._pos.clear()  # so __exit__ does not write a second time
        return self.path

    def _check_static_against_nodes(self) -> None:
        """Static arrays that arrive before append() and so can only be
        sized against the node count once rows exist."""
        # boundary_mask arrives before any append, so its length can only be
        # checked here. A mask sized to the wrong mesh silently clamps the wrong
        # particles, which looks like a physics bug rather than a bookkeeping one.
        if self.boundary_mask.size and self.boundary_mask.size != self._n_nodes:
            raise ValueError(
                f"boundary_mask has {self.boundary_mask.size} entries but the "
                f"episode has {self._n_nodes} nodes")

        # particle_ids arrives before any append, so like boundary_mask its
        # length can only be checked once the node count is known.
        if self.particle_ids.size and self.particle_ids.size != self._n_nodes:
            raise ValueError(
                f"particle_ids names {self.particle_ids.size} particles but the "
                f"episode recorded {self._n_nodes} nodes")
        if self.n_particles_simulated and self.n_particles_simulated < self._n_nodes:
            raise ValueError(
                f"n_particles_simulated={self.n_particles_simulated} is fewer than "
                f"the {self._n_nodes} nodes recorded; a subset cannot be larger "
                "than the set it came from")
        # WHY: per-node static arrays aligned to a different node set would be
        # silently broadcast against the wrong particles by any consumer.
        for name in ("rest_pos", "material_field"):
            arr = getattr(self, name)
            if arr.size and arr.shape[0] != self._n_nodes:
                raise ValueError(f"{name} has {arr.shape[0]} rows but the "
                                 f"episode has {self._n_nodes} nodes")
        if self.rest_pos_simulated.size and self.n_particles_simulated and \
                self.rest_pos_simulated.shape[0] != self.n_particles_simulated:
            raise ValueError(
                f"rest_pos_simulated has {self.rest_pos_simulated.shape[0]} rows, "
                f"n_particles_simulated is {self.n_particles_simulated}")

    def _write(self, target: str, partial_reason: Optional[str] = None) -> None:
        """Serialise the buffers to `target` through a file handle.

        Through a handle, not a filename, because np.savez_compressed appends
        ".npz" to any filename that lacks it -- which would turn both
        `<path>.tmp` and `<path>.partial` back into names the loader globs.
        """
        flat = np.concatenate(self._grasp_ids_flat) if self._grasp_ids_flat \
            else np.zeros(0, np.int32)
        full_flat = np.concatenate(self._grasp_full_flat) if self._grasp_full_flat \
            else np.zeros(0, np.int32)
        K = self.jaw_names.size

        def stacked(buf, dtype, tail):
            # Present-and-stacked, or the empty "not recorded" array.
            return np.stack(buf).astype(dtype) if buf else np.zeros((0,) + tail, dtype)

        # Encoded on disk, always float32 in memory (see the reader). Under
        # "delta16" what lands in the file is F - I, which is why f_encoding is
        # written alongside it -- the bytes alone do not say which it is.
        tissue_F = _encode_F(np.stack(self._F) if self._F else None,
                             self.f_encoding)
        extra = {} if partial_reason is None else {"partial_reason": partial_reason}

        with open(target, "wb") as fh:
            np.savez_compressed(
                fh,
                schema_version=SCHEMA_VERSION,
                f_encoding=self.f_encoding,
                particle_ids=self.particle_ids,
                n_particles_simulated=np.int32(self.n_particles_simulated),
                grid_dx=np.float32(self.grid_dx),
                simulator=self.simulator,
                task=self.task,
                dt=np.float32(self.dt),
                notes=self.notes,
                tissue_faces=self.tissue_faces,
                tissue_tets=self.tissue_tets,
                material_params=self.material_params,
                substep_dt=np.float32(self.substep_dt),
                n_substeps=np.int32(self.n_substeps),
                boundary_mask=self.boundary_mask,
                action_spec=self.action_spec,
                target_origin=self.target_origin,
                target_normal=self.target_normal,
                target_extent=self.target_extent,
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
                tissue_F=tissue_F,
                contact_mode=np.asarray(self._contact_mode, np.int8),
                exposure=np.asarray(self._exposure, np.float32),
                safety_strain=np.asarray(self._safety_strain, np.float32),
                # -- v3.0 --
                rest_pos=self.rest_pos,
                rest_pos_simulated=self.rest_pos_simulated,
                material_field=self.material_field,
                control_dt=self.control_dt,
                preroll_s=self.preroll_s,
                jaw_names=self.jaw_names,
                jaw_half_extents=self.jaw_half_extents,
                jaw_box_offset=self.jaw_box_offset,
                grasp_capture_radius=self.grasp_capture_radius,
                jaw_release_rad=self.jaw_release_rad,
                contact_radius=self.contact_radius,
                robot_model=self.robot_model,
                robot_scale=self.robot_scale,
                time=np.asarray(self._time, np.float64),
                jaw_box_pose=stacked(self._jaw_box_pose, np.float32, (K, 7)),
                jaw_box_vel=stacked(self._jaw_box_vel, np.float32, (K, 6)),
                jaw_joint=stacked(self._jaw_joint, np.float32, (K,)),
                node_contact=stacked(self._node_contact, bool, (self._n_nodes,)),
                node_grasp=stacked(self._node_grasp, bool, (self._n_nodes,)),
                phase=np.asarray(self._phase, np.int8),
                grasp_full_ids_flat=full_flat.astype(np.int32),
                grasp_full_ids_offset=(np.asarray(self._grasp_full_offset, np.int32)
                                       if self._optional_present.get("grasp_full_ids")
                                       else np.zeros(0, np.int32)),
                safety_strain_id=np.asarray(self._safety_strain_id, np.int32),
                safety_strain_pos=stacked(self._safety_strain_pos, np.float32, (3,)),
                **extra,
            )


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

    @property
    def n_particles(self) -> int:
        """Alias of `n_nodes`. MPM says "particles", FEM says "nodes"; they are
        the same axis of the same array. The stored field keeps its v1 name
        because four other files reference it -- consistency beats vocabulary."""
        return self.n_nodes

    @property
    def has_F(self) -> bool:
        """True when this episode carries deformation gradients.

        Callers should branch on this rather than on the simulator name: what
        matters is whether F was recorded, not which solver produced it.
        """
        return int(self._d["tissue_F"].size) > 0

    @property
    def is_subset(self) -> bool:
        """True when the file records fewer particles than the solver stepped.

        Anything computing a MAXIMUM over particles must branch on this: a max
        over a subset is biased low, so the stored per-particle arrays cannot
        reproduce the logged `safety_strain`. They are not supposed to -- see
        SUBSETS in the module docstring -- but code that assumes they can will
        report a safety violation that is really a sampling artefact.
        """
        n_sim = int(self._d["n_particles_simulated"])
        return n_sim > 0 and n_sim > self.n_nodes

    @property
    def subset_fraction(self) -> float:
        """Recorded particles / simulated particles. 1.0 when nothing was dropped."""
        n_sim = int(self._d["n_particles_simulated"])
        return 1.0 if n_sim <= 0 else self.n_nodes / n_sim

    def grasp_ids(self, t: int) -> np.ndarray:
        """Node indices grasped at step t (variable length)."""
        off = self._d["grasp_ids_offset"]
        return self._d["grasp_ids_flat"][off[t]:off[t + 1]]

    def grasp_full_ids(self, t: int) -> np.ndarray:
        """FULL-SOLVER particle indices held by the grasp at step t (v3.0).
        Empty for every step of a file that did not record them."""
        off = self._d["grasp_full_ids_offset"]
        if off.size == 0:
            return np.zeros(0, np.int32)
        return self._d["grasp_full_ids_flat"][off[t]:off[t + 1]]

    @property
    def has_jaws(self) -> bool:
        """True when per-jaw tool geometry was recorded (schema 3.0 robot episodes)."""
        return int(self._d["jaw_box_pose"].size) > 0

    def __repr__(self) -> str:
        sub = f" {self.n_nodes}/{int(self._d['n_particles_simulated'])}" \
            if self.is_subset else ""
        enc = self._d.get("f_encoding", "float32")
        return (f"<Trajectory {os.path.basename(self.path)} "
                f"sim={self._d['simulator']} task={self._d['task']} "
                f"v{self._d['schema_version']} "
                f"steps={len(self)} nodes={self.n_nodes}{sub} "
                f"dt={float(self._d['dt']):.5f}"
                f"{f' +F[{enc}]' if self.has_F else ''}>")


def load_trajectory(path: str) -> Trajectory:
    """Read one .npz episode back into memory, upgrading older schemas in place."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    # numpy stores strings as 0-d arrays; unwrap them so they behave like str.
    for k in ("schema_version", "simulator", "task", "notes", "action_spec",
              "f_encoding", "partial_reason", "robot_model"):
        if k in d:
            d[k] = str(d[k])

    version = d.get("schema_version", "0.0")
    if version not in KNOWN_SCHEMA_VERSIONS:
        raise ValueError(
            f"{path} is schema {version}; this reader knows "
            f"{KNOWN_SCHEMA_VERSIONS}. A file written by newer code cannot be "
            "interpreted by guessing -- update trajectory_io.py instead.")
    _upgrade_in_place(d)
    return Trajectory(d, path)


def _upgrade_in_place(d: dict) -> None:
    """Fill in fields a file predates, so every Trajectory looks like v2.

    WHY THIS IS NOT A WARNING PRINT: v1 files are legitimate data, not a
    mistake, and every consumer of this module would otherwise need its own
    `if "tissue_F" in ...` branch. Normalising once here means
    visualize_trajectory, validate_dataset and train_dynamics can read a v1 and
    a v2 file through identical code paths.

    Absent fields become EMPTY arrays rather than zeros -- see the module
    docstring for why a zero default would be a lie for most of these.
    """
    empty = {
        "material_params": np.float32,
        "boundary_mask": bool,
        "target_origin": np.float32,
        "target_normal": np.float32,
        "target_extent": np.float32,
        "tissue_F": np.float32,
        "contact_mode": np.int8,
        "exposure": np.float32,
        "safety_strain": np.float32,
        # v2.1. Empty particle_ids is the honest reading for every older file:
        # they recorded whatever they recorded and never said how it related to
        # a solver's particle array. Zeros would claim they all recorded
        # particle 0, which is worse than admitting the mapping is unknown.
        "particle_ids": np.int32,
        # v3.0. Every one of these is "not recorded" in an older file. In
        # particular an older file's frame 0 is NOT silently promoted to
        # rest_pos: the to-do list forbids substituting the first recorded
        # position for X unless it is the actual reference configuration, and
        # only a v3 file records whether it is (preroll_s).
        "rest_pos": np.float32,
        "rest_pos_simulated": np.float32,
        "material_field": np.float32,
        "control_dt": np.float64,
        "preroll_s": np.float64,
        "jaw_half_extents": np.float32,
        "jaw_box_offset": np.float32,
        "grasp_capture_radius": np.float64,
        "jaw_release_rad": np.float64,
        "contact_radius": np.float64,
        "time": np.float64,
        "jaw_box_pose": np.float32,
        "jaw_box_vel": np.float32,
        "jaw_joint": np.float32,
        "node_contact": bool,
        "node_grasp": bool,
        "grasp_full_ids_flat": np.int32,
        "grasp_full_ids_offset": np.int32,
        "phase": np.int8,
        "robot_scale": np.float64,
        # 20 Nov additions; every file before then has neither.
        "safety_strain_id": np.int32,
        "safety_strain_pos": np.float32,
    }
    for key, dtype in empty.items():
        if key not in d:
            d[key] = np.zeros(0, dtype)
    d.setdefault("robot_model", "")
    if "jaw_names" not in d:
        d["jaw_names"] = np.zeros(0, np.str_)

    # Scalars where 0 is unambiguous: a solver cannot take zero substeps, so
    # zero reads as "not recorded" without colliding with a real value.
    d.setdefault("substep_dt", np.float32(0.0))
    d.setdefault("n_substeps", np.int32(0))
    # v1's action was [target_xyz, grasp_flag] -- 4 wide, not 7, and not a pose.
    # Naming it "abs_pose_jaw" would misdescribe both its width and its meaning.
    d.setdefault("action_spec", "unknown")

    # Zero particles simulated, and zero grid spacing, are not physically
    # meaningful readings, so 0 carries "not recorded" here the same way it
    # does for n_substeps.
    d.setdefault("n_particles_simulated", np.int32(0))
    d.setdefault("grid_dx", np.float32(0.0))

    # v2.0 and earlier had no `f_encoding`; they chose between float32 and
    # float16 and left the dtype to say which. Inferring from the dtype
    # reproduces that exactly. No older file can be "delta16", because nothing
    # could write one -- so this inference is complete, not a guess.
    if "f_encoding" not in d:
        d["f_encoding"] = "float16" if d["tissue_F"].dtype == np.float16 \
            else "float32"

    # The encoding is a storage detail only. Decode on read so every consumer
    # sees float32 F: numpy.linalg promotes float16 to float64 anyway, and a
    # metric that silently changes precision with a writer flag is a debugging
    # trap. After this line `tissue_F` is F, never F - I.
    d["tissue_F"] = _decode_F(d["tissue_F"], d["f_encoding"])

    # `schema_version` is deliberately NOT rewritten to SCHEMA_VERSION. It
    # describes the file on disk, and a validator reporting "this v1 episode has
    # no F" is more useful than one reporting "this v2 episode has no F".


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

def _opt(arr, dtype, shape) -> np.ndarray:
    """`_as`, except None becomes the EMPTY "not recorded" array."""
    if arr is None:
        return np.zeros((0,) + tuple(0 if x == -1 else x for x in shape[1:]), dtype)
    return _as(arr, dtype, shape)


def _opt_scalar(value) -> np.ndarray:
    """A scalar stored as (1,) float64, or (0,) when not recorded.

    WHY NOT 0.0 FOR "NOT RECORDED": the scalars this is used for (preroll_s,
    the grasp radii) have 0 as a physically meaningful value, so the old
    n_substeps trick of letting 0 mean "absent" would be a lie here.
    """
    if value is None:
        return np.zeros(0, np.float64)
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"expected a finite scalar, got {value!r}")
    return np.array([v], np.float64)


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
    # round-trips correctly on this machine. SETUP_GUIDE.md tells the user to
    # run it, and verify_host.py / verify_container.py run it as a subprocess.
    import tempfile

    rng = np.random.default_rng(0)
    N, T = 40, 25
    with tempfile.TemporaryDirectory() as tmp:
        # -- 1. a v1-style caller: no v2 arguments anywhere ------------------
        # This is exactly how collect_retraction.py calls the writer. If this
        # path ever needs editing, the v2 migration broke the PyBullet collector.
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
        assert len(tr) == T and tr.n_nodes == N and tr.n_particles == N
        assert tr.grasp_ids(0).size == 0 and tr.grasp_ids(10).tolist() == [1, 2, 3]
        x, u, y = make_transition_pairs(tr, horizon=1)
        assert x.shape == (T - 1, N * 6 + 7) and u.shape == (T - 1, 4) and y.shape == (T - 1, N * 3)
        # Unsupplied v2 fields must be empty, not zero-filled: see the module
        # docstring. 0.0 is a real exposure value and must stay distinguishable.
        assert not tr.has_F and tr.tissue_F.size == 0
        assert tr.exposure.size == 0 and tr.safety_strain.size == 0
        assert tr.contact_mode.size == 0 and tr.material_params.size == 0
        assert tr.boundary_mask.size == 0 and tr.action_spec == "unknown"
        print(tr)
        print(f"pairs: x{x.shape} u{u.shape} y{y.shape}")

        # -- 2. a full v2 caller, with F stored as float16 -------------------
        q = os.path.join(tmp, "selftest_v2.npz")
        mask = np.zeros(N, bool)
        mask[:4] = True
        with TrajectoryWriter(
                q, "selftest_mpm", "tissue_retraction", 0.010,
                material_params=np.array([np.log(1500.0), np.log(2e5), 1050.0]),
                substep_dt=2.5e-4, n_substeps=40,
                boundary_mask=mask, action_spec="delta_pose_jaw",
                target_origin=[0.0, 0.0, 0.0], target_normal=[0.0, 0.0, 1.0],
                target_extent=[0.01, 0.01], store_F_as_float16=True) as w:
            for t in range(T):
                # F = a 25% uniaxial stretch, well clear of float16's ~1e-3
                # resolution near 1.0 so the round-trip below is meaningful.
                F = np.tile(np.diag([1.0 + 0.25 * t / T, 1.0, 1.0]), (N, 1, 1))
                w.append(
                    tissue_pos=rng.normal(size=(N, 3)),
                    ee_pose=np.array([0, 0, 0.1 * t, 0, 0, 0, 1]),
                    action=rng.normal(size=7),
                    tissue_F=F,
                    contact_mode=CONTACT_GRASP if t > 5 else CONTACT_NONE,
                    exposure=0.01 * t,
                    safety_strain=1.0 + 0.01 * t,
                )
        tv = load_trajectory(q)
        assert tv.has_F and tv.tissue_F.shape == (T, N, 3, 3)
        # float16 on disk, float32 in memory -- the writer flag must not leak
        # into the dtype every downstream metric sees.
        assert tv.tissue_F.dtype == np.float32
        assert np.allclose(tv.tissue_F[-1, 0], np.diag([1.0 + 0.25 * (T - 1) / T, 1, 1]),
                           atol=1e-3)
        assert tv.contact_mode.shape == (T,) and tv.exposure.shape == (T,)
        assert tv.boundary_mask.sum() == 4 and tv.action_spec == "delta_pose_jaw"
        assert CONTACT_MODE_NAMES[int(tv.contact_mode[-1])] == "grasp"
        assert abs(float(np.exp(tv.material_params[0])) - 1500.0) < 1.0
        print(tv)

        # -- 3. the mistakes the writer must refuse --------------------------
        # An optional field supplied on some steps but not others produces a
        # column shorter than tissue_pos, which nothing downstream detects.
        try:
            with TrajectoryWriter(os.path.join(tmp, "bad.npz"), "s", "t", 0.01) as w:
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7), exposure=0.5)
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7))
            raise AssertionError("writer accepted a half-populated optional field")
        except ValueError as e:
            assert "all-or-nothing" in str(e)
        try:
            with TrajectoryWriter(os.path.join(tmp, "bad2.npz"), "s", "t", 0.01,
                                  boundary_mask=np.zeros(N + 1, bool)) as w:
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7))
            raise AssertionError("writer accepted a mis-sized boundary_mask")
        except ValueError as e:
            assert "boundary_mask" in str(e)

        # -- 4. v2.1: a subset record with delta16 F -------------------------
        n_sim, n_rec = 24000, N
        SMALL = 1.0e-4                      # 0.01% stretch
        ids = np.linspace(0, n_sim - 1, n_rec).astype(np.int32)
        r = os.path.join(tmp, "selftest_subset.npz")
        with TrajectoryWriter(
                r, "taichi_mpm", "tissue_retraction", 0.0125,
                f_encoding="delta16",
                particle_ids=ids, n_particles_simulated=n_sim,
                material_params=np.array([np.log(3000.0), np.log(1.5e5), 1000.0]),
                action_spec="delta_pose_jaw") as w:
            for t in range(T):
                # A SMALL strain: 0.01% stretch. float16 spacing at 1.0 is
                # ~9.8e-4, so 1.0001 is below the first representable step
                # above 1.0 and rounds straight back to it: under "float16"
                # this deformation does not merely lose precision, it vanishes.
                F = np.tile(np.diag([1.0 + SMALL, 1.0, 1.0]), (N, 1, 1))
                w.append(tissue_pos=rng.normal(size=(N, 3)),
                         ee_pose=np.zeros(7), action=np.zeros(7), tissue_F=F)
        ts = load_trajectory(r)
        assert ts.is_subset and ts.n_nodes == n_rec
        assert int(ts.n_particles_simulated) == n_sim
        assert abs(ts.subset_fraction - n_rec / n_sim) < 1e-9
        assert ts.particle_ids.shape == (n_rec,) and ts.f_encoding == "delta16"
        assert ts.tissue_F.dtype == np.float32
        # The whole point, demonstrated against the alternative rather than
        # asserted: delta16 keeps the strain, plain float16 erases it.
        got = float(ts.tissue_F[0, 0, 0, 0])
        assert abs(got - (1.0 + SMALL)) < 1e-6, f"delta16 lost a small strain: {got}"
        naive = float(np.float32(np.float16(np.float32(1.0 + SMALL))))
        assert naive == 1.0, f"expected float16 to flatten 1+{SMALL} to 1.0, got {naive}"
        print(ts)
        print(f"delta16 kept a {SMALL:g} stretch as {got:.8f}; "
              f"plain float16 flattens it to {naive:.8f} (strain entirely lost)")

        # -- 5. subset bookkeeping the writer must refuse --------------------
        # Duplicate ids mean one particle recorded as two nodes: perfectly
        # correlated columns masquerading as independent data.
        for kwargs, want in (
            (dict(particle_ids=np.array([0, 1, 1], np.int32)), "duplicate"),
            (dict(particle_ids=np.array([0, 1, 99], np.int32),
                  n_particles_simulated=10), "only 10 particles"),
        ):
            try:
                TrajectoryWriter(os.path.join(tmp, "bad3.npz"), "s", "t", 0.01,
                                 **kwargs)
                raise AssertionError(f"writer accepted bad particle_ids ({want})")
            except ValueError as e:
                assert want in str(e), e
        # A subset smaller than the set it came from is the only legal ordering.
        try:
            with TrajectoryWriter(os.path.join(tmp, "bad4.npz"), "s", "t", 0.01,
                                  n_particles_simulated=3) as w:
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7))
            raise AssertionError("writer accepted more nodes than particles")
        except ValueError as e:
            assert "cannot be larger" in str(e)
        # The two spellings of the float16 choice must not disagree silently.
        try:
            TrajectoryWriter(os.path.join(tmp, "bad5.npz"), "s", "t", 0.01,
                             store_F_as_float16=True, f_encoding="delta16")
            raise AssertionError("writer accepted conflicting F encoding flags")
        except ValueError as e:
            assert "conflicts" in str(e)

        print("OK -- trajectory_io round-trip passed (v1-style, v2 float16 F, "
              "v2.1 subset + delta16 F, upgrade path)")
