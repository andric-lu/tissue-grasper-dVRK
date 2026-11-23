#!/usr/bin/env python3
"""
mpm_adapter.py -- drive the vendored Taichi MPM and write v2.1 episodes.

    conda activate tissue-host
    python host/mpm_adapter.py --out data_mpm/ --episodes 2

This is the seam between third_party/MPM/ (someone else's solver) and
src/trajectory_io.py (our data contract). It is small because DECISION_LOG.md
section 8 built the receiving end first, and because section 9.4 verified the field
mapping rather than assuming it:

    F_x -> tissue_pos     (N,3) float32
    F_v -> tissue_vel     (N,3) float32
    F   -> tissue_F       (N,3,3) float32

Six decisions are baked in here. Each one is a rule from CLAUDE.md meeting its
first real consumer, and each would be easy to get quietly wrong. Decisions 5
and 6 are here because the first version of this file got them wrong and the
data did not look wrong -- see DECISION_LOG.md section 9.5.

1. MU AND LAMBDA ARE WRITTEN DIRECTLY. `mpm3d.set_parameters(s_E, s_nu)` takes
   (E, nu) and internally computes lambda = E*nu/((1+nu)(1-2nu)) -- the singular
   conversion src/materials.py exists to keep out of the sampling path, in
   float32, singular at nu = 0.5 where tissue sits. `mu` and `la` are plain 0-d
   Taichi fields and the solver reads only those two, so this adapter assigns
   them and never calls set_parameters(). See section 9.4.

2. METRICS ARE COMPUTED OVER ALL 24,000 PARTICLES, NOT THE STORED SUBSET.
   `safety_strain` is a maximum over particles; a maximum over a subset is
   biased low. Measured: 3,000 of 24,000 particles underestimated peak stretch
   by 8% of the stretch above rest, erratically. Computing the metric before
   subsampling costs nothing -- the full state is already in memory -- and it
   is the only reason dropping particles is safe at all.

3. THE SUBSET IS FIXED FOR THE WHOLE EPISODE. Node identity has to be stable
   across time or consecutive frames describe different particles, which is
   meaningless to a dynamics model and silently wrong rather than loud.

4. THE SOLVER IS ALREADY IN SI UNITS. The MPM's domain is the unit cube and its
   gravity is 9.8, so one domain unit is one metre and one step is one second --
   confirmed in host/smoke_test_mpm.py, where free fall matched analytic -g*t to
   0.00%. No conversion happens here. If you ever rescale the domain you must
   rescale gravity with it, or the two silently disagree and everything falls at
   the wrong rate. `domain_scale` exists to make that coupling visible.

5. THE TIMEBASE IS `self.n_substeps`, NOT `mpm3d.steps`. Every recorded frame
   must advance the solver by exactly `frame_dt`, so
   n_substeps * substep_dt == frame_dt. The first version computed n_substeps
   from the P-wave bound, wrote it to disk, and then advanced the vendored
   module default of 25 instead: frames labelled 12.5 ms advanced 2.95 ms, and
   every number in the file agreed with every other one. Asserted here, in
   TrajectoryWriter, and in validate_dataset.py.

6. DENSITY REACHES THE SOLVER, VIA p_mass. `mpm3d.p_mass` is read inside the
   P2G kernel, so it is baked in at compile time exactly like `dt`, and until
   this file set it the solver ran every episode at the vendored rho = 1000
   while material_params recorded whatever was sampled. Both constants are
   covered by the per-process compile lock, because they are frozen by the same
   mechanism at the same moment.

ONE EPISODE PER PROCESS. Those two baked-in constants are why `--episodes N`
launches a child per episode rather than looping. See main().

WHAT THIS DOES NOT YET DO: there is no robot. PyBullet supplies rigid-body
collision through the SDF in `sdf.py`, and that is where the PSM plugs in later;
until then `ee_pose` and `action` are whatever the caller passes, and the tissue
is driven by gravity and its own elasticity. The file format does not care, which
is the point of having a format.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))   # so `import MPM.x` works
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "host"))           # so `import psm` works

import actions                                            # noqa: E402
import collection_config                                 # noqa: E402
import materials                                         # noqa: E402
import tool_geometry                                     # noqa: E402
import tissue_metrics                                    # noqa: E402
import trajectory_io                                     # noqa: E402
import psm                                                # noqa: E402 -- imports
# pybullet only, not taichi/MPM.mpm3d; safe at module load, before any
# MPMRecorder exists. See psm.py's own module docstring for why a PSM
# instance itself cannot be constructed this early.
import grasp_constraint                                   # noqa: E402 -- imports
# actions/trajectory_io/psm only (no taichi/MPM.mpm3d at its own module
# level either); same safe-at-load reasoning as psm.py above.

# Default number of particles to RECORD. The solver needs ~24,000 for stable
# physics; MeshGraphNets trains on 1.5k-5k nodes. These are different numbers
# and conflating them makes the dataset infeasible for the model long before it
# is inconvenient for the disk -- 24,000 particles puts 144,007 features into
# the MLP's input layer alone. See trajectory_io.py, SUBSETS.
DEFAULT_N_RECORD = 3000

# One domain unit in metres. 1.0 because the solver's gravity is 9.8 in domain
# units per second squared; any other value makes it not 9.8 m/s^2. Changing
# this WITHOUT changing mpm3d.gravity is a bug, which is why it is named.
DOMAIN_SCALE_M = 1.0


class MPMRecorder:
    """Runs the vendored MPM and records episodes in schema v2.1.

    One instance per episode. `capture()` reads solver state, computes the
    metrics over every particle, and appends the recorded subset.
    """

    def __init__(
        self,
        path: str,
        *,
        task: str = "tissue_settle",
        mu: float,
        lam: float,
        rho: float = 1000.0,
        n_record: int = DEFAULT_N_RECORD,
        seed: int = 0,
        target_origin: Optional[Sequence[float]] = None,
        target_normal: Optional[Sequence[float]] = None,
        target_extent: Optional[Sequence[float]] = None,
        f_encoding: str = "delta16",
        action_spec: str = "delta_pose_jaw",
        frame_dt: float = 0.0125,
        substep_safety: float = 0.3,
        n_substeps: Optional[int] = None,
        notes: str = "",
        particle_ids: Optional[np.ndarray] = None,
    ):
        import MPM.mpm3d as mpm3d          # imports run ti.init() -- see 9.4
        import taichi as ti

        self.m = mpm3d
        self.ti = ti
        self.mu, self.lam, self.rho = float(mu), float(lam), float(rho)

        # THE SUBSTEP IS CHOSEN FROM THE MATERIAL, NOT TAKEN FROM THE SOLVER.
        # mpm3d.py hardcodes dt = 5e-4 with the comment "sweet pot", which is
        # true for the one material it was tuned on. Across materials.py's
        # ranges it is not: lambda/mu reaches ~750, the pressure wave is ~16x
        # the bar wave, and at 5e-4 three of six sampled materials diverged --
        # one so badly that det(F) went non-finite and an SVD inside
        # max_principal_stretch refused to converge. That is the failure mode
        # check_substep_is_stable_for_stiffness was written to predict, and it
        # predicted it correctly.
        #
        # UNLESS THE CALLER NAMES A COUNT. `n_substeps` overrides the advisory,
        # which is what host/substep_study.py needs: a convergence study has to
        # VARY the thing the advisory computes, holding the material fixed. It
        # is not for collection -- passing it there discards the per-material
        # bound this whole path exists to apply. The advisory is still computed
        # either way, so `implied_safety` can report where the override sits
        # relative to it.
        advisory_dt = float(materials.suggested_substep_dt(
            self.mu, self.rho, mpm3d.dx * DOMAIN_SCALE_M,
            safety=substep_safety, lam=self.lam))
        self.advisory_substep_dt = advisory_dt

        if n_substeps is None:
            # Whole substeps per recorded frame, so the recorded dt stays a
            # round number the whole dataset shares regardless of material.
            # Episodes with different stiffness then differ in COST, not in
            # sampling rate -- a dataset whose dt varies per episode is one a
            # dynamics model cannot use.
            self.n_substeps = max(1, int(np.ceil(frame_dt / advisory_dt)))
        else:
            if int(n_substeps) < 1:
                raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")
            self.n_substeps = int(n_substeps)
        self.substep_dt = frame_dt / self.n_substeps
        self.frame_dt = float(frame_dt)
        # Where this substep sits relative to the P-wave bound, as the safety
        # factor that would have produced it. Reported rather than enforced:
        # the study's whole job is to run rows on both sides of 0.3.
        self.implied_safety = substep_safety * (self.substep_dt / advisory_dt)

        # THE TIMEBASE INVARIANT, asserted where it is established. Every
        # recorded frame must advance the solver by exactly frame_dt, so
        # n_substeps * substep_dt == frame_dt is not a coincidence to check
        # later but the definition of the two lines above. It is asserted here,
        # again in the writer, and again in the validator, because the first
        # version of this adapter satisfied none of them: it computed
        # n_substeps correctly and then advanced mpm3d.steps (= 25) instead,
        # so a frame labelled 12.5 ms actually advanced 2.95 ms. Nothing on
        # disk contradicted itself -- the file was internally consistent and
        # physically a lie.
        drift = abs(self.n_substeps * self.substep_dt - self.frame_dt)
        if drift > 1e-12:
            raise RuntimeError(
                f"timebase does not close: {self.n_substeps} x "
                f"{self.substep_dt:.6e} s != frame_dt {self.frame_dt:.6e} s "
                f"(off by {drift:.3e} s)")

        # Density: p_mass is what the P2G kernel actually weights particles by
        # (mpm3d.py:172,180,181), and it is a module-level Python float, so it
        # is baked in at COMPILE time exactly like dt. Until this line existed
        # the solver ran every episode at the vendored p_rho = 1000 while
        # material_params[2] recorded the sampled rho, which made the density
        # column of the dataset a decoration. Set both so anything reading
        # p_rho agrees with what p_mass implies.
        mpm3d.p_rho = self.rho
        mpm3d.p_mass = mpm3d.p_vol * self.rho

        # Taichi captures module-level Python constants when a kernel COMPILES,
        # so this must happen before the first substep() and cannot be changed
        # afterwards in the same process. Verified: setting it here and running
        # free fall reproduces -g*t at the NEW dt exactly. The consequence is
        # one (substep_dt, p_mass) pair per process -- see the guard below.
        #
        # BOTH are locked, not just dt. They are baked by the same mechanism at
        # the same moment, so a lock covering only dt would let a second episode
        # with the same stiffness but a different density run silently at the
        # first episode's mass -- the identical failure mode, minus the error
        # message.
        # GRAVITY IS THE THIRD BAKED CONSTANT. Boundary() reads it as a plain
        # Python global inside a @ti.func (`F_grid_v[I][2] -= dt * gravity`,
        # mpm3d.py:215), so it is frozen at compile time by the same mechanism
        # as dt and p_mass. The adapter never changes it -- but host/mpm_energy.py
        # does, because section 9.7's clean cell needs g = 0, and both import the
        # same module object. A lock covering only two constants would let a
        # collection run that happened to follow an energy-audit run in one
        # interpreter record a full dataset of tissue that never fell.
        self.gravity = float(mpm3d.gravity)
        if self.gravity != 9.8:
            raise RuntimeError(
                f"mpm3d.gravity is {self.gravity}, not 9.8 m/s^2. Something in "
                "this process changed it before collection started (see "
                "host/mpm_energy.py). Collect in a fresh interpreter.")

        locked = getattr(mpm3d, "_adapter_compile_lock", None)
        if locked is not None:
            l_dt, l_mass, l_g = locked
            if (abs(l_dt - self.substep_dt) > 1e-15
                    or abs(l_mass - mpm3d.p_mass) > 1e-30
                    or l_g != self.gravity):
                raise RuntimeError(
                    f"this process already compiled MPM kernels with substep "
                    f"dt={l_dt:.3e} s, p_mass={l_mass:.6e} kg and "
                    f"gravity={l_g} m/s^2, and Taichi bakes all three constants "
                    f"in at compile time; this episode needs "
                    f"dt={self.substep_dt:.3e} s, "
                    f"p_mass={mpm3d.p_mass:.6e} kg and gravity={self.gravity}. "
                    "Collect one episode per "
                    "process -- `--episodes N` does this by launching a child "
                    "per episode; calling MPMRecorder twice in one interpreter "
                    "does not.")
        mpm3d.dt = self.substep_dt
        # steps is NOT baked in -- it appears only in Python-level loops
        # (mpm3d.py:511,611), never inside a kernel -- but it is set here
        # anyway so that anything reading module state sees one coherent
        # timebase. `advance()` uses self.n_substeps, not this.
        mpm3d.steps = self.n_substeps
        mpm3d.timestep = self.n_substeps * self.substep_dt
        mpm3d._adapter_compile_lock = (self.substep_dt, mpm3d.p_mass,
                                       self.gravity)

        backend = str(ti.lang.impl.current_cfg().arch)
        if "metal" not in backend.lower():
            # Not fatal -- the physics is identical -- but 8x slower with no
            # error anywhere, which is exactly how a slow collection run gets
            # blamed on the solver. Say it out loud. See section 9.4.
            print(f"WARNING: Taichi is on {backend}, not metal. "
                  f"Collection will be ~8x slower.")
        self.backend = backend

        n_sim = int(self.m.n_particles)
        self.n_sim = n_sim

        if particle_ids is not None:
            # A FIXED NODE SET, chosen once from the reference cloud and shared
            # by every episode (TODO_ASSESSMENT section 4.4). Without it each
            # seed draws a different subset, so node i is a different particle
            # in every episode: the MLP cannot learn across episodes at all and
            # a graph model's per-node errors cannot be compared between them.
            ids = np.asarray(particle_ids, np.int64).ravel()
            if ids.size == 0 or ids.min() < 0 or ids.max() >= n_sim:
                raise ValueError(f"particle_ids must index [0, {n_sim})")
            if np.unique(ids).size != ids.size:
                raise ValueError("particle_ids contains duplicates")
            self.particle_ids = np.sort(ids).astype(np.int32)
            self.n_record = int(self.particle_ids.size)
        else:
            if n_record > n_sim:
                raise ValueError(f"n_record={n_record} exceeds the solver's {n_sim} "
                                 "particles; a subset cannot be larger than its set")
            self.n_record = int(n_record)
            # THE SUBSET, CHOSEN ONCE PER EPISODE. Sorted so particle_ids is
            # monotonic. Uniform without replacement: MPM particle index
            # carries no spatial structure (init_cube assigns random positions
            # per index), so a uniform draw is already spatially uniform.
            # NOTE: varies with `seed`, so node identity differs between
            # episodes -- pass `particle_ids` for a dataset.
            rng = np.random.default_rng(seed)
            self.particle_ids = np.sort(
                rng.choice(n_sim, size=self.n_record, replace=False)).astype(np.int32)

        self._init_solver()

        # THE REFERENCE CONFIGURATION, READ BEFORE THE FIRST SUBSTEP. init_cube()
        # sets F = I for every particle, so the positions it lays out ARE the
        # undeformed reference X of the to-do's node features -- by the
        # solver's own definition of F, not by assumption. Read now, because
        # one substep later the cloud has started to fall and F != I.
        # WHY ASSERT F = I: if a future init path starts from a pre-strained
        # state, "spawn positions" would stop being the reference, and x - X
        # would silently measure displacement from the wrong configuration.
        rest_full, _, F0 = self.state()
        dev = float(np.abs(F0 - np.eye(3, dtype=F0.dtype)).max())
        if dev > 1e-6:
            raise RuntimeError(f"F is not identity at initialisation (max |F-I| "
                               f"{dev:.3g}); spawn positions are not the reference "
                               "configuration")
        self.rest_pos_simulated = rest_full.astype(np.float32)
        self.preroll_s = 0.0

        # Default target: a small patch directly above the tissue, facing up.
        # Arbitrary but explicit -- exposure is meaningless without one, and a
        # silently-absent target would make the metric SKIP forever.
        if target_origin is None:
            target_origin = [0.25, 0.35, 0.02]
        if target_normal is None:
            target_normal = [0.0, 0.0, 1.0]
        if target_extent is None:
            target_extent = [0.05, 0.05]
        self.target = (np.asarray(target_origin, np.float64),
                       np.asarray(target_normal, np.float64),
                       np.asarray(target_extent, np.float64))

        # No particle is kinematically clamped by this solver. The `bound = 3`
        # band in Boundary() zeroes GRID velocity at the domain edge; it does
        # not pin particles, and the `anchor` block is disabled upstream
        # (`anchor = 0`). So all-False is the honest mask: recorded, and the
        # answer is none. An empty array would claim we never looked.
        boundary_mask = np.zeros(self.n_record, bool)

        self.w = trajectory_io.TrajectoryWriter(
            path,
            simulator="taichi_mpm",
            task=task,
            dt=self.frame_dt,                   # seconds per RECORDED step
            # SEED HONESTY. `seed` here drives numpy only: the material draw and
            # which particles get recorded. It does NOT vary the initial
            # particle cloud. mpm3d.py calls ti.init(arch=arch) with no
            # random_seed (line 22), so Taichi's RNG starts at 0 in every
            # process and init_cube()'s ti.random() lays out an identical cloud
            # every episode. Writing "seed=N" without this note would imply an
            # independent initial condition that does not exist, and a dataset
            # whose episodes all start from the same geometry is a narrower
            # dataset than its metadata suggests. Naming it here is the cheap
            # half of fixing it.
            notes=notes or (
                f"vendored MPM cb797f36, backend={backend}, "
                f"{self.n_record}/{n_sim} particles, seed={seed} "
                f"(material + subset only; taichi RNG fixed at 0, so the "
                f"initial particle cloud is identical across episodes), "
                f"rho={self.rho:.1f} kg/m^3 applied as p_mass={self.m.p_mass:.6e} kg, "
                f"domain_scale={DOMAIN_SCALE_M} m, "
                f"substep {self.substep_dt*1e6:.1f} us x {self.n_substeps} "
                + (f"(P-wave advisory, safety={substep_safety})"
                   if n_substeps is None else
                   f"(n_substeps FORCED to {self.n_substeps}; P-wave advisory "
                   f"was {self.advisory_substep_dt*1e6:.1f} us at "
                   f"safety={substep_safety}, so this is safety="
                   f"{self.implied_safety:.4f})")),
            # [log(mu), log(lambda), rho] -- the layout materials.unpack_material
            # expects. Logs because tissue stiffness spans orders of magnitude
            # and a network fed raw Pascals burns capacity on the exponent.
            material_params=np.array(
                [np.log(self.mu), np.log(self.lam), self.rho], np.float32),
            # The adapter's own values, not a read-back of module state. Reading
            # them off mpm3d is how n_substeps came to be recorded as the
            # vendored 25 while the episode was integrated at a different rate.
            substep_dt=self.substep_dt,
            n_substeps=self.n_substeps,
            # Measured off the solver, not assumed. dx = 1/n_grid = 15.6 mm
            # here; a validator guessing 1 mm rejects a stable episode by a
            # factor of 15.6, which is exactly what happened before this was
            # recorded. See section 3.4 and check_substep_is_stable_for_stiffness.
            grid_dx=float(self.m.dx) * DOMAIN_SCALE_M,
            boundary_mask=boundary_mask,
            action_spec=action_spec,
            target_origin=target_origin,
            target_normal=target_normal,
            target_extent=target_extent,
            f_encoding=f_encoding,
            particle_ids=self.particle_ids,
            n_particles_simulated=n_sim,
            # -- schema 3.0 --
            rest_pos=self.rest_pos_simulated[self.particle_ids],
            rest_pos_simulated=self.rest_pos_simulated,
            # Homogeneous tissue: the episode material broadcast to every node
            # (to-do Item 5). Natural logs of Pascals, density in kg/m^3.
            material_field=np.tile(np.array(
                [np.log(self.mu), np.log(self.lam), self.rho], np.float32),
                (self.n_record, 1)),
            # One action per recorded frame: the control interval IS the
            # recording interval in this collector.
            control_dt=self.frame_dt,
            preroll_s=0.0,
        )
        self.n_steps = 0
        self._closed = False

    # -- solver setup ------------------------------------------------------
    def _init_solver(self):
        m = self.m

        # RULE 1: write the Lame parameters straight into the fields. E and nu
        # are never formed, so the (1 - 2nu) singularity never appears.
        m.mu[None] = self.mu
        m.la[None] = self.lam
        # E and nu are only ever read by set_parameters(), which we do not call.
        # Filled in anyway so anything printing them reports this material
        # rather than the stale default from a previous run.
        E, nu = materials.E_nu_from_lame(self.mu, self.lam)
        m.E[None] = float(E)
        m.nu[None] = float(nu)

        m.set_base_position([0.0, 0.0, 0.0])

        # substep() reads these two module globals, which step() would normally
        # rebind from sdf.py (mpm3d.py:492-494). With no PyBullet scene there is
        # no collider, so bind them by hand and fill the SDF with a large
        # positive distance: the collision branch in Boundary() is then never
        # taken and no neighbour of SDF is ever indexed.
        m.SDF = m.sdf.tmp_sdf
        m.collision_mask = m.sdf.co_mask
        m.SDF.fill(1.0e9)
        m.collision_mask.fill(-1)

        m.init_cube()
        m.init_deformation_gradient()
        m.F_v.fill([0.0, 0.0, 0.0])
        m.F_C.fill([[0.0] * 3] * 3)
        m.F_grid_m.fill(0.0)
        m.F_grid_v.fill([0.0, 0.0, 0.0])
        m.init_collision_field()
        self.ti.sync()

    # -- driving -----------------------------------------------------------
    def preroll(self, seconds: float, **advance_kwargs) -> int:
        """Simulate `seconds` BEFORE row 0, without recording (settle pre-roll).

        WHY: the tissue spawns in the air. In the 80 Hz grasp script the
        approach overlapped the gravity drop, so the tool met tissue that was
        still falling -- not the settled tissue the to-do's grasp sequence
        assumes (TODO_ASSESSMENT 4.3). Rounded to whole frames; the duration
        actually simulated is what gets recorded as `preroll_s`, so row 0's
        state is honestly labelled as NOT the reference configuration.
        """
        if self.n_steps:
            raise RuntimeError("preroll() after capture(): the pre-roll must "
                               "come before row 0")
        k = int(round(float(seconds) / self.frame_dt))
        if k:
            self.advance(k, **advance_kwargs)
        self.preroll_s += k * self.frame_dt
        self.w.set_static(preroll_s=self.preroll_s)
        return k

    def advance(self, n_frames: int = 1, scale: float = 1.0,
                threshold: float = 0.05,
                grasp: Optional["grasp_constraint.GraspConstraint"] = None,
                pose_start: Optional[np.ndarray] = None,
                pose_end: Optional[np.ndarray] = None) -> None:
        """Run `n_frames` recorded steps' worth of substeps.

        self.n_substeps, NOT self.m.steps. The module global is the vendored
        default (25) and has nothing to do with this episode's material; using
        it here is what made recorded frames advance the wrong amount of time
        while every number written to disk agreed with every other one.

        If self.robot is set, collision geometry is refreshed ONCE per
        recorded frame here -- not once per substep, matching the vendored
        solver's own step()'s cadence -- against whatever pose the caller has
        already driven the PSM to (see record_grasp_episode(): the PSM is set
        to the frame's END target BEFORE advance() is called, so this reads
        that end-of-frame transform and holds it constant across the frame's
        substeps).

        If `grasp` is given, `grasp.apply(pose_start, pose_end, frame_dt,
        alpha)` runs once PER SUBSTEP (not once per frame like the collision-
        geometry hook above) -- the persistent grasp constraint is meant to
        be exact, not just approximately sufficient, so it needs sub-frame
        interpolation. `pose_start`/`pose_end` are required together with
        `grasp` and passed straight through -- explicit, every call, rather
        than an internally-cached "previous pose" -- specifically to avoid a
        third instance of the "looks primed but isn't" bug class (see
        host/psm.py's own _prev_box_pose priming fix)."""
        if grasp is not None and (pose_start is None or pose_end is None):
            raise ValueError(
                "advance() needs pose_start/pose_end when grasp is given -- "
                "fail loudly rather than silently skip the constraint")
        for _ in range(n_frames):
            robot = getattr(self, "robot", None)
            if robot is not None:
                robot.update_collision_geometry(self.m, self.m.sdf)
                robot.update_collision_velocity(self.m, self.frame_dt)
            # A grid-BC grasp (host/grasp_grid_bc.py, DECISION_LOG 14.22) owns
            # the whole substep: it runs the vendored P2G/Boundary/G2P itself
            # with its boundary condition inside. The teleport grasp runs after
            # the vendored substep, as before.
            own = getattr(grasp, "substep", None)
            for i in range(self.n_substeps):
                if own is not None:
                    own(scale, threshold, pose_start, pose_end, self.frame_dt,
                        i / self.n_substeps)
                    continue
                self.m.substep(scale, threshold)
                if grasp is not None:
                    grasp.apply(pose_start, pose_end, self.frame_dt,
                               (i + 1) / self.n_substeps)
        self.ti.sync()

    def state(self):
        """Full solver state as numpy: (pos, vel, F) over ALL particles."""
        return (self.m.F_x.to_numpy(), self.m.F_v.to_numpy(), self.m.F.to_numpy())

    def capture(self, *, ee_pose: np.ndarray, action: np.ndarray,
                ee_vel: Optional[np.ndarray] = None, jaw: float = 0.0,
                joint_pos: Optional[np.ndarray] = None,
                grasp_active: bool = False,
                grasp_node_ids: Optional[Sequence[int]] = None,
                contact_mode: Optional[int] = None,
                contact_force: Optional[np.ndarray] = None,
                **v3_fields) -> dict:
        """Record one step. Returns the metrics, computed over every particle.

        `v3_fields` passes schema-3.0 per-step fields (jaw_box_pose, node_grasp,
        phase, ...) straight to TrajectoryWriter.append(). `time` is supplied
        here, from the row count, so it cannot disagree with dt.
        """
        pos, vel, F = self.state()

        # RULE 2: metrics over ALL particles, before any subsampling. A maximum
        # over a subset is biased low, and this is the safety number.
        t_o, t_n, t_e = self.target
        exposure = float(tissue_metrics.compute_exposure(
            pos.astype(np.float64), t_o, t_n, t_e))
        F64 = F.astype(np.float64)
        strain = tissue_metrics.compute_safety_strain(F64)
        safety = float(strain["max"])
        # WHICH particle carried the peak, and where (DECISION_LOG 14.20).
        # Without it, a subset file cannot say whether its logged maximum was
        # recordable at all, and the validator's subset corroboration had to
        # fall back on a margin tuned on settling. Taken from the same stretch
        # array as the max (a second SVD pass would cost ~39 ms per row).
        peak = int(strain["argmax"])

        idx = self.particle_ids
        self.w.append(
            tissue_pos=pos[idx],
            tissue_vel=vel[idx],
            tissue_F=F[idx],
            ee_pose=ee_pose,
            ee_vel=ee_vel,
            action=action,
            jaw=jaw,
            joint_pos=joint_pos,
            grasp_active=grasp_active,
            grasp_node_ids=grasp_node_ids,
            contact_mode=contact_mode,
            # RULE: this solver has no force channel (sticky-velocity BC
            # only) -- contact_force is always the writer's zero default here,
            # a legitimate reading ("not measured"), not a value we computed.
            contact_force=contact_force,
            exposure=exposure,
            safety_strain=safety,
            safety_strain_id=peak,
            safety_strain_pos=pos[peak],
            time=self.n_steps * self.frame_dt,
            **v3_fields,
        )
        self.n_steps += 1
        return {"exposure": exposure, "safety_strain": safety,
                "n_above_threshold": int(strain["n_above_threshold"]),
                "soft": float(strain["soft"])}

    def close(self) -> str:
        """Write the episode. Idempotent: calling it inside a `with` block and
        then again from __exit__ must not raise, which is exactly what a caller
        that wants the path back before the block ends will do."""
        if not self._closed:
            self.w.close()
            self._closed = True
        return self.w.path

    def __enter__(self) -> "MPMRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # A CRASHED EPISODE IS SAVED AS `<path>.partial`, NEVER AS `<path>`.
        # This used to call close() on any exit, so an episode that raised at
        # row 40 of 200 was written as a normal-looking file with 40 rows --
        # and the validator's comparability check compares node count, action
        # width and dt, none of which a truncation changes. The rows are kept
        # for inspection under a name list_trajectories() never matches.
        if not self._closed:
            if exc_type is not None:
                partial = self.w.abort(f"{exc_type.__name__}: {exc}")
                if partial:
                    print(f"episode raised after {self.n_steps} row(s); kept as "
                          f"{partial} (not a dataset file)")
                self._closed = True
            elif self.n_steps:
                self.close()
        return False  # never swallow the exception


# --------------------------------------------------------------------------

def sample_episode_material(seed: int):
    """The (mu, lambda, rho) an episode seed produces. Shared so a caller that
    needs to hold the material FIXED while varying something else -- the
    convergence study -- gets exactly what collection would have used, rather
    than a lookalike drawn a different way."""
    rng = np.random.default_rng(seed)
    # Sample mu and lambda DIRECTLY, log-uniformly. Never (E, nu): lambda is
    # singular at nu = 0.5 and tissue sits at nu ~ 0.49. See materials.py.
    mat = materials.sample_material(rng)
    return float(np.exp(mat[0])), float(np.exp(mat[1])), float(mat[2])


def record_episode(path: str, *, n_steps: int = 100, seed: int = 0,
                   n_record: int = DEFAULT_N_RECORD, quiet: bool = False,
                   material=None, n_substeps: Optional[int] = None,
                   frame_dt: float = 0.0125, f_encoding: str = "delta16",
                   particle_ids: Optional[np.ndarray] = None) -> str:
    """One passive episode: tissue released under gravity, no tool.

    Passive because there is no robot yet (see the module docstring). It still
    exercises every part of the path -- large deformation, contact with the
    floor, metrics, subset bookkeeping, the writer -- which is what makes it
    useful before the PSM lands.

    `material` and `n_substeps` exist for host/substep_study.py, which must vary
    the substep while holding everything else identical. Collection passes
    neither: the material comes from the seed and the substep from the P-wave
    advisory, which is the entire point of both.
    """
    mu, lam, rho = sample_episode_material(seed) if material is None \
        else (float(material[0]), float(material[1]), float(material[2]))

    t0 = time.time()
    with MPMRecorder(path, task="tissue_settle", mu=mu, lam=lam, rho=rho,
                     n_record=n_record, seed=seed, n_substeps=n_substeps,
                     frame_dt=frame_dt, f_encoding=f_encoding,
                     particle_ids=particle_ids) as rec:
        # A zero delta-pose action is a real action ("hold still"), not a
        # placeholder, so action_spec="delta_pose_jaw" is honest here.
        ee_pose = np.array([0.25, 0.35, 0.20, 0, 0, 0, 1], np.float32)
        action = np.zeros(7, np.float32)
        # No pre-roll: row 0 is the spawn state, F = I, the reference
        # configuration itself (preroll_s = 0 says so in the file).
        for t in range(n_steps):
            rec.capture(ee_pose=ee_pose, action=action,
                        phase=collection_config.PHASE_ID["settle"])
            rec.advance(1)
        out = rec.close()
        n_rec, n_sim = rec.n_record, rec.n_sim
    wall = time.time() - t0

    if not quiet:
        E, nu = materials.E_nu_from_lame(mu, lam)
        size = os.path.getsize(out) / 1e6
        print(f"{os.path.basename(out)}: {n_steps} steps, {n_rec}/{n_sim} particles, "
              f"{size:.1f} MB, {wall:.1f}s")
        print(f"    material mu={mu:.0f} Pa lambda={lam:.0f} Pa "
              f"(E={float(E):.0f} Pa, nu={float(nu):.4f}), rho={rho:.0f}")
    return out


# -- robot-driven episodes ---------------------------------------------------

# Jaw half-extents measured via host/psm.py's proxy construction are ~6-9mm,
# well under one grid cell (dx=15.6mm). The vendored solver's default
# threshold=0.05 (5cm) would extend the contact "halo" roughly 2 grid cells
# beyond the jaw's true surface. This is the deliberately tighter, documented
# choice instead -- roughly (largest jaw half-extent + one grid cell) rounded
# down, since a tighter halo is the safe direction to be wrong in for a
# sticky-contact collider (see host/smoke_test_psm.py's premature-perturbation
# check). CONTACT_TOUCH_RADIUS_M is diagnostic-only for contact_mode
# labelling below; it has no effect on the physics.
GRASP_CONTACT_THRESHOLD_M = 0.02
CONTACT_STICK_RADIUS_M = GRASP_CONTACT_THRESHOLD_M
CONTACT_TOUCH_RADIUS_M = 2 * GRASP_CONTACT_THRESHOLD_M

JAW_OPEN_RAD = 1.0
JAW_CLOSED_RAD = 0.0


def legacy_grasp_config(n_approach: int = 20, n_close: int = 10,
                        n_retract: int = 20, n_release: int = 0,
                        frame_dt: float = 0.0125) -> dict:
    """The section 10 script as a config: FRAME counts at 80 Hz, no pre-roll.

    Kept so host/smoke_test_grasp.py and host/smoke_test_psm.py, which call
    record_grasp_episode(n_approach=..., ...), drive exactly the episode they
    were verified against. collection_config.DEFAULT_GRASP_CONFIG has the same
    geometry; this only pins the frame counts and the 80 Hz interval.
    tests/test_collection_config.py checks the two produce identical waypoints.
    """
    cfg = collection_config.load_config(None)
    cfg["frame_dt_s"] = frame_dt
    cfg["preroll_s"] = 0.0
    cfg["approach"]["duration_s"] = n_approach * frame_dt
    cfg["close"]["duration_s"] = n_close * frame_dt
    cfg["hold"]["duration_s"] = 0.0
    cfg["retract"]["duration_s"] = n_retract * frame_dt
    cfg["release"]["duration_s"] = n_release * frame_dt
    cfg["post_release"]["duration_s"] = 0.0
    collection_config.validate_config(cfg)
    return cfg


def _contact_mode(rec: "MPMRecorder", robot: "psm.PSM",
                   grasp: Optional["grasp_constraint.GraspConstraint"] = None) -> int:
    """CONTACT_GRASP while `grasp` is active; otherwise NONE/TOUCH/STICK from
    proximity alone -- never CONTACT_GRASP by proximity. Boundary()'s
    collision branch (third_party/MPM/mpm3d.py:241-249) is an unconditional
    zero-slip velocity constraint with no persistence -- the friction/slip
    code is commented out ("sticky trick") and a particle leaving the SDF
    radius is released immediately, no bookkeeping holds it. That alone is
    mechanically trajectory_io.CONTACT_STICK, never CONTACT_GRASP ("jaws
    closed, tissue kinematically attached") -- real persistent attachment
    only exists when `grasp` (host/grasp_constraint.py) says so.

    Distance is to each jaw box's SURFACE (accounting for half-extents), not
    its center -- jaw half-extents (2.5-5.75mm) are not negligible next to
    the ~20mm contact radius.

    RULE: computed over ALL simulated particles, never the recorded subset.
    A minimum distance taken over a subset can only be >= the true minimum
    (removing particles removes candidates that could have been closer), so
    subsetting first biases contact detection toward UNDER-reporting -- the
    same class of bug Rule 2 in MPMRecorder.__init__'s docstring already
    guards against for safety_strain/exposure. This function used to subset
    to rec.particle_ids before computing distance; fixed."""
    if grasp is not None and grasp.is_active():
        return trajectory_io.CONTACT_GRASP
    pos_full, _, _ = rec.state()
    pos = pos_full.astype(np.float64)   # ALL simulated particles, not the recorded subset
    poses = robot.jaw_box_poses()
    d_min = np.inf
    for link_name in robot.jaw_links:
        box_pos, box_quat = poses[link_name]
        half_extents = robot.jaw_half_extents(link_name)
        local = (pos - box_pos) @ psm.rotmat_from_quat(box_quat)
        d = np.linalg.norm(np.maximum(np.abs(local) - half_extents, 0.0), axis=-1)
        d_min = min(d_min, float(d.min()))
    if d_min < CONTACT_STICK_RADIUS_M:
        return trajectory_io.CONTACT_STICK
    if d_min < CONTACT_TOUCH_RADIUS_M:
        return trajectory_io.CONTACT_TOUCH
    return trajectory_io.CONTACT_NONE


def make_robot(cfg: dict, rec: "MPMRecorder"):
    """The arm the config names (DECISION_LOG 14.15).

    "si"       host/psm.py's Si PSM, life-size, our own IK (the section 10 arm)
    "classic"  SurRoL's Psm1 (dVRK Classic) via host/psm_classic.py, at the
               config's `scaling` with its RCM at `rcm_position_m`. Imported
               here, not at module level, because it needs SurRoL, which only
               the tissue-sim environment has.
    """
    arm = cfg.get("arm") or {"model": "si"}
    if arm["model"] == "classic":
        import psm_classic
        return psm_classic.ClassicPSM(rec.m, rec.m.sdf,
                                      rcm_position=tuple(arm["rcm_position_m"]),
                                      scaling=float(arm.get("scaling", 5.0)))
    if float(arm.get("scaling", 1.0)) != 1.0:
        raise ValueError("the Si arm (host/psm.py) has no scaling option")
    return psm.PSM(rec.m, rec.m.sdf, base_position=psm.DEFAULT_BASE_POSITION,
                   base_orientation=psm.DEFAULT_BASE_ORIENTATION)


def record_grasp_episode(path: str, *, n_steps: Optional[int] = None,
                          seed: int = 0, n_record: int = DEFAULT_N_RECORD,
                          quiet: bool = False, n_approach: Optional[int] = None,
                          n_close: Optional[int] = None,
                          n_retract: Optional[int] = None, n_release: int = 0,
                          config: Optional[dict] = None,
                          f_encoding: str = "delta16",
                          particle_ids: Optional[np.ndarray] = None) -> str:
    """A scripted approach -> close -> [hold] -> retract -> [release] episode,
    driven by the dVRK Si PSM (host/psm.py) with a real persistent grasp
    (host/grasp_constraint.py).

    TWO WAYS TO CALL IT.
      config=<dict>    the schema-3.0 path. Every motion parameter, the
                       recording interval and the settle pre-roll come from
                       a collection_config dict in physical units.
      n_approach=...   the section 10 path (frame counts at 80 Hz, no
                       pre-roll), kept for the smoke tests -- see
                       legacy_grasp_config(). Used when `config` is None.

    STATE/ACTION TIMING CONTRACT. Row t's ee_pose/jaw are the
    commanded/analytic scripted waypoint values, NOT the IK-achieved pose --
    this is what makes "applying row t's action reproduces row t+1's
    recorded state" exact rather than approximate (checked in
    host/smoke_test_psm.py and by validate_dataset's action-indexing check).
    IK-achieved accuracy is validated separately, inside psm.PSM.set_ee_pose(),
    which raises if achieved pose drifts from commanded beyond tolerance.
    joint_pos and the v3 jaw_box_pose are ACHIEVED quantities (FK of the IK
    solution). The commanded pose (poses[t]) is what
    GraspConstraint.maybe_freeze() uses to capture r_p, for the same reason.

    Collision geometry uses the END-of-frame transform: the PSM is driven to
    the frame's target pose BEFORE advance() is called, so advance()'s
    once-per-frame update_collision_geometry() reads that just-set new pose
    and holds it constant across the frame's substeps; collision velocity is
    the backward difference bracketing that same frame. The grasp
    constraint's own enforcement runs once per SUBSTEP with the frame's
    start/end poses interpolated -- see GraspConstraint.apply().

    RECORDS T = transitions + 1 ROWS. The final waypoint's resulting state is
    captured as its own row with a zero-padded action (the to-do's T+1
    states / T actions convention, plus one pad row; docs/FIELD_DICTIONARY.md).

    SCHEMA 3.0 PER-ROW FIELDS (see tool_geometry.py for the rules):
      jaw_box_pose   achieved world pose of each jaw collision box
      jaw_box_vel    backward difference of jaw_box_pose (row 0 zero)
      jaw_joint      [jaw, -jaw]: jaw_joint_1 and its mimic jaw_joint_2
      node_contact   node within contact_radius of a jaw box
      node_grasp     node-level two-jaw latch, evaluated on the RECORDED nodes
      grasp_full_ids the solver's true full-resolution attached particles
      phase          collection_config.PHASES index
    """
    if config is None:
        cfg = legacy_grasp_config(
            20 if n_approach is None else n_approach,
            10 if n_close is None else n_close,
            20 if n_retract is None else n_retract, n_release)
    else:
        if any(v is not None for v in (n_approach, n_close, n_retract)) or n_release:
            raise ValueError("pass either config= or the legacy n_* frame counts, not both")
        cfg = config
    wp = collection_config.build_waypoints(cfg)
    poses, jaws, phases = wp["poses"], wp["jaw"], wp["phase"]
    n_trans = len(poses) - 1
    if n_steps is not None and n_steps != n_trans:
        raise ValueError(f"n_steps={n_steps} but the scripted trajectory has "
                         f"{n_trans} transitions")
    frame_dt = float(cfg["frame_dt_s"])

    mu, lam, rho = sample_episode_material(seed)
    t0 = time.time()
    with MPMRecorder(path, task="tissue_grasp", mu=mu, lam=lam, rho=rho,
                     n_record=n_record, seed=seed, frame_dt=frame_dt,
                     f_encoding=f_encoding, particle_ids=particle_ids) as rec:
        robot = make_robot(cfg, rec)
        rec.robot = robot
        grasp_model = cfg.get("grasp_model", "teleport")
        if grasp_model == "grid_bc":
            import grasp_grid_bc
            grasp = grasp_grid_bc.GridGraspConstraint(rec.m, rec.ti)
        elif grasp_model == "teleport":
            grasp = grasp_constraint.GraspConstraint(rec.m, rec.ti)
        else:
            raise ValueError(f"grasp_model must be 'teleport' or 'grid_bc', got {grasp_model!r}")
        half = np.stack([robot.jaw_half_extents(n) for n in robot.jaw_links]
                        ).astype(np.float32).astype(np.float64)   # as stored
        rec.w.set_static(
            jaw_names=list(robot.jaw_links),
            robot_model=robot.model_name,
            robot_scale=robot.scaling,
            jaw_half_extents=half,
            jaw_box_offset=np.stack([robot.jaw_box_offset(n) for n in robot.jaw_links]),
            grasp_capture_radius=tool_geometry.GRASP_CAPTURE_RADIUS_M,
            jaw_release_rad=tool_geometry.JAW_RELEASE_TRIGGER_RAD,
            contact_radius=tool_geometry.CONTACT_RADIUS_M)
        try:
            robot.set_ee_pose(poses[0], jaws[0])
            robot.update_collision_geometry(rec.m, rec.m.sdf)
            # The real priming call -- set_ee_pose() alone does NOT touch
            # PSM._prev_box_pose (only update_collision_velocity() does),
            # so without this the loop's first REAL commanded transition
            # (poses[0]->poses[1]) was silently taking the "first frame,
            # zero velocity, no sweep-limit check" branch instead of this
            # call. This is where "before the episode starts" belongs.
            robot.update_collision_velocity(rec.m, rec.frame_dt)

            # SETTLE PRE-ROLL: the arm holds its start pose (well above the
            # tissue) while the tissue falls and settles. advance() refreshes
            # collision geometry each frame against the unchanged pose, so the
            # jaw velocity stays exactly zero throughout.
            rec.preroll(float(cfg["preroll_s"]),
                        threshold=GRASP_CONTACT_THRESHOLD_M)

            state = {"latched": np.zeros(rec.n_record, bool), "prev_box": None}

            def _capture_row(t: int) -> None:
                grasp.maybe_freeze(rec, robot, pose_t=poses[t], jaw_angle=float(jaws[t]))
                grasp.maybe_release(jaw_angle=float(jaws[t]))
                action_t = (actions.encode_action(poses[t], poses[t + 1], jaws[t + 1])
                            if t < n_trans else np.zeros(7, np.float64))

                box = robot.jaw_box_poses()
                # Rounded through float32 BEFORE the node flags use it, because
                # float32 is what the file stores. The validator re-derives
                # node_contact/node_grasp from the file and demands an exact
                # match; computing them from float64 poses would let a node
                # sitting within 1e-8 m of a radius flip between the two.
                box_pose = np.stack([np.concatenate(box[n]) for n in robot.jaw_links]
                                    ).astype(np.float32).astype(np.float64)
                if state["prev_box"] is None:
                    box_vel = np.zeros((len(robot.jaw_links), 6))
                else:
                    box_vel = tool_geometry.box_velocity_backward(
                        np.stack([state["prev_box"], box_pose]), rec.frame_dt)[1]
                state["prev_box"] = box_pose

                node_pos = rec.state()[0][rec.particle_ids].astype(np.float64)
                state["latched"] = tool_geometry.grasp_latch_step(
                    state["latched"], node_pos, box_pose, half, float(jaws[t]))
                rec.capture(
                    ee_pose=poses[t].astype(np.float32),
                    action=action_t.astype(np.float32),
                    jaw=float(jaws[t]),
                    joint_pos=robot.joint_positions(),
                    contact_mode=_contact_mode(rec, robot, grasp=grasp),
                    grasp_active=grasp.is_active(),
                    grasp_node_ids=grasp.recorded_node_ids(rec.particle_ids),
                    jaw_box_pose=box_pose,
                    jaw_box_vel=box_vel,
                    jaw_joint=robot.jaw_joint_values(float(jaws[t])),
                    node_contact=tool_geometry.node_contact_flags(node_pos, box_pose, half),
                    node_grasp=state["latched"],
                    grasp_full_ids=grasp.full_ids(),
                    phase=int(phases[t]),
                )

            for t in range(n_trans):
                _capture_row(t)
                robot.set_ee_pose(poses[t + 1], jaws[t + 1])
                rec.advance(1, threshold=GRASP_CONTACT_THRESHOLD_M,
                            grasp=grasp, pose_start=poses[t], pose_end=poses[t + 1])

            # The final row: poses[n_trans] was simulated by the last advance()
            # but not yet captured. Zero-padded action -- there is no
            # poses[n_trans + 1] to encode a transition to.
            _capture_row(n_trans)
        finally:
            robot.close()
        out = rec.close()
        n_rec, n_sim = rec.n_record, rec.n_sim
    wall = time.time() - t0

    if not quiet:
        E, nu = materials.E_nu_from_lame(mu, lam)
        size = os.path.getsize(out) / 1e6
        print(f"{os.path.basename(out)}: {n_trans + 1} rows (grasp, "
              f"{1/frame_dt:.0f} Hz, pre-roll {rec.preroll_s:.3f} s), "
              f"{n_rec}/{n_sim} particles, {size:.1f} MB, {wall:.1f}s")
        print(f"    approach {wp['speeds']['approach_m_s']:.3f} m/s, retract "
              f"{wp['speeds']['retract_m_s']:.3f} m/s")
        print(f"    material mu={mu:.0f} Pa lambda={lam:.0f} Pa "
              f"(E={float(E):.0f} Pa, nu={float(nu):.4f}), rho={rho:.0f}")
    return out


def _episode_path(out_dir: str, index: int) -> str:
    return os.path.join(out_dir, f"mpm_{index:04d}.npz")


def main(argv=None) -> int:
    """Collect episodes, ONE INTERPRETER EACH.

    WHY A CHILD PROCESS PER EPISODE. Taichi bakes module-level Python constants
    into a kernel when it compiles, and this adapter sets two of them per
    material: `dt` (from the P-wave bound) and `p_mass` (from the sampled
    density). Once the first substep() has compiled, neither can change. A
    second episode in the same interpreter therefore either runs at the first
    episode's physics -- silently, with truthful-looking metadata -- or trips
    the compile lock in MPMRecorder.__init__. The lock is the safety net; a
    fresh interpreter is the fix.

    REJECTED, os.fork(): it would skip the ~10 s kernel compile, which is the
    whole cost here. But at fork time this process holds a live Metal device,
    a compiled-kernel cache and Taichi's runtime threads. Only the forking
    thread survives into the child, so the child inherits a GPU context whose
    owning threads no longer exist. That is a crash if you are lucky.

    REJECTED, one worst-case substep for the whole dataset: it would let a
    single interpreter do everything, because nothing would vary. But the
    stiffest material in materials.py's ranges needs ~106 substeps per frame
    against the softest's ~24, so pinning every episode to the stiffest makes
    soft episodes roughly 4x more expensive for no gain in fidelity. Dataset
    collection is already the slow part.

    The cost accepted instead: each child recompiles, because Taichi's offline
    cache is keyed on the constants and every material has a different `dt`.
    Roughly 10 s per episode, paid once each, in parallel with nothing.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="data_mpm", help="output directory")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--task", choices=("settle", "grasp"), default="settle",
                    help="'settle': passive gravity settling from the spawn "
                         "(reference) state, no robot. 'grasp': scripted "
                         "approach/close/retract/release via host/psm.py, "
                         "driven by --config.")
    ap.add_argument("--steps", type=int, default=None,
                    help="--task settle only: recorded transitions (default "
                         "0.5 s worth at --frame-dt). Grasp length comes from "
                         "the config's durations.")
    ap.add_argument("--frame-dt", type=float, default=None,
                    help="recording interval, seconds (default 1/240, decision "
                         "D1). --task grasp takes it from the config instead, "
                         "and refuses a disagreeing value.")
    ap.add_argument("--config", default=None,
                    help="--task grasp: JSON config (configs/grasp_pilot.json). "
                         "With a non-empty 'randomize' block each episode draws "
                         "its own concrete config from its seed.")
    ap.add_argument("--f-encoding", choices=trajectory_io.F_ENCODINGS,
                    default="float32",
                    help="tissue_F on disk (default float32: at 240 Hz the "
                         "per-step dF of slow segments is comparable to "
                         "delta16's quantisation, TODO_ASSESSMENT 4.5)")
    ap.add_argument("--n-record", default=str(DEFAULT_N_RECORD),
                    help="particles to record, or 'all' (solver simulates "
                         "24000). Ignored when --particle-ids is given.")
    ap.add_argument("--particle-ids", default=None,
                    help=".npy of solver particle indices: ONE fixed node set "
                         "for every episode (configs/particle_ids_*.npy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", type=int, default=None,
                    help="collect exactly this one episode index and exit. Set "
                         "by the parent on each child; giving it by hand "
                         "collects a single episode without dispatching.")
    ap.add_argument("--episode-config", default=None,
                    help=argparse.SUPPRESS)   # set by the parent on grasp children
    args = ap.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)

    os.makedirs(args.out, exist_ok=True)
    particle_ids = np.load(args.particle_ids) if args.particle_ids else None
    n_record = (24000 if args.n_record == "all" else int(args.n_record))

    def episode_kwargs(index: int, cfg_path: Optional[str]) -> dict:
        kw = dict(seed=args.seed + index, n_record=n_record,
                  f_encoding=args.f_encoding, particle_ids=particle_ids)
        if args.task == "settle":
            frame_dt = args.frame_dt if args.frame_dt is not None else 1.0 / 240.0
            steps = args.steps if args.steps is not None \
                else int(round(0.5 / frame_dt))
            kw.update(n_steps=steps, frame_dt=frame_dt)
        else:
            with open(cfg_path) as fh:
                kw["config"] = json.load(fh)
            # The config's own seed wins: it is what makes Item 12's matched
            # variants share a material (and therefore an initial state) while
            # living under different episode indices.
            if "seed" in kw["config"]:
                kw["seed"] = int(kw["config"]["seed"])
        return kw

    record = record_episode if args.task == "settle" else record_grasp_episode

    # THE CHILD BRANCH. `--index` present means "you are the one episode".
    # Dispatch never happens here, so recursion is structurally impossible
    # rather than merely unlikely -- the parent always passes --index, and a
    # process holding --index never spawns.
    if args.index is not None:
        cfg_path = args.episode_config or (
            _write_episode_config(args, args.index) if args.task == "grasp" else None)
        record(_episode_path(args.out, args.index),
               **episode_kwargs(args.index, cfg_path))
        return 0

    # Grasp configs are resolved in the PARENT, one JSON per episode, saved
    # next to the episode it produced: the to-do's "save the collection
    # command/configuration" applies per episode once anything is randomised.
    cfg_paths = [(_write_episode_config(args, i) if args.task == "grasp" else None)
                 for i in range(args.episodes)]
    _write_collection_record(args)

    # A single episode needs no isolation: there is nothing to collide with,
    # and a subprocess would add an interpreter start-up for nothing.
    if args.episodes == 1:
        record(_episode_path(args.out, 0), **episode_kwargs(0, cfg_paths[0]))
        print(f"\nwrote 1 episode to {args.out}/")
        print(f"validate with: python host/validate_dataset.py --data {args.out}/")
        return 0

    # THE PARENT BRANCH. This interpreter must not import mpm3d -- doing so
    # would run ti.init() and compile kernels here, which is the state we are
    # isolating the children from. record_episode() is never called on this
    # path.
    for i in range(args.episodes):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--out", args.out,
               "--task", args.task,
               "--f-encoding", args.f_encoding,
               "--n-record", args.n_record,
               "--seed", str(args.seed),
               "--index", str(i)]
        if args.steps is not None:
            cmd += ["--steps", str(args.steps)]
        if args.frame_dt is not None:
            cmd += ["--frame-dt", repr(args.frame_dt)]
        if args.particle_ids:
            cmd += ["--particle-ids", args.particle_ids]
        if cfg_paths[i]:
            cmd += ["--episode-config", cfg_paths[i]]
        print(f"[{i + 1}/{args.episodes}] --index {i} "
              f"-> {os.path.basename(_episode_path(args.out, i))}")
        # check=True so a diverged or crashed episode stops the run loudly.
        # Silently continuing would leave a gap in the numbering and a dataset
        # whose episode count does not match what was asked for.
        subprocess.run(cmd, check=True)

    print(f"\nwrote {args.episodes} episode(s) to {args.out}/")
    print(f"validate with: python host/validate_dataset.py --data {args.out}/")
    return 0


def _write_episode_config(args, index: int) -> str:
    """Resolve and save the concrete config for episode `index`.

    Base config from --config (defaults if absent); if it has a randomize
    block, draw this episode's values from its seed. The frame interval is
    the config's -- a --frame-dt that disagrees is refused rather than
    silently overriding either one.
    """
    base = collection_config.load_config(args.config)
    if args.frame_dt is not None and abs(base["frame_dt_s"] - args.frame_dt) > 1e-12:
        raise SystemExit(f"--frame-dt {args.frame_dt} disagrees with the config's "
                         f"frame_dt_s {base['frame_dt_s']}; set it in one place")
    cfg = (collection_config.sample_episode_config(base, args.seed + index)
           if base.get("randomize") else dict(base, seed=args.seed + index))
    path = os.path.splitext(_episode_path(args.out, index))[0] + ".config.json"
    with open(path, "w") as fh:
        fh.write(collection_config.config_to_json(cfg))
    return path


def _write_collection_record(args) -> None:
    """Append the exact command line to <out>/collection_log.txt.

    WHY: the to-do's deliverable is "the collection command/configuration".
    Shell history is not a record; a file next to the data is.
    """
    with open(os.path.join(args.out, "collection_log.txt"), "a") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") +
                " ".join(["python", "host/mpm_adapter.py"] + args.argv) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
