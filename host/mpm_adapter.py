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
import os
import subprocess
import sys
import time
from typing import Optional, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))   # so `import MPM.x` works
sys.path.insert(0, os.path.join(REPO, "src"))

import materials                                         # noqa: E402
import tissue_metrics                                    # noqa: E402
import trajectory_io                                     # noqa: E402

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
        notes: str = "",
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
        self.substep_dt = float(materials.suggested_substep_dt(
            self.mu, self.rho, mpm3d.dx * DOMAIN_SCALE_M,
            safety=substep_safety, lam=self.lam))
        # Whole substeps per recorded frame, so the recorded dt stays a round
        # number the whole dataset shares regardless of material. Episodes with
        # different stiffness then differ in COST, not in sampling rate -- a
        # dataset whose dt varies per episode is one a dynamics model cannot use.
        self.n_substeps = max(1, int(np.ceil(frame_dt / self.substep_dt)))
        self.substep_dt = frame_dt / self.n_substeps
        self.frame_dt = float(frame_dt)

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
        locked = getattr(mpm3d, "_adapter_compile_lock", None)
        if locked is not None:
            l_dt, l_mass = locked
            if abs(l_dt - self.substep_dt) > 1e-15 or abs(l_mass - mpm3d.p_mass) > 1e-30:
                raise RuntimeError(
                    f"this process already compiled MPM kernels with substep "
                    f"dt={l_dt:.3e} s and p_mass={l_mass:.6e} kg, and Taichi "
                    f"bakes both constants in at compile time; this episode "
                    f"needs dt={self.substep_dt:.3e} s and "
                    f"p_mass={mpm3d.p_mass:.6e} kg. Collect one episode per "
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
        mpm3d._adapter_compile_lock = (self.substep_dt, mpm3d.p_mass)

        backend = str(ti.lang.impl.current_cfg().arch)
        if "metal" not in backend.lower():
            # Not fatal -- the physics is identical -- but 8x slower with no
            # error anywhere, which is exactly how a slow collection run gets
            # blamed on the solver. Say it out loud. See section 9.4.
            print(f"WARNING: Taichi is on {backend}, not metal. "
                  f"Collection will be ~8x slower.")
        self.backend = backend

        n_sim = int(self.m.n_particles)
        if n_record > n_sim:
            raise ValueError(f"n_record={n_record} exceeds the solver's {n_sim} "
                             "particles; a subset cannot be larger than its set")
        self.n_sim = n_sim
        self.n_record = int(n_record)

        # THE SUBSET, CHOSEN ONCE. Sorted so particle_ids is monotonic, which
        # makes "are these unique" answerable by inspection and keeps the stored
        # nodes in a stable, reproducible order across episodes with the same
        # seed. Uniform without replacement: MPM particle index carries no
        # spatial structure (init_cube assigns random positions per index), so
        # a uniform draw is already spatially uniform.
        rng = np.random.default_rng(seed)
        self.particle_ids = np.sort(
            rng.choice(n_sim, size=self.n_record, replace=False)).astype(np.int32)

        self._init_solver()

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
                f"(P-wave advisory, safety={substep_safety})"),
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
    def advance(self, n_frames: int = 1, scale: float = 1.0,
                threshold: float = 0.05) -> None:
        """Run `n_frames` recorded steps' worth of substeps.

        self.n_substeps, NOT self.m.steps. The module global is the vendored
        default (25) and has nothing to do with this episode's material; using
        it here is what made recorded frames advance the wrong amount of time
        while every number written to disk agreed with every other one.
        """
        for _ in range(n_frames):
            for _ in range(self.n_substeps):
                self.m.substep(scale, threshold)
        self.ti.sync()

    def state(self):
        """Full solver state as numpy: (pos, vel, F) over ALL particles."""
        return (self.m.F_x.to_numpy(), self.m.F_v.to_numpy(), self.m.F.to_numpy())

    def capture(self, *, ee_pose: np.ndarray, action: np.ndarray,
                ee_vel: Optional[np.ndarray] = None, jaw: float = 0.0,
                contact_mode: Optional[int] = None,
                contact_force: Optional[np.ndarray] = None) -> dict:
        """Record one step. Returns the metrics, computed over every particle."""
        pos, vel, F = self.state()

        # RULE 2: metrics over ALL particles, before any subsampling. A maximum
        # over a subset is biased low, and this is the safety number.
        t_o, t_n, t_e = self.target
        exposure = float(tissue_metrics.compute_exposure(
            pos.astype(np.float64), t_o, t_n, t_e))
        strain = tissue_metrics.compute_safety_strain(F.astype(np.float64))
        safety = float(strain["max"])

        idx = self.particle_ids
        self.w.append(
            tissue_pos=pos[idx],
            tissue_vel=vel[idx],
            tissue_F=F[idx],
            ee_pose=ee_pose,
            ee_vel=ee_vel,
            action=action,
            jaw=jaw,
            contact_mode=contact_mode,
            contact_force=contact_force,
            exposure=exposure,
            safety_strain=safety,
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
        # Write even if the episode raised: a partial trajectory is usually
        # still worth inspecting, and losing a long run to an exception in the
        # last step is infuriating. Mirrors TrajectoryWriter.__exit__.
        if self.n_steps and not self._closed:
            self.close()
        return False  # never swallow the exception


# --------------------------------------------------------------------------

def record_episode(path: str, *, n_steps: int = 100, seed: int = 0,
                   n_record: int = DEFAULT_N_RECORD, quiet: bool = False) -> str:
    """One passive episode: tissue released under gravity, no tool.

    Passive because there is no robot yet (see the module docstring). It still
    exercises every part of the path -- large deformation, contact with the
    floor, metrics, subset bookkeeping, the writer -- which is what makes it
    useful before the PSM lands.
    """
    rng = np.random.default_rng(seed)
    # Sample mu and lambda DIRECTLY, log-uniformly. Never (E, nu): lambda is
    # singular at nu = 0.5 and tissue sits at nu ~ 0.49. See materials.py.
    mat = materials.sample_material(rng)
    mu, lam, rho = float(np.exp(mat[0])), float(np.exp(mat[1])), float(mat[2])

    t0 = time.time()
    with MPMRecorder(path, task="tissue_settle", mu=mu, lam=lam, rho=rho,
                     n_record=n_record, seed=seed) as rec:
        # A zero delta-pose action is a real action ("hold still"), not a
        # placeholder, so action_spec="delta_pose_jaw" is honest here.
        ee_pose = np.array([0.25, 0.35, 0.20, 0, 0, 0, 1], np.float32)
        action = np.zeros(7, np.float32)
        for t in range(n_steps):
            rec.capture(ee_pose=ee_pose, action=action)
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
    ap.add_argument("--steps", type=int, default=100, help="recorded steps/episode")
    ap.add_argument("--n-record", type=int, default=DEFAULT_N_RECORD,
                    help="particles to record (solver simulates 24000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", type=int, default=None,
                    help="collect exactly this one episode index and exit. Set "
                         "by the parent on each child; giving it by hand "
                         "collects a single episode without dispatching.")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    # THE CHILD BRANCH. `--index` present means "you are the one episode".
    # Dispatch never happens here, so recursion is structurally impossible
    # rather than merely unlikely -- the parent always passes --index, and a
    # process holding --index never spawns.
    if args.index is not None:
        record_episode(_episode_path(args.out, args.index),
                       n_steps=args.steps, seed=args.seed + args.index,
                       n_record=args.n_record)
        return 0

    # A single episode needs no isolation: there is nothing to collide with,
    # and a subprocess would add an interpreter start-up for nothing.
    if args.episodes == 1:
        record_episode(_episode_path(args.out, 0), n_steps=args.steps,
                       seed=args.seed, n_record=args.n_record)
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
               "--steps", str(args.steps),
               "--n-record", str(args.n_record),
               "--seed", str(args.seed),
               "--index", str(i)]
        print(f"[{i + 1}/{args.episodes}] {' '.join(cmd[-2:])} "
              f"-> {os.path.basename(_episode_path(args.out, i))}")
        # check=True so a diverged or crashed episode stops the run loudly.
        # Silently continuing would leave a gap in the numbering and a dataset
        # whose episode count does not match what was asked for.
        subprocess.run(cmd, check=True)

    print(f"\nwrote {args.episodes} episode(s) to {args.out}/")
    print(f"validate with: python host/validate_dataset.py --data {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
