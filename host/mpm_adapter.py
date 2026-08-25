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

Four decisions are baked in here. Each one is a rule from CLAUDE.md meeting its
first real consumer, and each would be easy to get quietly wrong.

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

WHAT THIS DOES NOT YET DO: there is no robot. PyBullet supplies rigid-body
collision through the SDF in `sdf.py`, and that is where the PSM plugs in later;
until then `ee_pose` and `action` are whatever the caller passes, and the tissue
is driven by gravity and its own elasticity. The file format does not care, which
is the point of having a format.
"""

from __future__ import annotations

import argparse
import os
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

        # Taichi captures module-level Python constants when a kernel COMPILES,
        # so this must happen before the first substep() and cannot be changed
        # afterwards in the same process. Verified: setting it here and running
        # free fall reproduces -g*t at the NEW dt exactly. The consequence is
        # one substep_dt per process -- see the guard below.
        if getattr(mpm3d, "_adapter_dt_locked", None) is not None:
            locked = mpm3d._adapter_dt_locked
            if abs(locked - self.substep_dt) > 1e-15:
                raise RuntimeError(
                    f"this process already compiled MPM kernels with substep "
                    f"dt={locked:.3e} s and Taichi bakes that constant in at "
                    f"compile time; this episode needs {self.substep_dt:.3e} s. "
                    "Collect one episode per process (the CLI already does), or "
                    "fix the substep across the dataset.")
        mpm3d.dt = self.substep_dt
        mpm3d._adapter_dt_locked = self.substep_dt

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
            notes=notes or (
                f"vendored MPM cb797f36, backend={backend}, "
                f"{self.n_record}/{n_sim} particles, seed={seed}, "
                f"domain_scale={DOMAIN_SCALE_M} m, "
                f"substep {self.substep_dt*1e6:.1f} us x {self.n_substeps} "
                f"(P-wave advisory, safety={substep_safety})"),
            # [log(mu), log(lambda), rho] -- the layout materials.unpack_material
            # expects. Logs because tissue stiffness spans orders of magnitude
            # and a network fed raw Pascals burns capacity on the exponent.
            material_params=np.array(
                [np.log(self.mu), np.log(self.lam), self.rho], np.float32),
            substep_dt=self.m.dt,
            n_substeps=self.m.steps,
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
        """Run `n_frames` recorded steps' worth of substeps."""
        for _ in range(n_frames):
            for _ in range(self.m.steps):
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="data_mpm", help="output directory")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100, help="recorded steps/episode")
    ap.add_argument("--n-record", type=int, default=DEFAULT_N_RECORD,
                    help=f"particles to record (solver simulates 24000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.episodes):
        record_episode(os.path.join(args.out, f"mpm_{i:04d}.npz"),
                       n_steps=args.steps, seed=args.seed + i,
                       n_record=args.n_record)
    print(f"\nwrote {args.episodes} episode(s) to {args.out}/")
    print(f"validate with: python host/validate_dataset.py --data {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
