#!/usr/bin/env python3
"""mpm_energy.py -- drive the vendored MPM with a closed energy budget attached.

HOST ONLY. Importing this imports MPM.mpm3d, which calls ti.init() at module
level (section 9.4), so importing it IS the backend decision and nothing downstream
can pick an arch afterwards.

This is the Taichi half of the section 9.7 audit. The arithmetic half is
src/energy_ledger.py, which is numpy only and unit tested; everything here is
solver plumbing that a unit test cannot reach without a Metal device. Its checks
live in `host/energy_audit.py --selftest`, the same split host/smoke_test_mpm.py
already uses.

WHAT THIS ADDS OVER mpm_adapter.py
----------------------------------
The adapter records a DATASET: a subset of particles, in schema v2.1, through
TrajectoryWriter. This records a MEASUREMENT: every energy term, over all 24,000
particles, with initial conditions the collector would never produce (no
gravity, no contact, a prescribed homogeneous deformation). The two share the
solver and nothing else, which is why this is a separate file rather than a flag
on the adapter.

THREE COMPILE-TIME CONSTANTS, NOT TWO
-------------------------------------
Section 9.5 established that Taichi bakes `dt` and `p_mass` into kernels when they
compile, so one interpreter can hold only one of each. `gravity` is a third:
Boundary() reads it as a plain Python global inside a @ti.func
(`F_grid_v[I][2] -= dt * gravity`, mpm3d.py:215), so it is constant-folded at the
same moment by the same mechanism. Turning gravity off is therefore not a
runtime switch -- it is a property of the process, and the lock below covers all
three. A lock covering only two would let a cell that wanted g = 0 run silently
at 9.8 and report the difference as dissipation.

TWO WAYS TO STEP, AND WHY THE DEFAULT IS THE VENDORED ONE
---------------------------------------------------------
`advance_frame()` calls mpm3d.substep() unmodified. The headline number of the
audit -- total dissipation over a fixed simulated time -- is computed from
particle state pulled between frames, so it never depends on this file having
correctly reproduced anything.

`advance_frame_probed()` instead calls P2G, Boundary and G2P as three separate
kernels, which opens observation points on the GRID between the phases. That is
the only way to see the G2P projection loss on its own (G2P applies no forces,
so any energy it loses is pure interpolation) or to weigh the floor band's
sink. It is opt-in, it is gated on a bit-identity check in --selftest, and if
that check ever fails the probed path is the thing to distrust, not the result.

Splitting is legitimate because P2G, Boundary and G2P are already @ti.func
(mpm3d.py:146, 208, 279) and substep() already runs them as three sequential
top-level loops. Calling each from its own kernel preserves that ordering
exactly and edits nothing in third_party/ -- PROVENANCE.md stays "None yet".

WHY THE REDUCTIONS HAPPEN IN NUMPY
----------------------------------
Metal has no float64. Summing 24,000 particle energies in float32 on the GPU
would lose the small differences this study exists to measure. So state is
pulled with to_numpy() and summed in float64 on the CPU. That costs ~360 kB per
particle probe (cheap, once a frame) and ~4 MB per grid probe (not cheap, which
is why grid probes are opt-in and short).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))
sys.path.insert(0, os.path.join(REPO, "src"))

import energy_ledger                                     # noqa: E402
import materials                                         # noqa: E402

FRAME_DT = 0.0125            # matches mpm_adapter and section 9.6, so rows compare
DOMAIN_SCALE_M = 1.0         # mpm_adapter rule 4: one domain unit is one metre
SCALE, THRESHOLD = 1.0, 0.05  # substep()'s two arguments, as the adapter passes them

# Margin, in grid cells, that the clean cell keeps between the material and any
# node the solver would treat specially. `bound = 3` walls and the `I[2] <= 3`
# floor band both live at index <= 3, so 5 leaves two clear cells. Checked once a
# frame against the distance a particle could actually travel in a frame -- see
# assert_no_contact().
CLEAN_MARGIN_CELLS = 5


class CompileLockError(RuntimeError):
    pass


class Instrument:
    """One process, one (dt, p_mass, gravity), one measured trajectory."""

    def __init__(self, *, mu: float, lam: float, rho: float,
                 n_substeps: int, frame_dt: float = FRAME_DT,
                 gravity: float = 0.0, quiet: bool = False):
        import MPM.mpm3d as mpm3d                        # runs ti.init()
        import taichi as ti

        self.m, self.ti = mpm3d, ti
        self.mu, self.lam, self.rho = float(mu), float(lam), float(rho)
        self.gravity = float(gravity)
        self.frame_dt = float(frame_dt)
        self.n_substeps = int(n_substeps)
        if self.n_substeps < 1:
            raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")
        self.substep_dt = self.frame_dt / self.n_substeps

        # THE TIMEBASE INVARIANT, asserted where it is established -- same rule
        # as mpm_adapter decision 5. A study whose rows disagree about how much
        # time a frame represents is comparing different experiments.
        drift = abs(self.n_substeps * self.substep_dt - self.frame_dt)
        if drift > 1e-12:
            raise RuntimeError(f"timebase does not close, off by {drift:.3e} s")

        # THE THREE BAKED CONSTANTS. Set before any kernel is called; see the
        # module docstring for why gravity is one of them.
        mpm3d.p_rho = self.rho
        mpm3d.p_mass = mpm3d.p_vol * self.rho
        mpm3d.dt = self.substep_dt
        mpm3d.gravity = self.gravity
        mpm3d.steps = self.n_substeps
        mpm3d.timestep = self.n_substeps * self.substep_dt

        locked = getattr(mpm3d, "_energy_compile_lock", None)
        want = (self.substep_dt, mpm3d.p_mass, self.gravity)
        if locked is not None and not all(
                abs(a - b) <= 1e-15 * max(1.0, abs(b)) for a, b in zip(locked, want)):
            raise CompileLockError(
                f"this process already compiled MPM kernels with "
                f"(dt, p_mass, gravity) = {locked}; this run needs {want}. All "
                "three are baked in at compile time. One row per process -- "
                "energy_audit.py does that with --row.")
        mpm3d._energy_compile_lock = want

        self.p_mass = float(mpm3d.p_mass)
        self.p_vol = float(mpm3d.p_vol)
        self.dx = float(mpm3d.dx) * DOMAIN_SCALE_M
        self.n_grid = int(mpm3d.n_grid)
        self.n_particles = int(mpm3d.n_particles)

        self.backend = str(ti.lang.impl.current_cfg().arch)
        if "metal" not in self.backend.lower() and not quiet:
            print(f"WARNING: Taichi is on {self.backend}, not metal; ~8x slower.")

        self._kernels = None
        self._cloud = None
        self._prepare_solver()

    # -- setup -------------------------------------------------------------
    def _prepare_solver(self):
        m = self.m
        # Lame parameters straight into the fields; set_parameters() is never
        # called, because it takes (E, nu) and forms the (1 - 2nu) singularity
        # src/materials.py exists to avoid. mpm_adapter decision 1, section 9.4.
        m.mu[None] = self.mu
        m.la[None] = self.lam
        E, nu = materials.E_nu_from_lame(self.mu, self.lam)
        m.E[None] = float(E)
        m.nu[None] = float(nu)
        m.set_base_position([0.0, 0.0, 0.0])

        # No PyBullet scene, so bind what step() would normally rebind and fill
        # the SDF large and positive: the collision branch in Boundary() is then
        # never taken and no neighbour of SDF is ever indexed. Section 9.4.
        m.SDF = m.sdf.tmp_sdf
        m.collision_mask = m.sdf.co_mask
        m.SDF.fill(1.0e9)
        m.collision_mask.fill(-1)
        m.init_collision_field()

    # -- initial conditions ------------------------------------------------
    def init_slab(self, *, centre: Optional[Sequence[float]] = None,
                  stretch: float = 1.0, velocity: Optional[Sequence[float]] = None,
                  shear_mode: float = 0.0) -> Dict[str, float]:
        """Lay out the cloud, then apply a KNOWN deformation and/or velocity.

        Always starts from init_cube(), so the particle cloud is byte-identical
        to every other run in this repository. Taichi's RNG is seeded 0 in every
        process (mpm3d.py:22 calls ti.init with no random_seed), which section 9.5
        recorded as a limitation on dataset diversity and which here is the
        thing that makes rows comparable at all.

        `stretch` s applies A = diag(s, s^-1/2, s^-1/2) to BOTH the positions
        and F, about the cloud's own centroid. Because det A = 1 the initial
        state is exactly isochoric, so log(J) = 0 and the initial energy is
        (mu/2)(s^2 + 2/s - 3) per unit reference volume in closed form -- E(0)
        is known rather than measured. Applying it to positions without also
        applying it to F would be a body that has moved without deforming; the
        pair is what makes the state self-consistent.

        `shear_mode` adds a non-affine velocity field (a standing sine along z).
        Non-affine on purpose: APIC reproduces AFFINE fields exactly, so an
        affine initial velocity would be transferred losslessly and measure
        nothing. This is the content the transfers can actually destroy.
        """
        m = self.m
        # init_cube() IS NOT IDEMPOTENT. It fills F_x with ti.random(), and
        # Taichi's RNG state advances -- so a SECOND call in the same process
        # lays out a DIFFERENT cloud. Section 9.5's "the initial particle cloud is
        # identical in every episode" is true only because every episode is a
        # fresh process; it is false the moment two initial conditions are built
        # in one interpreter, which is exactly what --selftest does. The first
        # cloud is cached and replayed so every initial condition in a process
        # starts from the same particles.
        if self._cloud is None:
            m.init_cube()
            self.ti.sync()
            self._cloud = m.F_x.to_numpy().copy()
        else:
            m.F_x.from_numpy(self._cloud)
        m.init_deformation_gradient()
        m.F_v.fill([0.0, 0.0, 0.0])
        m.F_C.fill([[0.0] * 3] * 3)
        m.F_grid_m.fill(0.0)
        m.F_grid_v.fill([0.0, 0.0, 0.0])
        self.ti.sync()

        x = m.F_x.to_numpy().astype(np.float64)
        c = x.mean(axis=0)
        F = np.broadcast_to(np.eye(3), (self.n_particles, 3, 3)).copy()

        if stretch != 1.0:
            A = np.diag([stretch, stretch ** -0.5, stretch ** -0.5])
            x = c + (x - c) @ A.T
            F = np.broadcast_to(A, (self.n_particles, 3, 3)).copy()

        if centre is not None:
            x = x + (np.asarray(centre, np.float64) - x.mean(axis=0))

        v = np.zeros_like(x)
        if velocity is not None:
            v += np.asarray(velocity, np.float64)
        if shear_mode:
            span = max(x[:, 2].max() - x[:, 2].min(), 1e-12)
            phase = 2.0 * np.pi * (x[:, 2] - x[:, 2].min()) / span
            v[:, 0] += shear_mode * np.sin(phase)
            # Remove any net momentum the mode introduced. The clean cell's zero
            # linear momentum is a correctness signal (nothing in the solver can
            # create momentum), and it is only usable if it starts at zero.
            v -= v.mean(axis=0)

        m.F_x.from_numpy(x.astype(np.float32))
        m.F.from_numpy(F.astype(np.float32))
        m.F_v.from_numpy(v.astype(np.float32))
        self.ti.sync()
        return {"centroid": c, "stretch": stretch}

    # -- driving -----------------------------------------------------------
    def advance_frame(self, n_frames: int = 1) -> None:
        """One recorded frame, through the VENDORED substep(). n_substeps, never
        mpm3d.steps -- the module global is the vendored default and using it is
        what made frames advance the wrong amount of time in section 9.5."""
        for _ in range(n_frames):
            for _ in range(self.n_substeps):
                self.m.substep(SCALE, THRESHOLD)
        self.ti.sync()

    def _build_kernels(self):
        """P2G / Boundary / G2P as three kernels instead of one. See docstring."""
        if self._kernels is not None:
            return self._kernels
        ti, m = self.ti, self.m

        @ti.kernel
        def k_clear():
            for I in ti.grouped(m.F_grid_m):
                m.F_grid_v[I] = ti.zero(m.F_grid_v[I])
                m.F_grid_m[I] = 0

        @ti.kernel
        def k_p2g():
            m.P2G()

        # NO ANNOTATED ARGUMENTS. This file uses `from __future__ import
        # annotations`, which turns every annotation into a string, and Taichi
        # reads kernel argument annotations as types -- `scale: float` arrives
        # as the string "float" and is rejected. mpm3d.py has the same signature
        # and works only because it does not use the future import. SCALE and
        # THRESHOLD are genuine constants for this study, so they are captured
        # at compile time instead of passed, which sidesteps it entirely.
        @ti.kernel
        def k_boundary():
            m.Boundary(SCALE, THRESHOLD)

        @ti.kernel
        def k_g2p():
            m.G2P()

        self._kernels = (k_clear, k_p2g, k_boundary, k_g2p)
        return self._kernels

    def substep_split(self) -> None:
        """One substep through the split kernels. Must be bit-identical to
        mpm3d.substep(); --selftest check 1 is what establishes that."""
        k_clear, k_p2g, k_boundary, k_g2p = self._build_kernels()
        k_clear()
        k_p2g()
        k_boundary()
        k_g2p()

    def advance_frame_probed(self, n_frames: int = 1) -> List[Dict[str, float]]:
        """One frame through the split kernels, reading the grid between phases.

        Returns one record per substep. Expensive: two 4 MB grid readbacks per
        substep. Use on short runs only.
        """
        k_clear, k_p2g, k_boundary, k_g2p = self._build_kernels()
        out = []
        for _ in range(n_frames):
            for _ in range(self.n_substeps):
                before = self.ledger()
                k_clear()
                k_p2g()
                ke_p2g = self._grid_ke(momentum=True)
                k_boundary()
                ke_bnd = self._grid_ke(momentum=False)
                k_g2p()
                after = self.ledger()
                ke_particle_after = after["ke"] + after["ke_affine"]
                out.append({
                    "e_before": before["total"], "e_after": after["total"],
                    "ke_grid_after_p2g": ke_p2g,
                    "ke_grid_after_boundary": ke_bnd,
                    # G2P APPLIES NO FORCES. It only interpolates, so whatever
                    # energy it fails to hand back is pure projection loss with
                    # nothing else mixed in. This is the cleanest single number
                    # the instrument produces.
                    "loss_g2p": ke_bnd - ke_particle_after,
                    # With gravity off, Boundary()'s only remaining effects are
                    # the sticky SDF branch, the wall condition and the floor
                    # band, and the mass division is energy-neutral
                    # (|p|^2/2m == (1/2)m|v|^2). So this difference IS the
                    # boundary sink. With gravity on it also contains the
                    # gravity impulse work and must not be read as dissipation.
                    "loss_boundary": ke_p2g - ke_bnd,
                    "dissipated": before["total"] - after["total"],
                })
        self.ti.sync()
        return out

    def _grid_ke(self, *, momentum: bool) -> float:
        """Kinetic energy on the grid, joules, summed in float64 on the CPU.

        `momentum=True` for the state straight out of P2G, where F_grid_v holds
        MOMENTUM: Boundary() is what divides by mass (mpm3d.py:212). Reading
        that array as a velocity gives a number with no physical meaning and no
        outward sign of being wrong, which is the whole reason this is a flag
        rather than a comment.
        """
        gv = self.m.F_grid_v.to_numpy().astype(np.float64)
        gm = self.m.F_grid_m.to_numpy().astype(np.float64)
        live = gm > 0.0
        if not live.any():
            return 0.0
        sq = (gv[live] ** 2).sum(axis=1)
        return float((0.5 * sq / gm[live]).sum() if momentum
                     else (0.5 * gm[live] * sq).sum())

    # -- reading out -------------------------------------------------------
    def state(self):
        """(pos, vel, C, F) over ALL particles, float64. Metrics over the full
        set, never a subset -- mpm_adapter rule 2, for the same reason."""
        return (self.m.F_x.to_numpy().astype(np.float64),
                self.m.F_v.to_numpy().astype(np.float64),
                self.m.F_C.to_numpy().astype(np.float64),
                self.m.F.to_numpy().astype(np.float64))

    def ledger(self) -> Dict[str, float]:
        """The full budget of the current state.

        mu and lambda are read from the SOLVER'S FIELDS, not from self.mu /
        self.lam. They are ti.fields rather than baked constants, so a caller
        can change them at runtime -- the zero-stiffness checks do exactly that
        -- and a ledger that kept charging the constructor's material would
        report strain energy for a material the solver is not simulating. That
        is the same class of error as section 9.5's timebase: two numbers that agree
        with everything except each other.
        """
        pos, vel, C, F = self.state()
        return energy_ledger.energy_ledger(
            pos, vel, C, F, p_mass=self.p_mass, p_vol=self.p_vol,
            mu=float(self.m.mu[None]), lam=float(self.m.la[None]),
            dx=self.dx, gravity=self.gravity)

    def stencil_index_range(self):
        """(min, max) grid index any particle's quadratic stencil touches.

        The stencil is base .. base+2 with base = int(x/dx - 0.5), read straight
        off P2G (mpm3d.py:149-153). Computed from positions rather than from
        grid mass so it costs one 288 kB readback instead of 4 MB.
        """
        x = self.m.F_x.to_numpy().astype(np.float64)
        base = np.floor(x / self.dx - 0.5).astype(np.int64)
        return int(base.min()), int(base.max() + 2)

    def max_speed(self) -> float:
        v = self.m.F_v.to_numpy().astype(np.float64)
        return float(np.linalg.norm(v, axis=1).max())

    def assert_no_contact(self, margin_cells: int = CLEAN_MARGIN_CELLS) -> Dict:
        """No particle is anywhere the solver treats specially -- proven, not assumed.

        Three branches in Boundary() are non-conservative: the sticky SDF
        collision (never armed here, the SDF is filled 1e9), the `bound = 3`
        wall condition, and the `I[2] <= 3` floor band. All three live at index
        <= 3 or >= n_grid - 3, so keeping every stencil node inside
        [margin, n_grid - margin] with margin > 3 means none of them can fire.

        CHECKED ONCE A FRAME, AND THAT IS ENOUGH ONLY BECAUSE THE MARGIN IS
        CHECKED TOO. A particle travelling at the measured peak speed for a
        whole frame moves `max_speed * frame_dt`; if that is smaller than the
        clearance in metres, no particle can cross into a special region between
        two checks. The returned dict carries both numbers so the caller can
        assert the inequality rather than trust the sampling rate.
        """
        lo, hi = self.stencil_index_range()
        clearance_cells = min(lo - 3, (self.n_grid - 3) - hi)
        reach_m = self.max_speed() * self.frame_dt
        return {
            "min_index": lo, "max_index": hi,
            "ok": lo >= margin_cells and hi <= self.n_grid - margin_cells,
            "clearance_cells": clearance_cells,
            "clearance_m": clearance_cells * self.dx,
            "reach_m": reach_m,
            "sampling_safe": clearance_cells * self.dx > reach_m,
        }


# --------------------------------------------------------------------------

def advisory_n_substeps(mu: float, lam: float, rho: float, dx: float,
                        frame_dt: float = FRAME_DT, safety: float = 0.3) -> int:
    """The substep count the collector would choose. Shared with the ladder so
    the audit's rows are the SAME rows section 9.6 swept, not lookalikes."""
    dt = float(materials.suggested_substep_dt(mu, rho, dx, safety=safety, lam=lam))
    return max(1, int(np.ceil(frame_dt / dt)))
