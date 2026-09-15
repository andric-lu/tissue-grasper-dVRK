#!/usr/bin/env python3
"""
mpm_locality_study.py -- how far does a local kick travel in one recorded step?

    conda activate tissue-host
    python host/mpm_locality_study.py --selftest        # ~1 min, the gate
    python host/mpm_locality_study.py                   # ~10-15 min, full sweep

DECISION_LOG.md section 13.2 (brief study 0.1). HOST ONLY: the children import
MPM.mpm3d, which calls ti.init() at module level.

WHY THIS EXISTS
---------------
A graph network with M message-passing steps over radius R sees M*R of the
object. The wave-speed arithmetic (section 13.1) says a P-wave crosses the 300 mm
slab within one recorded step at every material in materials.py's range -- but
that is an UPPER bound on how far a signal travels, not on how far it travels
with enough amplitude to be worth predicting. The clamp-free floor band, free
surfaces and any damping all cut that down, and none of them appear in c_p. This
measures the real thing: the distance at which a local perturbation's effect
falls below 1% (and 0.1%) of its size at the source.

THE PROBE, AND WHY IT IS NOT THE BRIEF'S
----------------------------------------
The brief displaces one particle's POSITION by epsilon. In MPM that is nearly
inert: stress comes from each particle's carried F, not from where it sits, so
a moved particle with an unchanged F exerts no extra stress -- it perturbs the
quadrature, not the material. And it is one of ~20 particles in its grid cell,
so the first scatter to the grid dilutes it ~20x. Two alternatives were weighed:

  * perturb F on a patch   -- a genuine local strain, but its amplitude is hard
                              to relate to what a network sees as input
  * kick VELOCITY on a patch (chosen) -- a momentum source, resolved on the grid,
                              and velocity is what a GNN reads and predicts

Every particle within PATCH_RADIUS_CELLS * dx of a probe particle gets +dv along
one axis, in x (in-plane: the P-wave the section 13 argument is about) and z
(normal: bending, the direction a retraction pulls).

THE MEASUREMENT
---------------
From one base state, four runs of the same length:

    A    unperturbed
    A'   unperturbed again  -> |A' - A|
    B    kicked by dv
    B'   kicked by dv again -> |B' - B|; the floor is the larger of the two
    B2   kicked by 2 dv     -> |B2 - A| / |B - A| ~ 2, checked PER DISTANCE BAND

WHY CONTROL RUNS, AND WHY TWO. The vendored solver is not bitwise reproducible
against itself on Metal: P2G's atomic adds sum in a varying order (section 9.7,
check 1 measured ~2 ulp over 24 substeps). Without a control, GPU nondeterminism
amplified by the dynamics would be reported as influence. But the clean base
state is exactly at rest, where every sum is of zeros and |A' - A| is zero by
construction -- a vacuous floor. |B' - B| is the pair that actually carries the
noise of the arithmetic producing the response.

WHY THE KICK IS 1 m/s. float32 again, measured. See DV below: a 0.01 m/s kick
produced strain increments that round away in `1 + dt*C_ii` over most of the
slab, and the response went nonlinear with distance. At 1 m/s the physics is
still linear (patch response is the same 3.1e-3 of the kick at 0.1 and 1 m/s)
and nothing rounds.

WHY VELOCITY, NOT POSITION, IS THE PRIMARY RESPONSE. The far-field position
change is a small increment added to a coordinate near 0.5 m, where one float32
ulp is 6e-8 m, so position differences quantise first. Both are recorded;
position's floor includes the ulp at the largest coordinate so it cannot report
a radius inside rounding.

The response is |B - A| per particle, binned by distance from the probe in the
BASE state, and normalised by the kicked patch's own mean response at the same
time. `graph_design.influence_radius` turns that into a radius WITH A STATUS:
`saturated` means the influence reached the far edge of the object and the
number is a lower bound; `below_noise` means the threshold could not be
resolved at all.

RECORDING RATES FROM ONE RUN. frame_dt is 1/240 s and snapshots are taken after
1, 2, 3, 4 and 8 frames: dt_record = 1/240, 1/120, 1/80 (the current dataset),
1/60 and 1/30 s. The solver never changes between them.

THE SWEEP
---------
  materials   soft / mid / stiff corners of materials.py's ranges (mid is the
              geometric midpoint), rho at the middle of its range
  configs     clean        g = 0, lifted clear of walls and floor band, at rest,
                           no-contact ASSERTED every frame (section 9.7 cell 1)
              as_collected g = 9.8, init_cube as vendored, settled 0.75 s onto
                           the floor band -- the state data_mpm/ is recorded in
  damping     none, Kelvin-Voigt at two levels, mass-proportional at two levels
              (src/damping.py, Instrument.set_damping)

ONE SUBSTEP COUNT PER MATERIAL, ACROSS ALL DAMPING LEVELS. Viscosity adds a
diffusive stability bound, and the strongest level needs more substeps than the
elastic CFL does. Section 9.7 showed dissipation shifts with substep count, so a
per-row count would mix that shift into the damping comparison. Every row of a
material runs at the maximum any of its rows needs.

WHAT THIS DOES NOT ESTABLISH. The damping levels are chosen by decay time, not
from measured tissue viscosity, and materials.py's ranges are placeholders --
this measures what the SOLVER does across those settings. Whether real tissue
sits at any of them is a literature question section 13 leaves open.

Parent/child dispatch follows host/energy_audit.py: dt, p_mass and gravity are
baked into Taichi kernels at compile time, so each (material, config) row runs
in its own process. `--child` marks one and is what makes recursion impossible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "host"))

import damping as DMP                                    # noqa: E402
import graph_design as GD                                # noqa: E402
import materials                                         # noqa: E402
from MPM.config import n_grid                            # noqa: E402  (no ti.init)

# mpm_energy's TOP LEVEL imports numpy, energy_ledger and materials only;
# Instrument.__init__ imports MPM.mpm3d lazily. So the parent may read its
# constants without compiling anything -- and main() asserts that is still true
# rather than trusting this comment.
import mpm_energy as ME                                  # noqa: E402

DX = 1.0 / n_grid * ME.DOMAIN_SCALE_M
FRAME_DT = 1.0 / 240.0
SNAPSHOT_FRAMES = (1, 2, 3, 4, 8)
BASE_SAFETY = 0.3

# m/s kick. LARGE ON PURPOSE, and measured to be so (section 13.2). The first
# version kicked at 0.01 m/s and its selftest FAILED linearity (median 1.45).
# The cause is float32, not physics: at 55 substeps/frame (substep ~76 us) the
# per-substep strain increment dt*C of a 0.01 m/s kick is below half an ulp of
# 1.0 for 84-100% of particles beyond the near field, so `1 + dt*C_ii` rounds to
# exactly 1 and the increment is discarded. Response beyond 120 mm then had
# linearity 1.17. At 1 m/s no particle's increment rounds, linearity is
# 2.000 +- 0.03 in EVERY distance band, and the patch response is 3.1e-3 of the
# kick at 0.1 and at 1 m/s -- the physics is linear, the small kick was not
# resolved. The selftest now checks linearity per band and each row records how
# many significant particles sit under the rounding floor.
DV = 1.0
LIN_BANDS = (0.0, 0.02, 0.06, 0.12, float("inf"))   # m, for per-band linearity
LIN_TOL = 0.1                                      # band median must be 2 +- this
MIN_BAND_N = 20
PATCH_RADIUS_CELLS = 0.5    # patch = particles within dx/2 of the probe (~10 particles)
BIN_WIDTH = 0.005           # m. Particle spacing is ~5.7 mm; ~1 grid cell = 15.6 mm
FRACS = (0.01, 0.001)
ROW_TIMEOUT_S = 1800

# Decay times for the damping levels. Kelvin-Voigt levels are e-folding times of
# a P-wave at KV_WAVELENGTH, the shortest wavelength the grid resolves well
# (4 cells). "slow" = one frame of the current 80 Hz dataset; "fast" = 10x
# faster. Mass levels are wavelength-free, so they are set by motion time scale
# instead: 0.1 s and one 80 Hz frame.
KV_WAVELENGTH = 4.0 * DX
DAMPING_LEVELS = {
    "none":      (ME.DAMPING_NONE, None),
    "kv_slow":   (ME.DAMPING_KV, 1.0 / 80.0),
    "kv_fast":   (ME.DAMPING_KV, 1.0 / 800.0),
    "mass_slow": (ME.DAMPING_MASS, 0.1),
    "mass_fast": (ME.DAMPING_MASS, 1.0 / 80.0),
}
CONFIGS = {
    "clean": dict(gravity=0.0, centre=[0.5, 0.5, 0.5], settle_s=0.0,
                  assert_no_contact=True),
    "as_collected": dict(gravity=9.8, centre=None, settle_s=0.75,
                         assert_no_contact=False),
}

# Plot palette: the dataviz skill's ordinal blue ramp (steps 250-650, light
# surface), one step per recording interval, lightest = shortest interval.
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7", "#fcfcfb")


# --------------------------------------------------------------------------
# Shared definitions (parent and child)
# --------------------------------------------------------------------------

def material_corners() -> dict:
    """Soft/mid/stiff from materials.py's ranges -- measured from the module, not
    retyped, so they follow the ranges when the placeholders are replaced."""
    (mu_lo, mu_hi), (la_lo, la_hi) = materials.DEFAULT_MU_RANGE, materials.DEFAULT_LAM_RANGE
    rho = float(np.mean(materials.DEFAULT_RHO_RANGE))
    return {"soft": (mu_lo, la_lo, rho),
            "mid": (float(np.sqrt(mu_lo * mu_hi)), float(np.sqrt(la_lo * la_hi)), rho),
            "stiff": (mu_hi, la_hi, rho)}


def damping_params(level: str, rho: float) -> dict:
    kind, tau = DAMPING_LEVELS[level]
    if kind == ME.DAMPING_KV:
        return {"kind": kind, "eta": DMP.kv_eta_for_p_decay(tau, KV_WAVELENGTH, rho),
                "alpha": 0.0, "tau": tau}
    if kind == ME.DAMPING_MASS:
        return {"kind": kind, "eta": 0.0, "alpha": DMP.mass_alpha_for_decay(tau), "tau": tau}
    return {"kind": kind, "eta": 0.0, "alpha": 0.0, "tau": None}


def substeps_for_material(mu: float, lam: float, rho: float, levels) -> int:
    """Max over every damping level of (elastic CFL, viscous) substep counts."""
    n = ME.advisory_n_substeps(mu, lam, rho, DX, frame_dt=FRAME_DT, safety=BASE_SAFETY)
    for level in levels:
        p = damping_params(level, rho)
        dt_v = DMP.viscous_stable_substep(p["eta"], rho, DX, safety=BASE_SAFETY)
        if np.isfinite(dt_v):
            n = max(n, int(np.ceil(FRAME_DT / dt_v)))
    return n


# --------------------------------------------------------------------------
# Child: solver plumbing
# --------------------------------------------------------------------------

def _snapshot(inst) -> dict:
    m = inst.m
    return {"x": m.F_x.to_numpy().copy(), "v": m.F_v.to_numpy().copy(),
            "C": m.F_C.to_numpy().copy(), "F": m.F.to_numpy().copy()}


def _restore(inst, snap: dict, v=None) -> None:
    """Put the full particle state back and clear the grid. The grid is rebuilt
    from particles by P2G every substep, so particles ARE the state."""
    m = inst.m
    m.F_x.from_numpy(snap["x"])
    m.F_v.from_numpy(snap["v"] if v is None else v)
    m.F_C.from_numpy(snap["C"])
    m.F.from_numpy(snap["F"])
    m.F_grid_m.fill(0.0)
    m.F_grid_v.fill([0.0, 0.0, 0.0])
    inst.ti.sync()


def _require_no_contact(inst, where: str) -> None:
    c = inst.assert_no_contact()
    if not (c["ok"] and c["sampling_safe"]):
        raise RuntimeError(
            f"CLEAN CONFIG VIOLATED {where}: stencil {c['min_index']}..{c['max_index']}, "
            f"clearance {c['clearance_m']*1e3:.1f} mm vs per-frame reach "
            f"{c['reach_m']*1e3:.2f} mm")


def _trajectory(inst, snap: dict, cfg: dict, kick=None) -> dict:
    """Restore, optionally kick, advance to the last snapshot. float64 copies out."""
    v = None
    if kick is not None:
        patch, axis, dv = kick
        v = snap["v"].copy()
        v[patch, axis] += dv
    _restore(inst, snap, v)
    xs, vs, cs = [], [], []
    for f in range(1, max(SNAPSHOT_FRAMES) + 1):
        inst.advance_frame(1)
        if cfg["assert_no_contact"]:
            _require_no_contact(inst, f"at perturbation frame {f}")
        if f in SNAPSHOT_FRAMES:
            xs.append(inst.m.F_x.to_numpy().astype(np.float64))
            vs.append(inst.m.F_v.to_numpy().astype(np.float64))
            cs.append(inst.m.F_C.to_numpy().astype(np.float64))
    return {"x": xs, "v": vs, "C": cs}


def _probe_sites(base: np.ndarray, dx: float) -> dict:
    """Probe particles chosen from the MEASURED base cloud, not coordinates:
    the one nearest the centroid of the bounding box, and the one nearest a point
    two cells in from the -x face at mid-y, mid-z."""
    lo, hi = base.min(axis=0), base.max(axis=0)
    mid = 0.5 * (lo + hi)
    targets = {"centre": mid, "edge": np.array([lo[0] + 2 * dx, mid[1], mid[2]])}
    return {k: int(np.argmin(np.linalg.norm(base - t, axis=1))) for k, t in targets.items()}


def _measure(dist, patch, A, A2, B, Bp, B2, q, k, substep_dt) -> dict:
    """One (quantity, snapshot) cell: floor, reference, radii, linearity, profile.

    TWO CONTROL PAIRS, NOT ONE. |A' - A| at a state exactly at rest is zero by
    construction (summing zeros in any order gives zero), so on its own it would
    certify a vacuous floor -- the first version did exactly that. |B' - B|, the
    kicked run repeated, carries the real run-to-run noise of the arithmetic
    that produces the response. The floor is the larger of the two and of the
    float32 spacing of the values the difference was formed from.
    """
    resp = np.linalg.norm(B[q][k] - A[q][k], axis=1)
    resp2 = np.linalg.norm(B2[q][k] - A[q][k], axis=1)
    noise_rest = float(np.linalg.norm(A2[q][k] - A[q][k], axis=1).max())
    noise_kicked = float(np.linalg.norm(Bp[q][k] - B[q][k], axis=1).max())
    quant = float(np.spacing(np.float32(max(np.abs(A[q][k]).max(), np.abs(B[q][k]).max()))))
    floor = max(noise_rest, noise_kicked, quant)
    ref = float(resp[patch].mean())
    prof = GD.influence_profile(dist, resp, BIN_WIDTH, r_max=float(dist.max()))

    radii = {}
    for frac in FRACS:
        if ref <= 0:
            radii[str(frac)] = {"radius": float("nan"), "status": GD.NO_SIGNAL}
        else:
            r = GD.influence_radius(prof, ref=ref, frac=frac, noise_floor=floor)
            radii[str(frac)] = {"radius": r["radius"], "status": r["status"]}

    sig = (resp > GD.NOISE_FACTOR * floor) & (resp > FRACS[-1] * max(ref, 0.0))
    ratio = np.where(sig, resp2 / np.where(resp > 0, resp, 1.0), np.nan)
    bands = []
    for lo, hi in zip(LIN_BANDS[:-1], LIN_BANDS[1:]):
        s = sig & (dist >= lo) & (dist < hi)
        bands.append({"lo": lo, "hi": hi, "n": int(s.sum()),
                      "median": float(np.median(ratio[s])) if s.any() else float("nan")})

    # How many significant particles are under the float32 strain-increment
    # floor: 1 + dt*C_ii rounds to 1 when |dt*C_ii| < ulp(1)/2. A row where this
    # is large is measuring rounding, whatever its linearity says.
    ulp1 = float(np.spacing(np.float32(1.0)))
    diag = np.abs(np.diagonal(B["C"][k], axis1=1, axis2=2)).max(axis=1)
    rounds = substep_dt * diag < 0.5 * ulp1
    lin_all = ratio[sig]
    return {
        "ref": ref, "noise_rest": noise_rest, "noise_kicked": noise_kicked,
        "quant": quant, "floor": floor, "radii": radii,
        "linearity_median": float(np.median(lin_all)) if lin_all.size else float("nan"),
        "linearity_bands": bands,
        "rounding_frac": float(rounds[sig].mean()) if sig.any() else float("nan"),
        "n_significant": int(sig.sum()),
        "profile_centres": prof["centres"].tolist(),
        "profile_max": (prof["max"] / ref).tolist() if ref > 0 else [],
    }


def _bands_linear(cell) -> bool:
    populated = [b for b in cell["linearity_bands"] if b["n"] >= MIN_BAND_N]
    return bool(populated) and all(abs(b["median"] - 2.0) <= LIN_TOL for b in populated)


def run_row(out: str, *, config: str, level: str, mu: float, lam: float, rho: float,
            n_substeps: int) -> None:
    cfg = CONFIGS[config]
    dmp = damping_params(level, rho)
    inst = ME.Instrument(mu=mu, lam=lam, rho=rho, n_substeps=n_substeps,
                         frame_dt=FRAME_DT, gravity=cfg["gravity"], quiet=True)
    inst.set_damping(dmp["kind"], eta=dmp["eta"], alpha=dmp["alpha"])
    inst.init_slab(centre=cfg["centre"])

    t0 = time.time()
    settle_frames = int(round(cfg["settle_s"] / FRAME_DT))
    for f in range(settle_frames):
        inst.advance_frame(1)
        if cfg["assert_no_contact"]:
            _require_no_contact(inst, f"at settle frame {f}")
    snap = _snapshot(inst)
    if not np.all(np.isfinite(snap["x"])):
        raise RuntimeError("non-finite positions after settling -- solver diverged")
    base = snap["x"].astype(np.float64)
    t_settle = time.time() - t0

    A = _trajectory(inst, snap, cfg)
    A2 = _trajectory(inst, snap, cfg)
    cells = []
    for probe, pid in _probe_sites(base, inst.dx).items():
        dist = np.linalg.norm(base - base[pid], axis=1)
        patch = dist < PATCH_RADIUS_CELLS * inst.dx
        for axis_name, axis in (("x", 0), ("z", 2)):
            B = _trajectory(inst, snap, cfg, (patch, axis, DV))
            Bp = _trajectory(inst, snap, cfg, (patch, axis, DV))
            B2 = _trajectory(inst, snap, cfg, (patch, axis, 2.0 * DV))
            for k, frames in enumerate(SNAPSHOT_FRAMES):
                for q in ("v", "x"):
                    cells.append({"probe": probe, "axis": axis_name, "quantity": q,
                                  "frames": frames, "dt_record": frames * FRAME_DT,
                                  "patch_size": int(patch.sum()),
                                  **_measure(dist, patch, A, A2, B, Bp, B2, q, k, inst.substep_dt)})

    ext = base.max(axis=0) - base.min(axis=0)
    json.dump({
        "config": config, "level": level, "mu": mu, "lam": lam, "rho": rho,
        "damping": dmp, "n_substeps": n_substeps, "substep_dt": inst.substep_dt,
        "frame_dt": FRAME_DT, "gravity": cfg["gravity"], "settle_frames": settle_frames,
        "c_p": float(DMP.p_wave_speed(mu, lam, rho)), "c_s": float(DMP.s_wave_speed(mu, rho)),
        "extent_m": ext.tolist(), "max_distance_m": float(np.linalg.norm(ext)),
        "dx": inst.dx, "backend": inst.backend, "dv": DV,
        "wall_settle_s": t_settle, "wall_total_s": time.time() - t0,
        "cells": cells,
    }, open(out, "w"))


# --------------------------------------------------------------------------
# Child: selftest -- the instrument, before any result from it counts
# --------------------------------------------------------------------------

def run_selftest(out: str, *, mu: float, lam: float, rho: float, n_substeps: int) -> None:
    r = {}
    inst = ME.Instrument(mu=mu, lam=lam, rho=rho, n_substeps=n_substeps,
                         frame_dt=FRAME_DT, gravity=0.0, quiet=True)
    centre = CONFIGS["clean"]["centre"]
    ulp = float(np.spacing(np.float32(0.5)))

    # CHECK 1. WHY: catches a damping kernel that perturbs the solver when it
    # should add nothing -- a wrong stencil, a stray write, a reordered phase.
    # Kelvin-Voigt at eta = 0 runs the full extra scatter adding zeros; it must
    # sit within a small multiple of the VENDORED solver's own run-to-run floor
    # (bit-identity is the wrong bar on Metal, section 9.7 check 1).
    inst.init_slab(centre=centre, stretch=1.2)
    snap = _snapshot(inst)

    def positions_after(stepper, n=24):
        _restore(inst, snap)
        for _ in range(n):
            stepper()
        inst.ti.sync()
        return inst.m.F_x.to_numpy().copy()

    fused = lambda: inst.m.substep(ME.SCALE, ME.THRESHOLD)   # noqa: E731
    inst.set_damping(ME.DAMPING_NONE)
    a1, a2 = positions_after(fused), positions_after(fused)
    inst.set_damping(ME.DAMPING_KV, eta=0.0)
    b = positions_after(inst.substep_split)
    floor, delta = float(np.abs(a1 - a2).max()), float(np.abs(a1 - b).max())
    tol = max(4.0 * floor, 4.0 * ulp)
    r["kv_eta0_matches_solver"] = dict(
        passed=bool(delta <= tol),
        detail=f"24 substeps: KV(eta=0) vs vendored {delta/ulp:.1f} ulp; solver's own "
               f"floor {floor/ulp:.1f} ulp (tolerance {tol/ulp:.1f} ulp)")

    # CHECK 2. WHY: Kelvin-Voigt damps strain RATE, so a rigid translation
    # (C = 0) must lose nothing -- this is the property that makes it the
    # physical form. A kernel damping velocity instead of velocity gradient would
    # pass check 1 and fail here. Bar: 10x section 9.7's rigid-translation bar.
    kv_eta = damping_params("kv_fast", rho)["eta"]
    alpha = damping_params("mass_fast", rho)["alpha"]
    vel = [0.1, -0.05, 0.02]

    def rigid_energy_ratio(kind, **kw):
        inst.set_damping(kind, **kw)
        inst.init_slab(centre=centre, velocity=vel)
        e0 = inst.ledger()["total"]
        inst.advance_frame(1)
        return inst.ledger()["total"] / e0

    RIGID_BAR = 1e-5
    kv_ratio = rigid_energy_ratio(ME.DAMPING_KV, eta=kv_eta)
    r["kv_leaves_rigid_translation"] = dict(
        passed=bool(abs(kv_ratio - 1.0) <= RIGID_BAR),
        detail=f"eta={kv_eta:.1f} Pa*s: E(T)/E(0) = {kv_ratio:.9f} "
               f"(|change| must be <= {RIGID_BAR:g})")

    # CHECK 3. WHY: the same invariance test must be able to FAIL (CLAUDE.md), and
    # mass damping is the known violator. It must fail the rigid bar AND match
    # its closed form exp(-2 alpha T) -- which also proves the mass kernel's
    # per-substep factor is compounded the right number of times.
    mass_ratio = rigid_energy_ratio(ME.DAMPING_MASS, alpha=alpha)
    predicted = float(np.exp(-2.0 * alpha * FRAME_DT))
    r["mass_fails_rigid_and_matches_closed_form"] = dict(
        passed=bool(abs(mass_ratio - 1.0) > RIGID_BAR
                    and abs(mass_ratio / predicted - 1.0) <= 1e-3),
        detail=f"alpha={alpha:.1f}/s: E(T)/E(0) = {mass_ratio:.6f}, closed form "
               f"{predicted:.6f} (rigid bar violated as it must be)")

    # CHECK 4. WHY: catches a sign error (viscosity INJECTING energy) or a gross
    # scale error (a missing inv_dx^2 is 4096x) in the viscous scatter.
    # Zero stiffness, a non-affine shear velocity v_x(z), damped vs undamped over
    # one frame. Only the ORDER OF MAGNITUDE of the extra decay is checked: the
    # mode's wavelength is the slab thickness (~3 cells), where the grid's
    # effective Laplacian is well below the continuum k^2, so a tight bar would
    # be testing the discretisation, not the kernel. Stated as such.
    keep = (inst.m.mu[None], inst.m.la[None])
    inst.m.mu[None], inst.m.la[None] = 0.0, 0.0
    kv_slow = damping_params("kv_slow", rho)["eta"]

    def shear_energy(kind, **kw):
        inst.set_damping(kind, **kw)
        inst.init_slab(centre=centre, shear_mode=0.05)
        x = inst.m.F_x.to_numpy()
        e0 = inst.ledger()["total"]
        inst.advance_frame(1)
        return inst.ledger()["total"] / e0, float(x[:, 2].max() - x[:, 2].min())

    und, span = shear_energy(ME.DAMPING_NONE)
    dmp, _ = shear_energy(ME.DAMPING_KV, eta=kv_slow)
    inst.m.mu[None], inst.m.la[None] = keep
    measured_rate = -np.log(dmp / und) / (2.0 * FRAME_DT) if dmp > 0 and und > 0 else float("nan")
    predicted_rate = DMP.kv_velocity_decay_rate(kv_slow, span, rho, DMP.S_DIFFUSION)
    ratio = measured_rate / predicted_rate
    r["kv_dissipates_at_order_of_predicted_rate"] = dict(
        passed=bool(dmp < und and 0.1 <= ratio <= 10.0),
        detail=f"zero stiffness, shear mode over {span*1e3:.0f} mm: extra velocity decay "
               f"{measured_rate:.1f}/s vs continuum {predicted_rate:.1f}/s "
               f"(ratio {ratio:.2f}, must be in [0.1, 10] and damped < undamped)")

    # CHECK 5. WHY: the probe must respond LINEARLY and above its floor IN EVERY
    # DISTANCE BAND, or its radii describe rounding or noise. An overall median
    # hid exactly this once: 0.01 m/s gave 2.14 near the kick and 1.17 beyond
    # 120 mm, because far-field strain increments rounded away in float32.
    inst.set_damping(ME.DAMPING_NONE)
    inst.init_slab(centre=centre)
    snap = _snapshot(inst)
    base = snap["x"].astype(np.float64)
    cfg = CONFIGS["clean"]
    pid = _probe_sites(base, inst.dx)["centre"]
    dist = np.linalg.norm(base - base[pid], axis=1)
    patch = dist < PATCH_RADIUS_CELLS * inst.dx
    A, A2 = _trajectory(inst, snap, cfg), _trajectory(inst, snap, cfg)
    B = _trajectory(inst, snap, cfg, (patch, 0, DV))
    Bp = _trajectory(inst, snap, cfg, (patch, 0, DV))
    B2 = _trajectory(inst, snap, cfg, (patch, 0, 2 * DV))
    k80 = SNAPSHOT_FRAMES.index(3)
    cell = _measure(dist, patch, A, A2, B, Bp, B2, "v", k80, inst.substep_dt)
    band_txt = ", ".join(f"{b['lo']*1e3:.0f}-{b['hi']*1e3:.0f}mm {b['median']:.3f} (n={b['n']})"
                         for b in cell["linearity_bands"])
    r["probe_is_linear_in_every_band"] = dict(
        passed=bool(_bands_linear(cell)
                    and cell["radii"]["0.01"]["status"] in (GD.RESOLVED, GD.SATURATED)),
        detail=f"kick {DV} m/s, patch {int(patch.sum())} particles, 1/80 s: |B2-A|/|B-A| "
               f"by band: {band_txt} (each must be 2 +- {LIN_TOL}); rounding frac "
               f"{cell['rounding_frac']:.1%}; floor {cell['floor']:.2e} (rest "
               f"{cell['noise_rest']:.1e}, kicked {cell['noise_kicked']:.1e}); r_1% "
               f"{cell['radii']['0.01']['radius']*1e3:.0f} mm ({cell['radii']['0.01']['status']})")

    # CHECK 6. WHY: the refusal path end to end, on REAL nonzero solver noise.
    # The KICKED control pair's difference is fed in as if it were a response,
    # with the source response set equal to the floor: a 1% threshold then sits
    # under the floor and the pipeline must say below_noise. (The first version
    # used the at-rest pair, which is identically zero, and so never reached the
    # refusal branch at all.)
    noise_field = np.linalg.norm(Bp["v"][k80] - B["v"][k80], axis=1)
    quant = float(np.spacing(np.float32(np.abs(B["v"][k80]).max())))
    floor = max(float(noise_field.max()), quant)
    res = GD.influence_radius(GD.influence_profile(dist, noise_field, BIN_WIDTH),
                              ref=floor, frac=0.01, noise_floor=floor)
    r["noise_only_is_refused"] = dict(
        passed=bool(res["status"] == GD.BELOW_NOISE and noise_field.max() > 0),
        detail=f"kicked control pair: max|B'-B| {noise_field.max():.2e} m/s (must be > 0), "
               f"status {res['status']} (must be {GD.BELOW_NOISE})")

    json.dump(r, open(out, "w"))


# --------------------------------------------------------------------------
# Parent: dispatch, tables, figures
# --------------------------------------------------------------------------

def dispatch(cmd, timeout=ROW_TIMEOUT_S):
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, time.time() - t0, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, time.time() - t0, f"timed out after {timeout}s"


def _child_cmd(kind, out, mat, n_sub, config="clean", level="none"):
    mu, lam, rho = mat
    return [sys.executable, os.path.abspath(__file__), "--child", kind, "--out", out,
            "--material", f"{mu:.10g}", f"{lam:.10g}", f"{rho:.10g}",
            "--row", str(n_sub), "--config", config, "--level", level]


def _fmt_radius(rad: dict) -> str:
    s = rad["status"]
    if s == GD.RESOLVED:
        return f"{rad['radius']*1e3:.0f}"
    if s == GD.SATURATED:
        return f">={rad['radius']*1e3:.0f}"
    return "n/r" if s == GD.BELOW_NOISE else "none"


def _pivot(rows, probe, axis, quantity, frac):
    """(config, material, level) x dt_record -> radius string."""
    lines = []
    frames = SNAPSHOT_FRAMES
    head = "| config | material | damping | " + " | ".join(
        f"1/{round(1 / (f * FRAME_DT))} s" for f in frames) + " | c_p x 1/80 s |"
    lines += [head, "|" + "---|" * (4 + len(frames))]
    for row in rows:
        cells = {c["frames"]: c for c in row["cells"]
                 if c["probe"] == probe and c["axis"] == axis and c["quantity"] == quantity}
        vals = [_fmt_radius(cells[f]["radii"][str(frac)]) if f in cells else "-" for f in frames]
        lines.append(f"| {row['config']} | {row['material']} | {row['level']} | "
                     + " | ".join(vals) + f" | {row['c_p'] / 80 * 1e3:.0f} |")
    return "\n".join(lines)


def _linearity_table(rows):
    """Worst per-band linearity over every velocity cell of a row, how many cells
    fail the band test, the worst floor/ref, and the worst rounding fraction."""
    lines = ["| config | material | damping | worst band median | cells failing band test "
             "| worst floor/ref | worst rounding frac |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        vs = [c for c in row["cells"] if c["quantity"] == "v"]
        meds = [b["median"] for c in vs for b in c["linearity_bands"]
                if b["n"] >= MIN_BAND_N and np.isfinite(b["median"])]
        worst = max(meds, key=lambda m: abs(m - 2.0)) if meds else float("nan")
        n_bad = sum(1 for c in vs if not _bands_linear(c))
        nr = [c["floor"] / c["ref"] for c in vs if c["ref"] > 0]
        rf = [c["rounding_frac"] for c in vs if np.isfinite(c["rounding_frac"])]
        lines.append(f"| {row['config']} | {row['material']} | {row['level']} | {worst:.3f} | "
                     f"{n_bad}/{len(vs)} | {max(nr) if nr else float('nan'):.1e} | "
                     f"{max(rf) if rf else float('nan'):.1%} |")
    return "\n".join(lines)


def plot_config(rows, config, probe, axis, path):
    """Small multiples: rows = material, cols = damping. One line per recording
    interval, ordinal blue ramp. Response / patch response, log y; hairlines at
    the 1% and 0.1% thresholds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats = [m for m in material_corners() if any(r["material"] == m for r in rows)]
    levels = [lv for lv in DAMPING_LEVELS if any(r["level"] == lv for r in rows)]
    sel = {(r["material"], r["level"]): r for r in rows if r["config"] == config}
    if not sel:
        return None

    dpi = 150
    px = 72.0 / dpi                         # one pixel in points at this dpi
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
        "axes.linewidth": 1 * px, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(len(mats), len(levels), figsize=(2.3 * len(levels), 1.9 * len(mats) + 0.9),
                             sharex=True, sharey=True, squeeze=False, dpi=dpi)
    handles = []
    for i, mat in enumerate(mats):
        for j, lv in enumerate(levels):
            ax = axes[i][j]
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 50)      # stiff/mid ring up to ~20x ABOVE the source; do not clip it
            ax.grid(True, which="major", axis="y", color=GRID, linewidth=1 * px)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for frac in FRACS:
                ax.axhline(frac, color=MUTED, linewidth=1 * px, zorder=1)
            row = sel.get((mat, lv))
            if row is None:
                continue
            cells = sorted((c for c in row["cells"] if c["probe"] == probe and c["axis"] == axis
                            and c["quantity"] == "v"), key=lambda c: c["frames"])
            for c, color in zip(cells, RAMP):
                if not c["profile_max"]:
                    continue
                x = np.array(c["profile_centres"]) * 1e3
                y = np.array(c["profile_max"], dtype=float)
                y[y <= 0] = np.nan
                (h,) = ax.plot(x, y, color=color, linewidth=2 * px, solid_capstyle="round",
                               solid_joinstyle="round", zorder=3,
                               label=f"1/{round(1 / c['dt_record'])} s")
                if i == 0 and j == 0:
                    handles.append(h)
            if i == 0:
                ax.set_title(lv.replace("_", " "), fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(f"{mat}\nresponse / source", fontsize=8)
            if i == len(mats) - 1:
                ax.set_xlabel("distance from kick (mm)", fontsize=8)
    for frac, label in zip(FRACS, ("1%", "0.1%")):
        axes[0][-1].annotate(label, xy=(1.0, frac), xycoords=("axes fraction", "data"),
                             xytext=(3, 0), textcoords="offset points", va="center",
                             fontsize=7, color=MUTED)
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
               title="recording interval", title_fontsize=8, fontsize=8,
               bbox_to_anchor=(0.01, 1.0))
    fig.suptitle(f"Velocity response to a {DV*100:.0f} cm/s kick along {axis}, {probe} probe, "
                 f"{config.replace('_', ' ')} config", x=0.99, ha="right", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run only the gate")
    ap.add_argument("--no-selftest", action="store_true", help="skip the gate (not advised)")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild tables and figures from the JSON rows already in --runs; no solver")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--materials", nargs="+", default=list(material_corners()),
                    choices=list(material_corners()))
    ap.add_argument("--levels", nargs="+", default=list(DAMPING_LEVELS),
                    choices=list(DAMPING_LEVELS))
    ap.add_argument("--runs", default=os.path.join(REPO, "runs", "locality"))
    ap.add_argument("--figures", default=os.path.join(REPO, "figures", "phase0"))
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--row", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--config", default="clean", help=argparse.SUPPRESS)
    ap.add_argument("--level", default="none", help=argparse.SUPPRESS)
    ap.add_argument("--material", type=float, nargs=3, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    # THE CHILD BRANCH -- never dispatches, so recursion cannot happen.
    if args.child:
        mu, lam, rho = args.material
        if args.child == "row":
            run_row(args.out, config=args.config, level=args.level, mu=mu, lam=lam,
                    rho=rho, n_substeps=args.row)
        elif args.child == "selftest":
            run_selftest(args.out, mu=mu, lam=lam, rho=rho, n_substeps=args.row)
        else:
            raise SystemExit(f"unknown child kind {args.child}")
        return 0

    # THE PARENT must not have compiled anything. Checked, not assumed.
    assert "MPM.mpm3d" not in sys.modules, "parent imported the solver; children exist to isolate it"

    mats = material_corners()
    t0 = time.time()
    if args.replot:
        rows = []
        for config in args.configs:
            for mat_name in args.materials:
                for lv in args.levels:
                    p = os.path.join(args.runs, f"{config}_{mat_name}_{lv}.json")
                    if os.path.isfile(p):
                        row = json.load(open(p))
                        row["material"] = mat_name
                        rows.append(row)
        if not rows:
            raise SystemExit(f"no rows in {args.runs} -- run the sweep first")
        _write_report(rows, args)
        return 0
    print("=" * 96)
    print("MPM LOCALITY STUDY -- DECISION_LOG.md section 13.2".center(96))
    print("=" * 96)
    print(f"dx = {DX*1e3:.2f} mm, frame_dt = 1/{round(1/FRAME_DT)} s, snapshots after "
          f"{SNAPSHOT_FRAMES} frames, kick {DV} m/s on particles within "
          f"{PATCH_RADIUS_CELLS} dx")
    for name, (mu, lam, rho) in mats.items():
        n = substeps_for_material(mu, lam, rho, args.levels)
        print(f"  {name:>5}: mu={mu:8.0f} Pa  lambda={lam:9.0f} Pa  rho={rho:.0f}  "
              f"c_p={float(DMP.p_wave_speed(mu, lam, rho)):5.1f} m/s  "
              f"{n} substeps/frame for every damping level")
    for lv in args.levels:
        p = damping_params(lv, 1050.0)
        print(f"  damping {lv:>9}: {p['kind']:<12} eta={p['eta']:7.2f} Pa*s  "
              f"alpha={p['alpha']:6.1f} /s  tau={p['tau']}")

    scratch = tempfile.mkdtemp(prefix="locality_")
    if not args.no_selftest:
        mu, lam, rho = mats["soft"]
        n = substeps_for_material(mu, lam, rho, DAMPING_LEVELS)
        out = os.path.join(scratch, "selftest.json")
        print(f"\n{'SELFTEST -- the instrument, before any result from it counts':^96}")
        rc, wall, err = dispatch(_child_cmd("selftest", out, mats["soft"], n))
        if rc != 0 or not os.path.isfile(out):
            print(f"  selftest child failed (rc={rc}, {wall:.0f}s)\n{err[-3000:]}")
            return 1
        res = json.load(open(out))
        ok = True
        for name, c in res.items():
            ok &= bool(c["passed"])
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}\n         {c['detail']}")
        print(f"  SELFTEST {'PASSED' if ok else 'FAILED'} ({wall:.0f}s)")
        if not ok:
            print("  Refusing to run the sweep. The selftest is a gate.")
            return 1
        if args.selftest:
            return 0

    os.makedirs(args.runs, exist_ok=True)
    rows = []
    for config in args.configs:
        for mat_name in args.materials:
            mu, lam, rho = mats[mat_name]
            n = substeps_for_material(mu, lam, rho, args.levels)
            for lv in args.levels:
                out = os.path.join(args.runs, f"{config}_{mat_name}_{lv}.json")
                rc, wall, err = dispatch(_child_cmd("row", out, mats[mat_name], n, config, lv))
                if rc != 0 or not os.path.isfile(out):
                    tail = err.strip().splitlines()[-1][:120] if err.strip() else "?"
                    print(f"  {config:>12} {mat_name:>5} {lv:>9}: FAILED rc={rc} ({wall:.0f}s): {tail}")
                    continue
                row = json.load(open(out))
                row["material"] = mat_name
                rows.append(row)
                c = next(c for c in row["cells"] if c["probe"] == "centre" and c["axis"] == "x"
                         and c["quantity"] == "v" and c["frames"] == 3)
                print(f"  {config:>12} {mat_name:>5} {lv:>9}: {wall:5.0f}s  centre/x/v at 1/80 s "
                      f"r_1% {_fmt_radius(c['radii']['0.01']):>6} mm  "
                      f"linearity {c['linearity_median']:.3f}")

    _write_report(rows, args)
    print(f"Total wall time: {time.time() - t0:.0f}s")
    return 0


def _write_report(rows, args):
    report = ["# Locality study -- raw tables (DECISION_LOG.md section 13.2)", "",
              f"Radii in mm. `>=N` = saturated (influence reached the far edge; lower bound). "
              f"`n/r` = threshold below noise floor. `none` = nothing cleared the threshold.", ""]
    for probe in ("centre", "edge"):
        for axis in ("x", "z"):
            for frac in FRACS:
                report += [f"## {probe} probe, kick along {axis}, velocity, {frac:.1%} radius", "",
                           _pivot(rows, probe, axis, "v", frac), ""]
    report += ["## Linearity and noise (velocity cells)", "", _linearity_table(rows), ""]
    with open(os.path.join(args.runs, "tables.md"), "w") as fh:
        fh.write("\n".join(report))

    print("\n" + _pivot(rows, "centre", "x", "v", 0.01))
    print("\n" + _linearity_table(rows))
    figs = []
    for config in args.configs:
        for axis in ("x", "z"):
            p = plot_config(rows, config, "centre", axis,
                            os.path.join(args.figures, f"locality_{config}_{axis}.png"))
            if p:
                figs.append(p)
    print(f"\nTables: {os.path.join(args.runs, 'tables.md')}")
    print("Figures:\n  " + "\n  ".join(figs))


if __name__ == "__main__":
    raise SystemExit(main())
