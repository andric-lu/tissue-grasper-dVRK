#!/usr/bin/env python3
"""
substep_study.py -- is the MPM substep converged, or merely stable?

Run on the HOST (Taichi/Metal, see CLAUDE.md):

    conda activate tissue-host
    python host/substep_study.py                      # both materials, ~4 min
    python host/substep_study.py --frames 12 --materials soft    # quick check

THE QUESTION THIS ANSWERS
-------------------------
`mpm_adapter.py` picks its substep from the P-wave advisory in
`materials.suggested_substep_dt(..., lam=...)` with `safety = 0.3`. Nothing has
ever tested that 0.3. `check_substep_is_stable_for_stiffness` is careful to say
so in its own PASS message:

    Passing means "not obviously unstable", not "converged". Only a timestep
    study establishes the latter.

This is that study. It is the MPM counterpart of container/timestep_study.py,
which asks the same question of PyBullet and is still unrun (section 4).

Two ways 0.3 can be wrong, costing different things:

  TOO LOOSE -- episodes are stable but under-resolved. The deformation is a
    solver artifact rather than a property of the material. 17 August showed
    what that looks like at its worst: plausible output right up until det(F)
    went non-finite.

  TOO TIGHT -- every episode costs several times more than it needs to, for
    every dataset collected from here on.

METHOD
------
Hold EVERYTHING fixed except the substep count: same material, same seed, same
frame_dt, same recorded subset, same number of frames. Sweep n_substeps from a
quarter of the advisory count to eight times it, and watch whether the answer
stops changing.

Both materials from data_mpm/ are swept, not one. The advisory is a FORMULA
that scales with the material; testing it at a single material tests a
constant. The stiff-lambda case is the one where lambda/mu = 754 makes the
P-wave bound differ from the bar wave by 16x, and where the vendored 500 us
substep diverged outright.

WHY THE PRIMARY MEASURE IS A FIELD DIFFERENCE, NOT A PEAK
---------------------------------------------------------
`safety_strain` is a MAXIMUM over 3,000 particles -- a single order statistic.
It can move several percent between runs because one particle overtook another,
which reads as non-convergence when the field is actually fine. So the criterion
is the RMS difference in final particle position against the finest run,
normalised by how far the tissue moved in the first place. Peak stretch is still
reported, because it is the number the dataset carries.

That comparison is only meaningful because particle i is the SAME particle in
every run. mpm3d.py calls ti.init() with no random_seed (line 22), so Taichi's
RNG starts at 0 in every process and init_cube() lays out an identical cloud
every time; MPMRecorder's subset comes from np.random.default_rng(seed), so a
fixed seed selects the same particles. DECISION_LOG.md section 9.5 recorded that
fixed seed as a LIMITATION, because it narrows dataset diversity. Here it is the
opposite: it hands us a particle correspondence that convergence studies
normally have to work for. The parent asserts it rather than trusting it --
if Taichi's seeding ever changes, this must fail loudly, not silently compare
different lumps of material.

ONE PROCESS PER ROW
-------------------
Same constraint as the collector, for the same reason: Taichi bakes `dt` and
`p_mass` into kernels at compile time, so a row cannot change either after the
first substep. `--row N` marks a child and, exactly like `--index` in
mpm_adapter.py, makes recursion structurally impossible -- the parent always
passes it, and a process holding it never dispatches.

Children run with check=False. The coarse rows are EXPECTED to diverge; that is
what the study is for. A crash, a non-finite field, a timeout or a short file is
recorded as `diverged` and the sweep continues. A study that dies on its first
coarse row has tested nothing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))
sys.path.insert(0, os.path.join(REPO, "src"))

import materials                                          # noqa: E402
# n_grid ONLY -- MPM.config imports nothing, so this does NOT run ti.init().
# Reading dx from the solver's own config rather than hardcoding 1/64 is the
# section 3.4 rule: code should measure the world it operates in.
from MPM.config import n_grid                             # noqa: E402

FRAME_DT = 0.0125
DX = 1.0 / n_grid
BASE_SAFETY = 0.3          # the value under test
CONVERGED_AT = 0.05        # 5%, matching container/timestep_study.py
# Multiples of the advisory substep COUNT. Spans a factor of 32 so at least one
# row is expected to fail outright and at least one to be clearly converged; a
# sweep where every row agrees has not tested anything.
MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# Per-row wall-clock ceiling. The finest stiff row is ~850 substeps/frame, so
# this is generous; it exists so a hang is recorded as a divergence rather than
# stopping the study forever.
ROW_TIMEOUT_S = 900


def advisory_n_substeps(mu: float, lam: float, rho: float) -> int:
    """The substep count the collector would choose for this material."""
    dt = float(materials.suggested_substep_dt(mu, rho, DX, safety=BASE_SAFETY,
                                              lam=lam))
    return max(1, int(np.ceil(FRAME_DT / dt)))


def sweep_for(mu: float, lam: float, rho: float, multipliers=MULTIPLIERS):
    """(n_substeps, implied_safety) rows, coarsest first, duplicates removed."""
    n_adv = advisory_n_substeps(mu, lam, rho)
    rows = []
    for mult in multipliers:
        n = max(1, int(round(n_adv * mult)))
        if n not in [r[0] for r in rows]:
            rows.append((n, BASE_SAFETY * n_adv / n))
    return sorted(rows), n_adv


# --------------------------------------------------------------------------
# Child: one row
# --------------------------------------------------------------------------

def run_row(path: str, *, mu: float, lam: float, rho: float,
            n_substeps: int, frames: int, seed: int) -> None:
    """Collect one episode at a forced substep count. Runs in its own process.

    Deliberately goes through mpm_adapter.record_episode rather than driving the
    solver directly: the study must measure what the COLLECTOR does, and a
    parallel copy of the drive loop would eventually drift from it. The episode
    it writes is a real v2.1 file and validate_dataset.py can be pointed at it.
    """
    import mpm_adapter                                    # noqa: E402
    mpm_adapter.record_episode(path, n_steps=frames, seed=seed, quiet=True,
                               material=(mu, lam, rho), n_substeps=n_substeps)


# --------------------------------------------------------------------------
# Parent: diagnostics
# --------------------------------------------------------------------------

def diagnose(path: str, frames: int) -> dict:
    """Pull the convergence numbers out of one row's episode.

    Every failure mode lands here as `diverged` rather than an exception: a
    missing file (the child crashed before writing), a short file (it crashed
    partway and __exit__ saved what it had), or non-finite state.
    """
    from trajectory_io import load_trajectory

    if not os.path.isfile(path):
        return {"diverged": True, "why": "no file -- child died before writing"}
    try:
        tr = load_trajectory(path)
    except Exception as e:                                # noqa: BLE001
        return {"diverged": True, "why": f"unreadable: {type(e).__name__}"}

    pos = tr.tissue_pos.astype(np.float64)
    if len(tr) < frames:
        return {"diverged": True,
                "why": f"stopped at frame {len(tr)}/{frames}",
                "pos0": pos[0] if len(pos) else None}
    if not np.isfinite(pos).all():
        return {"diverged": True, "why": "non-finite positions", "pos0": pos[0]}

    F = tr.tissue_F.astype(np.float64) if tr.has_F else None
    if F is None or not np.isfinite(F).all():
        return {"diverged": True, "why": "non-finite F", "pos0": pos[0]}

    J = np.linalg.det(F)
    # Displacement from the initial frame, per particle, over the episode.
    disp = np.linalg.norm(pos - pos[0], axis=2)
    return {
        "diverged": False,
        "pos0": pos[0],
        "pos_final": pos[-1],
        # safety_strain is logged over ALL 24,000 particles (rule 2 in the
        # adapter), so this is the honest full-set peak, not a subset's.
        "peak_stretch": float(tr.safety_strain.max()) if tr.safety_strain.size
                        else float("nan"),
        "max_abs_J_minus_1": float(np.abs(J - 1.0).max()),
        "peak_disp_mm": float(disp.max() * 1000.0),
        # Characteristic motion scale, used to normalise the field error.
        "rms_disp_m": float(np.sqrt((np.linalg.norm(
            pos[-1] - pos[0], axis=1) ** 2).mean())),
        # HORIZONTAL AND VERTICAL, SEPARATELY. Not decoration: Boundary() damps
        # F_grid_v[I][0] and [1] near the floor -- x and y, never z. If the
        # substep dependence comes from that damping, the horizontal field must
        # fail to converge while the vertical one converges. If both fail
        # equally, the cause is somewhere else and this attribution is wrong.
        # See DECISION_LOG.md 9.6.
        "rms_disp_xy_m": float(np.sqrt((np.linalg.norm(
            pos[-1, :, :2] - pos[0, :, :2], axis=1) ** 2).mean())),
        "rms_disp_z_m": float(np.sqrt(((pos[-1, :, 2] - pos[0, :, 2]) ** 2).mean())),
        # SPECIFIC KINETIC ENERGY, as a dissipation probe. MPM's particle-grid
        # transfers lose energy PER TRANSFER, not per unit of simulated time, so
        # halving the substep doubles the number of transfers in the same 12.5 ms
        # and dissipates roughly twice as much. If that is what drives the
        # substep dependence, KE at a fixed simulated time must fall as
        # n_substeps rises. Mass is uniform across particles, so summing v^2 is
        # proportional to KE and the constant cancels in any comparison.
        "ke_final": float((np.linalg.norm(
            tr.tissue_vel.astype(np.float64)[-1], axis=1) ** 2).sum()),
        "ke_peak": float((np.linalg.norm(
            tr.tissue_vel.astype(np.float64), axis=2) ** 2).sum(axis=1).max()),
    }


def study(mu: float, lam: float, rho: float, *, label: str, frames: int,
          seed: int, scratch: str, keep: str = "",
          multipliers=MULTIPLIERS) -> list:
    """Run the whole sweep for one material and print its table."""
    rows, n_adv = sweep_for(mu, lam, rho, multipliers)
    print(f"\n{'=' * 78}")
    print(f"{label}:  mu = {mu:.0f} Pa   lambda = {lam:.0f} Pa   "
          f"rho = {rho:.1f} kg/m^3   lambda/mu = {lam / mu:.0f}")
    print(f"  advisory at safety={BASE_SAFETY}: {n_adv} substeps/frame "
          f"({FRAME_DT / n_adv * 1e6:.1f} us);  {frames} frames, seed {seed}")
    print("=" * 78)
    hdr = (f"{'n_sub':>7} {'substep':>10} {'safety':>8} {'peak stretch':>13} "
           f"{'max|J-1|':>9} {'peak disp':>10} {'KE final':>10} {'wall':>7}")
    print(hdr)
    print("-" * len(hdr))

    out = []
    for n, safety in rows:
        path = os.path.join(scratch, f"row_{label}_{n:05d}.npz")
        cmd = [sys.executable, os.path.abspath(__file__),
               "--row", str(n), "--out", path,
               "--material", f"{mu:.10g}", f"{lam:.10g}", f"{rho:.10g}",
               "--frames", str(frames), "--seed", str(seed)]
        t0 = time.time()
        try:
            # check=False: a diverged row is a RESULT, not an error.
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=ROW_TIMEOUT_S)
            rc = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc, timed_out = -1, True
        wall = time.time() - t0

        d = diagnose(path, frames)
        if timed_out:
            d = {"diverged": True, "why": f"timed out after {ROW_TIMEOUT_S}s"}
        elif rc != 0 and not d["diverged"]:
            d = {"diverged": True, "why": f"child exited {rc}", **d,
                 "diverged_override": True}
        d.update({"n": n, "safety": safety, "substep": FRAME_DT / n,
                  "wall": wall, "rc": rc})
        out.append(d)

        if d["diverged"]:
            print(f"{n:>7} {d['substep'] * 1e6:>8.1f}us {safety:>8.3f} "
                  f"{'DIVERGED':>13} {'':>9} {'':>10} {wall:>6.1f}s")
            print(f"        ^ {d['why']}")
        else:
            print(f"{n:>7} {d['substep'] * 1e6:>8.1f}us {safety:>8.3f} "
                  f"{d['peak_stretch']:>13.5f} {d['max_abs_J_minus_1']:>9.4f} "
                  f"{d['peak_disp_mm']:>7.2f}mm {d['ke_final']:>10.3e} "
                  f"{wall:>6.1f}s")

        if keep and not d["diverged"] and n == n_adv:
            # Keep the advisory row so validate_dataset.py can be pointed at a
            # real study output -- evidence the study drove the collection path.
            os.replace(path, os.path.join(keep, f"advisory_{label}.npz"))
        elif os.path.isfile(path):
            os.remove(path)                                # 7 MB a row otherwise

    return _assess(out, n_adv, label)


def _assess(rows: list, n_adv: int, label: str) -> list:
    """Convergence table and verdict for one material."""
    good = [r for r in rows if not r["diverged"]]
    if len(good) < 2:
        print("\n  Fewer than two rows survived; nothing to compare.")
        return rows

    # THE SHARED-CLOUD ASSERTION. Every row must start from an identical
    # particle cloud or the field comparison below is meaningless.
    ref0 = good[0]["pos0"]
    for r in good[1:]:
        if not np.array_equal(r["pos0"], ref0):
            raise SystemExit(
                f"FATAL: row n={r['n']} starts from a different particle cloud "
                "than the coarsest surviving row. The field comparison assumes "
                "particle i is the same particle in every run, which held "
                "because Taichi's RNG is seeded 0 in every process. That is no "
                "longer true, so this study cannot be trusted -- see 9.6.")

    finest = good[-1]
    scale = max(finest["rms_disp_m"], 1e-12)
    print(f"\n  Against the finest row (n={finest['n']}, "
          f"{finest['substep'] * 1e6:.1f} us). Field error is RMS final-position "
          f"difference\n  over all recorded particles, as a fraction of that "
          f"row's own RMS displacement ({scale * 1000:.2f} mm):")
    scale_xy = max(finest["rms_disp_xy_m"], 1e-12)
    scale_z = max(finest["rms_disp_z_m"], 1e-12)
    h = (f"  {'n_sub':>7} {'safety':>8} {'field err':>11} {'  horiz':>9} "
         f"{'vert':>9} {'d(stretch)':>11}")
    print(h)
    print("  " + "-" * (len(h) - 2))

    for r in good:
        err = float(np.sqrt((np.linalg.norm(
            r["pos_final"] - finest["pos_final"], axis=1) ** 2).mean()))
        err_xy = float(np.sqrt((np.linalg.norm(
            r["pos_final"][:, :2] - finest["pos_final"][:, :2], axis=1) ** 2).mean()))
        err_z = float(np.sqrt(
            ((r["pos_final"][:, 2] - finest["pos_final"][:, 2]) ** 2).mean()))
        r["field_err_rel"] = err / scale
        r["err_xy_rel"] = err_xy / scale_xy
        r["err_z_rel"] = err_z / scale_z
        r["stretch_rel"] = abs(r["peak_stretch"] - finest["peak_stretch"]) / \
            max(abs(finest["peak_stretch"] - 1.0), 1e-9)
        mark = "  <- advisory" if r["n"] == n_adv else ""
        print(f"  {r['n']:>7} {r['safety']:>8.3f} {r['field_err_rel'] * 100:>10.2f}% "
              f"{r['err_xy_rel'] * 100:>8.2f}% {r['err_z_rel'] * 100:>8.2f}% "
              f"{r['stretch_rel'] * 100:>10.2f}%{mark}")

    # Largest substep (fewest substeps, cheapest) that is within tolerance.
    converged = [r for r in good[:-1] if r["field_err_rel"] < CONVERGED_AT]
    print()
    if converged:
        best = converged[0]
        print(f"  CONVERGED at n={best['n']} ({best['substep'] * 1e6:.1f} us, "
              f"safety={best['safety']:.3f}, {best['wall']:.1f}s/episode): field "
              f"error {best['field_err_rel'] * 100:.2f}% < {CONVERGED_AT:.0%}.")
        if best["n"] < n_adv:
            print(f"  The advisory (n={n_adv}, safety={BASE_SAFETY}) is FINER "
                  f"than it needs to be by {n_adv / best['n']:.1f}x.")
        elif best["n"] == n_adv:
            print(f"  That is exactly the advisory. safety={BASE_SAFETY} is the "
                  "largest step that resolves this material.")
    else:
        print("  NOT CONVERGED at any tested substep coarser than the finest.")
        # Before telling anyone to refine further, check whether refining CAN
        # help. MPM's particle-grid transfers dissipate energy per TRANSFER,
        # not per unit of simulated time, so halving the substep doubles the
        # number of transfers in the same frame and roughly doubles the
        # damping. Where that dominates, refinement does not approach a
        # solution -- it walks steadily toward an over-damped one, and "extend
        # the sweep downward" is advice that burns GPU time forever. The KE
        # ratio is what distinguishes the two cases.
        # MONOTONICITY, not magnitude. A ratio threshold is arbitrary and got
        # this wrong once: the two materials here fall 296x and 9x across the
        # same sweep, showing identical behaviour, and a 10x cutoff told one of
        # them to keep refining. Steady energy loss as the substep shrinks is
        # the signature regardless of how much energy there was to lose.
        ke = [r["ke_final"] for r in good]
        ratio = max(ke) / max(min(ke), 1e-30)
        monotone_loss = all(b <= a * 1.02 for a, b in zip(ke, ke[1:]))
        if monotone_loss and len(ke) >= 3:
            print("  AND kinetic energy at a fixed simulated time falls "
                  f"{ratio:.0f}x across the sweep.")
            print("  That is the signature of transfer dissipation, not of a "
                  "solution being approached:")
            print("  more substeps means more particle-grid round trips in the "
                  "same frame, and each")
            print("  one damps. Refining further will not converge -- it will "
                  "go on quietly removing")
            print("  energy. See DECISION_LOG.md 9.6 before spending GPU time "
                  "on a finer sweep.")
        else:
            print("  Extend the sweep downward before trusting data collected "
                  "here.")

    # Monotonicity: called out rather than averaged away.
    errs = [r["field_err_rel"] for r in good[:-1]]
    if errs and any(b > a + 1e-12 for a, b in zip(errs, errs[1:])):
        print("  NOTE: field error is not monotone in n_substeps. Convergence "
              "is still plausible but the sweep deserves a closer look.")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60,
                    help="recorded frames per row, held equal across the sweep")
    ap.add_argument("--materials", default="both",
                    choices=("both", "soft", "stiff"),
                    help="which of data_mpm/'s two materials to sweep")
    ap.add_argument("--multipliers", type=float, nargs="+", default=None,
                    help="multiples of the advisory substep COUNT to test "
                         f"(default {' '.join(str(m) for m in MULTIPLIERS)}). "
                         "Extend upward when the finest row is not itself "
                         "converged -- everything is measured against it.")
    ap.add_argument("--keep", default="",
                    help="directory to keep each material's advisory-row "
                         "episode in, for validate_dataset.py")
    # Child-only. Its presence means "you are one row"; see the module docstring.
    ap.add_argument("--row", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--material", type=float, nargs=3, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    # THE CHILD BRANCH -- never dispatches, so recursion cannot happen.
    if args.row is not None:
        mu, lam, rho = args.material
        run_row(args.out, mu=mu, lam=lam, rho=rho, n_substeps=args.row,
                frames=args.frames, seed=args.seed)
        return 0

    # THE PARENT. Importing mpm_adapter here is safe -- its module-level imports
    # are numpy only (materials, tissue_metrics, trajectory_io), and mpm3d is
    # imported inside MPMRecorder.__init__, which the parent never constructs.
    # That is the line not to cross: importing mpm3d would run ti.init() and
    # compile kernels in the very process the children exist to isolate.
    import mpm_adapter

    # The two materials the collector actually produced, taken from the same
    # seeds, through the same sampler. Labelled by lambda/mu rather than by
    # seed order, because it is the ratio that decides how hard the P-wave
    # bound bites -- and a label that assumed seed 0 was the soft one would
    # quietly lie the moment the sampler changed.
    mats = {}
    for seed in (0, 1):
        mu, lam, rho = mpm_adapter.sample_episode_material(seed)
        mats[seed] = (mu, lam, rho)
    by_ratio = sorted(mats.items(), key=lambda kv: kv[1][1] / kv[1][0])
    labelled = [("soft-lambda", by_ratio[0][0], by_ratio[0][1]),
                ("stiff-lambda", by_ratio[1][0], by_ratio[1][1])]
    if args.materials == "soft":
        labelled = labelled[:1]
    elif args.materials == "stiff":
        labelled = labelled[1:]

    if args.keep:
        os.makedirs(args.keep, exist_ok=True)

    print("=" * 78)
    print("MPM SUBSTEP CONVERGENCE STUDY".center(78))
    print("=" * 78)
    print(f"dx = 1/{n_grid} = {DX * 1e3:.3f} mm (read from MPM/config.py), "
          f"frame_dt = {FRAME_DT * 1e3:.2f} ms")
    print(f"Testing safety = {BASE_SAFETY} in the P-wave advisory. One child "
          f"process per row.")

    mults = tuple(args.multipliers) if args.multipliers else MULTIPLIERS
    t0 = time.time()
    scratch = tempfile.mkdtemp(prefix="substep_study_")
    results = {}
    try:
        for label, seed, (mu, lam, rho) in labelled:
            results[label] = study(mu, lam, rho, label=label,
                                   frames=args.frames, seed=seed,
                                   scratch=scratch, keep=args.keep,
                                   multipliers=mults)
    finally:
        # Rows are ~7 MB each and there are up to twelve of them.
        for f in os.listdir(scratch):
            os.remove(os.path.join(scratch, f))
        os.rmdir(scratch)

    print(f"\n{'=' * 78}")
    print(f"Total wall time: {time.time() - t0:.1f}s")
    if len(results) == 2:
        print("\nBoth materials swept. If their verdicts disagree about the "
              "safety factor,\nthe factor is not material-independent and the "
              "FORMULA needs revisiting --\nwhich is the finding worth having, "
              "and why one material would not do.")
    print("Record the outcome in DECISION_LOG.md 9.6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
