#!/usr/bin/env python3
"""
validate_dataset.py -- read the dataset before the model does.

Run on the HOST:

    conda activate tissue-host
    python host/validate_dataset.py
    python host/validate_dataset.py --data data_synth/
    python host/validate_dataset.py --data data/ --verbose

WHY THIS FILE EXISTS
--------------------
A dynamics model cannot tell you that its training data was wrong. It fits
whatever it is given, reports a plausible loss, and fails later in a way that
looks like an architecture problem. Every hour spent tuning a model on a broken
dataset is an hour spent solving the wrong problem.

So every property that must hold for the data to be worth training on is
written down here as an executable check, and every check carries a `# WHY:`
comment naming the specific failure it catches. A check without a failure story
is a check nobody will trust enough to act on when it goes red.

visualize_trajectory.py is the complement to this: it inspects ONE episode in
depth and needs a human to look at it. This inspects EVERY episode and needs
nobody. Run this first; reach for the visualiser when this points somewhere.

STATUSES
--------
    PASS   the property holds
    WARN   suspicious, but legitimately possible -- look, do not necessarily act
    SKIP   the property is not applicable to this file (see below)
    FAIL   the property is violated; do not train on this data

SKIP IS NOT A SOFT FAIL. v1 episodes predate the deformation gradient, the
material parameters and the target region, and PyBullet's mass-spring cloth has
no constitutive model to have Lame parameters FOR. Those files are legitimate
data collected before the schema grew, not broken files. A check that cannot
apply says SKIP and says why; it does not fail, and it does not quietly pass
either, which would be worse -- a green tick against a property that was never
tested is how a dataset gets trusted for something it never demonstrated.

EXIT CODE is 0 unless something FAILed, so this can gate a collection run.
"""

import argparse
import os
import sys
from typing import Callable, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from materials import suggested_substep_dt, unpack_material  # noqa: E402
from tissue_metrics import (  # noqa: E402
    DEFAULT_GRID,
    DEFAULT_SIGMA,
    DEFAULT_THRESHOLD,
    compute_exposure,
    compute_safety_strain,
)
from trajectory_io import (  # noqa: E402
    CONTACT_GRASP,
    CONTACT_MODE_NAMES,
    CONTACT_NONE,
    list_trajectories,
    load_trajectory,
)

PASS, WARN, SKIP, FAIL = "PASS", "WARN", "SKIP", "FAIL"

# Thresholds. Collected here rather than buried in the checks so they can be
# argued with in one place.
UNSTABLE_SPEED = 1.0        # m/s; matches visualize_trajectory.py deliberately
MIN_MOTION_MM = 1.0         # below this an episode carries no dynamics signal
MAX_VOLUME_DRIFT = 0.05     # 5% -- see check_F_incompressible
METRIC_TOLERANCE = 1e-3     # logged vs recomputed exposure / safety_strain
MIN_MATERIAL_SPREAD = 0.05  # in log units, across the dataset
SUBSTEP_SLACK = 1.5         # how far over the advisory substep is tolerable
MAX_BOUNDARY_DRIFT = 1e-5   # 10 um; a kinematic clamp should be exact, but a
                            # solver that re-solves constraints each substep can
                            # leave float-level residue. Deliberately far below
                            # any real deformation -- see check_boundary_is_held

# The MPM grid spacing the substep check assumes. It is not in the schema, so
# it has to be assumed somewhere; assume it loudly rather than silently.
ASSUMED_MPM_DX = 1.0e-3


# --------------------------------------------------------------------------
# Check registry
# --------------------------------------------------------------------------

class Result:
    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message


PER_EPISODE: List[Callable] = []
PER_DATASET: List[Callable] = []


def check(scope: str = "episode"):
    """Register a check. `scope` is "episode" (one file) or "dataset" (all).

    The decorator exists so that adding a check is one function and one line of
    registration, with no central list to forget to update. The check's name and
    docstring become its label in the report, so both are user-facing.
    """
    def wrap(fn):
        fn.label = fn.__name__.replace("check_", "").replace("_", " ")
        (PER_EPISODE if scope == "episode" else PER_DATASET).append(fn)
        return fn
    return wrap


# --------------------------------------------------------------------------
# Per-episode checks: things that must hold within one file
# --------------------------------------------------------------------------

@check()
def check_shapes_agree(tr) -> Result:
    """Every per-step array has the same length as tissue_pos."""
    # WHY: an off-by-one between positions and actions produces a dataset where
    # every sample is paired with the wrong action. The model still trains, the
    # loss still falls, and the learned dynamics are shifted by one step --
    # which only shows up as a rollout that drifts, long after collection.
    T, N = len(tr), tr.n_particles
    expect = {"tissue_pos": (T, N, 3), "tissue_vel": (T, N, 3),
              "ee_pose": (T, 7), "ee_vel": (T, 6), "jaw": (T,),
              "grasp_active": (T,), "contact_force": (T, 3)}
    bad = [f"{k}{getattr(tr, k).shape}!={v}" for k, v in expect.items()
           if getattr(tr, k).shape != v]
    # `action` and `joint_pos` have a per-episode width the schema does not
    # fix, so only the leading axis can be asserted. Stating that explicitly
    # beats writing the width in terms of itself, which is an assertion that
    # cannot fail and reads as though it were checking something.
    for k in ("action", "joint_pos"):
        arr = getattr(tr, k)
        if arr.ndim != 2 or arr.shape[0] != T:
            bad.append(f"{k}{arr.shape} is not (T={T}, width)")
    # Optional v2 fields are either absent entirely or full length.
    for k in ("contact_mode", "exposure", "safety_strain"):
        arr = getattr(tr, k)
        if arr.size and arr.shape != (T,):
            bad.append(f"{k}{arr.shape}!=({T},)")
    if tr.has_F and tr.tissue_F.shape != (T, N, 3, 3):
        bad.append(f"tissue_F{tr.tissue_F.shape}!=({T},{N},3,3)")
    if bad:
        return Result(FAIL, "; ".join(bad))
    return Result(PASS, f"T={T} N={N} consistent")


@check()
def check_finite(tr) -> Result:
    """No NaN or inf anywhere in the numeric fields."""
    # WHY: one NaN in one node at one step poisons every gradient computed from
    # a batch containing it. The loss goes to NaN, the weights go to NaN, and
    # the traceback points at the optimiser rather than at the episode.
    bad = []
    for k in ("tissue_pos", "tissue_vel", "ee_pose", "ee_vel", "jaw",
              "action", "contact_force", "tissue_F", "exposure", "safety_strain"):
        arr = np.asarray(getattr(tr, k))
        if arr.size and not np.all(np.isfinite(arr)):
            bad.append(f"{k} ({np.count_nonzero(~np.isfinite(arr))} values)")
    return Result(FAIL, "non-finite in " + ", ".join(bad)) if bad \
        else Result(PASS, "all finite")


@check()
def check_no_divergence(tr) -> Result:
    """Node speeds stay physically plausible."""
    # WHY: a mass-spring or MPM solver stepped too coarsely for its stiffness
    # does not error -- it scatters the mesh over a few frames. The run
    # "completes successfully" and the episode is numerical garbage. Retraction
    # happens at centimetres per second; metres per second is divergence.
    speed = np.linalg.norm(tr.tissue_vel.astype(np.float64), axis=2)
    peak = float(speed.max()) if speed.size else 0.0
    if peak > UNSTABLE_SPEED:
        return Result(WARN, f"peak node speed {peak:.2f} m/s > {UNSTABLE_SPEED} "
                            f"at step {int(np.argmax(speed.max(axis=1)))}/{len(tr)-1}; "
                            "confirm with a timestep convergence study")
    return Result(PASS, f"peak node speed {peak:.3f} m/s")


@check()
def check_tissue_actually_moved(tr) -> Result:
    """The tissue deformed enough to carry information."""
    # WHY: a grasp that caught zero nodes, or a gripper that never reached the
    # sheet, still produces a complete, well-formed, entirely useless episode.
    # Nothing else in this file would notice.
    pos = tr.tissue_pos.astype(np.float64)
    moved_mm = float(np.linalg.norm(pos - pos[0], axis=2).max() * 1000)
    if moved_mm < MIN_MOTION_MM:
        return Result(WARN, f"peak displacement {moved_mm:.2f} mm < {MIN_MOTION_MM} mm; "
                            "episode carries almost no dynamics information")
    return Result(PASS, f"peak displacement {moved_mm:.1f} mm")


@check()
def check_grasp_is_consistent(tr) -> Result:
    """grasp_active agrees with the recorded grasp node ids."""
    # WHY: the two are written from different variables in the collector, and
    # one of them is cleared on release. They drift apart silently.
    grasp = tr.grasp_active.astype(bool)
    if not grasp.any():
        return Result(WARN, "grasp never active in this episode")
    empty = [t for t in np.flatnonzero(grasp) if tr.grasp_ids(int(t)).size == 0]
    if empty:
        return Result(FAIL, f"grasp_active True but no node ids at {len(empty)} step(s), "
                            f"first at {empty[0]}")
    return Result(PASS, f"grasped {tr.grasp_ids(int(np.argmax(grasp))).size} node(s) "
                        f"for {int(grasp.sum())}/{len(tr)} steps")


@check()
def check_boundary_is_held(tr) -> Result:
    """Particles marked kinematically clamped do not move."""
    # WHY: this is the file-level detector for the failure that ruined the first
    # dataset in this project -- anchors silently not holding, so the recorded
    # motion is rigid translation with no strain field in it (§3.5). That was
    # found by noticing peak displacement equalled the gripper's own travel,
    # which required knowing the gripper's travel and thinking to compare. This
    # asserts the property directly, from the file alone, referencing nothing
    # outside it.
    #
    # It catches the inverse error too: a mask claiming a clamp the motion does
    # not honour. That is a file lying about itself, and anything trusting
    # boundary_mask believes it -- a graph network treating those particles as
    # fixed anchors, or a loss that excludes them from the residual.
    if tr.boundary_mask.size == 0:
        return Result(SKIP, f"no boundary_mask recorded (schema "
                            f"{tr.schema_version}, simulator {tr.simulator})")
    mask = tr.boundary_mask.astype(bool)
    n = int(mask.sum())
    if n == 0:
        # A legitimate reading, not a missing one -- see the empty-vs-all-False
        # distinction in trajectory_io. Nothing is clamped, so nothing to check.
        return Result(PASS, "boundary_mask recorded; no particles are clamped")
    pos = tr.tissue_pos.astype(np.float64)
    drift = np.linalg.norm(pos[:, mask] - pos[0, mask], axis=2).max(axis=0)
    worst = float(drift.max())
    if worst > MAX_BOUNDARY_DRIFT:
        n_moved = int(np.count_nonzero(drift > MAX_BOUNDARY_DRIFT))
        return Result(FAIL, f"{n_moved} of {n} clamped particle(s) moved; worst "
                            f"{worst*1000:.4f} mm against a "
                            f"{MAX_BOUNDARY_DRIFT*1000:.4f} mm limit -- either the "
                            "constraint is not holding or boundary_mask is wrong")
    return Result(PASS, f"{n} clamped particle(s), max movement "
                        f"{worst*1e6:.3f} um")


@check()
def check_F_admissible(tr) -> Result:
    """det(F) > 0 for every particle at every step."""
    # WHY: a negative determinant is an INVERTED element -- the material has
    # been turned inside out. It is not a large deformation, it is a solver
    # failure, and every strain measure built on it is meaningless: the
    # Neo-Hookean energy takes ln(J) and is undefined for J <= 0. An episode
    # with inversions must be discarded, not clipped.
    if not tr.has_F:
        return Result(SKIP, f"no deformation gradient (schema {tr.schema_version}, "
                            f"simulator {tr.simulator})")
    J = np.linalg.det(tr.tissue_F.astype(np.float64))
    n_bad = int(np.count_nonzero(J <= 0.0))
    if n_bad:
        step = int(np.argmax((J <= 0).any(axis=1)))
        return Result(FAIL, f"{n_bad} inverted element(s), min det(F)={J.min():.4g}, "
                            f"first at step {step}")
    return Result(PASS, f"det(F) in [{J.min():.6f}, {J.max():.6f}], all positive")


@check()
def check_F_incompressible(tr) -> Result:
    """Volume change implied by det(F) is consistent with near-incompressibility."""
    # WHY: soft tissue is mostly water and barely changes volume. A solver that
    # reports 30% volume change is not modelling tissue -- usually the bulk
    # modulus is far too low for the timestep, or lambda was mis-scaled on the
    # way into the solver. SOFT check: a genuinely compressible phantom, or an
    # episode with real cavitation, is legitimate, so this warns and moves on.
    if not tr.has_F:
        return Result(SKIP, "no deformation gradient to check")
    J = np.linalg.det(tr.tissue_F.astype(np.float64))
    drift = float(np.abs(J - 1.0).max())
    if drift > MAX_VOLUME_DRIFT:
        return Result(WARN, f"volume change up to {drift*100:.1f}% "
                            f"(det(F) in [{J.min():.4f}, {J.max():.4f}]); "
                            "expected near-incompressible tissue")
    return Result(PASS, f"max volume change {drift*100:.3f}%")


@check()
def check_substep_is_stable_for_stiffness(tr) -> Result:
    """substep_dt is small enough for this episode's sampled stiffness."""
    # WHY: MPM stability follows the elastic wave speed, so the stable step
    # shrinks like 1/sqrt(E). Randomising stiffness over an order of magnitude
    # means the stiffest episodes need a substep several times smaller than the
    # softest. A collector with one fixed "default" substep produces episodes
    # that are fine at the soft end and silently diverging at the stiff end --
    # and the diverged ones still look like complete files.
    if tr.material_params.size == 0:
        return Result(SKIP, "no material parameters recorded")
    if float(tr.substep_dt) <= 0.0:
        return Result(SKIP, "substep_dt not recorded")
    mu, _, rho = unpack_material(tr.material_params)
    advised = float(suggested_substep_dt(mu, rho, ASSUMED_MPM_DX))
    actual = float(tr.substep_dt)
    ratio = actual / advised
    if ratio > SUBSTEP_SLACK:
        return Result(FAIL, f"substep {actual*1e6:.1f} us is {ratio:.1f}x the advisory "
                            f"{advised*1e6:.1f} us for mu={float(mu):.0f} Pa "
                            f"(dx assumed {ASSUMED_MPM_DX*1e3:.1f} mm)")
    # The advisory itself is an upper bound -- it uses the shear wave speed and
    # ignores the faster pressure wave -- so being well under it is expected.
    return Result(PASS, f"substep {actual*1e6:.1f} us vs advisory "
                        f"{advised*1e6:.1f} us (ratio {ratio:.2f})")


@check()
def check_contact_mode_transitions(tr) -> Result:
    """Contact modes change in a physically reachable order."""
    # WHY: two specific failures. (1) NONE -> GRASP in a single step means the
    # jaws closed on tissue they were not touching -- a labelling bug, since
    # contact has to precede a grasp. (2) A mode that changes on almost every
    # step is not contact, it is a threshold chattering around its cutoff, and
    # a model trained on it learns to predict noise.
    modes = tr.contact_mode
    if modes.size == 0:
        return Result(SKIP, "no contact_mode recorded")
    m = modes.astype(int)
    jumps = np.flatnonzero((m[:-1] == CONTACT_NONE) & (m[1:] == CONTACT_GRASP))
    if jumps.size:
        return Result(FAIL, f"{jumps.size} direct none->grasp transition(s) with no "
                            f"intervening contact, first at step {int(jumps[0])+1}")
    n_changes = int(np.count_nonzero(np.diff(m)))
    if len(tr) > 10 and n_changes > 0.5 * (len(tr) - 1):
        return Result(WARN, f"contact_mode changes on {n_changes}/{len(tr)-1} steps; "
                            "that is chattering, not contact")
    seen = ", ".join(sorted({CONTACT_MODE_NAMES[v] for v in np.unique(m)}))
    return Result(PASS, f"{n_changes} transition(s); modes seen: {seen}")


@check()
def check_logged_metrics_match_recomputation(tr) -> Result:
    """Logged exposure / safety_strain match what the state implies."""
    # WHY: the metrics are stored AND computable, which means there are two
    # sources of truth for the quantity the planner optimises. They drift: a
    # collector pinned to an old sigma, a metric whose default changed, an
    # exposure logged before the last solver substep rather than after. The
    # drift is invisible until a planner optimises a cost the model was never
    # trained against.
    if tr.target_origin.size == 0:
        return Result(SKIP, "no target region recorded")
    if tr.exposure.size == 0 and tr.safety_strain.size == 0:
        return Result(SKIP, "no logged metrics to compare")

    problems = []
    if tr.exposure.size:
        got = compute_exposure(tr.tissue_pos.astype(np.float64), tr.target_origin,
                               tr.target_normal, tr.target_extent,
                               sigma=DEFAULT_SIGMA, threshold=DEFAULT_THRESHOLD,
                               grid=DEFAULT_GRID)
        err = float(np.abs(got - tr.exposure.astype(np.float64)).max())
        if err > METRIC_TOLERANCE:
            problems.append(f"exposure differs by up to {err:.2e}")
    if tr.safety_strain.size:
        if not tr.has_F:
            problems.append("safety_strain logged but no F to recompute it from")
        else:
            got = compute_safety_strain(tr.tissue_F)["max"]
            err = float(np.abs(got - tr.safety_strain.astype(np.float64)).max())
            if err > METRIC_TOLERANCE:
                problems.append(f"safety_strain differs by up to {err:.2e}")
    if problems:
        return Result(FAIL, "; ".join(problems) +
                      f" (tolerance {METRIC_TOLERANCE:.0e})")
    return Result(PASS, f"logged metrics reproduce to {METRIC_TOLERANCE:.0e}")


# --------------------------------------------------------------------------
# Dataset-wide checks: things that can only be seen across episodes
# --------------------------------------------------------------------------

@check(scope="dataset")
def check_episodes_are_comparable(trs) -> Result:
    """Every episode shares a node count, an action width and a timestep."""
    # WHY: train_dynamics.py concatenates every episode into one array. A
    # mismatch there is a crash at best; at worst the shapes happen to align
    # and the model trains on two different meshes as though they were one.
    n = {t.n_particles for t in trs}
    a = {t.action.shape[1] if t.action.ndim > 1 else 1 for t in trs}
    dt = {round(float(t.dt), 9) for t in trs}
    bad = []
    if len(n) > 1:
        bad.append(f"node counts {sorted(n)}")
    if len(a) > 1:
        bad.append(f"action widths {sorted(a)}")
    if len(dt) > 1:
        bad.append(f"timesteps {sorted(dt)}")
    if bad:
        return Result(FAIL, "episodes disagree on " + "; ".join(bad))
    return Result(PASS, f"all {len(trs)} episodes: N={n.pop()} A={a.pop()} dt={dt.pop()}")


@check(scope="dataset")
def check_deformation_is_diverse(trs) -> Result:
    """Episodes differ from one another in how far the tissue moved."""
    # WHY: randomisation that silently does nothing -- a seed not threaded
    # through, a range collapsed to a point -- gives N copies of one episode.
    # The dataset looks the right size and the model appears to train; it has
    # simply memorised a single trajectory and generalises to nothing.
    peaks = np.array([float(np.linalg.norm(
        t.tissue_pos.astype(np.float64) - t.tissue_pos[0].astype(np.float64),
        axis=2).max()) for t in trs])
    if len(peaks) < 2:
        return Result(SKIP, "only one episode; diversity is undefined")
    spread = float(peaks.std() / max(peaks.mean(), 1e-12))
    if spread < 0.01:
        return Result(WARN, f"peak displacement varies by only {spread*100:.2f}% "
                            "across episodes; is randomisation actually active?")
    return Result(PASS, f"peak displacement {peaks.min()*1000:.1f}-{peaks.max()*1000:.1f} mm "
                        f"(spread {spread*100:.0f}%)")


@check(scope="dataset")
def check_material_is_diverse(trs) -> Result:
    """Material parameters actually vary across episodes."""
    # WHY: the same failure as deformation diversity, one level deeper and far
    # harder to see. Every episode can move differently while sharing one
    # stiffness -- the motion differs because the grasp point and direction were
    # randomised. A model trained on that learns THAT stiffness, not the
    # dependence on stiffness, and domain randomisation has bought nothing.
    with_mat = [t for t in trs if t.material_params.size]
    if not with_mat:
        return Result(SKIP, f"no episode records material parameters "
                            f"({len(trs)} file(s), all pre-v2 or non-constitutive)")
    if len(with_mat) < 2:
        return Result(SKIP, "only one episode records material parameters")
    params = np.stack([t.material_params.astype(np.float64) for t in with_mat])
    # log_mu and log_lambda are already in log units, so a plain spread is the
    # right measure; rho is linear and compared relative to its mean.
    spread_log = params[:, :2].std(axis=0)
    spread_rho = params[:, 2].std() / max(abs(params[:, 2].mean()), 1e-12)
    if spread_log.max() < MIN_MATERIAL_SPREAD and spread_rho < 0.001:
        return Result(WARN, f"material barely varies across {len(with_mat)} episode(s): "
                            f"sd(log mu)={spread_log[0]:.4f} "
                            f"sd(log lambda)={spread_log[1]:.4f}; "
                            "is sample_material being called per episode?")
    mu = np.exp(params[:, 0])
    return Result(PASS, f"mu spans {mu.min():.0f}-{mu.max():.0f} Pa across "
                        f"{len(with_mat)} episode(s), sd(log mu)={spread_log[0]:.2f}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_COLOUR = {PASS: "\033[32m", WARN: "\033[33m", SKIP: "\033[90m", FAIL: "\033[31m"}
_RESET = "\033[0m"


def _fmt(status: str, use_colour: bool) -> str:
    return f"{_COLOUR[status]}{status}{_RESET}" if use_colour else status


def run(data_dir: str, verbose: bool = False, colour: bool = True) -> int:
    files = list_trajectories(data_dir)
    if not files:
        print(f"No .npz files in {data_dir}/. Collect some first:\n"
              "  docker compose run --rm surrol python "
              "container/collect_retraction.py --episodes 5\n"
              "or generate synthetic ones:\n"
              "  python src/synthetic_traj.py --out data_synth/ --kinds all")
        return 1

    print(f"validating {len(files)} episode(s) in {data_dir}/\n")
    tally = {PASS: 0, WARN: 0, SKIP: 0, FAIL: 0}
    trs, failed_files = [], []

    for path in files:
        try:
            tr = load_trajectory(path)
        except Exception as exc:                       # noqa: BLE001
            # WHY not let it propagate: one corrupt file must not hide the
            # state of the other 199. Report it and keep going.
            print(f"{os.path.basename(path)}")
            print(f"  {_fmt(FAIL, colour)}  load  {type(exc).__name__}: {exc}")
            tally[FAIL] += 1
            failed_files.append(path)
            continue
        trs.append(tr)

        results = [(fn.label, fn(tr)) for fn in PER_EPISODE]
        worst = FAIL if any(r.status == FAIL for _, r in results) else \
            WARN if any(r.status == WARN for _, r in results) else PASS
        for _, r in results:
            tally[r.status] += 1
        if worst == FAIL:
            failed_files.append(path)

        shown = results if (verbose or worst != PASS) else []
        header = (f"{os.path.basename(path):24s} v{tr.schema_version} "
                  f"{tr.simulator:10s} T={len(tr):4d} N={tr.n_particles:5d}"
                  f"{'  +F' if tr.has_F else ''}")
        print(f"{header}   {_fmt(worst, colour)}")
        for label, r in shown:
            if r.status == PASS and not verbose:
                continue
            print(f"    {_fmt(r.status, colour)}  {label:38s} {r.message}")

    if trs:
        print("\ndataset-wide")
        for fn in PER_DATASET:
            r = fn(trs)
            tally[r.status] += 1
            if r.status == FAIL:
                failed_files.append(data_dir)
            print(f"    {_fmt(r.status, colour)}  {fn.label:38s} {r.message}")

    print(f"\n{tally[PASS]} passed, {tally[WARN]} warned, "
          f"{tally[SKIP]} skipped, {tally[FAIL]} failed")
    if tally[FAIL]:
        print("\nFAILures mean the data is wrong, not merely surprising. Do not\n"
              "train on it. Inspect the worst offender:\n"
              f"  python host/visualize_trajectory.py {failed_files[0]}")
        return 1
    if tally[SKIP]:
        print("SKIPs are legitimate: pre-v2 episodes have no deformation gradient,\n"
              "no material parameters and no target region to check against.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="directory of .npz episodes")
    ap.add_argument("--verbose", action="store_true",
                    help="show every check, not only the ones that are not PASS")
    ap.add_argument("--no-colour", action="store_true")
    args = ap.parse_args()
    raise SystemExit(run(args.data, verbose=args.verbose,
                         colour=not args.no_colour and sys.stdout.isatty()))


if __name__ == "__main__":
    main()
