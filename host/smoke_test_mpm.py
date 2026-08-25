#!/usr/bin/env python3
"""
smoke_test_mpm.py -- settle the three open questions in DECISION_LOG.md 9.3.

    conda activate tissue-host
    python host/smoke_test_mpm.py

9.3 vendored SurRoL's `Dev`-branch MPM at third_party/MPM/ and left three
questions marked "to be settled by a smoke test rather than argument":

  Q1  `ti._lib.core.with_metal()` is a PRIVATE Taichi API. Their code pins
      taichi 1.6.0; this host has 1.7.4. If that path moved, the guard raises
      or returns False, `arch` stays `ti.cpu`, and everything runs on the CPU
      with correct physics, wrong speed, and NO ERROR. This is the dangerous
      one precisely because nothing goes red.

  Q2  `pybullet` and `scikit-image` were not in the host environment.

  Q3  `from MPM.config import ...` means the directory must be importable *as*
      `MPM`, so its parent (third_party/) has to be on sys.path.

Two things worth knowing before reading the code:

  * `third_party/MPM/mpm3d.py` calls `ti.init()` at MODULE level (lines 14-22).
    Importing it IS the backend decision. So Q1 is probed in two parts: the
    private API is inspected BEFORE the import, and the arch Taichi actually
    settled on is read back AFTER it. Asking only the first question would
    confirm the call works while saying nothing about what it selected.

  * `substep()` reads the module globals `SDF` and `collision_mask`, which are
    `None` until `step()` rebinds them from fields in `sdf.py`. Driving the
    solver without a PyBullet scene therefore means binding them by hand. The
    SDF is filled with a large positive distance so the collision branch is
    never taken -- the tissue is in free space, which is the simplest state
    whose correct behaviour we already know.

Every check prints PASS / WARN / FAIL / SKIP and exits non-zero on any FAIL.
"""

import os
import platform
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(REPO, "third_party")

results = []


def record(status, name, detail):
    results.append((status, name, detail))
    icon = {"PASS": "  ok  ", "WARN": " warn ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
    print(f"[{icon}] {name}\n         {detail}")


def check(name):
    """Run a check, turning any exception into a FAIL rather than a traceback."""
    def wrap(fn):
        try:
            status, detail = fn()
        except ImportError as e:
            status, detail = "FAIL", f"not installed ({e.name}). Is the conda env active?"
        except Exception as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        record(status, name, detail)
        return fn
    return wrap


print("=" * 72)
print("MPM SMOKE TEST  (host, macOS-native)   DECISION_LOG.md 9.3")
print("=" * 72)


# ---------------------------------------------------------------------------
# Q3 -- importability. Done first because it needs no heavy dependency, and
# because a failure here explains every later failure.
# ---------------------------------------------------------------------------

@check("Q3: third_party/ exists and holds the four vendored MPM files")
def _():
    # WHY: catches a partial or mis-rooted vendoring. 9.3 committed to exactly
    # four files lifting out cleanly; if that set changed upstream, the "no
    # surrol.* imports" claim that justified vendoring is no longer tested.
    want = {"config.py", "mpm3d.py", "sdf.py", "requirements.txt"}
    d = os.path.join(THIRD_PARTY, "MPM")
    if not os.path.isdir(d):
        return "FAIL", f"{d} does not exist -- vendoring step not done"
    have = set(os.listdir(d))
    missing = want - have
    if missing:
        return "FAIL", f"missing {sorted(missing)} in {d}"
    return "PASS", f"{d} has all four: {sorted(want)}"


@check("Q3: MPM is importable as a package once third_party/ is on sys.path")
def _():
    # WHY: this is the whole of Q3. `mpm3d.py` says `from MPM.config import ...`,
    # an absolute import of a top-level package named MPM. Adding third_party/MPM
    # itself to sys.path -- the intuitive move -- makes `config` importable but
    # `MPM.config` still fail. The PARENT is what goes on the path.
    if THIRD_PARTY not in sys.path:
        sys.path.insert(0, THIRD_PARTY)
    import MPM.config as cfg
    # No __init__.py is present; this resolves as a PEP 420 namespace package.
    return "PASS", (f"MPM.config imported, n_grid={cfg.n_grid}, "
                    f"grid_shape={cfg.grid_shape} (namespace package, no __init__.py)")


# ---------------------------------------------------------------------------
# Q1 part one -- the private API, inspected BEFORE anything calls ti.init().
# ---------------------------------------------------------------------------

@check("Q1a: the private ti._lib.core.with_*() probes still exist on this Taichi")
def _():
    # WHY: this is the silent-failure path. mpm3d.py lines 14-20 call these three
    # to pick an arch. On taichi 1.6.0 (what they pin) they exist. If 1.7.4 moved
    # or removed them, the module raises AttributeError at import -- loud, fine --
    # but if any merely started returning False, arch silently stays ti.cpu.
    import taichi as ti
    found = {}
    for fn in ("with_metal", "with_vulkan", "with_cuda"):
        f = getattr(ti._lib.core, fn, None)
        found[fn] = f() if f is not None else None
    missing = [k for k, v in found.items() if v is None]
    ver = ".".join(str(x) for x in ti.__version__)
    if missing:
        return "FAIL", (f"taichi {ver}: {missing} no longer exist. mpm3d.py will "
                        f"raise AttributeError at import.")
    if not found["with_metal"]:
        return "FAIL", (f"taichi {ver}: with_metal() returned False. mpm3d.py will "
                        f"fall through to {'vulkan' if found['with_vulkan'] else 'cpu'} "
                        f"with no error. This is the failure 9.3 warned about.")
    return "PASS", (f"taichi {ver} (code pins 1.6.0): with_metal={found['with_metal']}, "
                    f"with_vulkan={found['with_vulkan']}, with_cuda={found['with_cuda']} "
                    f"-> mpm3d.py should select metal")


# ---------------------------------------------------------------------------
# Q2 -- the two missing dependencies.
# ---------------------------------------------------------------------------

@check("Q2: pybullet and scikit-image import on the host")
def _():
    # WHY: 9.3 flagged both as absent. pybullet has no macOS arm64 wheel and is
    # built from source; skimage ships a wheel. mpm3d.py imports both at module
    # level, so either being absent stops the MPM dead before any physics runs.
    import pybullet
    import skimage
    import skimage.measure  # what mpm3d.py actually imports, for get_mesh()
    return "PASS", (f"pybullet {getattr(pybullet, '__version__', 'built from source')}, "
                    f"scikit-image {skimage.__version__}")


# ---------------------------------------------------------------------------
# The import that decides the backend.
# ---------------------------------------------------------------------------

mpm3d = None


@check("Importing MPM.mpm3d (this is where ti.init() actually runs)")
def _():
    # WHY: the module runs ti.init() and allocates every field at import time.
    # Separating this from the checks that follow means a failure here is
    # reported as an import failure rather than as a mysterious physics failure.
    global mpm3d
    t0 = time.time()
    import MPM.mpm3d as _m
    mpm3d = _m
    return "PASS", (f"imported in {time.time() - t0:.1f}s, "
                    f"n_particles={_m.n_particles}, n_grid={_m.n_grid}, "
                    f"dt={_m.dt}, steps/frame={_m.steps}")


# ---------------------------------------------------------------------------
# Q1 part two -- what Taichi actually settled on. The question that matters.
# ---------------------------------------------------------------------------

@check("Q1b: Taichi actually initialised on the Metal (GPU) backend")
def _():
    # WHY: with_metal() reporting True only means the binary was BUILT with Metal
    # support. ti.init(arch=ti.metal) can still fall back to CPU at runtime and
    # say so in a log line nobody reads. This asks the live config what it chose.
    #
    # DEMONSTRATED AGAINST THE FAILURE, not just the success -- a check only ever
    # seen passing has not been shown to work:
    #
    #     TI_ARCH=arm64 python host/smoke_test_mpm.py     # -> this check FAILs
    #
    # (`TI_ARCH=cpu` is not a valid arch name on 1.7.4 and aborts the process
    # with SIGABRT; the CPU arch on Apple Silicon is `arm64`.) Forcing it that
    # way measured 5.17 ms/substep against 0.64 on Metal -- an 8.1x penalty --
    # while the settling distance came out within 2% of the GPU result. Correct
    # physics, wrong speed, no error anywhere: exactly the failure that gets
    # mistaken for "the MPM is just slow".
    if mpm3d is None:
        return "SKIP", "mpm3d did not import; nothing to interrogate"
    import taichi as ti
    arch = str(ti.lang.impl.current_cfg().arch)
    if "metal" in arch.lower():
        return "PASS", f"backend={arch} -- running on the M-series GPU as intended"
    return "FAIL", (f"backend={arch}, NOT metal. Physics will still be correct but "
                    f"~8x slower (measured 5.17 vs 0.64 ms/substep), with no error "
                    f"anywhere. Print this value in any script that depends on MPM speed.")


# ---------------------------------------------------------------------------
# The constitutive interface. 9.3 claimed materials.py maps onto set_parameters()
# exactly; these two checks test that claim rather than repeating it.
# ---------------------------------------------------------------------------

@check("set_parameters() takes (E, nu), NOT (mu, lambda) as 9.3 recorded")
def _():
    # WHY: CLAUDE.md's hard rule is "sample mu and lambda directly, never (E, nu)"
    # because lambda = E*nu/((1+nu)(1-2nu)) is singular at nu=0.5 and tissue sits
    # at nu~0.49. mpm3d.set_parameters(s_E, s_nu) performs exactly that singular
    # conversion internally, in float32. If a caller hands it (E, nu) obtained by
    # inverting a sampled (mu, lambda), the pair makes a round trip THROUGH the
    # singularity that materials.py exists to avoid. Recording the true signature
    # is the point of this check.
    if mpm3d is None:
        return "SKIP", "mpm3d did not import"
    import inspect
    sig = inspect.signature(mpm3d.set_parameters)
    names = list(sig.parameters)
    if names == ["s_E", "s_nu"]:
        return "WARN", (f"signature is set_parameters{sig} -- (E, nu), not (mu, lambda). "
                        f"9.3 says 'mu/la from set_parameters()'; that is inaccurate. "
                        f"Prefer writing mu[None]/la[None] directly (next check).")
    return "WARN", f"signature changed upstream: set_parameters{sig}, expected (s_E, s_nu)"


@check("A tissue-like (mu, lambda) survives the (E, nu) round trip set_parameters forces")
def _():
    # WHY: quantifies the cost of the detour above instead of asserting it is bad.
    # Takes materials.py's own ranges, converts to (E, nu), feeds set_parameters,
    # and reads back the mu/la Taichi ended up holding. Large error here means the
    # adapter MUST bypass set_parameters; small error means it is merely inelegant.
    # Either way the number belongs in the decision log rather than in a guess.
    if mpm3d is None:
        return "SKIP", "mpm3d did not import"
    sys.path.insert(0, os.path.join(REPO, "src"))
    import materials

    # Corners and centre of the sampling box, in log space, plus the stiffest and
    # most nearly-incompressible corner -- which is the one closest to nu = 0.5.
    lo_mu, hi_mu = materials.DEFAULT_MU_RANGE
    lo_lam, hi_lam = materials.DEFAULT_LAM_RANGE
    cases = [(lo_mu, lo_lam), (lo_mu, hi_lam), (hi_mu, lo_lam), (hi_mu, hi_lam),
             (float(np.sqrt(lo_mu * hi_mu)), float(np.sqrt(lo_lam * hi_lam)))]

    worst_rel, worst_desc = 0.0, ""
    for mu_in, lam_in in cases:
        E, nu = materials.E_nu_from_lame(mu_in, lam_in)
        mpm3d.set_parameters(s_E=float(E), s_nu=float(nu))
        mu_out = float(mpm3d.mu[None])
        lam_out = float(mpm3d.la[None])
        rel = max(abs(mu_out - mu_in) / mu_in, abs(lam_out - lam_in) / lam_in)
        if rel > worst_rel:
            worst_rel = rel
            worst_desc = (f"mu={mu_in:.4g},lam={lam_in:.4g} (nu={float(nu):.6f}) "
                          f"-> got mu={mu_out:.6g}, lam={lam_out:.6g}")
    # 1e-3 is the tolerance at which a material error would start to compete with
    # the spread of the placeholder ranges themselves.
    if worst_rel > 1e-3:
        return "FAIL", (f"worst relative error {worst_rel:.3e} over materials.py's range. "
                        f"{worst_desc}. The adapter must bypass set_parameters().")
    return "PASS", (f"worst relative error {worst_rel:.3e} across 5 corners of "
                    f"materials.py's ranges; worst case {worst_desc}")


@check("mu/la can be written directly, bypassing the (E, nu) detour entirely")
def _():
    # WHY: establishes the escape hatch the adapter will use. mu and la are plain
    # 0-d Taichi fields; nothing about the solver requires E or nu, which are only
    # ever read by set_parameters itself. Confirming this now means the adapter can
    # be written against a direct (mu, lambda) interface with no surprises later.
    if mpm3d is None:
        return "SKIP", "mpm3d did not import"
    mu_want, lam_want = 3000.0, 150000.0
    mpm3d.mu[None] = mu_want
    mpm3d.la[None] = lam_want
    got_mu, got_la = float(mpm3d.mu[None]), float(mpm3d.la[None])
    err = max(abs(got_mu - mu_want) / mu_want, abs(got_la - lam_want) / lam_want)
    if err > 1e-6:
        return "FAIL", f"direct write did not stick: mu={got_mu}, la={got_la}"
    return "PASS", (f"mu[None]={got_mu:.6g}, la[None]={got_la:.6g} written directly "
                    f"(rel err {err:.1e}); solver reads only these two")


# ---------------------------------------------------------------------------
# Physics. The point of a smoke test is that the thing runs, not that it imports.
# ---------------------------------------------------------------------------

state = {}
SCALE, THRESHOLD = 1.0, 0.05


def _fresh_cube(lift=0.0):
    """Reset to an undeformed cube at rest, optionally lifted clear of the floor.

    `init_cube()` spawns particles at z in [0.05, 0.10]. The floor in Boundary()
    is the `bound = 3` grid-cell band, i.e. z = 3*dx = 0.047 -- so the default
    cube is ALREADY resting on the floor, not free-falling. Anything wanting
    genuine free fall has to lift it first.
    """
    mpm3d.init_cube()
    mpm3d.init_deformation_gradient()
    mpm3d.F_v.fill([0.0, 0.0, 0.0])
    mpm3d.F_C.fill([[0.0] * 3] * 3)
    mpm3d.F_grid_m.fill(0.0)
    mpm3d.F_grid_v.fill([0.0, 0.0, 0.0])
    mpm3d.init_collision_field()
    if lift:
        x = mpm3d.F_x.to_numpy()
        x[:, 2] += lift
        mpm3d.F_x.from_numpy(x)


@check("The solver runs at all: kernels compile and launch on Metal")
def _():
    # WHY: import success proves nothing about whether kernels compile and run.
    # Metal rejects some Taichi constructs at compile time -- sparse SNodes above
    # all, which is why 9.3 checked for them -- and that only surfaces on the
    # first kernel launch, never on ti.init(). This is the first moment the MPM
    # has ever executed on this machine, so it is isolated from the timing below:
    # the first substep pays for compiling P2G, Boundary and G2P.
    #
    # Expect ~10s the very first time on a machine and ~0.5s afterwards. Taichi
    # caches compiled kernels under ~/.cache/taichi; measured here as 9.68s with
    # TI_OFFLINE_CACHE=0 against 0.56s warm, with steady-state speed identical
    # either way. A ten-second pause on a fresh clone is the cache being filled,
    # not a hang -- worth knowing before someone kills it.
    if mpm3d is None:
        return "SKIP", "mpm3d did not import"
    import taichi as ti

    mpm3d.set_parameters(s_E=8000.0, s_nu=0.2)   # the upstream defaults
    mpm3d.set_base_position([0.0, 0.0, 0.0])

    # substep() reads these two globals, which step() would normally rebind from
    # sdf.py (mpm3d.py:492-494). Bound by hand here so the solver can run with no
    # PyBullet scene. A large positive distance everywhere means "no collider
    # anywhere near", so the SDF branch in Boundary() is never taken and no
    # neighbour of SDF is ever indexed -- which also keeps it in bounds.
    mpm3d.SDF = mpm3d.sdf.tmp_sdf
    mpm3d.collision_mask = mpm3d.sdf.co_mask
    mpm3d.SDF.fill(1.0e9)
    mpm3d.collision_mask.fill(-1)

    _fresh_cube()
    t0 = time.time()
    mpm3d.substep(SCALE, THRESHOLD)
    ti.sync()                     # kernels are async; timing without this is fiction
    compile_s = time.time() - t0
    state["compile_s"] = compile_s
    return "PASS", (f"first substep took {compile_s:.2f}s, almost all of it one-time "
                    f"kernel compilation for P2G/Boundary/G2P")


@check("Gravity acts along -z at 9.8 m/s^2, the PyBullet convention in Boundary()")
def _():
    # WHY: Boundary() applies `F_grid_v[I][2] -= dt * gravity`, having abandoned
    # the Taichi convention of gravity along y on the commented line above it.
    # Getting this axis backwards would put the tissue's rest state on the wrong
    # side of the world and every episode with it. Checked against the analytic
    # free-fall value rather than just the sign, because a sign test would also
    # pass on a solver applying the wrong magnitude.
    #
    # The cube is lifted clear of the floor first. Without that it spawns inside
    # the `bound = 3` band and what gets measured is elastic rebound, not gravity.
    if mpm3d is None or "compile_s" not in state:
        return "SKIP", "solver did not run"
    import taichi as ti

    _fresh_cube(lift=0.30)        # z in [0.35, 0.40], grid rows 22-25, floor is row 3
    n_sub = 25
    for _ in range(n_sub):
        mpm3d.substep(SCALE, THRESHOLD)
    ti.sync()

    t = n_sub * mpm3d.dt
    vz = float(mpm3d.F_v.to_numpy()[:, 2].mean())
    want = -9.8 * t
    rel = abs(vz - want) / abs(want)
    # 2% covers the half-step offset between grid velocity update and G2P
    # transfer; anything larger means the magnitude itself is wrong.
    if rel > 0.02:
        return "FAIL", (f"after {t*1e3:.1f} ms of free fall mean v_z = {vz:+.5f} m/s, "
                        f"analytic -g*t = {want:+.5f} ({rel*100:.1f}% off). Gravity is on "
                        f"the wrong axis or the wrong magnitude.")
    return "PASS", (f"after {t*1e3:.1f} ms of free fall mean v_z = {vz:+.5f} m/s vs "
                    f"analytic -g*t = {want:+.5f} ({rel*100:.2f}% off) -- z is down")


@check("250 substeps of a cube settling on the floor, timed at steady state")
def _():
    # WHY: the free-fall case above exercises almost nothing -- no contact, no
    # large deformation. This is the case that stresses the constitutive model:
    # the cube lands on the boundary band and compresses under its own weight,
    # which is where an unstable solver blows up. Timing is taken here, after
    # compilation, so the number is the one that will govern collection runs.
    if mpm3d is None or "compile_s" not in state:
        return "SKIP", "solver did not run"
    import taichi as ti

    _fresh_cube()                 # default spawn: already in contact with the floor
    z0 = float(mpm3d.F_x.to_numpy()[:, 2].mean())

    n_frames = 10
    t0 = time.time()
    for _ in range(n_frames):
        for _ in range(mpm3d.steps):      # 25 substeps per frame
            mpm3d.substep(SCALE, THRESHOLD)
    ti.sync()
    wall = time.time() - t0

    n_sub = n_frames * mpm3d.steps
    state["x"] = mpm3d.F_x.to_numpy()
    state["v"] = mpm3d.F_v.to_numpy()
    state["F"] = mpm3d.F.to_numpy()
    state["dz"] = float(state["x"][:, 2].mean()) - z0
    sim_ms = n_sub * mpm3d.dt * 1e3
    # One "frame" is steps*dt = 12.5 ms of sim time, so realtime is 80 frames/s.
    fps = n_frames / wall
    return "PASS", (f"{n_sub} substeps ({sim_ms:.0f} ms sim time) in {wall:.2f}s = "
                    f"{wall/n_sub*1e3:.2f} ms/substep, {fps:.0f} frames/s "
                    f"(realtime would be 80); centroid settled {state['dz']*1e3:+.2f} mm")


@check("Particle state is finite -- no NaN or Inf anywhere")
def _():
    # WHY: an MPM that has gone unstable does not crash, it fills with NaN and
    # keeps running. CLAUDE.md records the same failure mode for quaternions:
    # one NaN propagates through a whole trajectory and surfaces much later, in
    # something that looks like an architecture problem.
    if "x" not in state:
        return "SKIP", "solver did not run"
    bad = {k: int((~np.isfinite(state[k])).sum()) for k in ("x", "v", "F")}
    if any(bad.values()):
        return "FAIL", f"non-finite entries: {bad}"
    return "PASS", (f"F_x, F_v, F all finite "
                    f"({state['x'].shape[0]} particles x {state['x'].shape[1]} dims)")


@check("det(F) > 0 for every particle -- no inverted elements")
def _():
    # WHY: det(F) is the local volume ratio J. J <= 0 means an element has turned
    # itself inside out, which is unphysical and makes the Neo-Hookean log(J) in
    # P2G() undefined -- log of a non-positive number. This is the same invariant
    # tests/ already enforces on the analytic episodes, applied here to state the
    # solver actually produced under contact.
    if "F" not in state:
        return "SKIP", "solver did not run"
    J = np.linalg.det(state["F"].astype(np.float64))
    if not np.all(J > 0):
        n = int((J <= 0).sum())
        return "FAIL", f"{n} of {J.size} particles have det(F) <= 0, min {J.min():.4g}"
    return "PASS", (f"J = det(F) in [{J.min():.4f}, {J.max():.4f}], mean {J.mean():.4f} "
                    f"(1.0 = undeformed; < 1 = compressed under its own weight)")


@check("Particle fields map onto the v2 schema without reshaping")
def _():
    # WHY: this is the whole reason 9.3 called the adapter small. The claim is
    # F_x -> tissue_pos, F_v -> tissue_vel, F -> tissue_F. Checking it here means
    # the adapter starts from a verified mapping instead of a hoped-for one; a
    # silent dtype or rank mismatch would otherwise surface inside the writer.
    if "x" not in state:
        return "SKIP", "solver did not run"
    n = state["x"].shape[0]
    want = {"F_x -> tissue_pos": (state["x"], (n, 3), "(T,N,3) float32"),
            "F_v -> tissue_vel": (state["v"], (n, 3), "(T,N,3) float32"),
            "F   -> tissue_F":   (state["F"], (n, 3, 3), "(T,N,3,3) float32")}
    problems, lines = [], []
    for label, (arr, shape, schema) in want.items():
        ok = arr.shape == shape and arr.dtype == np.float32
        lines.append(f"{label}: {arr.shape} {arr.dtype} -> one frame of {schema}")
        if not ok:
            problems.append(f"{label} is {arr.shape} {arr.dtype}, wanted {shape} float32")
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", "per-frame slices line up exactly:\n         " + "\n         ".join(lines)


@check("The v2 writer accepts a frame of real MPM state")
def _():
    # WHY: the strongest available end-to-end statement short of the adapter
    # itself. Everything above tests the MPM; this tests the seam, by pushing
    # actual solver output through src/trajectory_io.py's validation. If the two
    # halves disagree about shape, dtype or node count, it fails here rather than
    # after a twenty-minute collection run.
    if "x" not in state:
        return "SKIP", "solver did not run"
    import tempfile

    sys.path.insert(0, os.path.join(REPO, "src"))
    import materials
    import trajectory_io

    n = state["x"].shape[0]
    # materials.py's own packing: [log(mu), log(lambda), rho]. Pascals would be
    # the wrong units for a network input -- see materials.py, note 2.
    mat = np.array([np.log(3000.0), np.log(150000.0), 1000.0], np.float32)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "smoke.npz")
        w = trajectory_io.TrajectoryWriter(
            out, simulator="mpm", task="smoke_test", dt=mpm3d.dt * mpm3d.steps,
            material_params=mat,
            substep_dt=mpm3d.dt, n_substeps=mpm3d.steps,
            action_spec="unknown",
        )
        # Two frames, because a one-frame episode cannot catch the node-count
        # check that fires when N changes mid-episode.
        for _ in range(2):
            w.append(
                tissue_pos=state["x"],
                tissue_vel=state["v"],
                tissue_F=state["F"],
                ee_pose=np.array([0, 0, 0, 0, 0, 0, 1], np.float32),
                action=np.zeros(7, np.float32),
            )
        w.close()
        size_mb = os.path.getsize(out) / 1e6
        tr = trajectory_io.load_trajectory(out)
        got = tr.tissue_F.shape

    if got != (2, n, 3, 3):
        return "FAIL", f"round trip changed tissue_F to {got}, expected {(2, n, 3, 3)}"
    return "PASS", (f"wrote + reloaded 2 frames x {n} particles with tissue_F "
                    f"{got}, {size_mb:.1f} MB on disk")


@check("Sampled density reaches the solver as p_mass, and is what gets recorded")
def _():
    # WHY: material_params[2] is rho, and until the adapter set p_mass the
    # solver ignored it completely -- every episode ran at the vendored
    # p_rho = 1000 while the file claimed whatever was sampled. Nothing was
    # inconsistent on disk. The dataset simply had a density column that did
    # not describe the physics that produced it, which is worse than not
    # recording density at all: a model would learn to condition on a number
    # that never influenced anything.
    #
    # p_mass is a module-level Python float read inside the P2G kernel
    # (mpm3d.py:172,180,181), so it is baked in when kernels compile, exactly
    # like dt. This check verifies the arithmetic and the recording; the
    # ORDERING (set before first substep) is enforced in the adapter, whose
    # compile lock covers p_mass as well as dt.
    sys.path.insert(0, os.path.join(REPO, "src"))
    import materials
    import trajectory_io

    rng = np.random.default_rng(3)
    mat = materials.sample_material(rng)
    _, _, rho = materials.unpack_material(mat)
    rho = float(rho)

    applied = mpm3d.p_vol * rho
    vendored = mpm3d.p_vol * 1000.0

    # Demonstrated against the failure, per section 5: if the vendored default
    # happened to equal the sampled density, this check would pass while
    # proving nothing. Sampled rho spans 1000-1100, so it does not.
    if abs(applied - vendored) <= 0.0:
        return "FAIL", (f"sampled rho={rho:.1f} gives the same p_mass as the "
                        "vendored 1000 -- this check cannot distinguish the bug "
                        "it exists to catch")

    # And the number the file records must be the number that was applied.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "rho.npz")
        w = trajectory_io.TrajectoryWriter(
            out, simulator="mpm", task="rho_check", dt=0.0125,
            material_params=mat, substep_dt=0.0125 / 25, n_substeps=25,
            action_spec="unknown")
        w.append(tissue_pos=np.zeros((4, 3), np.float32),
                 ee_pose=np.array([0, 0, 0, 0, 0, 0, 1], np.float32),
                 action=np.zeros(7, np.float32))
        w.close()
        _, _, rho_back = materials.unpack_material(
            trajectory_io.load_trajectory(out).material_params)

    err = abs(float(rho_back) - rho)
    if err > 0.05:                      # float32 storage of ~1e3 kg/m^3
        return "FAIL", (f"recorded rho {float(rho_back):.4f} != applied rho "
                        f"{rho:.4f} (off by {err:.3e})")
    return "PASS", (f"rho={rho:.1f} kg/m^3 -> p_mass={applied:.6e} kg "
                    f"(vendored default would be {vendored:.6e}, "
                    f"{100*(vendored/applied - 1):+.1f}%); "
                    f"round-trips through material_params to {float(rho_back):.4f}")


# ---------------------------------------------------------------------------

print("=" * 72)
n_fail = sum(1 for s, _, _ in results if s == "FAIL")
n_warn = sum(1 for s, _, _ in results if s == "WARN")
n_skip = sum(1 for s, _, _ in results if s == "SKIP")
n_pass = sum(1 for s, _, _ in results if s == "PASS")
print(f"{n_pass} passed, {n_fail} failed, {n_warn} warnings, {n_skip} skipped.")
if n_fail:
    print("\nFAILURES:")
    for s, name, detail in results:
        if s == "FAIL":
            print(f"  - {name}\n      {detail}")
    sys.exit(1)
print("\nMPM runs on this host. Record the backend and the timing in DECISION_LOG.md 9.3.")
