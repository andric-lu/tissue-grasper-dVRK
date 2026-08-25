# BM2 — Soft Tissue Dynamics: Decision Log

**Sessions:**
1. 1 August 2026 · Environment setup and first working data pipeline (§1–§7)
2. 13–16 August 2026 · Schema v2, metrics, and the testable-before-the-simulator
   layer (§8) — *reconstructed from source on 17 August, see the note there*
3. 17 August 2026 · Documentation recovered into git, boundary_mask bug, MPM
   route settled (§9)

**Objective:** A dynamics model that predicts soft tissue response during
deformation and external disturbance, starting with basic tissue retraction.
**Hardware:** M3 Max MacBook Pro, 128 GB.

This is the substantive record: what was decided, why, what broke, and what is
still unresolved. Written to be useful in three months, when the reasoning
behind a parameter is no longer obvious.

---

## 1. Platform decision

### 1.1 Isaac Sim is not available on this hardware

Isaac Sim requires x86_64 Ubuntu 22.04/24.04 or Windows 10/11 with an NVIDIA
RTX GPU (minimum ~RTX 4080, 16 GB VRAM). There is no macOS build and no ARM
build. Emulation does not help — Rosetta translates instructions, it cannot
provide CUDA hardware.

**Consequence:** Isaac Sim work requires external resources. Options, in
increasing cost: NVIDIA Brev / Isaac Launchable containers with WebRTC
streaming (the client *does* have a macOS build); AWS g6 / Lambda / RunPod at
roughly $0.50–1.50/hr for an L4 or L40S; a dedicated Linux workstation with an
RTX 4090/5090, which becomes cheaper once runs go overnight.

**Deferred deliberately.** Nothing in the current phase needs it.

### 1.2 Two environments on one laptop: host-native + container

Two constraints pull in opposite directions:

- SurRoL targets Ubuntu with a dependency stack pinned to its 2021 publication
  era. Reproducing that on macOS means fighting the package manager repeatedly.
- Apple's GPU is reachable only through Metal, a macOS API. No container —
  even one running natively on ARM — can use it. Taichi's Metal backend and
  PyTorch's MPS backend both require running directly on macOS.

**Decision:** physics in an arm64 Ubuntu container, learning on the host,
one bind-mounted folder between them.

**Rejected: a full Linux VM.** Considered because it matched the previous
laptop's setup and would make ROS trivial. Rejected because Apple Silicon
virtualization has no GPU passthrough — inside a Linux guest there is no Metal,
so Taichi drops to `ti.cpu` and PyTorch loses MPS. Those are exactly the two
things that matter for the modelling work. A container gives the same Linux
userspace without a second kernel, and GPU access stays on the host where it
works.

**Revisit if** ROS/dVRK integration becomes central sooner than the soft-body
modelling. The tradeoff would then flip.

### 1.3 The trajectory format is the architectural decision that matters most

Every simulator writes the same `.npz` schema (`src/trajectory_io.py`); every
model reads it. The simulator becomes a swappable component.

Without this, moving PyBullet → Taichi/MPM → Isaac Sim is three rewrites of the
modelling code. With it, each is a new data *source*. This was set up before any
research code was written, which was the right order.

Chose `.npz` over HDF5: numpy is already everywhere, nothing extra to install,
and one file per episode is simpler. The reader/writer is isolated so the
backend can be swapped if episodes ever get large enough to need partial reads.

---

## 2. Environment specifics and the reasoning behind each pin

### 2.1 Host (macOS-native), conda env `tissue-host`

- **Python 3.11** — inside Taichi 1.7.4's supported range (3.9–3.13), fully
  supported by PyTorch, mature enough that every package has an ARM64 wheel.
- **Miniforge, not Anaconda** — Anaconda's installer can hand you an Intel build
  that runs under Rosetta. Everything appears to work, but Metal backends may
  refuse to initialise, silently dropping you to CPU at 10–30× slower with no
  error. `verify_host.py` checks `platform.machine() == "arm64"` first for
  this reason.
- **Entire numerical stack from pip, nothing compiled from conda.** See §3.2.

### 2.2 Container (Ubuntu 22.04, arm64)

Three pins are load-bearing:

| Pin | Why |
|---|---|
| `numpy==1.23.5` | 1.24 removed `np.bool` / `np.int` / `np.float`, which gym 0.21 and SurRoL still use |
| `gym==0.21.0` | 0.26 changed the core API — `reset()` returns a tuple, `step()` returns five values. SurRoL is written against the old API, so a newer gym imports fine and fails at runtime |
| `pip install --no-deps -e .` for SurRoL | Its `setup.py` requests `gym>=0.15.6` with no upper bound; without `--no-deps`, pip "helpfully" upgrades and undoes the pin above |

**General principle established:** research code is pinned to the software
landscape of its publication date. "Install the latest version" is wrong far
more often than right when working with published research code.

Note: PyBullet has no prebuilt wheel for ARM Linux, so it compiles from source
during the image build — 5–8 minutes with no output. Not a hang.

### 2.3 SurRoL branch

**`--branch main` is required.** The repository's *default* branch is
`SR-VPPV` (Science Robotics'25), which is organised as paper artifacts with no
`setup.py` at the root. `main` is the IROS'21 SurRoL: installable, with the dVRK
robots, gym-style API, and ten surgical tasks.

Branches for later:

- **`Dev`** — the Taichi MPM soft-body work ([arXiv:2402.01181](https://arxiv.org/abs/2402.01181)). **This is the Phase 1 destination.**
- `SurRoL-v2` — RA-L'23, human-in-the-loop interactive simulation
- `SR-VPPV` — Science Robotics'25, full surgical autonomy framework

**Lesson:** on research repositories the default branch is whatever the lab
published most recently, not the version the documentation describes. Always
check after cloning.

---

## 3. Bugs found and root causes

Recorded because the root causes generalise.

### 3.1 SurRoL clone got the wrong branch
`pip install -e .` failed with "does not appear to be a Python project."
Root cause: default branch is not `main` (§2.3). **Fix:** `--branch main`, plus
an explicit `test -f setup.py` guard immediately after the clone — the original
error surfaced three instructions after the actual mistake.
*Generalises to:* assert your assumptions where they're made, not where they
eventually fail.

### 3.2 OpenMP duplicate runtime (`OMP: Error #15`)
Python aborted on import. Root cause: conda-forge numpy links against conda's
`libomp.dylib`; the PyPI torch wheel bundles its own. Two OpenMP runtimes in one
process. **Fix:** entire numerical stack from pip; conda provides only the
interpreter.
**Explicitly rejected** the common `KMP_DUPLICATE_LIB_OK=TRUE` workaround — it is
documented as unsupported and can silently produce incorrect numerical results.
In a project whose whole output is numerical predictions, a flag that quietly
corrupts arithmetic is worse than a crash.
*Generalises to:* one package manager per environment for anything compiled.

### 3.3 `surrol.__file__` is None
`os.path.dirname(surrol.__file__)` raised TypeError. Root cause: SurRoL ships no
`surrol/__init__.py`, making it a namespace package, and those have
`__file__ = None`. **Fix:** use `__path__`, which is populated for both regular
and namespace packages.

### 3.4 Grasp caught zero nodes on every episode
Root cause: grasp height and xy were hardcoded. The sheet falls under gravity
during the approach and settles ~5 cm below where the gripper stopped, and the
asset's true extent is a property of the file, not inferable from `scale=0.30`.
**Fix:** settle first, read actual node positions, derive the grasp target from
the mesh. Plus a nearest-N fallback so a grasp can never silently catch nothing.
*Generalises to:* code that measures the world it operates in survives changes of
asset, scale and resolution. Hardcoded coordinates break silently.

Secondary bug: the warning printed 72 times because the retry condition was
"no anchors yet," which stays true when the grasp catches nothing. Needed a
separate "already attempted" flag.

### 3.5 Tissue was being dragged, not retracted
Peak displacement ≈ 140 mm, almost exactly the gripper's own travel
(100 mm lift + 120 mm retract). The sheet was unattached, so the recorded data
was rigid translation with no strain field in it.
**Fix:** pin the perimeter to the world (`createSoftBodyAnchor` with
`bodyUniqueId = -1`). Real tissue is continuous with surrounding structure;
pulling it stretches it.
**Diagnostic signature to remember:** peak tissue displacement ≈
`LIFT_HEIGHT + RETRACT_DIST` means the anchoring has failed.

### 3.6 Demo cloth asset was unusable for this purpose
`cloth_z_up.obj` is 60 cm across with 25 simulation nodes — 15 cm spacing. Three
problems: cannot resolve a deformation gradient; 60 cm is not a surgical length
scale; and its render mesh has a different vertex count from its simulation
mesh, so node ordering cannot be matched back to the `.obj` topology (no mesh
connectivity → no graph network).
**Fix:** `make_tissue_mesh.py` authors the sheet — default 10 cm, 20×20 = 400
nodes at 5 mm spacing, one vertex per simulation node so topology maps cleanly
and is stored in every trajectory.

All motion parameters were rescaled ~5× down with the tissue (lift 20 mm,
retract 25 mm, 8 mm jaw). Lengths in this system are only meaningful relative to
the tissue's own size.

---

## 4. OPEN — timestep convergence not yet resolved

**Status: still blocking as of 17 August 2026. Do not collect a large PyBullet
dataset until settled.** See §8.8 for why session 2 routed around this rather
than resolving it.

Measured peak node speed **0.661 m/s at dt = 1/1000**. That is **39× faster than
the gripper ever moves** (25 mm over 1.5 s ≈ 1.7 cm/s), so it is not the tissue
following the gripper. Two candidate explanations:

- **Elastic recoil at release** — real physics, keep it. Stored energy dumps when
  the anchors are removed.
- **Solver divergence** — numerical, data is garbage.

A single run cannot distinguish them.

### Background: why divergence happens here

Explicit integrators are conditionally stable. Roughly `dt < 2/ω` where
`ω = sqrt(k_eff / m_node)`. Stiffer springs and lighter nodes both raise ω and
shrink the largest stable timestep.

Moving from 25 nodes to 400 dropped per-node mass from ~4×10⁻³ kg to
~1.25×10⁻⁴ kg — 32× lighter — while stiffness was simultaneously raised. That
pushes ω up roughly 8×. Order-of-magnitude estimate puts the stable limit near
1 ms, which would make dt = 1/1000 marginal.

This is the central tension in soft-tissue simulation: tissue is stiff and
nearly incompressible, which is exactly what forces small timesteps. Implicit
integrators and MPM exist largely to escape it — a real part of why the
Taichi/MPM branch is the destination.

### Resolution path

```
docker compose run --rm surrol python container/timestep_study.py
```

Runs identical episodes at 1/240 … 1/4000 with logging held at 30 Hz, and reports
peak speed, peak deformation, *when* the peak occurred, and percent change
between refinements.

- Peak at the release step **and converging** → recoil, keep it.
- Peak mid-pull, or still shrinking with every refinement → divergence; go finer.

Then set `DT` in `collect_retraction.py` to the largest converged value.

**Why this matters beyond correctness:** "we used dt = 1/2000" invites the
question of why. "Peak deformation changed by under 5% below 1/2000" answers it.

---

## 5. Validation framework

Divergence is the loudest failure but not the only one. Stable-but-wrong is more
dangerous because nothing warns you.

**Failure categories identified:**

1. **Constraint failures** — anchors silently not holding; grasp slipping partway
   so the action no longer explains the outcome (precisely the relationship the
   model is learning)
2. **Parameter failures** — a stiffness value misspelled or overwritten and
   having no effect; weeks spent "tuning" something inert
3. **Discretization failures** — timestep convergence resolves *time*; mesh
   convergence resolves *space*. A result that shifts with resolution is a
   property of the discretization, and a model trained at one resolution will
   not transfer
4. **Logging failures, invisible in any video** — velocity divided by physics
   timestep instead of logging interval; node ordering permuted between mesh and
   trajectory, so a graph network trains on an essentially random graph and the
   failure reads as "graph nets don't work here"

**`container/validate_physics.py`** — nine experiments whose correct answer is
known in advance: rest stability, boundary held, grasp held, 4-fold rotational
symmetry, mirror symmetry, Saint-Venant decay, stiffness monotonicity,
determinism, mesh convergence. **Never run — see §10.**

**`host/validate_dataset.py`** — data-integrity checks on recorded data. Reads
only the shared format, so it applies unchanged to Taichi/MPM and Isaac Sim
later. Eight checks as of session 1; thirteen as of session 2 (§8.5);
fourteen as of session 3 (§9.2).

**The symmetry test is the strongest single check.** Grasp the centre node of an
odd-resolution mesh, pull straight up: geometry, boundary condition and load are
all invariant under 90° rotation, so the solution must be too. A correct answer
known without any reference simulation. Verified sensitive — a scrambled node
ordering produces 94% error against a 2% threshold, while a genuinely symmetric
field passes to 3×10⁻¹⁶.

**What none of this covers:** everything above checks *internal consistency*. It
cannot establish that a mass-spring cloth resembles real tissue, because it does
not — no volume, no incompressibility, stiffness in arbitrary units. A simulation
can pass all seventeen checks and be physically meaningless. Closing that gap
needs MPM/Neo-Hookean (parameters with physical units) and eventually validation
against measured tissue data — indentation or uniaxial tension curves from the
literature for the relevant tissue type. *"My simulation is self-consistent" and
"my simulation predicts reality" are different claims.*

---

## 6. Modelling decisions already made

- **Predict position deltas, not absolute positions.** Absolute targets force the
  network to memorise where tissue sits in world coordinates, and that large
  constant offset dominates the loss.
- **Split train/validation by episode, never by random timestep.** Consecutive
  frames are near-identical; a random split leaks near-duplicates into validation
  and produces a beautiful, meaningless curve. Common error in learned-dynamics
  work.
- **Always report the constant-velocity baseline.** Soft tissue at 30 Hz is
  smooth, so "assume every node keeps moving as it was" is strong. A model that
  doesn't clearly beat it has learned nothing, however small the loss looks.
- **The MLP is a placeholder.** Flattening the mesh discards connectivity and
  locks the model to one resolution. MeshGraphNets ([arXiv:2010.03409](https://arxiv.org/abs/2010.03409))
  is the target architecture; the trajectory format already carries
  `tissue_faces` / `tissue_tets` for it.
- **Randomise material parameters per episode from the start.** A model trained
  on one stiffness learns that stiffness, not the dynamics. Cheaper than
  regenerating a dataset later. *(Session 2 found the obvious way to do this is a
  trap — see §8.3.)*

### Known gap: topology recovery is fragile

PyBullet's `getMeshData` returns node positions but not connectivity, so
topology is recovered by parsing the source `.obj` and assuming node ordering
matches. This holds for self-authored meshes (vertex counts are checked) but is
not guaranteed in general. **If a graph model produces nonsense while an MLP
works, suspect this first.**

---

## 7. Roadmap

1. **Resolve the timestep question** (§4). Still open.
2. **Swap the kinematic block for the dVRK PSM.** Replace with
   `surrol.robots.psm.Psm1` driven through `psm.move()`. Read
   `/opt/SurRoL/tests/test_psm.ipynb` first — it mirrors the real dVRK Python
   interface. The block was deliberate: introducing soft-body physics and a
   7-DOF arm together makes failures unattributable.
3. **Scale up collection.** Hundreds of episodes across grasp points, retraction
   directions, stiffnesses, mesh resolutions. Parallelise across containers.
4. **Replace the MLP with a graph network.**
5. **Move to MPM / Neo-Hookean** (SurRoL `Dev` branch) for physically meaningful
   ground truth. Where the host Taichi Metal backend earns its place.
   *Session 2 built the schema, metrics and validation this step will need,
   ahead of the simulator itself (§8).*
6. **Isaac Sim** on cloud GPU or a Linux workstation — a new data source, not a
   rewrite.

---

## 8. Session 2 — 13–16 August 2026: the schema grows a constitutive model

> **Note on provenance.** This section was **reconstructed on 17 August 2026 from
> the source files themselves**, not written during the session it describes. No
> transcript of 13–16 August was kept. The reasoning below is recovered from the
> `WHY THIS FILE EXISTS` headers and `# WHY:` comments in the code, which are
> unusually complete and were written at the time — so the *content* is
> first-hand even though the *narrative* is after the fact. Where this section
> states a conclusion the code does not explicitly support, it says so.
>
> The lesson worth taking from that: §1–7 were written the same day as the work
> and §8 was written sixteen days late, and the difference in effort was
> substantial. The decision log is cheap to write while the reasoning is live and
> expensive to reconstruct afterwards.

Session 1 ended with a PyBullet mass-spring pipeline and an honest note that it
"is not yet meaningful as physics." Session 2 did not try to make the mass-spring
cloth better. It built the layer that a *constitutive* simulator — MPM /
Neo-Hookean, roadmap item 5 — will need, and made that layer testable before the
simulator exists.

That ordering is the substantive decision of the session, and it is the same bet
as §1.3: define the contract first, make the component swappable.

### 8.1 Schema 1.0 → 2.0

`src/trajectory_io.py` roughly doubled. The additions all exist because a
mass-spring cloth has no constitutive model and an MPM solver does.

**Static, new in v2:**

| Field | Why it exists |
|---|---|
| `material_params` (3,) | `[log μ, log λ, ρ]` — see §8.3 |
| `substep_dt`, `n_substeps` | The solver's internal step, which is *not* the logging interval. Recording it makes the stability check in §8.5 possible at all |
| `boundary_mask` (N,) bool | Which particles are kinematically clamped. Session 1 recomputed this geometrically every time it was needed |
| `action_spec` str | `"abs_pose_jaw"` / `"delta_pose_jaw"` / `"unknown"` — so a file states its own action convention instead of the reader assuming one |
| `target_origin` / `target_normal` / `target_extent` | The region the retraction is meant to expose. Without it, "success" is not defined in the file |

**Per-step, new in v2:** `tissue_F` (T,N,3,3) deformation gradient,
`contact_mode` (T,) int8, and the two logged metrics `exposure` and
`safety_strain`.

**The empty-array convention is the part worth remembering.** v1 files still
load. Fields introduced in v2 come back as **empty arrays, never as zeros**,
because `0.0` is a legitimate reading for `exposure` (fully occluded) and
all-`False` is a legitimate `boundary_mask`. A zero default would be
indistinguishable from a real measurement. `arr.size == 0` means "this simulator
did not record it," and callers branch on it honestly.

`substep_dt` / `n_substeps` are the deliberate exception: zero substeps is not a
physically meaningful reading, so `0` carries "not recorded" without ambiguity.

`tissue_F` is also the field that will force the storage-backend question. Nine
floats per particle per step is ~144 MB per episode at MPM scale (10,000
particles × 400 steps × 9 × 4 bytes) before compression. `store_F_as_float16`
halves it as a stopgap; the header names h5py as the move when halving stops
being enough.

### 8.2 `src/actions.py` — the action space is chosen for the planner

Two decisions, both made because MPPI and CEM work by drawing thousands of
perturbations around a nominal action sequence, and that sampling is only well
behaved when the space is small, bounded and centred at zero.

- **Deltas, not absolute poses.** An absolute pose is a point in a workspace
  whose origin is arbitrary; a Gaussian around it is not a natural distribution
  over anything. A delta is zero-centred by construction. (This is the same
  reasoning as §6's "predict position deltas," now applied to the input side.)
- **Axis-angle for rotation deltas, quaternions for absolute poses.**
  Quaternions double-cover the rotation group — `q` and `−q` are the same
  rotation — so a network regressing a quaternion sees two correct answers for
  every input and is penalised for picking either. Small rotations as axis-angle
  are smooth through zero. Absolute poses stay quaternions because that is what
  the schema and every simulator already use.

**The `atan2` rule.** Extract a rotation angle with

```
theta = 2 * atan2(|q_v|, q_w)        NOT        theta = 2 * acos(q_w)
```

Normalising a quaternion in floating point can leave `q_w` a few ulps above 1.0,
and `acos(1.0000000000000002)` is NaN. The NaN then propagates through a whole
trajectory of composed rotations and is discovered much later as a dataset full
of holes. `atan2` is total — defined for every pair of finite inputs, needing no
clamping.

Conventions are stated once and enforced in one file: quaternion layout is
**scalar-last** `[qx, qy, qz, qw]` (what `ee_pose` already stored), rotation
deltas are **world-frame** with the delta applied on the left, so translation
deltas add and rotations multiply with no per-step change of basis to get wrong.

### 8.3 `src/materials.py` — why (E, ν) is the wrong thing to randomise

§6 already committed to randomising material parameters per episode. This file is
the discovery that doing it the obvious way is a trap.

```
lambda = E*nu / ((1 + nu)(1 - 2nu))
```

is singular at ν = 0.5, and soft tissue sits at ν ≈ 0.49 — right next to the
singularity. ν = 0.49 and ν = 0.499 differ by 0.2% and give λ values roughly 10×
apart. **A uniform sweep over ν is therefore a wildly non-uniform, heavy-tailed
sweep over the quantity the solver actually uses.** So μ and λ are sampled
directly, log-uniformly. Conversions to and from (E, ν) still exist, because that
is the parameterisation every paper and every indentation test reports.
`NU_SINGULARITY_LIMIT = 0.4999` refuses to convert above the cutoff rather than
returning a number nobody should trust.

The stored representation is `[log μ, log λ, ρ]` — logs for the two that span
decades, linear for density, which varies by a few percent and whose log would be
a near-constant. Fed raw Pascals, a network spends its capacity encoding the
exponent.

**The ranges in this file are placeholders and are labelled as such in the
source.** They are order-of-magnitude stand-ins for generic soft tissue. Liver,
bowel and fat differ from each other by more than the width of these ranges.
Before any result is claimed from a model trained on them, they need replacing
with values fitted to indentation or uniaxial-tension curves for the specific
tissue being retracted. This is the same gap §5 named and it has not closed.

Material is one global triple per episode today and will be per-particle when
heterogeneous tissue arrives. Every function that *consumes* material accepts
`(3,)` or `(N, 3)` and broadcasts, so that becomes a data-generation change rather
than a rewrite.

### 8.4 `src/tissue_metrics.py` — report the max, optimise the surrogate

A model that predicts particle positions is not yet useful: an MPC loop cannot
score a rollout on positions, it needs a scalar cost. Two metrics, defined once so
the simulator, the validator, the training targets and the planner cannot drift
into disagreeing about what "success" and "unsafe" mean:

- `exposure` — success. Fraction of the target region the tissue no longer
  occludes. Higher is better.
- `safety_strain` — safety. A soft maximum over per-particle stretch, standing in
  for tearing. Lower is better.

Two splits run through the file, both for the same reason: **the physically
meaningful quantity has a bad gradient.**

1. **Eigenvalues vs invariants.** Maximum principal stretch is what maps to a
   tissue-tear threshold in the literature, so it is what gets *reported*. But it
   needs an eigendecomposition of `FᵀF`, and eigenvalue gradients diverge when
   eigenvalues coincide — and at rest `F = I`, where all three coincide exactly.
   *The undeformed state sits precisely on the degenerate point.* So anything in
   a loss path uses the Neo-Hookean energy or the raw invariants `J = det(F)` and
   `I₁ = tr(FᵀF)`, which are polynomial in F and smooth everywhere `J > 0`.
2. **Hard max vs soft max.** Injury is a maximum phenomenon — tissue tears at the
   single worst point, not on average. But a hard max over ~10,000 particles is
   non-smooth and hands the entire gradient to one particle per batch. So
   aggregation uses a power mean, and the true max is returned alongside it.

A naming trap recorded in the source: **`safety_strain` is a stretch ratio, not a
strain.** Undeformed is 1.0, not 0.0. It is named for its role in the cost
function, not for its units.

These metrics are deterministic functions of state, so they are computed rather
than stored as an independent source of truth — but the schema logs them anyway,
precisely so `validate_dataset.py` can recompute and compare (§8.5). And each
lives behind a named function so that later, when the privileged inputs they need
are no longer available at control time, the body of `compute_exposure` becomes a
learned head and every caller stays as it is.

### 8.5 `host/validate_dataset.py` — 8 checks became 13

The session-1 version had eight checks. It now has ten per-episode and three
dataset-wide. The new ones all test properties that only exist once there is a
constitutive model:

- **`F` is admissible** — `det(F) > 0`. A negative determinant is an *inverted*
  element: the material has been turned inside out. That is not a large
  deformation, it is a solver failure, and the Neo-Hookean energy takes `ln(J)`
  and is undefined there.
- **`F` is near-incompressible** — soft tissue is mostly water. A solver
  reporting 30% volume change is not modelling tissue; usually λ was mis-scaled
  on the way in. Deliberately a soft check, since a compressible phantom is a
  legitimate thing to simulate.
- **Substep is stable for the stiffness** — MPM stability follows the elastic
  wave speed, so the stable step shrinks like `1/√E`. Once stiffness is
  randomised over an order of magnitude, the stiffest episodes need a substep
  several times smaller than the softest, and a collector with one fixed
  "default" substep silently produces garbage at the stiff end. This check is why
  `substep_dt` had to go into the schema.
- **Contact-mode transitions are legal** — `NONE → GRASP` in a single step means
  the jaws closed on tissue they were not touching, which is a labelling bug
  since contact must precede a grasp. A mode changing on almost every step is a
  threshold chattering around its cutoff.
- **Logged metrics match recomputation** — the metrics are both stored *and*
  computable, which means two sources of truth for the quantity the planner
  optimises. They drift: a collector pinned to an old σ, a metric whose default
  changed, an exposure logged before the last solver substep rather than after.
- **Material is diverse** — the deeper twin of the existing deformation-diversity
  check. Every episode can move differently while sharing one stiffness, because
  the motion differs through grasp point and direction. A model trained on that
  learns *that* stiffness, not the dynamics.

**`SKIP` is not a soft fail.** v1 episodes predate `F`, the material parameters
and the target region, and PyBullet's mass-spring cloth has no constitutive model
to have Lamé parameters *for*. Those are legitimate data collected before the
schema grew. A check that cannot apply says SKIP and says why — it does not fail,
and it does not quietly pass either, which would be worse: a green tick against a
property that was never tested is how a dataset gets trusted for something it
never demonstrated.

Exit code is 0 unless something FAILs, so the script can gate a collection run.

### 8.6 `src/synthetic_traj.py` — breaking the "untestable until the simulator lands" deadlock

This is the move that makes the rest of the session verifiable, and it is worth
generalising.

The MPM simulator does not exist yet, so no episode anywhere carries a deformation
gradient. Everything built to consume one — the schema, the metrics, the
validator, eventually the model — is therefore untestable against real data. And
"untestable until the simulator lands" means every bug surfaces at once, tangled
together with the simulator's own.

Four episodes break that deadlock. Each is a deformation **written down
analytically, so `F` is not estimated from the motion — the motion is generated
from `F`.** The expected value of every metric is then exact rather than
approximate, and a discrepancy points at the code under test rather than at
discretisation error.

| kind | deformation | what it pins down |
|---|---|---|
| `rest` | nothing moves, `F = I` | zero strain, constant exposure |
| `uniaxial` | isochoric stretch 1.0 → 1.4 | `J = 1` exactly at every step |
| `rotation` | rigid-body rotation, `F = R(t)` | **zero strain despite large motion** |
| `retract` | clamped slab sheared off target | exposure rises monotonically |

**The rotation case is the one that matters.** It is the end-to-end version of the
pure-rotation unit test: that test proves the metric is frame-indifferent in
isolation; this proves nothing between there and the metric reintroduces the error
— not the writer, not the float32 cast, not the reader, not the validator. *A
stack that reports strain for a rigidly rotating body is measuring its own
coordinate system.*

**What this is not:** a simulator. Nothing here solves an equation of motion, and
none of these deformations is a response to the forces logged alongside it. The
end-effector poses are plausible decoration so that consumers expecting a complete
v2 file get one. **Do not train a dynamics model on these episodes and conclude
anything** — the dynamics are prescribed, so a model would learn the ramp, not the
physics.

### 8.7 `tests/` — 200 unit tests, and why they live in the main environment

Four files: `test_actions.py` (46), `test_materials.py` (60),
`test_tissue_metrics.py` (61), `test_validate_dataset.py` (33).

`pytest` was added to `host/environment.yml` rather than to a separate dev
environment, with the stated reason that the metric and rotation tests **encode
physics that must hold** — a pure rotation is not strain; `det(F) > 0` — so they
are part of the working setup rather than an optional extra.

### 8.8 What session 2 did *not* do

Recorded so this is not mistaken for more progress than it was.

- **§4 is still open.** `DT = 1.0 / 1000.0` remains hardcoded at line 67 of
  `container/collect_retraction.py`, and `timestep_study.py` has still never been
  run. The 20 episodes in `data/` carry `dt = 0.008` (1/1000 physics step, logged
  every 8th), so **the entire existing dataset was collected at the timestep the
  log flagged as marginal.** Session 2 routed around this rather than resolving
  it: the MPM path has its own stability story (§8.5), and PyBullet's may simply
  stop mattering.
- **No MPM simulator.** Nothing yet writes a real `F`. `synthetic_traj.py` is
  scaffolding, explicitly not a simulator.
- **No new physics data.** `data/` is unchanged since 1 August.
- **`validate_physics.py` was never run** — see §10.
- **The v1 dataset is thin.** Across sampled episodes, peak displacement spans
  26.5–29.4 mm — a 4% spread, with no material randomisation at all. The
  diversity check passes, but 20 near-identical retractions is not much signal.
  The machinery to fix this now exists (`sample_material`) and has not been
  applied to a collection run.

### 8.9 State as verified on 17 August 2026

```
python host/validate_dataset.py --data data_synth/
    42 passed, 1 warned, 0 skipped, 0 failed
    (the WARN is rest.npz reporting no motion — which is that episode's purpose)

python host/validate_dataset.py --data data/          # 3 episodes sampled
    17 passed, 0 warned, 16 skipped, 0 failed
    (SKIPs are the v2-only checks correctly declining on v1 files)
```

The 200-unit suite was **not** re-run on 17 August. The pytest cache records a
clean last run (`lastfailed: {}`, 200 collected), but that is a stale record, not
a verification. Run `pytest tests/ -v` to confirm.

---

## 9. Session 3 — 17 August 2026: recovery, one real bug, and the MPM route

Written the same day as the work, which §8 is the argument for.

### 9.1 The reasoning was not under version control

`DECISION_LOG.md` and `SESSION_TRANSCRIPT.md` existed only in the claude.ai
project. Never in git, never diffable against a commit, invisible to anyone
cloning the repository — the two documents carrying every "why" in the project
lived outside it. `container/validate_physics.py` was in the same position: cited
in §5 for sixteen days, never written to disk.

Also found: four days of work (13–16 August, all of §8) had never been committed.
Roughly 120 KB across seven new files with no git history behind it.

All now committed, in four commits. The near-loss is the point — **the work most
worth keeping was the work least protected**, because writing code feels like
progress and committing it does not.

### 9.2 `boundary_mask` was wrong in three of four synthetic episodes

The first bug found *by* the v2 validation layer, and it was found by asking a
question the layer could not answer.

All four kinds in `synthetic_traj.py` shipped the same 22-particle clamp mask,
because `_clamped_edge()` was applied unconditionally. Only two honour it:

```
episode     clamped   max disp CLAMPED
rest             22          0.000 mm    consistent
uniaxial         22         12.868 mm    *** file lied ***
rotation         22         60.000 mm    *** file lied ***
retract          22          0.000 mm    consistent
```

**The deformations were right; the metadata was wrong.** `rotation` *must* move
every particle — clamp one and it stops being rigid, which destroys the exact
property that episode exists to demonstrate. Same for `uniaxial`: a clamped edge
makes the stretch non-uniform and `J = 1` exactly stops holding. So the fix was
never to change the physics, it was to stop the file asserting a constraint the
motion does not honour.

The other three now carry an **all-False** mask, not an empty one. Empty means
"this simulator did not record it"; all-False means "recorded, nothing clamped."
We know the answer, so claiming ignorance would be its own small lie — the same
distinction §8.1 draws for every other v2 field.

**Nothing caught this because no check read `boundary_mask`.** Thirteen checks,
none touching the field. Now fourteen:

> **`check_boundary_is_held`** — particles marked kinematically clamped do not
> move. SKIPs when no mask is recorded.

This is the §3.5 failure — anchors silently not holding — made detectable **from
the file alone**. In August that was found by noticing peak displacement equalled
the gripper's own travel, which required knowing the gripper's travel and
thinking to compare. The check asserts the property directly.

`collect_retraction.py` now writes the mask it already computes in
`build_scene`, and writes what was *actually* anchored rather than what the
geometry identifies — with `ANCHOR_BOUNDARY` off the mask is all-False, a real
reading rather than a statement of intent. The 20 episodes in `data/` predate
this and record no mask; the check SKIPs on them.

**Verified against the broken files, not just the fixed ones:** the old
`rotation.npz` and `uniaxial.npz` FAIL with the exact displacements above, while
the corrected set passes 46/46. A check only demonstrated on good data has not
been demonstrated.

*Generalises to:* a schema field with no check on it is a field that will drift.
The empty-vs-zero doctrine of §8.1 protects the *reader* from a fabricated
default; it does nothing about a *writer* that fabricates a real-looking value.

### 9.3 The MPM route: Taichi decides it, and it decides it on architecture

Investigated whether SurRoL's own MPM makes further simulator work unnecessary.
It largely does — but the branch matters, and so does where it can run.

**Two different MPMs in that repository:**

- `SR-VPPV/Data_driven_scene_simulation` — [arXiv:2405.00956](https://arxiv.org/abs/2405.00956),
  physics-embedded 3D Gaussians reconstructed **from stereo endoscopic video**.
  MPM in service of visual realism. Wrong tool here: there is no video, and the
  deformation is fitted to look right rather than to a constitutive law with
  parameters you set.
- **`Dev`** — [arXiv:2402.01181](https://arxiv.org/abs/2402.01181), Taichi MPM,
  Neo-Hookean, dVRK-compatible. This is the destination, as §2.3 already said.

**The blocker is architectural and absolute: Taichi has no Linux ARM64 wheel.**
Not in the pinned 1.6.0, not in current 1.7.4. Wheels cover Windows x86-64,
Linux x86-64, macOS arm64 — and nothing for Linux ARM. `panda3d==1.10.11` ships
aarch64 wheels and `pymeshlab` covers Linux ARM64; Taichi is the sole holdout.

So the MPM **cannot run in the container**, at all, without building Taichi from
source against LLVM for aarch64. It runs on the host, where the macOS arm64 wheel
already sits in `environment.yml`.

§1.2 chose host+container to keep Metal reachable, expecting a speed argument.
The real consequence is stronger: **the host is not where Taichi runs best, it is
the only place it runs.** This was already recorded in a comment in
`host/environment.yml` on 16 August — "There is no ARM Linux build of Taichi at
all" — and was still rediscovered from PyPI a day later, because a code comment
is not where anyone looks for a platform decision. That is this section's second
argument for itself.

**`MPM/` lifts cleanly.** Four files — `config.py`, `mpm3d.py`, `sdf.py`,
`requirements.txt` — importing `taichi`, `pybullet`, `skimage`, `numpy`. No
`surrol.*`, no `panda3d`. Notable:

- Backend selection already prefers Metal, then Vulkan, then CUDA, then CPU.
- **Dense grids only** — no `pointer`/`bitmasked` SNodes, which is the family of
  Taichi feature Metal does not support. The main portability risk is absent.
- Per-particle fields `F_x`, `F_v`, `F_C`, `F` (deformation gradient), `FJ` map
  almost one-to-one onto the v2 schema.
- Neo-Hookean: `stress = mu*(F@Fᵀ − I) + I*la*log(J)`, with μ/λ from
  `set_parameters()`. `materials.py` is not approximately right for this
  interface, it is exactly right — and §8.3's ν → 0.5 warning finally has a
  concrete consumer.
- API is `init()` / `step()` / `reset()` / `get_mesh()` / `substep()`. PyBullet
  supplies rigid-body collision via an SDF, which is the seam the PSM plugs into
  later.

**Decision: vendor it** at `third_party/MPM/`, recording the upstream commit SHA.
Not a submodule — these four files will be edited to expose particle state and to
script the tool, and a submodule you cannot edit is friction without benefit. The
SHA is what makes the fork honest.

**Open, to be settled by a smoke test rather than argument:**

1. `ti._lib.core.with_metal()` is a **private** Taichi API and their code pins
   1.6.0 while the host has 1.7.4. If that path moved, the guard fails silently
   and everything runs on `ti.cpu` — correct physics, wrong speed, no error.
   Print the selected backend and confirm it.
2. `pybullet` and `scikit-image` are not yet in the host environment. PyBullet
   has no macOS arm64 wheel and compiles from source, 5–8 minutes with no
   output. Not a hang (§2.2 documents the same for the container).
3. `from MPM.config import ...` means the directory must be importable *as*
   `MPM`, so its parent has to be on `sys.path`.

**Still not done:** nothing writes MPM state to `.npz`. That adapter — read
particle state, write a v2 trajectory — is the next piece of code, and it is
small precisely because §8 built its receiving end first.

---

### 9.4 The MPM smoke test: all three questions closed, two new facts

`host/smoke_test_mpm.py`, 17 August 2026. **15 checks pass, 0 fail, 1 warns.**
The MPM runs on this machine, on the GPU, and its output goes into a v2 `.npz`.

The four files were vendored at `third_party/MPM/` from `Dev` at
**`cb797f360bb16ea629b449cb902a1dae60c46e81`** (2024-05-21), byte-identical to
upstream, with `third_party/PROVENANCE.md` recording the SHA, the sparse-checkout
command that fetched them, and a `sha256` per file.

**Q1 — the private Taichi API. Closed, favourably.** `ti._lib.core.with_metal()`,
`with_vulkan()` and `with_cuda()` all still exist on 1.7.4 despite the code
pinning 1.6.0, returning `True / True / False`. `mpm3d.py` selects `ti.metal`,
and — the question that actually mattered — `ti.lang.impl.current_cfg().arch`
reads back `Arch.metal` after import. The silent CPU fallback did not happen.

Both halves of that are checked, because they are different claims.
`with_metal()` returning `True` says only that the Taichi binary was *built*
with Metal support; `ti.init()` can still fall back at runtime and mention it
in a log line nobody reads. Asking only the first question would have confirmed
the guard works while saying nothing about what it selected.

Worth keeping in mind: `mpm3d.py` calls `ti.init()` at **module level**
(lines 14–22). Importing it *is* the backend decision. Nothing downstream can
choose an arch, and any script wanting a different one must set it before the
import.

**The backend check was demonstrated against the failure, per §5.** A check only
ever seen passing has not been shown to work, so the CPU fallback was forced:

```bash
TI_ARCH=arm64 python host/smoke_test_mpm.py    # Q1b FAILs, exit 1
```

(`TI_ARCH=cpu` is *not* a valid arch name on 1.7.4 — it aborts the process with
`RHI Error: Unknown architecture name: cpu` and SIGABRT. On Apple Silicon the CPU
arch is `arm64`. Worth knowing before reading that abort as a broken install.)

That run reported `backend=Arch.arm64, NOT metal` and exited 1, which is the
behaviour wanted. It also produced the number that had been guesswork: **Metal
is 8.1× faster**, 0.64 ms/substep against 5.17 on the CPU. Earlier drafts of this
check asserted "10–30×" from general knowledge; the measured figure is lower, and
is now what the code says.

The other half of that result is the more important one: **the CPU run settled
the cube to −4.39 mm against Metal's −4.46 mm**, within 2%. Correct physics,
wrong speed, no error anywhere — the failure mode §9.3 predicted, reproduced on
demand and confirmed to be invisible without this check.

**Q2 — the missing host dependencies. Closed, but not by a plain install.**
`scikit-image` 0.26.0 came from a wheel. **PyBullet did not build**, and §9.3's
prediction that it would merely be slow was wrong — it *fails*:

```
zutil.h:128:   #ifndef fdopen
               #define fdopen(fd, mode) NULL   /* No fdopen() */
               #endif
_stdio.h:322:  error: expected identifier or '('
```

PyBullet vendors a very old zlib. `gzguts.h` includes the system `<stdio.h>`
*after* that macro is defined, so the macro rewrites the SDK's real declaration
of `fdopen` into `FILE *((void*)0)(int, const char *)`. Modern macOS SDK, 2013
zlib.

The guard is `#ifndef fdopen` — not zlib's later `#ifndef HAVE_FDOPEN` — so the
usual `-DHAVE_FDOPEN` does nothing. What works is defining the macro to itself:

```bash
export CFLAGS="-Dfdopen=fdopen" CXXFLAGS="-Dfdopen=fdopen"
pip install pybullet
```

C does not rescan a macro that expands to its own name, so `fdopen` stays
`fdopen`, `#ifndef fdopen` is false, and the bad `#define` is skipped. Built
`pybullet==3.2.7` in about four minutes. This is now in `host/environment.yml`
with the reasoning, but the flag has to be exported *before* `conda env create`
— a YAML file cannot set it, so an unprepared `conda env create` still fails
here. That is the one rough edge left in the host recipe.

Rejected: conda-forge's `pybullet`, which has an osx-arm64 build and would have
been one line. It would have put a compiled package from a second package
manager into an environment whose entire numerical stack comes from pip — the
thing §3.2 exists to prevent. A four-minute build is cheaper than re-litigating
`OMP: Error #15`.

**Q3 — the import path. Closed, as expected.** `third_party/` goes on
`sys.path`, not `third_party/MPM/`. `from MPM.config import ...` is an absolute
import of a top-level package named `MPM`; putting the inner directory on the
path makes `config` importable and `MPM.config` still fail. There is no
`__init__.py` and none is needed — it resolves as a PEP 420 namespace package,
which is also how the vendored tree stays byte-identical to upstream.

#### New fact 1: `set_parameters()` takes (E, ν), and §9.3 said otherwise

§9.3 recorded "Neo-Hookean … with μ/λ from `set_parameters()`" and concluded
`materials.py` "is not approximately right for this interface, it is exactly
right." The first half is wrong. The signature is:

```python
def set_parameters(s_E=8000, s_nu=0.2):
    mu[None] = E[None] / (2 * (1 + nu[None]))
    la[None] = E[None] * nu[None] / ((1 + nu[None]) * (1 - 2 * nu[None]))
```

It takes **(E, ν)** and performs internally the exact conversion §8.3 exists to
keep out of the sampling path — the one singular at ν = 0.5, in float32, with a
default `s_nu=0.2` nowhere near tissue. Feeding it a sampled (μ, λ) means
inverting to (E, ν) and letting it convert back: a round trip *through* the
singularity, to recover numbers we already had.

Measured rather than assumed, over the five corners of `materials.py`'s ranges:
**worst relative error 2.7e-4**, at μ = 200, λ = 2×10⁶ (ν = 0.499950). Small —
smaller than the placeholder ranges' own uncertainty — so this was never going
to be the bug that ate a week. But it is avoidable for nothing: `mu` and `la`
are plain 0-d Taichi fields, the solver reads *only* those two, and writing them
directly round-trips exactly (relative error 0.0).

**The adapter will write `mu[None]` and `la[None]` directly and never call
`set_parameters()`.** §8.3's rule survives contact with its first real consumer,
which is the outcome that matters; it just needs one more line of code than §9.3
thought.

The wider point is that §9.3 reached "exactly right" by reading the paper's
description and the field names, without running anything. Four of its claims
held and one did not, and the one that did not is the one that touches the
project's sharpest rule.

#### New fact 2: the first run takes ten seconds and is not hung

Steady state is **0.64 ms/substep** — 24 000 particles on a 64³ grid, 62
frames/s against 80 for realtime, so roughly 0.8× realtime on the M3 Max GPU.

But the *first* substep on a fresh machine takes **~10 s**, compiling P2G,
Boundary and G2P for Metal. Taichi caches the result under `~/.cache/taichi`,
after which it is ~0.56 s. Confirmed by disabling the cache: `TI_OFFLINE_CACHE=0`
puts it straight back to 9.68 s, with steady-state speed unchanged at 0.64 ms.
So it is compilation, not a slow first step, and the timing number to quote is
the steady-state one.

This matters only because a ten-second silent pause on a fresh clone looks
exactly like a hang, and §2.2 and §9.3 both already had to say the same thing
about PyBullet's build. Third instance of the pattern: **on this project, a long
silence is usually a compiler.**

#### What the test does that is worth keeping

`substep()` reads the module globals `SDF` and `collision_mask`, which are
`None` until `step()` rebinds them from fields in `sdf.py` (`mpm3d.py:492–494`).
The test binds them by hand and fills the SDF with a large positive distance,
which means the solver can be driven **with no PyBullet scene at all** — the
collision branch is never taken, so no neighbour of `SDF` is ever indexed. That
is the seam the PSM plugs into later, and being able to run without it is what
makes this a smoke test rather than an integration test.

Two physics checks, not one, because they fail differently:

- **Free fall**, cube lifted clear of the floor, checked against analytic
  −g·t: −0.12250 m/s vs −0.12250, 0.00 % off. This confirms both the axis
  (`Boundary()` applies gravity to `F_grid_v[I][2]`, the PyBullet convention,
  having abandoned Taichi's y on the line above) *and* the magnitude. A sign
  test alone would pass on a solver using the wrong g.
- **Settling under contact**, the default spawn, which deforms the material:
  J = det(F) ∈ [0.8575, 1.0750], mean 0.9971, all finite, none inverted.

The first version of the gravity check **failed**, reporting `v_z = +0.0667 m/s`.
That was the test's fault, not the solver's: `init_cube()` spawns particles at
z ∈ [0.05, 0.10] while the floor is the `bound = 3` band at z = 3·dx = 0.047, so
the cube starts already in contact and what was being measured was elastic
rebound. Recorded because it is the §5 discipline working — the check was
specific enough to be wrong in an informative way, rather than vague enough to
pass. Fixing it produced a strictly better check, against an analytic value
instead of a sign.

Finally, the mapping §9.3 called "small" is now verified rather than hoped for:
`F_x → tissue_pos` (24000, 3), `F_v → tissue_vel` (24000, 3), `F → tissue_F`
(24000, 3, 3), all float32, all lining up with the v2 schema without a reshape,
and two frames written through `TrajectoryWriter` and read back at
(2, 24000, 3, 3) for 2.6 MB.

**One number that should worry us:** 2.6 MB for two frames, essentially all of
it `tissue_F`. Nine floats per particle per step, at 24 000 particles, is
864 KB/frame. A 100-step episode is ~86 MB and twenty of them is ~1.7 GB.
`trajectory_io.py` already anticipated this — its `store_F_as_float16` flag and
the "WHEN TO ACTUALLY MAKE THAT SWAP" note name `tissue_F` as the trigger. It
has now been reached. Not resolved here, but it is no longer hypothetical.

**Still not done, unchanged from §9.3:** nothing writes MPM state to `.npz` as
part of a real episode. The adapter is still the next piece of code — but it now
starts from a verified mapping, a known-good parameter interface, and a solver
that has actually run.

---
## 10. Files

Every path below was checked against the repository on 17 August 2026.

**Correction.** Earlier versions of this table listed files that had been written
during a session but never saved to disk. `container/validate_physics.py` was one
of them: it was described in §5 for sixteen days while not existing in the
repository at all. It has now been restored from the session artifact and
committed — but **it has never been executed**, and it was written against the
session-1 PyBullet/v1 world, so treat its correctness as unverified until it has
actually been run.

Two documents were in the same position and are also now in the repo:
`DECISION_LOG.md` (this file) and `SESSION_TRANSCRIPT.md` existed only in the
claude.ai project knowledge base. They were never under version control, could
not be diffed against a commit, and would not have reached anyone cloning the
repository. From here on they live alongside the code they describe; the project
copy is the shareable view, not the only copy.

| Path | Runs on | Purpose | Status |
|---|---|---|---|
| `README.md` | — | Orientation | |
| `SETUP_GUIDE.md` | — | Full annotated setup walkthrough | |
| `CHEATSHEET.md` | — | Day-to-day commands | |
| `DECISION_LOG.md` | — | This file | added to repo 17 Aug |
| `SESSION_TRANSCRIPT.md` | — | Chronological session-1 record | added to repo 17 Aug |
| `src/trajectory_io.py` | **both** | The data contract, schema v2.0 | |
| `src/actions.py` | **both** | Action space: pose deltas, axis-angle | §8.2 |
| `src/materials.py` | **both** | Lamé sampling, log-space packing | §8.3 |
| `src/tissue_metrics.py` | **both** | `exposure`, `safety_strain` | §8.4 |
| `src/synthetic_traj.py` | **both** | Analytically-known episodes | §8.6 |
| `docker/Dockerfile` | — | Container environment, fully commented | |
| `host/environment.yml` | macOS | Host environment | |
| `host/verify_host.py` | macOS | GPU reachability checks | |
| `host/train_dynamics.py` | macOS | MLP baseline on MPS | |
| `host/visualize_trajectory.py` | macOS | 3D animation + stability diagnostics | |
| `host/validate_dataset.py` | macOS | 14 data-integrity checks | §8.5, §9.2 |
| `host/smoke_test_mpm.py` | macOS | MPM smoke test, 15 checks | §9.4, **passes** |
| `third_party/PROVENANCE.md` | — | Upstream SHA, checksums, local edits | §9.4 |
| `third_party/MPM/` | macOS | Vendored SurRoL `Dev` MPM, 4 files, unmodified | §9.3, §9.4 |
| `container/verify_container.py` | Linux | Physics + SurRoL checks | |
| `container/make_tissue_mesh.py` | Linux | Author the tissue sheet | |
| `container/collect_retraction.py` | Linux | Scripted retraction episodes | |
| `container/timestep_study.py` | Linux | Convergence study | **never run** (§4) |
| `container/validate_physics.py` | Linux | Nine controlled physics experiments | **never run** |
| `tests/` | macOS | 200 unit tests, four files | §8.7 |

**Data and artifacts** (git-ignored):

| Path | Contents |
|---|---|
| `data/` | 20 PyBullet retraction episodes, schema v1.0, 1 Aug |
| `data_synth/` | 4 analytic episodes, schema v2.0, regenerate with `synthetic_traj.py` |
| `models/dynamics_mlp.pt` | MLP baseline, 1 Aug |
| `assets/tissue_20x20.obj` | Authored tissue mesh (tracked) |

---

## 11. References

- Xu et al., *SurRoL*, IROS 2021 — [arXiv:2108.13035](https://arxiv.org/abs/2108.13035)
- *Efficient Physically-based Simulation of Soft Bodies in Embodied Environment for Surgical Robot* — [arXiv:2402.01181](https://arxiv.org/abs/2402.01181)
- Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021 — [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
- *Autonomous Soft Tissue Retraction Using Demonstration-Guided RL* — [arXiv:2309.00837](https://arxiv.org/abs/2309.00837)
- Long et al., *Surgical embodied intelligence for generalized task autonomy*, Science Robotics 2025
- [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) · [Isaac Sim cloud deployment](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_cloud.html)
