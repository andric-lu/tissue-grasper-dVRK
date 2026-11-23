# BM2 — Soft Tissue Dynamics: Decision Log

**Sessions:**
1. 1 August 2026 · Environment setup and first working data pipeline (§1–§7)
2. 13–16 August 2026 · Schema v2, metrics, and the testable-before-the-simulator
   layer (§8) — *reconstructed from source on 17 August, see the note there*
3. 17 August 2026 · Documentation recovered into git, boundary_mask bug, MPM
   route settled (§9)
4. 31 August 2026 · The dVRK PSM and a persistent grasp (§10)
5. 15 September 2026 · Phase 0 of the MPM→GNN brief: locality, coarsening,
   graph statistics, MPS scatter-add — measured before building (§13)

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

### 9.5 The adapter: four bugs, three of them invisible

25 August 2026. `host/mpm_adapter.py` reads MPM particle state and writes a v2.1
episode. The adapter itself is small, as §9.3 predicted. What was not predicted
is that writing it would expose four defects, three of which produced data that
was **internally consistent and physically wrong** — the failure mode this
project's entire validation discipline exists to catch, arriving all at once.

Verified at the end of this section: **249 tests pass**, `trajectory_io.py`'s
self-test passes, `smoke_test_mpm.py` is **16/16 on Metal**, and two fresh
100-step episodes validate with **0 FAIL**.

#### What the adapter does

`MPMRecorder`, one instance per episode. It samples a material, writes μ and λ
straight into `mpm3d.mu[None]` / `mpm3d.la[None]` (never `set_parameters()` —
§9.4), chooses a substep from the material, drives `substep()` directly with a
hand-bound SDF so no PyBullet scene is needed, and appends frames through
`TrajectoryWriter`. Four rules from CLAUDE.md meet their first real consumer
here, and they are recorded in the module docstring rather than only here.

**Schema 2.1** carries what 2.0 could not:

- `f_encoding`, with **`delta16`** the new default for `tissue_F`. F sits *at*
  1.0 at rest, and float16's spacing at 1.0 is ~9.8e-4, so a small strain
  rounds away entirely: a 0.01% stretch stores as exactly 1.0 under plain
  float16. `delta16` stores `F − I`, putting the quantisation near zero where
  float16 is dense, and the same stretch survives as 1.00010002. This is
  demonstrated against the alternative in `tests/test_trajectory_io.py`, not
  asserted — the float16 counterpart is run through the same writer and shown
  to lose it.
- `particle_ids` / `n_particles_simulated`. The solver needs 24,000 particles;
  MeshGraphNets takes 1.5k–5k nodes. **3,000 are recorded, and the subset is
  fixed for the whole episode** — node identity must be stable across time or
  consecutive frames describe different particles. Metrics are computed over
  **all 24,000 before subsampling**, because `safety_strain` is a maximum and a
  maximum over a subset is biased low.
- `grid_dx`, recorded rather than assumed. §9.2's rule about measuring the world
  you operate in; the validator previously guessed 1 mm against the solver's
  1/64 m and rejected a stable episode by a factor of 15.6.

Size, the §9.4 worry: 11.5 MB per 100-step episode at 3,000 particles in
delta16, against the ~86 MB that 24,000 in float32 would have cost. Subsetting
did more than the encoding, but both were needed.

#### Bug 1 — the timebase, which no file contradicted

`self.n_substeps` was computed correctly from the P-wave bound. It was then
**never used**. `advance()` looped `range(self.m.steps)` — the vendored module
default of 25 — and the writer was handed `n_substeps=self.m.steps` too.

For the stiffest sampled material the recorded frame claimed `dt` = 12.5 ms and
actually advanced 25 × 117.9 µs = **2.95 ms**. A factor of **4.2**.

Nothing on disk contradicted anything. `dt`, `substep_dt` and `n_substeps` were
each individually plausible; only their product was wrong. Every check then in
existence passed. A model trained on it would have learned tissue roughly four
times too stiff, and no diagnostic would have pointed here — it would have
looked like an architecture problem, which is exactly what §8.5 says a validator
exists to prevent.

The fix is that **`self.n_substeps` is the single source of truth**: `advance()`
loops it, the writer receives it, and `mpm3d.steps` is set from it so module
state stays coherent for anything else that reads it. `mpm3d.dt` was always
being set correctly; the bug was that `steps` was not set with it.

Three guards now, because one was demonstrably not enough:

1. `MPMRecorder.__init__` asserts `n_substeps * substep_dt == frame_dt` where
   the two are established.
2. `TrajectoryWriter.__init__` raises when `dt` disagrees with
   `substep_dt * n_substeps` — the caller with the bad numbers is still on the
   stack. Guarded on both being non-zero, so v1 and PyBullet callers, which
   record neither, are untouched.
3. `check_timebase_is_consistent` in the validator, for files already on disk
   and for any writer that is not ours. SKIPs when the fields are absent.

**`TIMEBASE_TOLERANCE` is relative (1e-6), not absolute, and that was measured
rather than guessed.** The first version used an absolute 1e-9 s. A real
episode — 12.5 ms in 24 substeps — drifts **4.7e-10 s**, because `substep_dt` is
stored as float32 and reconstructing `dt` from it accumulates `n_substeps`
roundings. A factor of two of margin is one unlucky material away from
rejecting good data for arithmetic reasons. The relative form is scale-free:
1e-6 sits four orders above float32 reconstruction error and four below the
smallest error worth catching (one substep miscounted in a thousand is 1e-3).

#### Bug 2 — density was decoration

`material_params[2]` is ρ, sampled per episode. The solver never saw it. `p_mass`
is a module-level Python float read inside the P2G kernel (`mpm3d.py:172,180,181`)
and it was left at the vendored `p_vol * 1000` for every episode.

So the dataset had a density column that did not describe the physics that
produced it. That is *worse* than not recording density: a model conditioning on
it would learn to depend on a number that never influenced anything. Measured
magnitude: at a sampled ρ = 1080 kg/m³ the vendored default is **7.4% off**.

The adapter now sets `p_rho` and `p_mass` before the first `substep()`. Like
`dt`, `p_mass` is baked in at kernel-compile time, so the ordering is not
optional — which is why the per-process lock was extended from `dt` alone to the
**`(substep_dt, p_mass)` pair**. A lock covering only `dt` would let a second
episode with the same stiffness but a different density run silently at the
first episode's mass: the identical failure mode, minus the error message.

Demonstrated in `smoke_test_mpm.py`'s 16th check, which also asserts that the
vendored default and the sampled ρ actually differ — a check that could pass
while proving nothing is not a check.

#### Bug 3 — the exposure bound was wrong, and passing on luck

`check_logged_metrics_match_recomputation` demanded that recomputed exposure
**equal** the logged value. For a subset record that is the wrong claim in the
wrong direction.

Exposure is the fraction of the target *not* occluded. Removing particles can
only remove occluders, so a subset can only ever look **more** exposed than the
full set it came from. The relation is a monotone inequality, not an equality.
Measured, to be sure rather than to argue it: dropping 24,000 → 3,000 moves
exposure by +1.4e-3, and 24,000 → 300 by +0.12. Always upward, never down.

The equality bound was rejecting correct data. The first fresh 100-step episode
logged exposure 0.0 over 24,000 particles and recomputed 1.44e-3 over the stored
3,000 — a real, expected, physically necessary gap, called drift by a 1e-3
tolerance. The two stale episodes from 17 August had passed the same bound at
7.4e-4, which is not correctness; it is 26% of margin and a smaller subset away
from failing.

`safety_strain` already had this right — §9.2 built it as a bounded inequality
because a maximum over a subset cannot exceed the maximum it is drawn from.
Exposure simply never got the same treatment. It has it now, in both directions:
`got < logged` is a **hard failure** (a subset cannot add occlusion, so either
the logged metric was computed over the subset, or `particle_ids` and
`tissue_pos` disagree about which particles these are), and `got >> logged` past
`SUBSET_EXPOSURE_MARGIN` means the subset has stopped representing the tissue.

The success message changed too, from "logged metrics reproduce to 1e-3" to
"consistent with full-set bounds". **"Reproduce" is a claim about equality and it
was false.** Wording that trains the reader to expect the wrong relation makes
the correct check look like a weakened one.

#### Bug 4 — `--episodes N` could not work at all

Recorded in `aeadc2c`'s commit message and fixed here. Taichi bakes module-level
constants into kernels at compile time, and this adapter now sets two of them
per material. A second episode in one interpreter therefore either runs at the
first episode's physics or trips the compile lock. It tripped, correctly — but
the error message claimed "the CLI already does" collect one episode per
process, and `main()` did no such thing; it looped in-process.

**The parent now launches one child per episode** with `sys.executable`, an
explicit `--index`, and `check=True`. `--index` is what makes recursion
structurally impossible rather than merely unlikely: the parent always passes
it, and a process holding it never dispatches. `--episodes 1` stays in-process,
since there is nothing to collide with.

**Rejected, `os.fork()`:** it would skip the ~10 s kernel compile, which is the
entire cost. But at fork time the process holds a live Metal device, a compiled
kernel cache and Taichi's runtime threads, and only the forking thread survives
into the child. The child would inherit a GPU context whose owning threads no
longer exist. A crash if you are lucky.

**Rejected, one worst-case substep for the whole dataset:** it would let a single
interpreter do everything, because nothing would vary. But the stiffest material
in `materials.py`'s ranges needs 106 substeps per frame against the softest's 24,
so pinning every episode to the stiffest makes soft episodes ~4× more expensive
for no gain in fidelity.

The cost accepted instead is that each child recompiles, since Taichi's offline
cache is keyed on the constants and every material has a different `dt`.
Measured: 15.9 s and 21.3 s for the two 100-step episodes, 37.4 s wall total,
most of it compilation.

#### The stale tests, and what they were actually saying

`aeadc2c` shipped with two failing tests, deliberately. Both were in
`TestSubstepStability` and both were the fixture, not the check: `write_episode`
computed its default substep with `suggested_substep_dt(20000.0, 1050.0, dx)` —
**no `lam`** — so it picked a bar-wave step ~16× too large and the check
correctly failed an episode the fixture called valid.

The fixture now derives its advisory from whatever material the test passes,
with `lam`. It also no longer hardcodes `n_substeps=40` against an unrelated
`dt`; the writer's new invariant refused that immediately, which is the guard
working on its first contact with existing code.

The more interesting repair is `test_the_same_substep_is_fine_for_soft_tissue`.
Its "soft" material dropped μ tenfold, 20 kPa → 200 Pa, but left λ at 200 kPa.
Under a P-wave bound that moves the advisory by **9%** — 19.8 µs to 21.7 µs — so
the soft episode failed at the same substep as the stiff one and the test was
asserting something that had stopped being true. **Under this bound, softness is
λ, not μ.** The pair now shares a 50 µs substep that genuinely straddles the two
advisories:

| material | λ + 2μ | P-wave advisory | ratio at 50 µs | verdict |
|---|---|---|---|---|
| stiff, μ = 20 kPa, λ = 2e5 | 240 000 | 19.84 µs | 2.52 | FAIL |
| soft, μ = 200, λ = **2e4** | 20 400 | 68.06 µs | 0.73 | PASS |
| soft, μ = 200, λ = 2e5 *(the old one)* | 200 400 | 21.72 µs | 2.30 | FAIL |

A third test pins the straddle itself, so a later edit to either material cannot
quietly make both tests trivially agree.

#### Coverage moved out of `__main__`

`trajectory_io.py`'s self-test stays — `SETUP_GUIDE.md` tells the user to run it
and both `verify_*.py` scripts run it as a subprocess, which is a different job.
But it was the *only* cover for delta16, the subset bookkeeping and the schema
upgrade path, and a self-test behind `if __name__ == "__main__"` runs only when
someone remembers, stops at the first failure, and cannot be collected or
counted. `tests/test_trajectory_io.py` now holds 29 tests over those properties.

#### The seed does less than it looks like it does

Recorded because the metadata would otherwise imply something false. `mpm3d.py`
calls `ti.init(arch=arch)` with **no `random_seed`** (line 22), so Taichi's RNG
starts at 0 in every process and `init_cube()`'s `ti.random()` lays out an
**identical particle cloud in every episode**. The episode seed drives numpy
only: the material draw and which particles get recorded.

So a dataset collected this way varies in material and in nothing else. The
episode `notes` now say so in as many words. Fixing it means seeding Taichi per
episode, which is a change to how the vendored solver is initialised and belongs
with the PSM work rather than here; naming it is the cheap half.

#### The stale files, kept

`data_mpm/stale/` holds the two 17 August episodes. `mpm_0000.npz` ran 60 frames
at a 500 µs substep its material happened to tolerate; `mpm_0001.npz` is a
one-frame partial written by `__exit__` after its material diverged at the same
substep. They are the evidence behind the P-wave finding and behind Bug 3's
"passing on luck", and they are not a baseline for anything: they predate the
density fix, so they ran at ρ = 1000 regardless of what they record. Ignored by
git, kept on disk, and deliberately not deleted until this section existed.

#### Verification, run in order

```
pytest tests/ -q                                 249 passed
python src/trajectory_io.py                      OK (v1, v2 float16, v2.1 subset + delta16)
python host/smoke_test_mpm.py                    16 passed, 0 failed, 1 warning, arch=metal
python host/mpm_adapter.py --out data_mpm --episodes 2 --steps 100
                                                 mpm_0000.npz 11.5 MB 15.9s
                                                 mpm_0001.npz 11.5 MB 21.3s
python host/validate_dataset.py --data data_mpm/ 24 passed, 3 warned, 2 skipped, 0 failed (exit 0)
```

Per-file invariants, asserted directly rather than inferred from a green run:

| | `mpm_0000` | `mpm_0001` |
|---|---|---|
| timebase | 12.5 ms = 24 × 520.833 µs, rel drift 3.7e-8 | 12.5 ms = 106 × 117.925 µs, rel drift 0.0 |
| μ, λ | 3758 Pa, 69 279 Pa | 2112 Pa, 1 592 053 Pa |
| ρ → `p_mass` | 1004.1 → 4.787909e-4 ✓ | 1014.4 → 4.837112e-4 ✓ |

`mpm_0001` is the material that diverged on 17 August. It now runs 106 substeps
of 117.9 µs per frame and completes 100 frames.

The two remaining WARNs are expected and are not defects: no grasp is ever
active (there is no robot yet), and `mpm_0000` shows 18.7% volume change, which
is a real property of a soft λ = 69 kPa material settling under gravity, flagged
by a threshold tuned for near-incompressible tissue. Both are honest reports
about a passive-settling episode.

#### Still not done

- **No robot.** `ee_pose` and `action` are whatever the caller passes; the tissue
  is driven by gravity and its own elasticity. The SDF seam in `sdf.py` is where
  the PSM plugs in, and `smoke_test_mpm.py` establishing that `substep()` runs
  with no PyBullet scene is what makes that seam testable in isolation.
- **Material ranges are still placeholders** (§8.3). Liver, bowel and fat differ
  by more than the width of those ranges. Two episodes with σ(log μ) = 0.29 is a
  demonstration that collection works, not a dataset.
- **Taichi's seed is fixed**, as above.
- **§4 is still open** for the PyBullet side. Nothing here touches it.

---

### 9.6 The substep study: a negative result, and the right kind

25 August 2026. `host/substep_study.py`. The question was whether `safety = 0.3`
in the P-wave advisory is *converged* or merely *stable* — §9.5 and
`check_substep_is_stable_for_stiffness` both flagged that nothing had ever
tested it.

**The answer is that it is stable, that it is not converged, and that no value
of it can be — because a substep-refinement study is the wrong instrument for
MPM at a fixed grid resolution.** That last part was not the expected outcome
and is the reason this section is worth its length.

#### Design

Hold everything fixed except the substep count: same material, same seed, same
`frame_dt`, same recorded subset, same frame count. Sweep `n_substeps` from ¼ of
the advisory count to 8×. Both materials from `data_mpm/`, because the advisory
is a *formula* that scales with the material — testing it at one material tests
a constant.

One child process per row, for the §9.5 reason: `dt` and `p_mass` are baked into
kernels at compile time. Children run with `check=False`, because the coarse rows
are *expected* to fail and a study that dies on its first divergence has tested
nothing.

**The measurement is unusually clean here, and §9.5's limitation is why.**
`mpm3d.py` calls `ti.init()` with no `random_seed`, so every process lays out an
identical particle cloud, and `MPMRecorder`'s subset is drawn from a fixed seed.
Particle *i* is therefore the same lump of material in every row, which allows a
**field-by-field** comparison rather than a comparison of summary scalars.
§9.5 recorded that fixed seed as a defect narrowing dataset diversity; for this
study it is a gift. The parent asserts it rather than trusting it.

The primary measure is the RMS final-position difference from the finest row,
normalised by that row's own RMS displacement — not peak stretch. A maximum over
3,000 particles is a single order statistic and can move several percent because
one particle overtook another.

#### The tables

```
soft-lambda:  mu = 3758 Pa   lambda = 69279 Pa   rho = 1004.1   lambda/mu = 18
  advisory at safety=0.3: 24 substeps/frame (520.8 us);  60 frames, seed 0

  n_sub    substep   safety  peak stretch  max|J-1|  peak disp   KE final    wall
      6   2083.3us    1.200      DIVERGED                         1.4s   (frame 3/60)
     12   1041.7us    0.600       1.18208    0.2050    8.85mm  1.208e-01    9.6s
     24    520.8us    0.300       1.16868    0.1872    8.49mm  2.484e-02   10.1s
     48    260.4us    0.150       1.14124    0.1719    8.08mm  1.630e-03   11.0s
     96    130.2us    0.075       1.10763    0.1595    7.50mm  6.887e-04   12.7s
    192     65.1us    0.037       1.07401    0.1356    6.70mm  4.077e-04   16.3s

    n_sub   safety   field err     horiz      vert  d(stretch)
       12    0.600      29.79%   135.03%    26.34%     146.03%
       24    0.300      18.73%    82.19%    16.71%     127.93%  <- advisory
       48    0.150      11.23%    48.57%    10.06%      90.85%
       96    0.075       5.20%    25.05%     4.51%      45.42%
      192    0.037       0.00%     0.00%     0.00%       0.00%

stiff-lambda: mu = 2112 Pa   lambda = 1592053 Pa  rho = 1014.4   lambda/mu = 754
  advisory at safety=0.3: 106 substeps/frame (117.9 us);  60 frames, seed 1

  n_sub    substep   safety  peak stretch  max|J-1|  peak disp   KE final    wall
     26    480.8us    1.223      DIVERGED                         1.1s   (frame 1/60)
     53    235.8us    0.600       1.16448    0.0311    6.72mm  4.807e-03   11.2s
    106    117.9us    0.300       1.11837    0.0298    5.77mm  2.714e-03   13.1s
    212     59.0us    0.150       1.08410    0.0270    4.85mm  1.631e-03   17.1s
    424     29.5us    0.075       1.05889    0.0227    4.03mm  9.225e-04   25.1s
    848     14.7us    0.037       1.03692    0.0184    3.35mm  5.311e-04   40.6s

    n_sub   safety   field err     horiz      vert  d(stretch)
       53    0.600     100.82%   151.44%    93.19%     345.46%
      106    0.300      67.22%   103.19%    61.71%     220.58%  <- advisory
      212    0.150      40.33%    68.55%    35.65%     127.76%
      424    0.075      18.18%    33.11%    15.55%      59.50%
      848    0.037       0.00%     0.00%     0.00%       0.00%
```

169.9 s wall for the pair.

#### What the stability half confirms

**The P-wave bound does its stability job, at both materials.** The only rows
that diverged are the ones above `safety ≈ 1.2` — n=6 for the soft material
(frame 3 of 60) and n=26 for the stiff one (frame 1 of 60). Everything at
`safety ≤ 0.6` completed 60 frames with finite `F`. The bound is not decorative,
and the safety margin of 0.3 buys a factor of four below the observed cliff.

Note also that `max|J-1|` is **0.187 for the soft material and 0.030 for the
stiff one** — the λ/μ = 754 material resists volume change 6× harder, exactly as
a near-incompressible material should. Unprompted physical sanity, from a
quantity nothing in the study was tuning.

#### What does not converge, and why refining will not fix it

Every quantity moves monotonically with refinement and none of them settles.
Peak stretch above rest falls from 0.169 to 0.074 (soft) and 0.118 to 0.037
(stiff) — factors of **2.3 and 3.2** — across a 8× substep refinement, with no
sign of a plateau. Field error at the advisory is 18.7% and 67.2%.

An earlier draft of this section blamed the floor friction. `Boundary()` applies

```python
if I[2] <= 3:
    F_grid_v[I][0] *= 0.1
    F_grid_v[I][1] *= 0.1
```

— a **per-substep multiplicative tangential damping with no `dt` scaling**
(`mpm3d.py:274-276`), so the time in which lateral motion is arrested is
proportional to the substep. That is a genuine defect in the vendored solver and
it is real. **It is not the main cause here, and the study was made to say so.**
Splitting the field error into horizontal and vertical components was added
specifically to test the attribution, because the damping touches `[0]` and `[1]`
and never `[2]`. The soft material's *vertical* error (16.71%) is comparable to
its total and its horizontal error is larger, but the stiff material's vertical
error is 61.71% — far too large to be explained by a term that never touches z.
The hypothesis survived contact with one column and died against the other.

> **WITHDRAWN 26 August 2026 — see §9.7.** Everything from here to the end of
> this subsection is wrong. The mechanism below was inferred from kinetic energy
> alone, in a configuration with gravity and floor contact both active, at a
> time by which the body has already settled — so the two KE numbers being
> compared are both residual jitter near zero. Measured against a closed energy
> budget in this same configuration, dissipation *falls* as the substep shrinks
> (by 2.1× at the stiff material across a 16× ladder) and the positions *do*
> converge, at an observed order of about 0.3. The paragraphs below are kept
> rather than deleted because the reasoning that produced them is the thing
> worth being able to re-read.

The claimed cause is **numerical dissipation in the particle-grid transfers**, and
the probe that identifies it is kinetic energy at a fixed simulated time:

| | soft-λ | stiff-λ |
|---|---|---|
| KE at coarsest surviving row | 1.208e-1 | 4.807e-3 |
| KE at finest row | 4.077e-4 | 5.311e-4 |
| ratio across the sweep | **296×** | **9×** |

Both fall **monotonically**. MPM's P2G/G2P round trip is a projection onto the
grid basis and loses energy every time it happens; the loss is per *transfer*,
not per unit of simulated time. Halving the substep doubles the number of
transfers inside the same 12.5 ms frame and therefore roughly doubles the
damping. Refining `dt` does not approach a solution — it walks steadily toward an
over-damped one. That is why the tissue deforms *less* at every refinement rather
than converging on an amount.

**So the classical instrument does not apply.** "Refine `dt` until the answer
stops changing" assumes the discretisation error vanishes as `dt → 0` with
everything else held fixed. In MPM at a fixed grid, it does not: `dx` sets the
dissipation per transfer, and shrinking `dt` alone only buys more transfers. A
genuine MPM convergence study must refine `dx` and `dt` **together**, which is a
different and much more expensive experiment, and one that changes the particle
count and therefore the schema.

> **END OF THE WITHDRAWN ARGUMENT (§9.7).** The classical instrument does apply:
> a `dt` refinement at fixed `dx` converges here, slowly. The zero-stiffness test
> in §9.7 — where nothing but the transfers can act — holds total loss constant
> to 0.2% across a factor of 64 in `dt`, which is the direct refutation. The
> stability half of this section, and its finding that `safety_strain` is not
> comparable between episodes, are unaffected and stand.

#### The consequence that actually matters for the dataset

**`safety_strain` is not a material property under this solver.** It varies by a
factor of 2.3–3.2 with a choice of substep that no downstream consumer can see.
Worse, the collector assigns substeps *per material*, so the two episodes in
`data_mpm/` were integrated at 24 and 106 substeps per frame — meaning their
`safety_strain` values carry different amounts of numerical damping and **are not
comparable to each other**. A model conditioned on material and trained to
predict deformation would be fitting the substep schedule as much as the tissue.

That is a dataset-design finding, not a solver bug, and it is the most valuable
thing this study produced. It does not have a fix inside the current
architecture; it has to be recorded and carried.

#### What changed as a result

**Nothing about `safety = 0.3`.** It is retained, now on its actual grounds:
it is a *stability* bound with a measured factor-of-four margin below the
divergence cliff, and it was never a convergence claim. The study's contribution
is that this is now stated rather than assumed, and that the alternative reading
has been ruled out rather than left open.

`check_substep_is_stable_for_stiffness` keeps its wording — "passing means not
obviously unstable, not converged" was exactly right — and now cites §9.6 for
what convergence turned out to mean.

#### Two things the study got wrong first, kept because they are the method

- **The floor-friction attribution**, above. A plausible mechanism, found by
  reading the source, that the data refused. The horizontal/vertical split
  exists because of it and stays in the tool.
- **A magnitude threshold for the dissipation verdict.** The first version
  reported transfer dissipation when KE fell more than 10× across the sweep.
  The two materials fall 296× and 9×, showing *identical* behaviour, so the
  cutoff told the stiff material to "extend the sweep downward" — advice that
  would have burned GPU time indefinitely on a sweep that cannot converge. The
  test is now **monotonicity**, which is scale-free and was right for both.

#### Verification

```
pytest tests/ -q                                   249 passed
python host/substep_study.py --frames 60           169.9s, both materials
python host/validate_dataset.py --data <kept>/     24 passed, 0 failed (exit 0)
```

The third is the one worth noting: the study's own advisory-row episodes are
real v2.1 files that pass the full validator, which is the evidence that it
drove the collector rather than a lookalike copy of its drive loop.

Also asserted inside the study, not merely assumed: every row starts from an
identical particle cloud. If Taichi's seeding ever changes, that comparison
becomes meaningless and the study now fails loudly instead of quietly comparing
different lumps of material.

#### Still open

- **§4 remains untouched.** `container/timestep_study.py` has still never run;
  the Docker daemon was not running on this machine. That study asks the same
  question of PyBullet, where the classical instrument *does* apply, because
  PyBullet's soft body has no particle-grid transfer.
- **A real MPM convergence study** — refining `dx` and `dt` together — has not
  been attempted and would change the particle count. (§9.7: it is no longer the
  *only* valid instrument, but it is still the right one for a spatial claim.)
- **The `*= 0.1` floor friction is still a defect** in the vendored solver, now
  documented. It is not the dominant error here but it is dt-dependent and
  should be fixed or replaced with a dt-scaled law before anything depends on
  lateral behaviour near the floor. `third_party/PROVENANCE.md` records that the
  tree is unmodified; fixing this means editing it and saying so there.

---
### 9.7 The energy audit: §9.6's mechanism was wrong, and its probe was the reason

26 August 2026. `host/energy_audit.py`, `host/mpm_energy.py`,
`src/energy_ledger.py`.

§9.6 concluded that MPM's particle-grid transfers dissipate energy *per
transfer* rather than per unit of simulated time, and therefore that refining
the substep at a fixed grid cannot converge. That became load-bearing
immediately: CLAUDE.md tells future work not to refine, and §9.6 used it to
declare `safety_strain` non-comparable between episodes.

**It is wrong.** Measured against a closed energy budget, in §9.6's own
configuration, dissipation *falls* as the substep shrinks and the particle
positions *do* converge. The per-transfer mechanism is not what the sweep was
showing, and the reason it looked that way is the probe.

#### Why the original probe could not carry the claim

§9.6's evidence is one number: specific kinetic energy of a 3,000-particle
subset at a fixed simulated time, falling monotonically across the sweep
(`substep_study.py:216`). Four things are wrong with it, and the fourth is
fatal on its own.

1. **Kinetic energy is not the energy budget.** The rows settle under gravity,
   so KE is pumped by gravitational release and drained by contact, elasticity
   and the transfers simultaneously. A lower KE is as consistent with "settled
   sooner" as with "damped harder".
2. **Contact was live in every row.** The `*= 0.1` band and the `bound = 3`
   wall condition were both firing, so nothing in the sweep isolated the
   transfers at all.
3. **The finest row was the reference**, while being — by §9.6's own argument —
   the most over-damped row in the table.
4. **At T the body has already settled, so the probe was comparing two numbers
   that are both essentially zero.** This is the one that ends the argument.
   Under the full ledger, KE at T in the §9.6 configuration is 8e-5 J at the
   coarsest row and below 1e-5 J at every finer one, out of a total of 8.0 J.
   §9.6's 296× and 9× ratios are ratios between residual jitter of a body at
   rest. They measure how fast each row stopped moving, not how much energy it
   destroyed.

#### The instrument

`src/energy_ledger.py` computes the closed budget over **all** 24,000 particles
in float64:

    E = KE_particle + KE_affine + Psi_elastic + PE_gravity

Two properties make it exact rather than indicative, and both are checked:

- **The energy already in the repository is the exact conjugate of the
  solver's stress.** `tissue_metrics.strain_energy_neohookean` is
  Psi = (mu/2)(I1-3) - mu·lnJ + (lambda/2)(lnJ)^2, and (dPsi/dF)F^T =
  mu(FF^T - I) + lambda·lnJ·I, which is character for character `mpm3d.py:165`.
  Verified by numerical differentiation in `tests/test_mpm_energy.py`, together
  with the demonstration that perturbing lambda makes the identity fail. That
  test did not exist before and the identity had only ever been asserted by
  reading the two lines side by side.
- **The APIC affine term is included.** Each particle carries
  v_p + C_p(x - x_p); the energy in C is real and invisible to any sum over
  `F_v`. D = (dx^2/4)I is derived from the solver's own quadratic weights
  rather than quoted, and the derivation is a test.

`host/mpm_energy.py` drives the vendored solver. It splits `substep()` into its
three phases by calling `P2G`, `Boundary` and `G2P` — already `@ti.func` — from
three kernels of our own, which opens observation points on the grid without
editing `third_party/`. `PROVENANCE.md` still reads "None yet" and the four
SHA-256s still match `cb797f36`.

**Gravity is a third compile-time constant.** `Boundary()` reads it as a plain
Python global inside a `@ti.func` (`mpm3d.py:215`), so it is baked exactly like
`dt` and `p_mass`. §9.5's compile lock covered two of the three; it now covers
all three, and `mpm_adapter` refuses outright to collect in a process where
gravity is not 9.8.

#### D1: the instrument was gated before anything was read from it

Five checks, all of which must pass before the cells run:

```
[PASS] gravity_off            max|v| after one frame from rest = 0.000e+00 m/s
[PASS] rigid_translation      relative energy change +4.5e-07 (stress is exactly zero)
[PASS] affine_velocity        relative energy change +2.7e-07 (zero stiffness; APIC is exact)
[PASS] split_matches_solver   fused-vs-split 3.0 ulp, solver's own floor 3.0 ulp
[PASS] boundary_inert_in_clean_cell   worst boundary energy change 2.1e-08 of grid KE
```

The last one is the honest form of "contact and friction work": in the clean
cell it is not estimated, it is shown to be zero. The fourth is the more
interesting: **this solver is not bitwise reproducible against itself on
Metal.** Two runs of the *vendored* `substep()` from a byte-identical state
differ by ~1 ulp after 24 substeps, because P2G scatters into the grid with
atomic adds and the summation order varies between launches. A bit-identity
check would have failed forever for a reason unrelated to the split, so the
noise floor is measured in the same process and the split is held to it.

#### The zero-stiffness transfer test: the sharpest form of the question

With `mu = lambda = 0` no force acts on anything. Particles advect and their
velocity field is repeatedly filtered through the grid, so any energy that
disappears is transfer loss and nothing else — and with no stress there is no
CFL condition, so `dt` can be swept over decades.

```
 n_sub    substep  transfers         E(0)         loss  loss/transfer
     4   3125.0us         16 7.209437e-03  3.77957e-03    2.36223e-04
    16    781.2us         64 7.209437e-03  3.78365e-03    5.91195e-05
    64    195.3us        256 7.209437e-03  3.78514e-03    1.47857e-05
   256     48.8us       1024 7.209437e-03  3.78560e-03    3.69687e-06
```

**The total loss is constant to 0.2% across a factor of 64 in `dt`.** Loss per
transfer falls by exactly 64×. If dissipation counted transfers, the loss column
would have risen 64-fold; it did not move. This alone refutes the mechanism, in
the cleanest possible setting, and it costs four short runs.

Why the naive picture fails: the transfer destroys the non-affine content of the
velocity field inside a cell, and what *regenerates* that content between
transfers is inter-transfer motion of size |v|·dt. Halving `dt` doubles the
transfers but quarters what each one has left to remove.

#### The three cells

`T = 0.75 s` (60 frames × 12.5 ms), the ladder is 0.5× to 8× the P-wave
advisory count, both collected materials, one process per row.

The verdict is the exponent q in (dissipation over fixed T) ~ dt^q, fitted on
the **decay constant** k = -ln(E(T)/E(0))/T rather than the loss fraction. A
fraction of E(0) cannot exceed 1 and the clean cell reaches 78%, so the ceiling
would squeeze the exponent toward zero — toward this study's own conclusion,
which is the one direction it must not be biased. Where nothing is saturated the
two agree to two decimals (cell 4: +0.097 vs +0.095, +0.266 vs +0.264).

| cell | material | q | verdict | measured across the 16× ladder |
|---|---|---|---|---|
| 1 clean | soft-λ | **-0.337** | INTERMEDIATE | 2.5× (per-transfer predicts 16×) |
| 1 clean | stiff-λ | **-0.205** | INTERMEDIATE | 1.8× |
| 3 band | soft-λ | **-0.068** | RATE_LIKE | 1.2× |
| 3 band | stiff-λ | **+0.237** | INTERMEDIATE | 1.9× (r² = 0.68, non-monotone) |
| 4 §9.6 | soft-λ | **+0.097** | RATE_LIKE | 1.3× |
| 4 §9.6 | stiff-λ | **+0.266** | INTERMEDIATE | 2.1× |

**Not one cell is near q = -1.** The pre-registered hypothesis predicted the
dissipation would change by the full span of the ladder, 16×. The largest
change measured anywhere is 2.5×, and in half the cells it goes the other way.

#### Cell 4 is the one that settles it, because it is §9.6's own configuration

```
   n_sub    substep        E(0)        E(T)  dissipated    frac
      12   1041.7us     8.45575     7.98641     0.46934   5.55%
      24    520.8us     8.45575     8.02540     0.43035   5.09%
      48    260.4us     8.45575     8.05553     0.40021   4.73%
      96    130.2us     8.45575     8.07779     0.37796   4.47%
     192     65.1us     8.45575     8.09533     0.36042   4.26%   (soft-lambda)

      53    235.8us     8.54264     8.32707     0.21558   2.52%
     106    117.9us     8.54264     8.36315     0.17949   2.10%
     212     59.0us     8.54264     8.39284     0.14981   1.75%
     424     29.5us     8.54264     8.41594     0.12670   1.48%
     848     14.7us     8.54264     8.43984     0.10281   1.20%   (stiff-lambda)
```

**Dissipation falls monotonically with refinement, by 2.1× at the stiff
material.** §9.6 says it roughly doubles at every halving of the substep. The
measurement moves in the opposite direction, in the configuration §9.6 ran.

And the positions converge:

```
          pair     RMS diff  as % of disp   order p
        53/106     0.5438mm        21.26%         -
       106/212     0.4360mm        20.28%      0.32
       212/424     0.3715mm        20.34%      0.23
       424/848     0.2873mm        18.23%      0.37
```

Successive differences shrink monotonically at a consistent observed order of
about 0.3. That is slow — it is not the first-order convergence one would want —
but "nothing converges under substep refinement, and nothing can" is false.

#### Why §9.6's KE probe pointed the wrong way

The term breakdown at T, from the same runs, with §9.6's probe as one column:

```
   n_sub          KE   KE_affine     elastic     PE_grav       TOTAL
      12     0.00008     0.00000     0.17121     7.81512     7.98641
      24     0.00002     0.00000     0.15865     7.86673     8.02540
      48     0.00000     0.00000     0.15323     7.90230     8.05553
      96     0.00000     0.00000     0.15029     7.92750     8.07779
     192     0.00000     0.00000     0.14854     7.94679     8.09533
```

KE falls across the ladder, exactly as §9.6 reported. The total rises. What
actually happens as the substep shrinks is that **less** energy is destroyed, so
the slab sinks **less** far into the floor — `PE_grav` at T rises from 7.815 to
7.947 J — and being less compressed it stores less strain energy, 0.171 down to
0.149 J. §9.6 observed that the tissue deforms less at every refinement and that
part is real; it attributed it to more damping when the cause is less.

Cell 1 makes the same point from the other side: there KE at T *rises* with
refinement (0.068 → 0.219 J) while elastic energy falls (0.496 → 0.073 J) and
the total falls. KE tracks the phase of the oscillation, not the dissipation.
**In neither cell does KE alone have the sign of the energy loss.**

#### The floor band: right conclusion, wrong reason, and now measured

§9.6 dismissed the `*= 0.1` band because it touches `[0]` and `[1]` and never
`[2]`, while the stiff material's vertical error was the largest in the sweep.
That reasoning is unsound: at λ/μ = 754 the material is nearly incompressible,
so suppressing lateral flow at the base suppresses vertical settling almost
rigidly, and the material with the strongest coupling showing the largest
vertical error is what the floor hypothesis *predicts*. The alternative reading
was rejected using its own best evidence.

Cell 3 arms the band alone — slab positioned so its lowest stencil node sits
exactly on k = 3, gravity off, lateral velocity only — and settles it properly.
The band removes 89–93% of the energy, which is enormous, but it removes the
same amount whatever the substep: q = -0.068 at the soft material, per-second
spread 1.05 across the whole ladder. **A 0.1× multiplier per substep is so
aggressive that it saturates into a hard clamp at every substep count tested**,
which makes it dt-independent rather than transfer-counting. §9.6's conclusion
about the band stands; its argument for that conclusion does not.

#### What this changes

- **§9.6's mechanism paragraph is withdrawn.** Amended in place, with a pointer
  here. `CLAUDE.md`'s "Nothing converges under substep refinement, and nothing
  can" is replaced, and the comments at `substep_study.py:210` and `:353` — which
  told the next reader not to spend GPU time refining — now say why that advice
  was wrong.
- **`safety = 0.3` is still kept, and still on stability grounds.** Nothing here
  touches the divergence cliff at `safety ≈ 1.2`; §9.6's stability half is
  unaffected and remains correct.
- **`safety_strain` is still not comparable between `data_mpm/`'s two
  episodes.** §9.6's most useful finding survives its mechanism being wrong.
  The two episodes ran at 24 and 106 substeps per frame, dissipation *does*
  vary with substep count — weakly, and in the opposite direction from what
  §9.6 said, but it varies — and the collector chooses that count per material.
  A model conditioned on material still partly fits the substep schedule.
- **A conventional refinement study is valid here after all.** It converges at
  order ~0.3, so it is expensive, but "the classical instrument does not apply"
  was wrong. Whether to spend that GPU time is a separate decision.

#### Recommendation, not implemented

Pinning `n_substeps` dataset-wide (at the stiffest material's requirement) would
make dissipation identical across episodes and restore comparability. It
reverses the cost argument recorded in `mpm_adapter.main()`'s docstring — soft
episodes become ~4× more expensive — and it is now a correctness argument rather
than a convenience one. Recorded here; not done. This section is an audit.

#### Four things this got wrong first, kept because they are the method

- **`init_cube()` is not idempotent.** It draws from `ti.random()` and the RNG
  advances, so a second call in one process lays out a *different* cloud. §9.5's
  "the initial particle cloud is identical in every episode" is true only
  because every episode is a fresh process, and false the moment two initial
  conditions are built in one interpreter — which is what the selftest does. The
  first probe of solver reproducibility reported a 0.35 m discrepancy that was
  entirely this.
- **The first affine-transfer check blamed APIC for the check's own bug.** It
  set particle velocities to an affine field but left `F_C` at zero, which makes
  the represented field piecewise *constant*; the transfer then legitimately
  destroyed what it could not see, and it was reported as a 0.75% APIC fault.
  With `F_C` set, conservation is 2e-7.
- **`ledger()` charged the wrong material.** It read the Python-side `mu`
  instead of the solver's live `ti.field`, so a runtime-zeroed stiffness still
  accrued strain energy. Same class as §9.5's timebase: two numbers that agree
  with everything except each other.
- **The verdict function twice overstated its own result**, in the tool written
  to catch overstatement. Its first band filed a measured q = -0.34 as
  "dissipation is essentially independent of dt" while the decay rate had moved
  2.5× across that ladder; then the degenerate-fit fallback relabelled a
  q = +0.24 ladder as RATE_LIKE because the collapse diagnostic preferred "per
  time". Both are the same error §9.6 made. The verdict now quotes the measured
  factor against both predictions, and the r² gate may caveat or withdraw a
  verdict but never substitute one. Regression tests for both.

#### Verification

```
pytest tests/ -q                              279 passed (249 before, 30 new)
python host/energy_audit.py --selftest        5/5, D1 gate
python host/energy_audit.py --frames 60       304.9 s, three cells, both materials
shasum -a 256 third_party/MPM/*.py            all four match PROVENANCE.md
python host/smoke_test_mpm.py                 unchanged
python host/validate_dataset.py --data data_mpm/   exit 0
```

The SHA check is the one worth repeating: the whole instrument works by calling
the vendored `@ti.func`s from kernels of our own, and it has not touched the
vendored tree.

#### Still open

- **§4 remains untouched.** `container/timestep_study.py` has still never run.
  It now matters more, not less: the PyBullet side has no particle-grid
  transfer, and the excuse that convergence studies do not apply to this class
  of solver has just been removed.
- **A coupled `dx`/`dt` study** is still unattempted, and still the right
  instrument for a genuine spatial convergence claim. It is no longer the *only*
  valid instrument.
- **Order ~0.3 convergence is poor and unexplained.** A first-order scheme
  should show p ≈ 1. The gap is a real question this study raises and does not
  answer.
- **The `*= 0.1` band is still a defect** — now measured, at 89–93% of the
  energy in cell 3. It is dt-independent, so it is not a *convergence* problem,
  but it is an enormous and physically arbitrary sink sitting under every
  episode in `data_mpm/`.

---
## 10. Session 4 — 31 August 2026: the PSM lands

Every MPM episode before this section was passive gravity settling —
`_init_solver()` hand-bound `SDF`/`collision_mask` to empty space because
there was no robot. This section replaces that with a real, moving,
colliding tool: `psm_Si_model/psm_si_surrol.urdf`, a dVRK **Si**-variant PSM
(13 links), driven kinematically through `host/psm.py` and wired into
`host/mpm_adapter.py`'s `record_grasp_episode()`.

### 10.1 The plan was sent back once, and the correction caught four real bugs

The first design draft was reviewed before any code was written and rejected
with nine specific, technical objections. Four of them were not style
notes — they were bugs that would have shipped silently:

1. **The proxy collision-frame design was wrong.** The draft planned to fold
   the jaw mesh's AABB offset into PyBullet's `collisionFramePosition` and
   let the solver "just work." `sdf.py`'s box path reads only
   `p.getCollisionShapeData(...)[0][3]` — the shape's extents — and **never**
   reads a collision shape's local frame offset. The box's assumed world
   pose is entirely whatever `i_rot_list`/`i_pos_list` the caller supplies;
   an offset placed via `collisionFramePosition` would have been silently
   invisible to the SDF math while still being a real geometric fact about
   the body. Fixed by building each proxy as a zero-offset box and composing
   `T_world_box = T_world_jaw . T_jaw_box` by hand, in numpy, every frame.
2. **Both jaws would have shared one collision velocity**, derived from the
   single commanded EE-pose delta. Jaw closure moves the two jaws
   differently from each other and from the EE frame; a shared velocity was
   wrong for both. Fixed by finite-differencing each proxy's own
   `T_world_box` across consecutive recorded frames, independently.
3. **The IK joint-index mapping was wrong.** `p.calculateInverseKinematics`
   returns one value per **movable** joint (fixed joints skipped entirely),
   in increasing raw-joint-index order — not one value per raw joint index.
   The draft would have read IK's output at the wrong offset for every joint
   after the first fixed one in the chain. Fixed by resolving a
   `movable-index -> raw-index` map once, at construction.
4. **The design was about to fabricate `CONTACT_GRASP`.** The schema defines
   `CONTACT_GRASP` as tissue **kinematically attached** to the tool.
   `Boundary()`'s collision branch (`mpm3d.py:241-249`) is an unconditional
   zero-slip velocity constraint with no persistence — the friction/slip code
   is commented out ("sticky trick"), and a particle leaving the SDF radius
   is released immediately, no bookkeeping holds it. That is mechanically
   `CONTACT_STICK`, never `CONTACT_GRASP`. The draft's plan to label
   jaw-closed-and-near as `CONTACT_GRASP` — manufactured specifically to make
   `check_grasp_is_consistent` stop warning — is exactly the kind of check-
   gaming this project's own validation discipline exists to catch. Every
   episode from this path emits only `NONE`/`TOUCH`/`STICK`; `grasp_active`
   is always `False`, `grasp_node_ids` always empty, and
   `check_grasp_is_consistent` keeps WARNing on every episode, honestly,
   until real persistent attachment is built.

The remaining five corrections fixed the PyBullet client lifecycle (`sdf.py`'s
calls carry no `physicsClientId`, so there can be exactly one live client per
process — no injectable client parameter), the `init_pos()`/`MPMRecorder`
construction ordering (a `PSM` cannot exist before `MPMRecorder.__init__` has
already imported and `ti.init()`'d `MPM.mpm3d`), the exact state/action
timing contract (row *t*'s `ee_pose`/`action` are the commanded/analytic
waypoint, never the IK-achieved pose — kept separate so "applying row *t*'s
action reproduces row *t+1*'s recorded state" is exact, not approximate),
the `grasp_node_ids` indexing spec for whenever real attachment lands
(recorded-subset indices, not full-solver indices — the visualizer uses them
directly against `tissue_pos`), and required provenance for the 16 MB asset
tree before committing it (`psm_Si_model/PROVENANCE.md`).

### 10.2 The vendored solver's collision path has one hard, undocumented trap

Read directly from `third_party/MPM/sdf.py` and `mpm3d.py`, not assumed:

- **`MAX_COLLISION_OBJECTS = 3`** (`mpm3d.py:26`), compile-time-sized. At most
  3 simultaneous colliders without editing the vendored solver. This episode
  uses 2 (the jaw links) plus one permanently-inert dummy far outside the
  domain, placed once and never moved — every `co_obj` slot is queried
  unconditionally every step, so an unused slot still needs a real body.
- **A `co_obj` slot with `link_id == -1`** — PyBullet's "base" convention —
  makes `sdf.py` take the "needle" precomputed-SDF path
  (`static_sdf[idx]`). Nothing in this repository ever calls
  `sdf.init_static_sdf(...)`, so `static_sdf` sits at Taichi's zero default:
  every in-range grid node would read distance 0.0, a phantom zero-distance
  surface across the whole domain, the instant that slot is queried. Every
  proxy in `host/psm.py` is therefore built as a **one-link** multibody
  (`link_id = 0`), never a bare base — this is the single most important
  invariant in the file, and `host/smoke_test_psm.py` asserts it directly.
- **`sdf.position`** (the grid field the collision kernel transforms) is only
  populated by `mpm3d.init_pos()`. `MPMRecorder._init_solver()` never called
  it, because it never needed the collision path before. Owned by
  `PSM.__init__` now, once, and only there — not duplicated into
  `_init_solver()`, which stays unchanged whether or not a robot exists.
- **Taichi's Metal backend has no f64 primitive type.** The first real run
  failed at `reverse_rotation_matrix.from_numpy(i_rot)` with
  `RuntimeError: Type f64 not supported` — `numpy`'s default float dtype is
  float64, and `.from_numpy()` goes through a compiled kernel that requires
  the array's dtype to match the field's, unlike plain `field[i] = ...`
  assignment, which casts silently. Every array fed to
  `switch_reference_frame_and_update_sdf` is explicitly `float32` now. Not
  predicted by the design review — found by running it, on the first attempt.

### 10.3 What was built

- **`psm_Si_model/`** — the URDF (13 links: `link_0 -j1-> link_1`, then two
  branches off `link_1`: `-j4->` the real tool chain to
  `tool_gripper_center`, and `-j2->link_2-j3->link_3`, the dVRK's decorative
  parallelogram linkage, confirmed **not** an ancestor of the tool from the
  URDF's own `<parent>`/`<child>` tags) plus all referenced mesh assets.
  `PROVENANCE.md` records sha256 for all 122 files; **source and license are
  still an open TODO**, filled in with a placeholder pending confirmation —
  the URDF's own header comments point at a `ros2_dvrk_model`-style package
  and a personal development path, not a confirmed upstream.
- **`host/psm.py`** — the `PSM` class: URDF load (`useFixedBase=True`,
  required since `link_0`'s world-fixed joint is commented out in the URDF),
  movable-DOF-correct IK with a post-solve residual gate (raises past 2 mm /
  0.02 rad — an unreachable target fails loudly, not silently), jaw proxy
  construction and per-frame `T_world_box` composition, per-proxy
  finite-difference collision velocity with sweep-limit assertions
  (0.5·dx translation, 0.2 rad/frame rotation), `workspace_aabb()` for
  calibration. `j2`/`j3` pinned to 0; the only enforced `<mimic>` is
  `jaw_joint_2 = -jaw_joint_1` (PyBullet has no native `<mimic>` support).
- **`host/mpm_adapter.py`** — `record_grasp_episode()`, a scripted
  approach → close → retract (no planner), plus `capture()`/`advance()`
  changes to carry `joint_pos`/`grasp_active`/`grasp_node_ids` and to refresh
  collision geometry once per recorded frame (matching the vendored
  `step()`'s own cadence) rather than once per substep. `--task
  {settle,grasp}` added to `main()`; `settle` is the unchanged default.
- **Base placement** (`psm.DEFAULT_BASE_POSITION`/`ORIENTATION`) was picked
  empirically, not derived — the chain has too many compounded rotated
  offsets (`j1`'s own origin alone translates 0.834 m along a rotated axis)
  to hand-derive. `workspace_aabb()` at the chosen placement (500 samples)
  gives `x∈[0.006,0.516], y∈[0.117,0.564], z∈[-0.058,0.445]` domain-metres,
  comfortably containing the tissue's spawn region
  (`x∈[0.1,0.4], y∈[0.2,0.5], z∈[0.05,0.10]`, `init_cube()`), on the first
  placement tried.
- **Contact threshold, measured not inherited.** Jaw half-extents came out to
  ~6–9 mm — smaller than one grid cell (`dx=15.6 mm`). The vendored default
  `threshold=0.05` (5 cm) would have extended the contact "halo" roughly two
  grid cells beyond the jaw's true surface. `record_grasp_episode()` uses
  `threshold=0.02` (2 cm) instead — documented, deliberately tighter, the
  safe direction to be wrong in for a sticky-contact collider.

### 10.4 Verification

`host/smoke_test_psm.py`, 19 checks in the same idiom as
`smoke_test_mpm.py`, split into a shared-PSM phase (nothing that calls
`substep()`) and a real-episode phase (`record_grasp_episode()` run once,
inspected). Three of the 19 directly reproduce the bugs §10.1 caught, run
against fixed versions and confirmed they'd catch the original mistake:
the three-way AABB/live-proxy-AABB/SDF-argmin consistency check (would have
caught the `collisionFramePosition` bug — two independently-computed poses
could agree while both being wrong, but the SDF's own minimum cannot),
per-proxy velocity independence (would have caught the shared-EE-delta
bug), and an independent geometric recomputation of `contact_mode` from
`joint_pos` alone, matching all 50 recorded frames of a real episode exactly
— not inferred from aggregate tissue motion, which is confounded by
ordinary gravity settling that happens with or without the tool nearby.

```
pytest tests/ -q                                        278 passed, 1 skipped
python host/smoke_test_psm.py                            19 passed, 0 failed
python host/mpm_adapter.py --task grasp --out data_mpm_grasp --episodes 2 --steps 50
python host/validate_dataset.py --data data_mpm_grasp/    0 failed, 4 WARN (honest)
```

The two WARNs on every grasp episode are both expected, not regressions:
`grasp is consistent` ("grasp never active in this episode" — §10.1 point 4,
by design) and `F incompressible` (a pre-existing material-model property,
unrelated to this section). Dataset-wide checks over the two episodes
**PASS**: `deformation is diverse` (peak displacement 33.1–34.9 mm, 3%
spread) and `material is diverse` (μ spans 2112–3758 Pa). A real episode
shows tissue displacement up to 8.6 mm attributable to the tool, and
`contact_mode` stages cleanly `NONE → TOUCH → STICK → TOUCH → NONE` across
approach/close/retract with no direct `NONE → GRASP` jump — because `GRASP`
never appears.

### 10.5 Still open

- **Trocar-pivot (RCM-constrained) motion.** The URDF's `RCM` link exists
  (a visual-only sphere) but nothing constrains the tool through it; IK is
  free-space 6-DOF. A real trocar constraint needs either a nullspace/
  secondary-objective IK term or a reduced action parameterization.
- ~~Persistent grasp attachment.~~ **Done, §10.7** — `host/grasp_constraint.py`.
- **`psm_Si_model/PROVENANCE.md`'s source/license fields are still TODO.**
  Blocks nothing technically, but should be resolved before this asset is
  treated as a permanent part of the repository's history.
- **§4 remains untouched.** The PyBullet/container track and
  `container/timestep_study.py` are unrelated to this section and still
  unrun.
- **Material ranges are still placeholders** (`materials.py`). The two grasp
  episodes collected here are a pipeline demonstration, not a claim about
  real tissue response to grasping.

### 10.6 `host/visualize_grasp.py` — the recorded episode replayed with real mesh geometry

`visualize_trajectory.py` renders `ee_pose` as a single square marker because
it predates the PSM. That's honest but not the geometry a reader needs to
judge whether the tool is actually where it looks like it should be relative
to the tissue. `visualize_grasp.py` instead loads the wrist+jaw assembly's
STL meshes (`tool_main_link` through the two jaw links — `link_0`–`link_4`,
the arm base, are excluded: they sit ~0.5 m away at
`psm.DEFAULT_BASE_POSITION` and would force the view to zoom out until the
tissue is a speck), bakes each link's `<visual><origin>` transform into its
vertices once (parsed from the URDF XML directly, not transcribed by hand),
and replays `joint_pos` frame-by-frame through a throwaway PyBullet `DIRECT`
connection to get live FK — no `host/psm.py`'s `PSM` class, no
`MPM.mpm3d` import, no ~10 s Taichi compile, since pure rendering needs
none of it.

Checked directly, not assumed: at `jaw_joint_1 = 1.0` rad the two jaw
meshes visibly splay open on either side of the yaw link; at `0.0` rad they
close together into a single compact shape — the `jaw_joint_2 =
-jaw_joint_1` mimic (host/psm.py) rendering correctly through independently
loaded meshes, not just matching in the recorded numbers. A static frame
mid-approach shows the instrument tip sitting directly at the tissue's top
surface, matching `contact_mode`'s own timing.

### 10.7 Four real bugs, and a real persistent grasp (`host/grasp_constraint.py`)

§10's PSM was reviewed again — by name, four specific bugs, plus a request to
implement the persistent grasp attachment §10.5 had left as future work. All
four bugs were confirmed by reading the code directly before any fix; all
four were real.

**1. Jaw collision boxes used a world-frame AABB as if it were local-frame.**
`_build_jaw_proxy()` computed `half_extents` from `p.getAABB()` — world-axis-
aligned — and reused those three numbers directly as the box's LOCAL
half-extents (the box's own axes are identity relative to the jaw). Since the
jaw's local frame is not world-axis-aligned at the reference pose, this
scrambled which measured extent belonged to which local axis. Measured true
local half-extents: `[2.50, 5.75, 1.58]` mm; the world-AABB-derived values
were `[5.94, 5.50, 9.38]` mm — oversized on two axes, short on the third.
Fixed by adding `psm.mesh_vertices_in_link_frame()` (moved out of
`host/visualize_grasp.py`'s `_load_link_meshes()`, which already did this
exact transform for rendering — now the single, shared, correctness-critical
source, reading `<collision>` origin, not `<visual>`) and taking a plain
axis-aligned min/max of vertices already expressed in the link's own local
frame. The fixed value reproduces the independently-measured
`[2.50, 5.75, 1.58]` mm exactly; `host/smoke_test_psm.py` now asserts this as
a named regression value, not just "some number now."

**2. Collision velocity was never actually primed.** A comment on
`set_ee_pose(poses[0], jaws[0])` claimed it "primes the per-proxy prev-pose
cache" — false. `set_ee_pose()` never touches `PSM._prev_box_pose`; only
`update_collision_velocity()` does, and it was never called before the loop.
So the very first REAL commanded transition took the "first frame, zero
velocity, no sweep-limit check" branch instead of a true priming call — the
actual first move was recorded with zero jaw velocity and skipped its safety
check entirely. Fixed: `robot.update_collision_velocity(rec.m, rec.frame_dt)`
now runs once, explicitly, right after the initial `update_collision_geometry()`
call, before the loop — "before the episode starts" is now the real priming
moment.

**3. The episode's actual final state was simulated but never recorded.** 51
waypoints, but the loop only ever captured 50 rows — the last action's
resulting state (`poses[50]`) was executed but discarded. Fixed: one more
`capture()` call after the loop, with a zero-padded action, matching
`src/actions.py`'s own documented final-row convention. Episodes are now
`T = n_steps + 1` rows.

**4. Contact detection used the recorded subset instead of all particles.**
`_contact_mode()` subsetted to `rec.particle_ids` (3000 of 24000) *before*
computing the minimum box-surface distance — biasing that minimum upward
(never too small), against the project's own established rule (used
everywhere else for `safety_strain`/`exposure`): compute extrema over the
full particle set, subset only for what's written to disk. Fixed to use
`pos_full` directly.

**The new feature: `host/grasp_constraint.py`.** A host-owned, persistent
rigid attachment: on freeze, select full-solver particles within
`GRASP_CAPTURE_RADIUS_M` of *both* jaw box surfaces simultaneously (requiring
both is what turns "near a jaw" into "in the gap between them," and makes
freeze self-gating on jaw angle with no extra threshold — the two boxes are
geometrically far apart while open, so nothing can satisfy both proximities
regardless of radius); freeze membership and each particle's position in the
gripper's frame (`r_p`, captured relative to the **commanded** pose, not the
IK-achieved one — same reasoning as the state/action timing contract
elsewhere in this file); enforce `x_p(t) = T_gripper(t) . r_p`,
`v_p = v_gripper + omega_gripper x (x_p - c)` **every substep** (not once per
frame — `MPMRecorder.advance()` gained a `grasp`/`pose_start`/`pose_end`
parameter set, threaded explicitly through every call rather than cached
internally, specifically to avoid a third instance of the "looks primed but
isn't" bug class bugs 2 and the original collision-frame review both hit).
`CONTACT_GRASP` is emitted only while the mask is nonempty; release is a
separate, explicit jaw-angle trigger (`JAW_RELEASE_TRIGGER_RAD = 0.3` rad —
release cannot legitimately be geometric drift, since a forced particle
ending up outside its own now-irrelevant proximity radius must not
auto-release). A 4th scripted phase (`n_release`) was added so release is
actually exercised at least once — the original 3-phase script never reopens
the jaw.

`substep()` (`third_party/MPM/mpm3d.py:309-320`) is one Taichi kernel —
clear grid, P2G, Boundary, G2P, all inlined — with no interception point
from Python between them. Enforcement is therefore a **post-substep
correction**: let the vendored kernel run normally, then overwrite `F_x`/`F_v`
for attached indices via a small new kernel, closing over the real field
objects the same way `host/mpm_energy.py`'s `_build_kernels()` already does
in this codebase — including that file's own documented workaround for
`from __future__ import annotations` breaking Taichi's kernel-argument type
resolution (no annotated arguments; the gripper transform lives in small
persistent 0-d fields instead, updated via plain Python assignment, the same
idiom `mpm3d.py` uses for `co_v`/`co_w`/`centroid`).

**`F_C[p]` is zeroed for attached particles, and this reverses the first
design's stated judgment call — because the first design's reasoning was
empirically wrong.** The original plan explicitly argued for leaving `F_C[p]`
untouched (zeroing it would falsely claim zero local strain). Built and run:
with `F_C[p]` left alone, a real 60-row episode showed logged `safety_strain`
spike to **26.6x and 15.9x** at several frames — physically absurd, and
`validate_dataset.py` correctly FAILed on it. Diagnosis: every substep, `G2P()`
resamples `F_C[p]` from the grid velocity *gradient* at the particle's
*current* position — but that position was just teleported to the rigid
target by this file's own correction, into a grid neighbourhood the
particle's momentum was never part of. The resulting sampled gradient is a
numerical artifact of the teleport, not a real strain-rate, and
`P2G()`'s `F[p] = (I + dt*F_C[p]) @ F[p]` compounds it every subsequent
substep. Zeroing `F_C[p]` for attached particles breaks the loop entirely:
the same episode's peak `safety_strain` dropped to a physically reasonable
2.25x. `F[p]` itself is never written directly and is not indirectly frozen
either — this measured correction, not the first draft's a priori reasoning,
is what shipped.

**`check_grasp_is_consistent` needed a real fix, not a weakening.**
`grasp_node_ids` indexes the recorded N-particle subset (3000 of 24000);
a real, small, frozen attachment (measured: often just 1-3 full-solver
particles) can legitimately never intersect that subset at all, which used
to read as `grasp_active=True` with empty ids — the exact pattern the
existing check FAILed on. But since `host/grasp_constraint.py` freezes
membership once and never recomputes it while active, `grasp_node_ids` must
be **identical** across every step of one continuous active run — which can
legitimately mean "always empty," but can never legitimately flicker. The
check now enforces exactly that (a strictly stronger invariant than before,
catching a failure mode — mid-grasp membership changing — the old check
couldn't see at all) and downgrades the empty-subset case to an explanatory
WARN rather than a FAIL.

**Verification.**
```
pytest tests/ -q                                        278 passed, 1 skipped
python host/smoke_test_psm.py                            17 passed, 0 failed
python host/smoke_test_grasp.py                           8 passed, 0 failed
python host/mpm_adapter.py --task grasp --out data_mpm_grasp \
    --episodes 2 --n-release 10
python host/validate_dataset.py --data data_mpm_grasp/
```
Both real episodes show `contact_mode` staging cleanly through
`none -> touch -> stick -> grasp -> ... -> none`, `check_boundary_is_held`
and `check_contact_mode_transitions` both PASS, and `CONTACT_GRASP` is
genuinely present — not fabricated, not proximity-only.

**Still open, honestly.** `check_logged_metrics_match_recomputation`'s
`safety_strain` corroboration can legitimately FAIL for a grasp episode: its
margin was calibrated against passive settling, where peak stretch is
diffuse, but a grasp concentrates peak stretch at the 1-3-particle grasp
point, which a random 3000/24000 subset can easily miss regardless of how
correct the physics is. Verified directly (not assumed): the FULL-particle
logged value itself is a modest, physically reasonable ~2.1x in the case
this was checked against — no blowup — so this is a subset-coverage gap in a
pre-existing, unmodified check, not a sign the grasp constraint is wrong.
Not fixed here — it touches a check that also protects passive-episode data
integrity, and the right fix (task-aware margin? bias `n_record` sampling
toward the grasp region? something else?) is a real design decision, not a
bug fix, left for whoever collects `--task grasp` data at scale next.

---
## 11. Files

Every path below was checked against the repository on 17 August 2026; the
four §9.7 rows were added on 26 August 2026; the `psm_Si_model`/`psm.py`/
`smoke_test_psm.py` rows were added on 31 August 2026 (§10); the
`grasp_constraint.py`/`smoke_test_grasp.py` rows and the `psm.py`/
`mpm_adapter.py`/`validate_dataset.py` updates were added the same day (§10.7);
the Phase 0 rows were added on 15 September 2026 (§13).

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
| `host/validate_dataset.py` | macOS | 17 data-integrity checks | §8.5, §9.2, §9.5, §10.7 |
| `host/smoke_test_mpm.py` | macOS | MPM smoke test, 16 checks | §9.4, §9.5, **16/16** |
| `third_party/PROVENANCE.md` | — | Upstream SHA, checksums, local edits | §9.4 |
| `third_party/MPM/` | macOS | Vendored SurRoL `Dev` MPM, 4 files, unmodified | §9.3, §9.4 |
| `host/mpm_adapter.py` | macOS | MPM -> v2.1 episodes, one child process each | §9.5, §10, §10.7 |
| `psm_Si_model/` | — | dVRK Si PSM URDF + meshes; source/license still TODO | §10.1, §10.3 |
| `host/psm.py` | macOS | PSM: IK, jaw proxy colliders (local-frame AABB, §10.7), SDF/co_v/co_w sync, shared mesh-in-link-frame helper | §10, §10.7 |
| `host/grasp_constraint.py` | macOS | Persistent grasp: freeze/release, per-substep F_x/F_v correction kernel | §10.7 |
| `host/smoke_test_psm.py` | macOS | PSM smoke test, 17 checks | §10.4, §10.7, **17/17** |
| `host/smoke_test_grasp.py` | macOS | GraspConstraint smoke test, 8 checks | §10.7, **8/8** |
| `host/visualize_grasp.py` | macOS | Renders --task grasp episodes with real PSM mesh geometry (wrist+jaw assembly), not a point marker | §10.6 |
| `host/substep_study.py` | macOS | Substep convergence sweep, one child per row | §9.6, **run**; its dissipation verdict is superseded by §9.7 |
| `src/energy_ledger.py` | **both** | Closed mechanical-energy budget, exponent fit, verdict | §9.7 |
| `host/mpm_energy.py` | macOS | Instrumented MPM driver: split kernels, grid probes, known ICs | §9.7 |
| `host/energy_audit.py` | macOS | Energy audit, 5 selftests + 3 cells, one child per row | §9.7, **run** |
| `src/graph_design.py` | **both** | Influence radius with status, node samplers, radius graph, components, hop diameter | §13 |
| `src/damping.py` | **both** | Kelvin–Voigt / mass damping decay rates, viscous stability limit | §13.2 |
| `host/mps_scatter_check.py` | macOS | MPS scatter-add correctness vs CPU/float64, message-passing throughput | §13.5, **run** |
| `host/mpm_locality_study.py` | macOS | Local-kick influence radius; 3 materials × 2 configs × 5 damping; `--selftest`, `--replot` | §13.2, **run** |
| `host/locality_attribution.py` | macOS | Same probe at 55/110/220 substeps: physical vs numerical coupling | §13.2, **run** |
| `host/coarsening_study.py` | macOS | Safety/exposure fidelity vs node count and sampler | §13.3, **run** |
| `host/graph_stats.py` | macOS | Degree (surface/interior), components, hop diameter vs R/h; M criterion | §13.4, **run** |
| `figures/phase0/` | — | Committed Phase 0 figures | §13 |
| `container/verify_container.py` | Linux | Physics + SurRoL checks | |
| `container/make_tissue_mesh.py` | Linux | Author the tissue sheet | |
| `container/collect_retraction.py` | Linux | Scripted retraction episodes | |
| `container/timestep_study.py` | Linux | Convergence study | **never run** (§4) |
| `container/validate_physics.py` | Linux | Nine controlled physics experiments | **never run** |
| `tests/` | macOS | 320 unit tests, seven files | §8.7, §9.7, §13 |

**Data and artifacts** (git-ignored):

| Path | Contents |
|---|---|
| `data/` | 20 PyBullet retraction episodes, schema v1.0, 1 Aug |
| `data_synth/` | 4 analytic episodes, schema v2.0, regenerate with `synthetic_traj.py` |
| `data_mpm_full/` | 1 full-resolution (24,000-particle) grasp episode for §13.3, regenerable |
| `runs/` | Phase 0 study outputs: `locality/`, `coarsening/` (incl. `nodesets.npz`), `graph_stats*/`, `phase0/` |
| `models/dynamics_mlp.pt` | MLP baseline, 1 Aug |
| `assets/tissue_20x20.obj` | Authored tissue mesh (tracked) |

---

## 12. References

- Xu et al., *SurRoL*, IROS 2021 — [arXiv:2108.13035](https://arxiv.org/abs/2108.13035)
- *Efficient Physically-based Simulation of Soft Bodies in Embodied Environment for Surgical Robot* — [arXiv:2402.01181](https://arxiv.org/abs/2402.01181)
- Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021 — [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
- *Autonomous Soft Tissue Retraction Using Demonstration-Guided RL* — [arXiv:2309.00837](https://arxiv.org/abs/2309.00837)
- Long et al., *Surgical embodied intelligence for generalized task autonomy*, Science Robotics 2025
- [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) · [Isaac Sim cloud deployment](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_cloud.html)

---

## 13. Session 5 — 15 September 2026: Phase 0 of the MPM→GNN brief, measured before building

A brief arrived proposing a graph network over the MPM tissue, organised as
Phases 0–6. Its central argument is sound: a GNN is a local model (M
message-passing steps over radius R see M·R of the object), while
near-incompressible tissue is globally coupled over one recorded step. Its
Phase 0 is four measurements that choose the graph node count, the connection
radius, the recording rate and the damping before anything is built.
**This session did Phase 0 only, on the MPM scene as it exists.** Both scope
limits were agreed before any code was written, because Phases 1–6 cannot be
specified honestly until these four numbers exist. Work is on branch
`phase0-measure`, one commit per step.

The short version, before the detail:

- **The brief's geometry, rate and several repo facts were wrong** (§13.1).
  Every number below uses the real scene.
- **At the current 80 Hz, one recorded step couples the whole slab**, and no
  node set passes at M = 10. **At 240 Hz every candidate node set passes at
  M = 10.** But the far-field part of that coupling is the solver's transfer
  stencil, not elastic waves: it grows with substep count at fixed physical
  time (§13.2).
- **Damping does not localise the response** at any tested level. Mass damping
  cannot, by construction; Kelvin–Voigt diffuses momentum outward (§13.2).
- **float32 discards volumetric strain increments below
  ~ulp(1)/(2·dt)** — 7.9e-4 s⁻¹ at 55 substeps/frame — while keeping shear
  (§13.2). This is a solver finding, not a GNN one.
- **No node set up to 3,000 carries the peak safety stretch** (best worst-seed
  0.84 of the excess). Weighted exposure is fine from ~450 nodes. Unweighted
  exposure is broken by construction on any subset (§13.3).
- **MPS scatter-add is correct.** Batched inference is **0.05%** of an MPPI
  budget whose unit the brief also got wrong (§13.5).

### 13.1 The brief against the repository

Checked before planning, by reading the code rather than the brief:

| Brief assumes | Repository has | Evidence |
|---|---|---|
| 100×100×10 mm clamped sheet | **300×300×50 mm** random cloud, 24,000 particles, **no clamp** | `mpm3d.py:363` `init_cube`; adapter writes `boundary_mask` all False |
| particle spacing 2.5 mm | mean spacing **5.72 mm**; grid dx **15.6 mm**; slab **3.2 cells thick** | `MPM/config.py` n_grid = 64 |
| 30 Hz recording | **80 Hz** (`frame_dt = 0.0125`) | `mpm_adapter.py:135` |
| damping is a new option | the `*= 0.1` floor band already removes 89–93% of energy | §9.7 cell 3 |
| schema v2.0; `src/checks.py`; "§12.1 cKDTree" | **v2.1**; no `checks.py`; §12 is References; scipy used nowhere; `src/` must stay numpy-only | `trajectory_io.py:190` |
| `host/collect_mpm.py` | `host/mpm_adapter.py` exists and is the adapter | — |

**Wave reach, recomputed.** ρ = 1050 kg/m³; R = 1.5·h = 8.6 mm if the nodes
were the particles.

| material | c_p (m/s) | reach in 1/80 s | 1/30 s | 1/240 s | M needed at 80 Hz if nodes = particles |
|---|---|---|---|---|---|
| soft (μ = 200, λ = 2e4) | 4.4 | 55 mm | 147 mm | 18 mm | 6 |
| mid (μ = 2e3, λ = 2e5) | 13.9 | 174 mm | 465 mm | 58 mm | 20 |
| stiff (μ = 2e4, λ = 2e6) | 44.1 | 551 mm | 1469 mm | 184 mm | 64 |

For M = 10 over a 300 mm object the brief's own coarsening formula gives
h_graph ≥ 20 mm, ~560 nodes — coarser than the solver grid, and only ~2.5 nodes
through the slab's thickness.

**Three findings from the arithmetic alone:**

1. **`compute_exposure` is not resolution-invariant.** `_coverage` *sums* unit
   Gaussians (σ = 4 mm) and compares the sum against an absolute 0.5. Under the
   full slab coverage is ~27; the same geometry on 600 nodes sums to ~0.67, on
   150 to ~0.17 — every query point reads "clear" under complete cover. Fixed
   by an optional `weights` argument (commit `635e367`); measured in §13.3.
2. **`p_vol = (dx/2)³` assumes 8 particles per cell; the cloud has 20.3.**
   Total mass, gravity load and energy are 2.54× the geometric body. Stress and
   mass both scale with `p_vol`, so wave speeds and accelerations are
   unaffected; any *extensive* readout (mass, contact force, absolute energy —
   including §9.7's joules) is 2.54× off. **Not fixed.** It must be decided
   before Phase 1 records a `particle_volume` field.
3. **The brief's MPC budget has the wrong unit.** It states S·H·30 = 230,000
   "node updates per second" (S = 256, H = 30, 30 Hz). That is *forward passes*.
   Each pass evaluates every edge in every block, so the requirement is
   S·H·f·E·M = **1.9 × 10¹⁰ edge-block evaluations/s** at n = 600, E = 8,400,
   M = 10 — five orders of magnitude above the brief's figure. §13.5 measures
   against the corrected number.

### 13.2 Locality: how far a local kick travels in one recorded step (study 0.1)

`host/mpm_locality_study.py`, reusing `mpm_energy.Instrument` and
`energy_audit.py`'s one-child-per-row dispatch.

**The probe is not the brief's.** The brief displaces one particle's
*position*. In MPM stress comes from each particle's carried F, so a moved
particle with unchanged F exerts no extra stress — that probe measures the
quadrature, not the material — and it is one of ~20 particles in its cell, so
the first scatter dilutes it ~20×. Perturbing F on a patch was the other
option; rejected because its amplitude is hard to relate to network inputs.
**Chosen: a velocity impulse on the ~12 particles within dx/2 of a probe
particle**, along x (in-plane, the P-wave) and z (normal, bending), at a centre
probe and an edge probe.

From one base state, five runs: A, A′ (unkicked, twice), B, B′ (kicked, twice),
B2 (kicked ×2). Response = |v_B − v_A| per particle, binned by distance, divided
by the patch's own response. `graph_design.influence_radius` returns the radius
with a status: `saturated` = still above threshold at the far edge (a lower
bound), `below_noise` = threshold under the noise floor (refused). Snapshots
after 1, 2, 3, 4 and 8 frames of 1/240 s give every recording interval from one
run.

**Sweep:** soft/mid/stiff corners of `materials.py` × `clean` (g = 0, lifted
clear, no-contact asserted every frame) and `as_collected` (g = 9.8, settled
0.75 s onto the floor band) × damping {none, Kelvin–Voigt at two levels, mass
at two levels}. One substep count per material across all damping levels (55
per 1/240 s frame, set by the strongest viscosity's stability limit), because
§9.7 showed dissipation shifts with substep count.

#### The selftest failed twice, and both failures were real

**Failure 1: linearity 1.45 at a 0.01 m/s kick.** Diagnosed by measurement
(soft, clean, 55 substeps/frame, substep 75.8 µs):

| kick | band | linearity median | `dt·max|C|` < ulp(1)/2 | diagonal ΔF exactly 0 | off-diagonal ΔF exactly 0 | median \|J − 1\| |
|---|---|---|---|---|---|---|
| 0.01 m/s | 0–20 mm | 2.14 | 99% | 0.6% | 0% | 7.8e-6 |
| 0.01 m/s | 60–120 mm | 1.89 | 94% | 81% | 0% | 9.7e-13 |
| 0.01 m/s | 120–350 mm | **1.17** | 100% | **100%** | **0%** | **1.3e-14** |
| 1 m/s | 120–350 mm | 2.000 | 0% | 0.9% | 0% | 1.4e-5 |

`P2G` updates `F ← (I + dt·C)·F` in float32. When |dt·C_ii| < ulp(1)/2 =
5.96e-8, `1 + dt·C_ii` rounds to exactly 1 and the increment is lost. The
off-diagonal entries start at 0, where a tiny number is represented exactly,
so they survive. **The solver therefore drops volumetric/normal strain
increments below a strain-rate floor while keeping shear — and volumetric is
precisely the pressure channel that carries the P-wave.** The floor is
ulp(1) / (2·dt):

| setting | substep | normal strain-rate floor |
|---|---|---|
| locality study, 55 per 1/240 s | 75.8 µs | 7.9e-4 s⁻¹ |
| `data_mpm/` soft-λ episode, 24 per 1/80 s | 521 µs | 1.1e-4 s⁻¹ |
| `data_mpm/` stiff-λ episode, 106 per 1/80 s | 118 µs | 5.1e-4 s⁻¹ |

It *rises* with substep count. That qualifies §9.7's "refining dt at fixed dx is
a legitimate instrument": it is, until the floor bites. Metal has no float64
(§10), so the obvious fix is unavailable. At 1 m/s the physics is still linear
— the patch response is 3.1e-3 of the kick at 0.1 and at 1 m/s — so the kick
is now **1 m/s**, and linearity is checked **per distance band**, because an
overall median hid a far-field failure once.

**Failure 2: the at-rest control pair is vacuous.** The clean base state is
exactly at rest, so A and A′ sum only zeros and |A′ − A| = 0 by construction.
Selftest check 6 (refusal on noise) never reached the refusal branch as a
result. The repeated kicked pair |B′ − B| carries the real run-to-run noise
(8.3e-8 m/s); the floor is now the larger of both pairs and the float32
spacing.

**Selftest, as passed:**
```
[PASS] kv_eta0_matches_solver             KV(eta=0) vs vendored 1.0 ulp; solver's own floor 1.0 ulp
[PASS] kv_leaves_rigid_translation        eta=83.1 Pa*s: E(T)/E(0) = 1.0000011
[PASS] mass_fails_rigid_and_matches_closed_form   E(T)/E(0) = 0.513416 vs exp(-2 alpha T) = 0.513417
[PASS] kv_dissipates_at_order_of_predicted_rate   0.22x continuum at a 3-cell wavelength
[PASS] probe_is_linear_in_every_band      2.000 / 2.000 / 1.999 / 2.000; floor 8.3e-8 m/s
[PASS] noise_only_is_refused              kicked control pair -> below_noise
```

#### Results

1% velocity-influence radius (mm), centre probe, kick along x, **no damping**.
`>=N` = saturated. Centre-to-corner distance is ~234 mm.

| config | material | 1/240 s | 1/120 s | 1/80 s | 1/60 s | 1/30 s |
|---|---|---|---|---|---|---|
| clean | soft | 130 | 195 | ≥215 | ≥215 | ≥215 |
| clean | mid | 160 | ≥215 | ≥215 | ≥215 | ≥215 |
| clean | stiff | 170 | ≥215 | ≥215 | ≥215 | n/r |
| as_collected | soft | 145 | 225 | ≥230 | ≥230 | ≥230 |
| as_collected | mid | 175 | ≥215 | ≥215 | n/r | ≥215 |
| as_collected | stiff | n/r | ≥215 | ≥215 | n/r | n/r |

Full tables for both probes, both axes, both thresholds and all damping levels:
`runs/locality/tables.md`. Figures: `figures/phase0/locality_*.png`.

**The soft material's influence outruns its P-wave by 7×.** At 1/240 s,
c_p·t = 18 mm but r₁% = 130 mm, so something other than elastic waves carries
the far field. Two candidates were separated by holding physical time fixed and
changing only the substep count (`host/locality_attribution.py`):

| soft, t = 4.2 ms (c_p·t = 18 mm) | resp/src @ 50 mm | @ 150 mm | @ 200 mm | r₁₀% | r₁% |
|---|---|---|---|---|---|
| 55 substeps/frame | 0.77 | 5.9e-3 | 1.1e-3 | 80 | 130 |
| 110 | 0.82 | 2.0e-2 | 3.4e-3 | 90 | 185 |
| 220 | 0.84 | 7.3e-2 | 3.6e-2 | 100 | ≥215 |

- **The far tail is numerical.** Physical propagation over 4.2 ms cannot depend on
  the substep count; this tail grows up to 30×. Each P2G/G2P pair spreads a
  particle's momentum over ±1.5 dx twice, and more substeps are more transfers.
- **The near field (≤ ~3 dx ≈ 47 mm) is set by the grid, not by c_p**, and does
  not change with substep count: one transfer already couples everything within
  the stencil.
- **Stiff is physically global**: c_p·t exceeds the object at every interval,
  responses exceed the source (ringing), and many stiff cells are `n/r` because
  run-to-run noise reaches 2–37% of the response.
- **A momentum-free kick was tried and rejected.** Subtracting the patch impulse
  from every particle puts a uniform ~5e-4 m/s into B − A itself, so its far
  field rose instead of falling.

**For the learning problem the distinction matters twice.** The network learns
the solver, so the numerical tail *is* in the target, and it gets worse as
substeps increase. And "record faster" helps by shortening the interval, not by
removing a physical wave.

**Damping does not localise, and neither form could have been expected to:**

- **Mass damping** multiplies every velocity by the same factor, so
  response/source keeps its shape. The `mass_slow` curves match `none`.
- **Kelvin–Voigt viscosity** damps high wavenumbers but *diffuses* momentum:
  at η = 83 Pa·s, √(νt) is ~18–26 mm at 1/240 s and ~31–44 mm at 1/80 s
  (transverse–longitudinal). The `kv_fast` far field sits higher, not lower.

The brief's option B1 (add damping to make the dynamics local) is rejected on
this evidence. The hook stays in `Instrument.set_damping()` (commit `59f17e4`)
because tissue *is* viscoelastic; what η real tissue has is a literature
question this session does not answer.

**What this does not establish.** Damping levels were chosen by decay time, not
measured tissue viscosity, and the material ranges are still placeholders.

### 13.3 Coarsening: how few nodes carry the metrics (study 0.2)

`host/coarsening_study.py` on one full-resolution grasp episode:
`mpm_adapter.py --task grasp --n-record 24000 --out data_mpm_full` (51 rows,
46.6 MB, validates 14 PASS / 1 WARN / 0 FAIL — the WARN is det F ∈ [0.40, 1.25]
around the grasp). A grasp and not a settle, because a settling slab stretches
~1.0 everywhere and any subset would agree. Frame 0 is asserted to be the rest
state (max|F − I| = 0.0). Node sets are chosen once in rest space.

**Acceptance, fixed before any result was seen:** in every seed, peak excess
ratio (max_c − 1)/(max_f − 1) ≥ 0.90, soft-stretch error ≤ 0.01, weighted hard
exposure error ≤ 0.05. Exposure is scored on the episode — whose target is
covered in every frame — **and** on an 11-step synthetic lateral shift of the
rest cloud across the target (geometry only), so the partially exposed range is
tested too.

| sampler | n | peak excess (worst / mean) | soft err | weighted exposure err | unweighted exposure err | OK |
|---|---|---|---|---|---|---|
| FPS | 450 | 0.49 / 0.65 | 0.021 | 0.039 | 0.66 | no |
| FPS | 600 | 0.49 / 0.74 | 0.024 | 0.035 | 0.54 | no |
| FPS | 3000 | 0.81 / 0.91 | 0.005 | 0.023 | 0.06 | no |
| surface-quota FPS | 450 | 0.61 / 0.64 | 0.009 | 0.035 | 0.65 | no |
| surface-quota FPS | **600** | **0.66 / 0.69** | **0.007** | **0.023** | 0.51 | no |
| surface-quota FPS | 3000 | 0.73 / 0.82 | 0.006 | 0.027 | 0.06 | no |
| lattice | 1800 | 0.84 / 0.84 | 0.003 | 0.055 | 0.09 | no |
| random (the adapter's method) | 3000 | 0.74 / 0.81 | 0.010 | 0.020 | 0.07 | no |

Full table: `runs/coarsening/table.md`; figure `figures/phase0/coarsening.png`.

- **No sampler and n ≤ 3,000 is acceptable. The failing criterion is always the
  peak.** A maximum over a subset is biased low (§9.5's rule 2), and here the peak
  concentrates on a handful of particles.
- **56% of the 146 over-threshold particles at the peak frame are within one grid
  cell of a grasp-constrained particle.** The safety peak is partly the grasp
  constraint's own artefact (§10.7), not tissue response.
- **Weighted exposure is within 0.035 from n = 450** with FPS or surface-quota
  sampling. **Unweighted exposure reads 0.90 "clear" under full cover at
  n = 150** — the §13.1 collapse, shown.
- **The lattice sampler puts 0–8% of its nodes on the surface**, against ~29% of
  particles. That matters for exposure (lattice needs n ≈ 3,000) and for §13.4's
  surface degree.

**The safety peak is a decision for Andric, not a parameter.** Options:
(a) keep full-resolution metrics as ground truth and train a learned
peak-correction head on node features; (b) upsample F from nodes to particles
(PhysWorld's skinning) — cannot recover a peak no node sees unless the upsampler
is learned; (c) redefine the constrained quantity as a coarse-stable statistic
(neighbourhood power mean, top-k fraction); (d) treat the grasp-neighbourhood
peak separately, since over half of it is the constraint. **Recommendation:
investigate (d) first — it is cheap and may remove most of the gap — then (a).**

### 13.4 Graph statistics: what graph those nodes make (study 0.3)

`host/graph_stats.py`. Pitch h is measured (median nearest-neighbour distance),
R = ratio·h over [1.2, 2.5]. Acceptance: one component, mean degree 10–20,
minimum degree ≥ 3, M_needed = ⌈r₁%/R⌉ ≤ M, and diameter ≤ M whenever the
influence saturates.

**A bug in the first version, caught before it reached this log.** It took the
largest influence radius over all cells and kept *that cell's* status. A
resolved 245 mm edge-probe cell then outranked five saturated ≥215–230 mm
centre-probe cells, the diameter criterion was never applied, and graphs with
13–18-hop diameters passed at M = 10. Saturation is now sticky.

**Results for surface-quota FPS, n = 600** (h = 18.7 mm):

| R/h | R | mean degree (interior / surface) | min | components | hop diameter |
|---|---|---|---|---|---|
| 1.4 | 26.1 mm | 5.8 (7.4 / 4.3) | 0 | 3 | — |
| 1.5 | 28.0 mm | 7.4 (9.0 / 5.7) | 1 | 1 | 21 |
| **1.7** | **31.7 mm** | **10.5 (12.4 / 8.5)** | **3** | **1** | **18** |
| 2.0 | 37.3 mm | 15.5 (18.4 / 12.5) | 4 | 1 | 15 |
| 2.5 | 46.7 mm | 28.0 (32.9 / 23.1) | 6 | 1 | 11 |

- **Surface nodes run ~30% below interior degree** at every R: the slab is only a
  few node pitches thick, so half a surface node's neighbourhood sphere is empty.
- **At 1/80 s (current data): no node set passes at M = 10.** Five of twelve
  influence cells saturate, and diameters are 13–29 hops. Sweeping M: M = 12
  passes only the n = 450 sets; M = 14 passes all four tested; M = 18 reaches the
  smallest R (1.7h).
- **At 1/240 s: every node set passes at M = 10** with R = 1.7–1.9h. No cell
  saturates; the largest radius is 185 mm (stiff, edge probe), so M_needed = 6
  at R = 31.7 mm.
- **Broken-input demonstration built in:** R = 0.5h shatters every set into n
  components, or the script exits non-zero.

Figure: `figures/phase0/graph_stats.png`; results `runs/graph_stats/` (1/80 s)
and `runs/graph_stats_240/` (1/240 s).

### 13.5 MPS scatter-add: correct, and far too slow for MPPI (study 0.4)

`host/mps_scatter_check.py`, torch 2.13.0. Message passing is a scatter-add,
written by hand because torch_geometric has no Metal path, and a scatter that
silently drops duplicate-index writes trains to a plateau with no error.

- **Exact-integer `index_add_` is bit-identical to CPU** up to E = 10⁶, including
  every edge into one node. Integers < 2²⁴ sum exactly in any order, so there is
  no tolerance to argue about.
- **Float forward and gradients: MPS error vs float64 is no worse than CPU
  float32's** (ratio 1.05 forward, 0.23 gradients; limit 4). **The first version
  FAILED both** with hand-picked 1e-5 / 1e-4 tolerances; measured against
  float64, the CPU was further from the truth than MPS at E = 10⁶. Summing
  ~1,700 float32 terms per node accumulates rounding a fixed tolerance cannot
  follow. The first gradient run also crashed — a test bug (`.to("cpu")` returns
  the same tensor, so `requires_grad_` leaked into the shared input).
- **Each criterion is shown to fail:** a permuted index, one dropped message
  (8,169× CPU rounding) and one mis-routed edge in the gradients (3,012×) are all
  flagged.

**Throughput**, one GNS-shaped block (3-layer 128-wide edge and node MLPs,
LayerNorm, residual):

| graph | MPS inference | MPS train | CPU inference | CPU train |
|---|---|---|---|---|
| n = 600, E = 8,400 | 1.1e7 edges/s | 5.0e6 | 1.7e6 | 7.7e5 |
| n = 1800, E = 36,000 | 1.4e7 | 5.8e6 | 3.0e6 | 1.3e6 |
| MPPI batch, S = 256 × n = 600 | 9.2e6 | 3.6e6 | — | — |

**Against the corrected budget of 1.94 × 10¹⁰ edge-blocks/s, batched MPS
inference is 0.05% — about 2,100× short.** Levers, none chosen: timestep bundling
(m = 8 cuts H by 8), fewer samples or a shorter horizon, fewer blocks, a coarser
graph, a real GPU, or a grid processor. Training on MPS at this scale is viable
for prototyping; the real runs belong on the cluster, as the brief expected.

### 13.6 The four numbers, with what each costs

| decision | value | status | the error being accepted |
|---|---|---|---|
| node set | **n = 600, surface-quota FPS** (50% surface) | provisional | peak excess stretch seen 0.66 (worst seed) — `safety_strain.max` is **not** carried; weighted exposure within 0.023; soft stretch within 0.007 |
| radius | **R = 1.7h ≈ 32 mm** | provisional | mean degree 10.5; surface nodes 8.5; minimum 3 |
| recording rate | **1/240 s** | recommended | 3× the frames of the current 80 Hz. The far-field coupling it resolves is partly the solver's stencil and grows with substeps |
| damping | **none** | measured | neither form localises; physical tissue viscosity unknown |
| message passing | **M = 10** at 1/240 s (M_needed = 6) | follows | at 1/80 s the same graph needs M ≥ 14–18 |

**Why 600 and not 450:** at 450, weighted exposure (0.035) and peak ratio
(0.61) are both worse, for a hop diameter only two smaller. The alternative is
recorded; the difference is not large.

### 13.7 Phases 1–6, annotated by what Phase 0 changed

Not implemented. Each line is a correction to the brief as written.

- **Phase 1.** Extend `mpm_adapter.py`, do not add `collect_mpm.py`.
  - Record at 240 Hz.
  - Decide particles-per-cell before writing `particle_volume` (§13.1 finding 2).
  - The brief's §1.5 boundary retune is moot until a clamp exists — there is none.
  - Lattice seeding interacts with surface starvation (§13.3, §13.4).
  - Record `tissue_C`, knowing its small diagonal increments are already below
    float32 resolution at high substep counts.
- **Phase 2.** Schema 2.1 → 3.0, not 2.0 → 3.0. Storage at 240 Hz is 3× the
  current frames before `tissue_C`.
- **Phase 3.** `check_coarse_metrics_track_fine` cannot use a 0.9 peak criterion:
  no node set up to 3,000 passes it. Check weighted exposure, and *report* the
  peak ratio. `check_graph_diameter` must know the recording rate — at 80 Hz it
  fails by construction.
- **Phase 4.** Brute force was enough for Phase 0's ≤ 3,000 static nodes; world
  edges rebuilt every step are the case that may need scipy in `src/` (and the
  container's numpy 1.23).
- **Phase 5.** The MPPI budget (§13.5) should shape the architecture from the
  start, not be discovered in Phase 6. Timestep bundling is no longer optional.
  Exposure heads or losses on a node subset must use `weights`.
- **Phase 6.** Unchanged.

### 13.8 Still open

- **The safety peak** (§13.3): options (a)–(d), Andric's decision.
- **The float32 volumetric strain-rate floor** (§13.2): no fix chosen. Metal has
  no f64, and more substeps raise the floor.
- **`p_vol` vs particles per cell** (§13.1): extensive quantities 2.54× off.
- **The MPPI budget** (§13.5): 2,100× short.
- **Tissue viscosity** from literature, and whether that η localises anything.
- **The locality study ran on one probe patch size and one kick amplitude.** The
  1 m/s choice is justified by linearity, not by a sweep.
- **Not verified this session:** `src/graph_design.py` and `src/damping.py`
  importing under the container's numpy 1.23 (no numpy-2-only API is used, but
  it was not run); the dataviz palette's colour-vision checks (no Node runtime on
  this machine; contrast was computed instead — every ramp step ≥ 2.06:1).

### 13.9 Verification

```
pytest tests/ -q                                          320 passed
python host/energy_audit.py --selftest                    SELFTEST PASSED (after the damping hook)
python host/mps_scatter_check.py                          6 PASS, 1 WARN (budget), 0 FAIL
python host/mpm_locality_study.py --selftest              6/6 PASS
python host/mpm_locality_study.py                         30/30 rows, 342 s
python host/locality_attribution.py                       6 children, 27 s
python host/mpm_adapter.py --task grasp --n-record 24000 --out data_mpm_full
python host/validate_dataset.py --data data_mpm_full/     14 PASS, 1 WARN, 0 FAIL
python host/coarsening_study.py --data data_mpm_full/     no sampler acceptable (as reported)
python host/graph_stats.py --samplers surface_quota fps lattice --n 450 600 900 --frames 1
                                                          all 9 acceptable; broken-input PASS
```

Checks shown failing on broken input this session: four source mutations of
`graph_design.py`, each caught by exactly its target test; below-noise refusal
on real solver noise; mass damping failing the rigid-translation invariance
Kelvin–Voigt passes; a permuted index, a dropped message and a mis-routed edge
in the scatter check; R = 0.5h shattering every node set; unweighted exposure
reading clear under full cover.

---

## 14. Session 6 — 22 October 2026: the to-do list, schema 3.0, and the graph builder (Items 1–6)

> Written the same day. Covers the gap since §13 (the 25 Sep cleanup and the
> 5 Oct decisions were never logged here) and the implementation of the PhD
> to-do list's Items 1–6 (`Soft_Tissue_World_Model_TODOLIST.docx`). Assessment
> and plan: Project doc `claude/TODO_ASSESSMENT_AND_PLAN.md`.

### 14.1 What happened between §13 and today

- **25 Sep cleanup.** The §9.6–§13 study scripts were removed (`substep_study`,
  `energy_audit`, `mpm_energy`, `mpm_locality_study`, `locality_attribution`,
  `coarsening_study`, `graph_stats`, `mps_scatter_check`, `src/energy_ledger.py`,
  `src/damping.py`); docs moved to `logs/`. **History was rewritten on 22 Oct to
  strip AI attribution trailers, so hashes older than that are void**; the
  removed code is recoverable from `git show 8749e36:<path>` (tip of
  `phase0-measure`).
- **5 Oct decisions.** (a) Run the simulator natively on macOS; Docker later
  only on x86 + NVIDIA. (b) The target arm is the **dVRK Classic**, matching the
  lab; the current `psm_Si_model/` does not. (c) Remote sessions reach the repo
  through a Linux ARM VM that cannot run Taichi, Metal or PyBullet.
- **22 Oct.** `cleanup` merged into `main` (`2d45c77`). Work for Items 1–6 on
  branch `todo-items-1-6`.

### 14.2 Decisions D1–D5, and the precedence rule

Alternatives and tradeoffs are in the assessment doc §5. Decided 22 Oct, with
the rule **"where an answer conflicts with the to-do list, the list's
configuration wins"**:

| # | Decision | Outcome |
|---|---|---|
| D1 | Recording rate | **240 Hz** (Δt = 1/240 s). `--frame-dt`, default 1/240. At 80 Hz Phase 0 needs M ≥ 14–18. |
| D2 | Arm | Pilot on the Si `psm.py`; **the Item 6 debug set goes through SurRoL's robot interface** (`Psm1`, Classic), because Item 6 says so. That puts the Classic swap on the critical path for the debug set. `host/check_surrol_interface.py` is its first step. |
| D3 | Grasp flag g | Two **separate** node-level indicators (Item 1): `g_contact` (proximity) and `g_grasp` (the solver's two-jaw latch evaluated on nodes). "g omitted" is the ablation. |
| D4 | Node count | 3,000 baseline; the extra comparison row is **1,000** (the list: "reduce toward 1,000 only if needed"), not Phase 0's 600. Radii 1.5/2.0/2.5 × grid_dx as listed. |
| D5 | Boundary flag b | Constant-zero slot, documented; boundary handling is not tested. |

### 14.3 Phase 0 integrity fixes (assessment §4.6)

- **A crashed episode is saved as `<name>.npz.partial`, never as `<name>.npz`.**
  `TrajectoryWriter.__exit__` and `MPMRecorder.__exit__` used to save whatever
  rows existed under the real name; the comparability check compares N, action
  width and dt, none of which a truncation changes. The `.partial` file holds a
  `partial_reason`; `list_trajectories` does not match it; `validate_dataset`
  prints how many exist.
- **`append()` validates the whole row before touching any buffer.** It used to
  store `tissue_pos` first, so a rejected row left buffers of unequal length,
  and the optional-field lock was recorded even for a rejected first row.
- **`close()` is atomic** (write `<path>.tmp`, `os.replace`). Writes go through a
  file handle because `np.savez_compressed` appends `.npz` to bare names.
- The `zeros(7)` vs `jaw[-1]` last-row pad difference is left: Item 5's
  convention never uses the last row's action.

### 14.4 Schema 3.0

Field-by-field meaning in `docs/FIELD_DICTIONARY.md` (the Item 5 deliverable).
Only additions; 1.0/2.0/2.1 files load with every new field empty.

- **`rest_pos` is read off the solver before the first substep**, where
  `init_cube()` has just set F = I — the reference configuration by the solver's
  own definition of F, asserted at collection. It is **not** frame 0 in general:
  `preroll_s` records how long was simulated before row 0, and the validator
  FAILs a file whose row 0 differs from `rest_pos` while claiming no pre-roll
  (the to-do's "do not substitute the first recorded position for X").
- `rest_pos_simulated` (all 24,000, 288 KB) lets node sets be re-chosen and holds
  the true rest positions of grasped particles the subset misses.
- `time`, `control_dt`, `phase`; per-jaw `jaw_box_pose` (achieved), `jaw_box_vel`
  (**backward** difference, row 0 zero — known at time t without the future),
  `jaw_joint`; static `jaw_names`, `jaw_half_extents`, `jaw_box_offset`, and the
  rule radii; `node_contact`, `node_grasp`; `grasp_full_ids` (ground truth).
- **Indexing** unchanged and now written down: T rows, `(G_t, a_t) → G_{t+1}`
  for t ≤ T−2, one zero pad action on the last row. Checked.
- **F defaults to `float32`** for collection (assessment §4.5).
- **One fixed node set**: `--particle-ids <file>` for every episode. A new
  dataset-wide check WARNs when episodes record different node sets.
- New validator checks, each demonstrated failing on corrupted data in
  `tests/test_schema_v3.py`: positive time intervals, reference configuration,
  material field, action/state indexing, jaw geometry, node indicators (must
  re-derive exactly), full-resolution grasp, integration residual (Item 7's
  pre-check, reported now so the number exists before Item 7), node identity.

### 14.5 The node-level grasp rule (`src/tool_geometry.py`)

- One module computes box distance, `g_contact`, the `g_grasp` latch and jaw
  velocities, imported by the collector, the validator, the graph builder and —
  later — inference. `grasp_constraint.py` now takes its two constants from it.
- The latch mirrors `_capture_row`'s order: freeze (if nothing held, latch every
  node within 20 mm of BOTH jaws), then release (jaw ≥ 0.3 rad).
- **Known divergence from the solver.** The solver freezes once, over all
  particles. The node rule freezes over nodes; if the solver's 1–3 particles
  are not near any node, the node latch can stay empty and fire on a later row
  when a node comes within range of both closed jaws. That is the documented
  node-level stand-in, not a bug; Item 11 compares it with `grasp_full_ids`.
- Flags are computed from float32-rounded geometry (what the file stores), so
  the validator's re-derivation matches exactly rather than to within rounding.

### 14.6 The grasp script as a config (`src/collection_config.py`, `configs/`)

- Units are seconds and metres; frame counts are derived. At 80 Hz the old
  frame counts were also physical durations, and moving to 240 Hz would
  otherwise have tripled every speed silently.
- The default geometry reproduces §10's verified waypoints exactly at 80 Hz
  (test), except that the orientation quaternion is now normalised (the literal
  had norm 1.000058; components move by < 1e-4).
- `record_grasp_episode(n_approach=…)` (the smoke tests' call) still runs the
  §10 script at 80 Hz with no pre-roll.
- **Inherited, flagged, not chosen:** the approach/retract speed is ~0.56 m/s,
  fast for surgical retraction. **Placeholder:** `preroll_s = 0.5` until the
  settle pilot shows how long the tissue takes to come to rest.
- Randomisation (`configs/grasp_debug_set.json`): grasp point ±2 cm in x, y;
  retraction within a 20° cone; 6–12 cm at the nominal speed; material from the
  seed. Matched-initial-state variants for Item 12 share a seed and differ only
  in retraction (test: their waypoints are identical up to the retract phase).
- Each episode is saved next to its exact config; each run appends its command
  line to `collection_log.txt`. `src/split_manifest.py` picks held-out episodes
  before windowing, hashes every file, and keeps matched variants on one side.

### 14.7 The graph builder (`src/graph_builder.py`)

- Edge i→j stores receiver-minus-sender quantities. Reference edges (8 features)
  from X, fixed per episode; spatial edges (7) from x, rebuilt every row; a pair
  in both keeps both. Tool edges (14) jaw→tissue only, chosen by unsigned surface
  distance < r_c, carrying the signed distance. Node features 21 (19 with g
  omitted; omitted columns are dropped, not zeroed). Tool nodes 25, including the
  commanded action. No absolute position anywhere.
- **The radius search is exactly symmetric by construction**: the expanded
  |a|²−2a·b+|b|² form only proposes candidates; the decision uses Σ(pⱼ−pᵢ)²,
  which is bitwise the same for (i, j) and (j, i).
- Tests show each property check failing on a broken graph first (one-way
  edges, self-loops, out-of-range ids, an absolute-position feature).

### 14.8 Item 3 on a proxy cloud (before any pilot)

`host/sampling_study.py --proxy-cloud`: 24,000 points uniform in init_cube's box —
the same distribution, not the same points. Reference graph at rest:

| n | sampler | 1.0 dx | 1.5 dx | 2.0 dx | 2.5 dx |
|---|---|---|---|---|---|
| 3000 | FPS | deg 7.7, min 1 | deg 24.5, min 7 | 54.4 | 97.5 |
| 3000 | voxel | deg 7.8, min 1 | 25.0, min 6 | 56.4 | 100.2 |
| 3000 | random | **4 components, 2 isolated** | 27.7, min 3 | 60.4 | 106.4 |
| 1000 | random | — | **2 components, 1 isolated** | 19.6 | 34.7 |

Coverage (worst particle-to-node distance): FPS 9.1 mm, voxel 11.9 mm, random
18.0 mm at n = 3,000. Measured FPS pitch ≈ 10.1 mm (the assessment's 11.5 mm
estimate was high), so 1.5/2.0/2.5 dx = 2.3/3.1/3.9 pitches. All 3,000-node
graphs at the to-do's radii are connected; even the smallest has mean degree
~25, above the 10–20 that GNS-style models use.

**The open tradeoff, not decided here.** The `--recommend` rule (fewest edges
with no isolated nodes) picks FPS at 1.5 dx = 23 mm. With the to-do's 4
message-passing blocks that is a receptive field of 94 mm, short of Phase 0's
1% influence radii at 1/240 s (130 mm soft, 160–170 mm mid/stiff; §13). 2.5 dx
gives 156 mm at ~4× the edges. More blocks at a small radius is the other route.
To be decided on the pilot, with deformed rows, and recorded here.

### 14.9 Item 4: `host/visualize_graph.py`

Images at rest / contact / grasp / strong retraction / release (settle: rest,
peak motion, end), statistics over time, and a PASS/WARN/FAIL checklist:
indexing, spatial edges re-checked against a brute-force distance matrix,
reference edges constant while stretched, fold edges, degree blow-up, material
fragmentation of the reference and combined graphs, and tool-edge coverage of
contact-flagged nodes. Shown to FAIL fragmentation at a radius below the node
spacing.

### 14.10 A collection-cost finding

The adapter logs `exposure` every row over all 24,000 particles. Profiled in the
container: **~0.47 s per row**, 70% of an episode's wall time. At 240 Hz a 1.5 s
grasp episode has ~360 rows. Options, not decided: log it every k rows (a
schema change), speed up `_coverage`, or stop logging it during debug
collection (Item 10 defers exposure metrics anyway). Measure on the Mac first.

### 14.11 Found, not fixed

`python src/synthetic_traj.py --kinds all` fails on `main` before any of this
work: one synthetic kind's substep_dt × n_substeps misses dt by 0.27% and the
writer's timebase guard refuses it. Separate bug.

### 14.12 Not verified this session

Nothing here ran Taichi, Metal, PyBullet or SurRoL. The adapter's new paths
were exercised against numpy stand-ins for the solver, arm and grasp constraint
(`tests/test_adapter_fake_solver.py`), which checks bookkeeping, not physics.
**Re-run on the Mac before collecting:** `smoke_test_mpm.py`,
`smoke_test_psm.py`, `smoke_test_grasp.py` (adapter and `grasp_constraint.py`
import changes; the correction kernel is untouched), then the pilots.

### 14.13 Verification

```
pytest tests/ -q                       (Python 3.13, numpy 2.2.6)   366 passed, 1 skipped
pytest tests/ -q                       (Python 3.10, numpy 1.23.5)  365 passed, 2 skipped
python src/trajectory_io.py            OK
python host/sampling_study.py --proxy-cloud --recommend             self-check PASS
python host/visualize_graph.py <stand-in grasp episode> --video     7/7 PASS
python host/check_surrol_interface.py  (container: no taichi/pybullet/surrol, fails cleanly)
```

### 14.14 The first schema-3.0 pilots (run on the Mac, same evening)

Smoke tests unchanged: MPM 16/16 (1 expected WARN), PSM 17/17, grasp 8/8.
Pilots at 240 Hz, all 24,000 particles recorded, float32 F, seed 0
(μ = 3758 Pa, λ = 69 kPa, ν = 0.474, ρ = 1004):

| | settle | grasp |
|---|---|---|
| rows, size, wall time | 240, 315 MB, 43 s | 241 (+0.5 s pre-roll), 319 MB, 53 s |
| validator | 0 FAIL, 3 WARN | 0 FAIL, 3 WARN; every new v3 check PASS |

The §14.10 exposure-cost worry does not hold on the Mac: a whole grasp
pilot, pre-roll included, took 53 s.

**Settling (sets `preroll_s`).** Mean particle speed 188 mm/s at 21 ms
(the drop), 9.1 mm/s at 0.5 s, 1.9 mm/s at 0.75 s, 0.42 mm/s at 1.0 s.
The 0.5 s placeholder leaves the tissue moving at ~9 mm/s when the tool
arrives.

**Integration residual (Item 7's pre-check).** The to-do's rule
x_{t+1} = x_t + Δt·v_{t+1} misses recorded positions by RMS 29.4 µm against
216.6 µm of motion per row (settle; 13.6%), 75.6 µm during the drop. The
trapezoid rule x_{t+1} = x_t + Δt·(v_t + v_{t+1})/2 gives 4.7 µm (2.2%), six
times smaller, at no extra cost (v_t is a model input already). Grasp
episode, grasped nodes excluded: 13.6 µm vs 8.6 µm against 67 µm of motion.
Decide at Item 7.

**The grasp tears the tissue.** The solver held **12** particles (not 1–3),
and the node-level latch selected exactly those 12. Retraction lifted them the
full 0.1397 m while only 32 of 24,000 particles moved more than 20 mm: the
grasp drew out a thin strand, reference edges stretched to 13.3× rest. On
release the strand recoiled at 1.6 m/s (the validator's divergence WARN, and
the 8.3 mm integration-residual maximum, are both this) and **did not rejoin**:
the material stays in two pieces to the end of the episode. In MPM, particles
pulled out of each other's kernel support stop interacting; this is numerical
fracture, not tissue mechanics. It is the jaw-smaller-than-a-grid-cell problem
recorded on 5 Oct, now measured. Retraction over 14 cm at 0.56 m/s makes it
worst, but any pull on 12 particles separates them eventually.

**A bug in my Item 4 checklist, found by this pilot.** "No material
fragmentation" tested the combined graph, which contains the fixed reference
edges and therefore can never come apart; it PASSed the torn strand. It now
treats reference edges stretched beyond 3× rest as torn and checks what is
still connected: on this pilot it FAILs (2 pieces from row 130 on). A test
reproducing the strand fails on the old check and passes on the new one.

**Item 3 on the real cloud** agrees with the proxy: FPS at 1.5 × grid_dx
(145,832 edges at rest) recommended, voxel runner-up; written to
`configs/particle_ids_3000.npy`. The receptive-field caveat of §14.8 stands.

**SurRoL check, `tissue-host` env:** gym and surrol not installed (expected:
SurRoL `main` needs numpy < 1.24, `tissue-host` has numpy 2.4.6). Step 3 PASSED:
in one process with PyBullet, the MPM exposes x, v, F, particle_ids and
rest_pos (Item 6's condition, solver side).

**Blocking the Item 6 debug set:** the grasp-scale decision. Collecting it
now would train Stage B on numerical tearing.

### 14.15 The grasp-scale decision: short slow pulls (option 1) and the Classic arm at ×5 (option 3)

**Decided 22 Oct**, from four options set out after 14.14: (1) shorter, slower
pulls; (2) a larger capture radius (a fictional "suction cup"); (3) scale the
robot ×5, as SurRoL's own tasks do, through SurRoL's Classic `Psm1`; (4) a
128³ grid (~16× cost, jaw still sub-cell). Andric chose **1 and 3**.

**What was built.**
- `host/psm_classic.py` — `ClassicPSM`: SurRoL `main` a903fa6 `Psm1` at
  `scaling=5`, behind the same interface as `host/psm.py`, which was made
  arm-agnostic (`jaw_links`, `_register_proxies`, `jaw_joint_values`).
  Commands go through SurRoL's RCM-frame path (`pose_world2rcm` → SurRoL's own
  `inverse_kinematics`, exactly what `Psm.move()` does internally) and are
  applied with SurRoL's `reset_joint` (its parallel-link and jaw mapping).
  `move()` itself is not called: it only sets motor targets for a dynamic
  physics step, which this project never takes — measured: a 5 mm `move()`
  left the arm where it was (residual 5.000 mm). The RCM is placed by
  measurement at (0.25, 0.35, 0.60) m; at zero joints the Classic tool points
  straight down, so retraction is a pure insertion-axis pull.
- **Measured Classic jaw**: a box primitive, 1.4 × 10.2 × 1.8 mm at scale 1 —
  smaller than the Si jaw. At ×5: half-extents 3.5 × 25.5 × 4.5 mm, centred
  25.5 mm along each jaw. PyBullet reports the box frame relative to the
  link's INERTIAL frame; composed with the inertial offset by measurement.
- **Jaw conventions differ, and the file says which**: Si `jaw` is the
  per-jaw angle (joints `[+jaw, −jaw]`); Classic `jaw` is SurRoL's total
  opening (joints `[−jaw/2, +jaw/2]`, 80° open). New static fields
  `robot_model` and `robot_scale`; `tool_geometry.JAW_JOINT_FACTORS`; the
  validator checks `jaw_joint` against the recorded model's convention.
- **Config `arm` block** (`model`, `scaling`, `rcm_position_m`) and three
  configs: `grasp_pilot.json` (Si, option 1), `grasp_pilot_classic.json`
  (Classic ×5 + option 1), `grasp_debug_set.json` (Classic ×5, randomised).
  All: 1.0 s pre-roll (14.14), 4 cm approach at 0.1 m/s, close 0.125 s, hold
  0.1 s, **2 cm retraction at 4 cm/s** (debug set: 1–3 cm), release 0.125 s,
  0.25 s recoil window. 361 rows at 240 Hz.
- `host/setup_tissue_sim.sh` — the simulation environment (Py 3.10,
  numpy 1.23.5, taichi 1.7.4, pybullet 3.2.6, gym 0.21 `--no-deps
  --no-build-isolation`, roboticstoolbox-python 1.4.4, SurRoL `-e`).
- `host/grasp_pilot_report.py` — the retraction-vs-tearing numbers.

**Physical reading of ×5 (not applied, recorded).** The MPM world is
unchanged (1 m domain, real gravity, placeholder materials). With the arm at
5× life size, the 300 × 300 × 50 mm slab stands for ~60 × 60 × 10 mm of
tissue, and the 2 cm pull for a 4 mm one. Mapping material, density and time
to real tissue needs a similarity scaling that is not applied; no result
should be read in real-tissue units until it is.

**Verification — real solver, real SurRoL, in the cloud container.** The
container is x86_64 Linux, where Taichi 1.7.4 and PyBullet have wheels, so the
real adapter ran on CPU (27 ms/substep, ~40× slower than Metal). Seed 0
material (μ = 3758 Pa, λ = 69 kPa), all 24,000 particles recorded:

| | Si, §10 pull (14 cm, 0.56 m/s) | Si, option 1 (2 cm, 4 cm/s) | **Classic ×5 + option 1** |
|---|---|---|---|
| particles held | 12 | 30 | **123** |
| followed > 25 / 50 / 75 % of the lift | 12 / 12 / 12 | 232 / 77 / 32 | **764 / 432 / 270** |
| max reference-edge stretch | 13.3× | 1.90× | 2.94× |
| material pieces (edges > 3× torn) | **2** | 1 | 1 |
| peak speed | 1.6 m/s, release recoil | 0.31 m/s, retract | 0.72 m/s, jaw close |

`check_surrol_interface.py`: 5/5 PASS (imports; `Psm1`; MPM x/v/F/ids; the
×5 `ClassicPSM` posed, closed and stepped on the live solver; SurRoL's URDF) —
the 5 Oct deciding check, run end to end. Validator on both new pilots: every
v3 check PASS; node-level grasp rule reproduces exactly, and at full
resolution selects exactly the solver's held set.

**What the Classic ×5 pilot still shows, open.**
- **The bite crushes the tissue between the jaws.** The highest stretches are
  all on edges touching held particles and are already 2.5× at the moment of
  grasp, before any pull; det F reaches 0.18–2.88 locally (395 particles,
  1.7%, at grasp; bulk median |J − 1| = 0.003). A sticky 7 mm-thick box
  closing on the tissue to a zero gap, and the grasp constraint zeroing the
  held particles' F_C, both act here. Not tearing; the next thing to tune
  (bite depth: the closed jaws reach ~3 cm into a 5 cm slab; closing speed:
  80° in 0.125 s puts 274 particles above 0.3 m/s).
- **Residual distortion after release** (max stretch 2.90× on the last row,
  44 edges > 2×): the bite region has not recovered within the 0.25 s window.
- **The validator needs ~6 GB on a 361-row full-resolution episode** (killed
  at the container's 8 GB cap; ran on the Mac for 241 rows). Debug-set
  episodes record 3,000 nodes and are unaffected. Chunking deferred.

**Found and fixed on the way.**
- `check_action_state_indexing` took rotation error from a dot product,
  2·atan2(√(1−d²), d), ill-conditioned at d ≈ 1: the Classic tool-down
  quaternion (qw = 0) stored as float32 read as a 3.7e-4 rad misalignment.
  Now the relative quaternion's 2·atan2(|q_v|, |q_w|) (CLAUDE.md's rule); a test
  shows both the false FAIL gone and a real 10 mrad misalignment still caught.

Tests: 370 passed, 1 skipped, on Python 3.13/numpy 2.2.6 and on the tissue-sim
stack (Python 3.10, numpy 1.23.5, real Taichi + PyBullet + SurRoL).

### 14.16 The Classic ×5 pilot on the Mac, and the cloud depends on the backend

Mac (Metal, tissue-sim env), `configs/grasp_pilot_classic.json`, seed 0: 361 rows
in 106 s. `check_surrol_interface.py` 5/5 PASS on macOS.

| | container (CPU, x64) | Mac (Metal) |
|---|---|---|
| held | 123 | 134 |
| followed > 25 / 50 / 75 % | 764 / 432 / 270 | 724 / 415 / 266 |
| max stretch / at end | 2.94 / 2.90 | 3.84 / 3.06 |
| pieces | 1 | 1 |
| peak speed | 0.72 m/s, close | 0.76 m/s, close |

Same conclusion: retraction, not tearing. **The difference has a cause worth
recording: the initial particle cloud depends on the Taichi backend.**
`rest_pos_simulated` differs between the two files (md5 56ad2d7e… Metal vs
cc2cd481… x64; first particle (0.335, 0.228, 0.099) vs (0.295, 0.308, 0.077)).
`ti.random()` is deterministic per backend, not across backends, so "the cloud
is identical in every episode" (§9.7) holds **only on one backend**.
Consequences: `configs/particle_ids_3000.npy` (chosen on the Metal cloud) is
valid only for Metal-collected episodes; collect a dataset on one backend, and
re-run `sampling_study.py --write-ids` before collecting anywhere else (a CUDA
cluster). `check_node_identity_is_shared` WARNs if rest_pos differs across
episodes, which is exactly this case.

On the Mac run the max stretch (3.84×) exceeds the 3× tear threshold on some
edges at the bite, yet the material stays one piece: the over-stretched edges
are local to the jaw-crushed region (§14.15), not a cut through the tissue.

### 14.17 19 November 2026: state check before the Item 6 debug set

Nothing was committed after §14.16; all of §14 lives as uncommitted changes on
`todo-items-1-6`. Re-checked today:

- `pytest tests/ -q`: 370 passed, 1 skipped in both `tissue-host`
  (numpy 2.x) and `tissue-sim` (Python 3.10, numpy 1.23).
- **The Mac's `tissue-sim` env is not what `host/setup_tissue_sim.sh` builds.**
  It lives under `/opt/anaconda3`, not the miniforge install that holds
  `tissue-host`; it has no `gym`, SurRoL is **not** pip-installed (no editable
  link), and `roboticstoolbox-python` is 1.5.0, not the pinned 1.4.4.
  `check_surrol_interface.py` run plainly: 1/5 PASS (`No module named
  'surrol'`). With `PYTHONPATH=~/SurRoL`: 4/5 PASS, only step 1's `gym`
  import failing — the arm itself never imports gym. §14.16's 5/5 on macOS was
  therefore run some other way than the script describes. Until the env is
  rebuilt from the script, collection with the Classic arm needs
  `PYTHONPATH=~/SurRoL`.
- Not yet collected: the Item 6 debug set (`data_debug/` does not exist). The
  open §14.15 bite-crush finding is the reason to decide before collecting:
  Stage B would otherwise learn the crushed bite as normal grasp behaviour.
- The `TODO_ASSESSMENT` document cited in §14 header and in code comments
  (`mpm_adapter.py`, `validate_dataset.py`) is not in the repo.

### 14.18 19 November 2026: tuning the bite (Classic ×5), before the debug set

Andric chose to tune the §14.15 bite before collecting. Every pilot below is
the `grasp_pilot_classic.json` script (seed 0, Metal, all 24,000 particles,
`PYTHONPATH=~/SurRoL` per §14.17) with one or two values changed; the
trial configs were throwaway, so the values are recorded here.

**Geometry, measured from the first Classic pilot.** The Classic jaw boxes
are 51 mm long (half-extent 25.5 mm) and hang straight down from the EE frame,
so the **jaw tips sit at EE z − 51 mm**. The settled slab top is at
z ≈ 0.0955 m. At the original `grasp_position_m` z = 0.12 the tips reached
**~25 mm** into the 50 mm slab. Crushing starts during the *approach*, before
the jaws close: the open jaws plough down into the surface.

`host/grasp_pilot_report.py` gained four crush columns, measured before any
pull: particles with |det F − 1| > 0.1 at the end of the approach and at the
end of the close, min det F at end of close, max reference-edge stretch at
end of close. (Row 0 background: 26–32 such particles, left over from the
drop.)

| bite depth (EE z) | close | jaw stops at | held | follow 75% | crushed approach / close | min J | stretch at close | end stretch |
|---|---|---|---|---|---|---|---|---|
| 25 mm (0.1200), §14.16 | 0.125 s | 0 | 134 | 266 | 494 / 824 | 0.22 | 2.60 | 3.06 |
| 15 mm (0.1315) | 0.125 s | 0 | 85 | 123 | 267 / 651 | 0.26 | 2.04 | 2.66 |
| 8 mm (0.1385) | 0.125 s | 0 | 66 | 86 | 192 / 510 | 0.34 | 1.94 | 2.64 |
| 15 mm | 0.5 s | 0 | 80 | 108 | 267 / 512 | 0.35 | 1.98 | 2.18 |
| **8 mm** | **0.5 s** | **0** | **64** | **71** | **192 / 444** | **0.40** | **1.97** | **2.10** |
| 8 mm | 0.5 s | 0.1 rad | 61 | 71 | 192 / 427 | 0.38 | 1.90 | 2.04 |
| 8 mm | 0.5 s | 0.2 rad | 58 | 70 | 192 / 424 | 0.38 | 1.84 | 2.43 |

All seven: one material piece throughout, lift exactly 20 mm.

**Reading.**
- Depth and closing speed both help; neither removes the crush. The best
  (8 mm, 0.5 s) roughly halves the crushed count and lifts min J from 0.22 to
  0.40, at the cost of half the held particles (134 → 64) and a quarter of the
  tissue that follows the pull at 75% (266 → 71).
- **Stopping the jaws short of closed does nothing useful** (444 → 424–427),
  so the full close stays.
- **What is left is a broad dent, not a pinch.** On the 8 mm / 0.5 s pilot,
  361 of the 444 are compressed (J < 0.9), their median distance from the jaw
  centre is 54 mm (p90 97 mm), about 3.5 grid cells, and only 52 are in the
  held set. The jaws are thinner than one 15.6 mm cell, so the solver acts on
  them through whole cells. Config knobs cannot go below that; only a finer
  grid can (option 4 of §14.15, ~16× cost).
- Validator on the 8 mm / 0.5 s pilot: 21 PASS, 2 WARN (det F in
  [0.40, 1.67]; integration residual ratio 0.169, §14.14's open Item 7 rule),
  0 FAIL.

**Decided 19 Nov (Andric): 8 mm / 0.5 s, full close**, applied to
`configs/grasp_pilot_classic.json` and `configs/grasp_debug_set.json`
(`grasp_position_m` z 0.12 → 0.1385, `close.duration_s` 0.125 → 0.5;
nothing else changed). Episodes are now 451 rows, not 361. The edited pilot
config equals the trial that produced the bold row, comment aside (checked).
Alternatives set out and not taken: keep the original bite (crush trains into
Stage B); a finer grid (only fix for the residual dent, ~16× cost, deferred
until Stage B shows it matters). **Carried as a known limit:** the broad
~3-cell dent at grasp, det F down to ~0.4 locally. The debug set's ±2 cm
grasp offset moves the grasp point over a nearly flat top, so the 8 mm depth
should hold across episodes; check `crushed_close` on the collected set.
`grasp_pilot.json` (the Si arm) is untouched.

### 14.19 20 November 2026: the Item 6 debug set, collected; one check FAILs

```
PYTHONPATH=~/SurRoL /opt/anaconda3/envs/tissue-sim/bin/python host/collect_debug_set.py \
    --out data_debug --particle-ids configs/particle_ids_3000.npy
```

Defaults: 4 settle (seeds 1000–1003, 240 rows), 8 grasp (seeds 1100–1107,
413–490 rows), 4 matched variants (seed 1200; vertical and ~12° off
vertical × 1 cm and 3 cm), all on the fixed 3,000-node set, float32 F, Metal.
About 25 min, 1.1 GB. Every episode has its `.config.json` next to it, and
each directory has its `collection_log.txt`.

**Validator.** settle: 0 FAIL (the WARNs are "grasp never active", correct for
settle, plus integration residual). grasp: **6 of 8 FAIL**; variants: **4 of 4
FAIL**, all on **one** check, `logged metrics match recomputation`, in its
lower-bound half: the 3,000-node subset's max principal stretch falls
51–76% short of the logged full-set value (limit 50%). The upper-bound half,
the one that catches a metric computed over the wrong particles, PASSes
everywhere. Every other check PASSes; dataset-wide checks PASS (comparable,
diverse deformation 31–55 mm, μ 279–11,949 Pa, one shared node set). The
collector stops at the first FAILing directory, so **no split manifest was
written** and the variants were validated by hand.

**Why it FAILs: the §10.7 open item, now on every grasp episode.** The
margin was calibrated at 8% on diffuse settling; a grasp concentrates the
peak at the held particles. The solver held 25–62 particles per episode, of
which only 3–10 are in the node set. The logged peak stretch builds up during
retraction (e.g. seed 1103: 1.5× approach, 5.4× close, 16.9× retract,
18.6× release) while the node graph's reference edges peak at 2.6–6.4× and
stay in one piece. That is consistent with the peak sitting on held particles
the subset does not record, but **it is not proven by anything in the file**:
the file has no record of *which* particle carried the logged maximum.

**The crush is mostly gravity, not the bite.** `crushed_close` varies 16–276
nodes across episodes, uncorrelated with bite depth (2.3–10.7 mm measured
against the local slab top). It tracks λ instead: episodes sampled with
λ ≈ 20–30 kPa (ν 0.35–0.46) are already compressed under their own weight
at row 0 (90–266 nodes |J−1| > 0.1); those with λ ≥ 200 kPa have zero. The
bite's own share (end-of-close minus row 0) is 10–131 nodes against the
tuned pilot's 88 on the same nodes, so 14.18's tuning holds across the set.
`materials.py` ranges are placeholders and draw μ and λ independently, so
λ/μ reaches ~2; recorded, not changed.

**Bug fixed in `grasp_pilot_report.py`.** On subset episodes it measured the
lift over an empty set of held particles, so lift read correctly only by
accident and every `followed` count was 0. It now uses the held particles
that were recorded (`held_recorded` column).

**Not yet done:** split manifest, robot+tissue video, and a decision on the
FAIL.

**§10.7's mechanism does not hold on the tuned pilot (checked 20 Nov).** On
the full-resolution 8 mm / 0.5 s pilot (`data_pilot/bite_8mm_slow`), the
logged peak (4.06× end of retract, 4.15× release) sits on **non-held**
particles ~50–60 mm from the jaw-box centre, i.e. at the bite; the held
particles never exceed 2.24× (their F is frozen by the `F_C` zeroing,
§10.7). Only 4 of 24,000 particles exceed 3×. And on that pilot the fixed
node set *does* contain the peak particle: subset max = full max. So the
debug set's 51–76% shortfall is not "the subset misses the 1–3 held
particles"; it is most likely a handful of particles at the jaw contact,
which the node set happens to catch on one grasp point and miss on others.
Unverified for the debug episodes themselves, which store no full F and no
peak-particle id.

### 14.20 20 November 2026: where the logged peak stretch is, measured

Andric asked for the two measurements set out after 14.19: record which
particle carries the logged peak, and re-run the highest-peak episode with
every particle recorded.

**New fields (schema 3.0, additive; empty in every earlier file).**
`safety_strain_id` (T,) i32, the **solver** index of the particle carrying
the logged maximum, and `safety_strain_pos` (T,3) f32, its world position.
Distance to the jaws is derived from `jaw_box_pose`, not stored.
`tissue_metrics.compute_safety_strain` now also returns `argmax`, taken from
the same stretch array as `max` (a second SVD pass would add ~39 ms per row,
~18 s per episode). The writer refuses an id without a position, an id
without `safety_strain`, or an id outside [0, n_sim). In `FIELD_DICTIONARY.md` §8.

**New check `safety peak is consistent`.** On rows where the peak particle is
a recorded node: its stored position must equal `safety_strain_pos` bit for
bit, and its stored F must reproduce `safety_strain` to 1e-3. Otherwise:
FAIL. Rows where it is not recorded are counted, not passed; SKIP if it never
is. Tests (`tests/test_schema_v3.py::TestSafetyPeak`): PASS on a consistent
episode, SKIP when the peak is off the node set and on older files, FAIL on a
subset index written as a solver index and on a position one row stale.
**The subset-index test failed against the first draft**: such an index
usually lands on an unrecorded particle, so the exact comparison never ran
and the check SKIPped. It now also FAILs when the id claims "not recorded"
but the position is exactly a recorded node's. 376 passed, 1 skipped, in
both envs.

On the real solver (seed 1103's config, 3,000 nodes): the id matches the
full-resolution re-run's argmax on 456/456 rows; 18 distinct particles carry
the peak over the episode; the peak is a recorded node on 33 rows, all
consistent. `check_logged_metrics_match_recomputation` is unchanged.

**The re-run (seed 1103, μ = 573 Pa, λ = 616 kPa, all 24,000 particles,
`data_pilot/debug1103_full`).** It reproduces the collected subset episode
to 16 µm in position and 2.2e-3 in logged peak: close, **not bitwise**. The
Metal solver is not bit-reproducible run to run (likely the order of atomic
grid adds); §9.7's "identical cloud" is about the initial state only.

| end of | logged peak | particles > 3× / > 5× | p99.9 / p99 | peak particle | recorded nodes' max |
|---|---|---|---|---|---|
| close | 5.42 | 86 / 2 | 3.87 / 2.19 | not held, 21 mm from jaw | 3.96 |
| hold | 5.47 | 94 / 6 | 3.98 / 2.23 | not held, 26 mm | 4.16 |
| retract | 16.87 | 257 / 81 | 8.31 / 3.07 | not held, 46 mm | 9.21 |
| release | 15.66 | 233 / 69 | 7.65 / 2.96 | not held, 42 mm | 8.61 |
| post-release | 15.29 | 222 / 66 | 7.23 / 2.93 | not held, 39 mm | 8.69 |

Held particles' own max stretch: 3.5–3.7 while held (F frozen, §10.7),
4.6 after release.

**What this establishes.**
- **The 14.19 addendum's guess is withdrawn.** It said the shortfall was "a
  handful of particles at the jaw contact". At this material it is a
  **zone**: 80 particles above 5×, and the peak particles were, at rest,
  5–6 mm from the nearest held particle, i.e. the held cluster's immediate
  neighbours, 20–60 mm from the jaw surface. The grasp constraint drags the
  held cluster rigidly and the material next to it takes the stretch.
- **§10.7's mechanism is wrong too**: the peak is never on held particles.
- **The logged maximum is not noise.** p99.9 is 8.3× on the same row, so a
  percentile definition would report the same story at about half the number.
  Stretches of 5–17× are far past any tissue failure stretch: the simulation
  is reporting, correctly, that the scripted grasp produces locally
  non-tissue-like deformation next to the held cluster even after 14.18.
- **The node set sees part of the zone** (9.2× on that row) and misses its
  peak. The old margin check's FAILs on 6 of 8 grasp episodes were a true
  statement that the node set cannot corroborate the peak, not a sign the
  logged number was wrong; the new check is the exact version of that test.

Note: seed 1103 (`grasp/mpm_0003`) has the debug set's highest logged peak
but was **not** one of its 6 FAILs; its per-row shortfall stays under 50%.

**Open, for Andric.** (a) What `safety_strain` should report (max, p99.9, or
max outside a zone around the held cluster), now that the max is known to be
a real zone, not a few particles. (b) Whether to keep the old margin half of
`check_logged_metrics_match_recomputation` now that the exact check exists.
(c) The bigger one: 5–17× next to the grasp means the held-cluster-dragged-
rigidly constraint is itself the source. Re-collecting the debug set picks up
the new fields but not a fix for that.

### 14.21 20 November 2026: the grasp constraint detaches the held cluster

Andric asked to investigate the constraint before re-collecting. Two full-
resolution episodes: `data_pilot/debug1103_full` (μ = 573 Pa) and
`data_pilot/bite_8mm_slow` (μ = 3758 Pa). Scripts were throwaway; method and
numbers are here.

**How the constraint works (`host/grasp_constraint.py`, read).** After every
vendored `substep()` it overwrites the held particles' `F_x`, `F_v` with the
gripper's rigid motion and zeroes their `F_C`, so their F is frozen from the
moment of capture. Their neighbours are left to the solver.

**Test 1: does stored F agree with how the particles actually moved?** For a
particle i, fit F_geo by least squares to its rest neighbours within 12 mm:
min Σ |(x_j − x_i) − F (X_j − X_i)|². On 30 random particles stored and
geometric stretch agree at every phase (median 1.08 vs 1.08; 1.11 vs 1.10),
so the method is sound. On the 30 most-stretched particles:

| | stored stretch, median / max | geometric, median / max | J stored / J geometric (median) |
|---|---|---|---|
| μ 573, end of retract | 9.05 / 16.87 | 4.18 / 6.98 | **0.65 / 7.50** |
| μ 573, end of episode | 8.59 / 15.29 | 3.00 / 4.71 | 0.86 / 4.26 |
| μ 3758, end of retract | 2.15 / 4.06 | 2.12 / 3.79 | 1.04 / 1.49 |

At μ 573 the neighbourhood's volume has grown 7.5× while the particles' own F
says compressed: the material has come apart, and F no longer describes the
motion. At μ 3758 the two mostly agree, with a milder volume opening.

**Test 2: does the surrounding tissue come with the grasp?** "Shell" = the
non-held particles within 8 mm of a held one at rest (58 and 75 particles;
rest spacing 3.2 mm; grid dx 15.6 mm).

| | μ 573 | μ 3758 |
|---|---|---|
| held → nearest shell gap, end of close | 7.4 mm | 6.6 mm |
| same, end of hold (no pull yet) | 11.3 mm | 6.7 mm |
| same, end of retract | 36.6 mm | 15.1 mm |
| shell lift when held has risen 20 mm | 4.0 mm | 9.5 mm |
| non-held particles within one cell of the cluster, from | end of hold on: **0** | mid-retract on: **0** |
| after release | held cluster coasts on to 37 mm; never rejoins | rejoins partly (gap 12.6 mm) |

**Reading.** The held set is 25–64 particles, about one to three grid cells
of material. The jaws closing push the surrounding tissue away from it (the
gap doubles before any pull at μ 573), and the pull then lifts the cluster
out of its neighbours' interpolation reach (~1.5 cells, ~23 mm). Past that
the cluster and the tissue no longer interact: numerical fracture, the
§14.14 tearing at a smaller scale. The 5–17× peak is the torn edge. It is
driven by the constraint, not by tissue mechanics: (i) the constraint moves
the cluster by teleport, so the tissue is dragged only through the grid nodes
it shares with the cluster, by a small mass; (ii) the cluster's F is frozen,
so it develops no tension as it is pulled; (iii) the cluster is barely one
grid cell, which the grid cannot attach to the rest of the tissue.

**Options, not chosen.**
- **A. Grasp as a grid boundary condition.** The standard MPM approach.
  A host-owned substep kernel that calls the vendored `@ti.func`s in order
  (`P2G()`, `Boundary()`, then a grasp BC, then `G2P()`) and sets every grid
  node in the held particles' stencil to the gripper's rigid velocity.
  No teleport, no `F_C` zeroing, so every particle's F evolves from the same
  grid velocity it moves with. No vendored edit is needed because the three
  are `ti.func`s, not kernels. The grip footprint becomes about 1.5 cells
  around the held set, which is the grid's resolution, stated honestly.
  Costs: a new kernel; proof that with no grasp it is bitwise the vendored
  `substep()`; re-running the three smoke tests. Unknown: whether it drags
  enough tissue at this resolution. Boundary()'s own sticky halo is already
  a grid BC near the jaws, and the shell still did not follow, so this must
  be measured, not assumed.
- **B. A larger particle-held region** (everything between the jaws plus a
  cell of margin). It keeps the teleport and the frozen F, so it moves the
  torn edge outward without removing it. Close to §14.15's rejected option 2.
- **C. A 128³ grid.** Halves the cell, so a cluster spans more cells; it
  does not fix the teleport or the frozen F; ~16× cost.
- **D. Keep the constraint, cap the pull** below detachment (a few mm at
  μ 573). Stage B would see almost no retraction.

Recommendation: prototype A on these two configs and judge it by the same
three numbers (gap, shell lift, stored vs geometric F) before deciding.

### 14.22 20 November 2026: option A prototyped, the grasp as a grid boundary condition

Andric chose to prototype option A of 14.21.

**What was built.**
- `host/grasp_grid_bc.py`, `GridGraspConstraint`: a subclass of
  `GraspConstraint` with the same membership (capture rule, release rule,
  `full_ids()`, CONTACT_GRASP). It owns the whole substep in one kernel of its
  own: clear grid → `P2G()` → `Boundary()` → set every grid node in the held
  particles' 3×3×3 stencil that has mass to v_g + ω_g × (x_node − c) →
  `G2P()`. There is no teleport and no `F_C` zeroing; `apply()` is a no-op.
  `third_party/` is untouched: the three steps are `@ti.func`s, callable from
  any kernel. Two Taichi constraints shaped the code. The file must not use
  `from __future__ import annotations`, because the kernel has annotated
  scalar arguments. And every loop stays at the kernel's top level, with the
  "is a grasp active" test inside each loop body, because Taichi parallelises
  only top-level loops.
- Config key `grasp_model`: `"teleport"` (default, unchanged) or `"grid_bc"`.
  It is recorded in each episode's sibling `.config.json`; **the .npz itself
  does not say which was used**.
- `MPMRecorder.advance()` calls the grasp's own `substep()` when it has one.
- `host/smoke_test_grid_grasp.py`, 4 checks. With nothing held, the new path
  differs from the vendored substep by 8.9e-8 m over 60 substeps, against the
  vendored substep's own repeat spread of 6.0e-8 m (Metal is not
  bit-reproducible, 14.20). Held particles follow the rigid path to 0.001 mm
  over a 40 mm lift. Their F changes by ≤ 1e-7 under pure translation, with
  no forced zeroing. And the tissue is dragged: 26.8 mm of rise 2–3 cells
  out and 14.1 mm 3–4 cells out, for a 40 mm lift. **The first draft of the
  drag check measured a ring 12–20 mm out**, which lies inside the BC
  footprint, so it passed trivially (39.8 of 40 mm). The rings now start
  beyond 2 cells.
- `host/grasp_pilot_report.py` gained the 14.21 diagnostics as columns
  (`gap_mm`, `shell_follow`, `top_stretch_stored/fit`, `top_J_stored/fit`;
  full-resolution episodes only). They reproduce 14.21's teleport numbers
  exactly.

**Result.** Same configs as 14.21, all 24,000 particles, Metal.

| | 1103 teleport | **1103 grid_bc** | pilot teleport | **pilot grid_bc** |
|---|---|---|---|---|
| held → shell gap, end of retract | 36.5 mm | **7.4 mm** | 15.9 mm | **6.6 mm** |
| shell rise / held rise | 0.14 | **1.00** | 0.36 | **1.00** |
| particles following > 75% of the pull | 210 | **741** | 71 | **982** |
| top-30 stretch, stored / fitted | 9.05 / 4.18 | 3.31 / 2.65 | 2.15 / 2.12 | 2.03 / 1.70 |
| top-30 J, stored / fitted | 0.65 / **7.50** | 1.03 / **1.07** | 1.04 / 1.49 | 0.93 / 0.97 |
| logged peak, retract / release | 16.87 / 18.58 | **3.70 / 4.01** | 4.06 / 4.15 | **2.33 / 2.80** |
| node-graph edge stretch, max / end | 6.36 / 4.58 | 2.89 / 1.79 | 3.17 / 2.10 | 2.15 / 1.42 |
| last-row det F range | 0.19–2.56 | 0.91–1.14 | 0.69–1.39 | 0.78–1.29 |
| validator | | 0 FAIL, 2 WARN | | 0 FAIL, 2 WARN |
| wall time | 124 s | 164 s (+32%) | | |

The tissue now comes with the grasp, F agrees with the motion (fitted J ≈ 1,
not 7.5), and the peak no longer grows during retraction: it is set by the
close (3.7×) and stays there. The remaining WARNs are the known ones (det F
range from the bite, 14.18; integration residual, Item 7). Smoke tests after
the adapter change: MPM 16/16 (1 expected WARN), PSM 17/17, grasp 8/8, grid
grasp 4/4. Unit tests 376 passed, 1 skipped, both envs.

**Still open.**
- Stored stretch of the most-stretched particles still exceeds the fitted one
  by ~20% (3.31 vs 2.65; 2.03 vs 1.70). The fit averages over 12 mm, so some
  difference is expected; random particles agree exactly. Not investigated.
- The grip footprint is the held particles' stencil, ~1.5 cells (~23 mm)
  around them: larger than the jaw. It is the finest grip this grid can
  impose; a model trained on it learns a 23 mm-wide grasp.
- Peak speed now occurs at release (0.68 and 0.88 m/s): elastic recoil once
  the BC stops. Not examined.
- **Not applied to any config.** `grasp_debug_set.json` and
  `grasp_pilot_classic.json` still default to `teleport`; `data_debug/` was
  collected with it.

### 14.23 20 November 2026: the debug set re-collected with the grid-BC grasp

Andric chose to switch and re-collect. `configs/grasp_debug_set.json` now has
`"grasp_model": "grid_bc"` (`grasp_pilot_classic.json` is unchanged, still
`teleport`). The teleport set from 14.19 is kept as evidence in
`data_pilot/debug_teleport/` (gitignored). Same command, seeds and node set
as 14.19; the dry run confirmed the matched variants inherit `grid_bc`.

**Validator.** settle 4/4: 0 FAIL. variants 4/4: 0 FAIL. grasp: **2 of 8
FAIL** (14.19: 6 of 8), both on the old margin half of
`logged metrics match recomputation` (57% short of the logged peak, limit
50%). The collector stops at a FAIL directory, so **no split manifest was
written again**. Every other check PASSes. The new exact check
`safety peak is consistent` PASSes on all 12 grasp and variant episodes: the
peak particle is a recorded node on 4–230 rows per episode, consistent every
time.

**Where the peak is now** (from `safety_strain_id`, the first collection to
have it):

| | teleport (14.19) | grid_bc |
|---|---|---|
| logged peak, range over 12 episodes | 4.3–18.6× | **2.55–5.07×** |
| peak particle | the held cluster's neighbours (14.20) | a **held** particle in 10 of 12, mostly at release; 0001 and 0004: a non-held particle at the close |
| peak displacement range across grasp episodes | 31–55 mm | 25–32 mm |

The two FAILs: **0001** (μ 11,949 Pa) peaks at 2.55× during the close on a
non-held particle 23 mm (rest) from the held set, 24 mm from the jaw: the
bite (14.18), not the pull. **0006** (μ 7,379 Pa) peaks at 5.07× at release
on a held particle 12 mm from the jaw. In both the peak particle is usually
not a recorded node, which is exactly what the margin check is guessing at.
The exact check answers it directly.

**Not yet investigated:** why held particles reach 3–5× at release. With
the grid BC their F evolves (it was frozen under the teleport), so release
is the first time they are free to spring back; 14.22 also saw the peak
speed move to release.

**Open, for Andric (14.20 (b), now concrete).** The margin half of the old
check was a stand-in for "is the logged peak on a recorded node". The exact
check now answers that, and it passes everywhere. Options: (1) downgrade the
margin half to WARN when `safety_strain_id` is present, and keep it as a
FAIL for older files; (2) keep it as a FAIL and drop 0001 and 0006 from the
set; (3) keep it and widen the margin. (1) is the recommendation: (2)
throws away valid data, and (3) is a number with no failure story.

### 14.24 21 November 2026: the margin check downgraded; the Item 6 set validates

Andric chose option (1) of 14.23.

**The change (`check_logged_metrics_match_recomputation`).** The
subset-margin half (the subset's peak stretch falling more than 50% short of
the logged full-set value) is now a **WARN when the file records
`safety_strain_id`**, with the reason in the message. For every older file
it is still a FAIL. Untouched: the impossible direction (subset peak *above*
the logged value) is a FAIL with or without the id, and so are the exposure
bounds. Tests in `tests/test_schema_v3.py::TestMarginDowngrade`: without the
id the margin FAILs; with it, it WARNs; with the id, a logged value below the
subset maximum still FAILs. 379 passed, 1 skipped, both envs.

`host/collect_debug_set.py --validate-only` skips simulation, then validates
and writes the manifest. I used it to resume after the 14.23 stop; the
variant configs it rewrites were byte-identical (md5 checked).

**The Item 6 debug set, validated.**

| directory | episodes | PASS / WARN / FAIL (all checks) |
|---|---|---|
| settle | 4 | 63 / 12 / 0 |
| grasp | 8 | 172 / 16 / 0 (0001 and 0006: margin WARN, as above) |
| variants | 4 | 87 / 9 / 0 |

The remaining WARNs are known: integration residual (Item 7's rule), det F
range from the bite, "grasp never active" on settle, and "material barely
varies" on the variants (one shared seed, by design for Item 12).

**`data_debug/split_manifest.json`** picks held-out episodes before any
windowing and hashes every file. Held out: `settle/mpm_0002` and
`grasp/mpm_0006`. The 4 variants are one group on the train side; the
manifest notes that nothing of that task is held out, so Item 12 results on
them are a debug result, not generalisation.

**Video (Item 6's last deliverable).** `runs/video/debug_grasp_0003_gridbc.gif`
(gitignored, regenerable): `python host/render_video.py
data_debug/grasp/mpm_0003.npz --out runs/video/debug_grasp_0003_gridbc.mp4`,
229 frames. It is a **GIF, not MP4**, 28 MB: neither env has ffmpeg, so the
script falls back to a GIF. `pip install imageio-ffmpeg` or `brew install
ffmpeg` fixes it; not installed. Checked by eye at t = 1.04 s and 1.48 s
(retract): the slab stays in one piece, tissue tents up under the jaws, and
spatial-only edges collect where it folds near the grip.

Item 6's deliverables now exist: the collection command and configs, the
validated files, the split manifest, and the video. Nothing is committed.
