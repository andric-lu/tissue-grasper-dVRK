# tissue-dynamics

Learned dynamics models for soft tissue under robotic manipulation, starting with
tissue retraction. Read `DECISION_LOG.md` before changing any parameter — it
records why each one is what it is. Section references below are to that file.

## Working with Andric

- **Not a computer science major.** Explain what a thing does and why it exists,
  not just what to type. Assume no familiarity with build systems, packaging, or
  git internals; assume real fluency in the physics and the research. He should be able to explain the CS decisions and implementations in precise detail.
- **Show the reasoning trace.** He should be able to reproduce the work himself
  afterwards. Name the commands you ran and what their output meant.
- **Do not jump to the recommended implementation.** Give the alternatives and
  the tradeoff, then the recommendation. A decision he can't reconstruct in three
  months is a decision he can't defend.

## Where code runs — check the imports first

Two environments. A script's imports decide which one it belongs to, and running
it in the wrong one fails on the first import line.

| Imports | Runs on | How |
|---|---|---|
| `pybullet`, `surrol` | container | `docker compose run --rm surrol python <script>` |
| `torch`, `matplotlib`, `taichi` | host (macOS) | `conda activate tissue-host` |
| numpy only | either | everything in `src/` |

`host/` and `container/` are not organisational tidiness — they name the
interpreter. `src/` is the shared layer and must import cleanly on both sides,
which means numpy only, and must work under numpy 1.23 (container pin) and 2.x
(host).

Apple's GPU is reachable only through Metal, a macOS API. No container can use
it. **Taichi has no Linux ARM64 wheel in any version**, so MPM work is
host-only — not a preference, the wheel does not exist (§9.3).

## Commands

```bash
# host
conda activate tissue-host
pytest tests/ -q                                    # 279 tests
python host/validate_dataset.py --data data/        # exit != 0 on FAIL
python host/visualize_trajectory.py data/x.npz      # --stride 3 for long episodes
python src/synthetic_traj.py --out data_synth/ --kinds all
python host/smoke_test_mpm.py                       # 16 checks, §9.4; ~10s first run
python host/mpm_adapter.py --out data_mpm --episodes 2 --steps 100   # §9.5
python host/substep_study.py --frames 60            # §9.6; ~3 min, both materials
python host/energy_audit.py --selftest              # §9.7 D1 gate, ~40s
python host/energy_audit.py --frames 60             # §9.7; ~5 min, 3 cells

# container
docker compose run --rm surrol python container/collect_retraction.py --episodes 20
docker compose run --rm surrol python container/verify_container.py
```

Run `validate_dataset.py` before training on any new data. It exists because a
model cannot tell you its training data was wrong — it fits whatever it is given
and fails later in a way that looks like an architecture problem.

## Conventions that have already caused bugs

Each of these cost real time. Do not quietly change one.

- **Quaternions are scalar-last `[qx, qy, qz, qw]`.** A file mixing conventions
  is unrecoverable after the fact.
- **Extract angles with `2*atan2(|q_v|, q_w)`, never `2*acos(q_w)`.**
  Normalisation can leave `q_w` a few ulps above 1.0 and `acos` returns NaN,
  which propagates through a whole trajectory and surfaces much later.
- **Rotation deltas are axis-angle; absolute poses are quaternions.** Quaternions
  double-cover SO(3), so a network regressing one sees two correct answers.
- **Sample μ and λ directly, log-uniformly. Never sample (E, ν) uniformly.**
  λ is singular at ν = 0.5 and tissue sits at ν ≈ 0.49 (§8.3).
- **In the schema, an empty array means "not recorded"; all-False/zero means
  "recorded, and the value is zero."** Never write a zero default for something
  you did not measure, and never write an empty array for something you know.
- **`safety_strain` is a stretch ratio, not a strain.** Undeformed is 1.0.
- **Report max principal stretch; optimise the invariants.** Eigenvalue gradients
  diverge where eigenvalues coincide, and at rest `F = I` they all coincide
  exactly. Loss paths use `J` and `I₁`.
- **Predict position deltas, never absolute positions.**
- **Split train/validation by episode, never by random timestep.** Consecutive
  frames are near-identical; a random split leaks duplicates and produces a
  beautiful meaningless curve.
- **Always report the constant-velocity baseline.** A model that doesn't clearly
  beat it has learned nothing, however small the loss.
- **Code should measure the world it operates in.** Hardcoded coordinates break
  silently when the asset, scale or resolution changes (§3.4).
- **Research code is pinned to its publication era.** "Install the latest
  version" is usually wrong here. One package manager per environment for
  anything compiled (§3.2).

## Validation discipline

- Every check carries a `# WHY:` comment naming the specific failure it catches.
  A check without a failure story is one nobody will trust when it goes red.
- **Demonstrate a check against broken data, not just good data.** A check only
  shown passing has not been shown to work. `check_boundary_is_held` was verified
  by pointing it at the buggy episodes and confirming the exact failure (§9.2).
- **`SKIP` is not a soft fail.** v1 episodes legitimately lack v2 fields. A check
  that cannot apply says SKIP and says why; it never quietly passes.
- **A schema field with no check on it will drift.** That is how three synthetic
  episodes came to assert clamps they did not honour.
- All of this checks *internal consistency only*. It cannot establish that the
  simulation resembles real tissue. "Self-consistent" and "predicts reality" are
  different claims.

## Decision log discipline

`DECISION_LOG.md` is the substantive record; `SESSION_TRANSCRIPT.md` is the
chronological one. Both are in the repo and must stay there.

- **Write the log the same day.** §1–7 were written during the work; §8 was
  reconstructed sixteen days later from source, and the difference in effort was
  large. Anything not written down evaporates.
- **A code comment is not where a platform decision goes.** The Taichi ARM Linux
  gap was recorded in a comment in `environment.yml` and still had to be
  rediscovered from PyPI a week later, because nobody looks there.
- Number sections. They get cited later.

## Current state

- **§4 is open and blocking for PyBullet.** `DT = 1/1000` in
  `collect_retraction.py` is unvalidated; `timestep_study.py` has never run. All
  20 episodes in `data/` were collected at that timestep.
- **`container/validate_physics.py` has never been executed.** It targets the
  session-1 v1 world; check its imports still resolve before trusting a FAIL.
- **Material ranges in `materials.py` are placeholders.** Liver, bowel and fat
  differ by more than the width of those ranges. No result should be claimed
  from a model trained on them.
- **`data/` is schema v1** — no `F`, no material params, no `boundary_mask`.
  Those checks SKIP on it, correctly.
- **`data_mpm/` is gitignored and regenerable** — two v2.1 episodes from
  `mpm_adapter.py`, validating clean. `data_mpm/stale/` keeps the two 17 August
  files as §9.5's evidence: they predate both the P-wave substep and the density
  fix, so they ran at ρ = 1000 whatever they record. Evidence, never a baseline.
- **`data_mpm_grasp/` is gitignored and regenerable** — `--task grasp`
  episodes (§10), a real PSM approach/close/retract. Two episodes validate
  clean (0 FAIL); `grasp is consistent` WARNs honestly on both, by design —
  see "The PSM" below before treating that WARN as a bug.
- **The MLP is a placeholder.** MeshGraphNets is the target; the schema already
  carries `tissue_faces` / `tissue_tets`.

### MPM — vendored and smoke-tested (§9.4)

`third_party/MPM/` holds SurRoL's `Dev` MPM, four files, byte-identical to
upstream at `cb797f36`. `third_party/PROVENANCE.md` has the SHA and checksums;
update its "Local modifications" section the moment you edit one of them.

`host/smoke_test_mpm.py` passes 15/15. All three of §9.3's open questions are
closed. What it established that you need to know:

- **Runs on Metal, 0.64 ms/substep** (24k particles, 64³ grid, ~0.8× realtime).
  Metal is **8.1×** faster than the CPU here — verified by forcing the fallback
  with `TI_ARCH=arm64`, which makes the backend check FAIL as it should. Note
  `TI_ARCH=cpu` is not a valid name and aborts; the CPU arch is `arm64`.
- The *first* substep on a fresh machine takes ~10 s compiling kernels, then
  ~0.5 s once `~/.cache/taichi` is warm. Not a hang. On this project a long
  silence is usually a compiler — same as PyBullet's build.
- **`mpm3d.py` calls `ti.init()` at module level.** Importing it *is* the
  backend decision; nothing downstream can pick an arch.
- **`third_party/` goes on `sys.path`, not `third_party/MPM/`.** No
  `__init__.py` — PEP 420 namespace package, which keeps the tree unmodified.
- **`substep()` needs `SDF` and `collision_mask` bound by hand** if you drive it
  without a PyBullet scene; `step()` normally rebinds them from `sdf.py`. Fill
  the SDF large and positive and the collision branch never fires.
- **PyBullet does not build with a plain `pip install`** on this macOS SDK. Needs
  `export CFLAGS="-Dfdopen=fdopen" CXXFLAGS="-Dfdopen=fdopen"` first, *before*
  `conda env create` — a YAML file cannot set it. Reasoning in §9.4.

**`set_parameters()` takes (E, ν), not (μ, λ) — §9.3 got this wrong.** It runs
`la = E*nu/((1+nu)(1-2nu))` internally, the singular conversion §8.3 exists to
avoid, defaulting to `s_nu=0.2`. Measured round-trip error is small (2.7e-4) but
pointless: **the adapter writes `mpm3d.mu[None]` and `mpm3d.la[None]` directly
and never calls `set_parameters()`.**

### The adapter — done and verified (§9.5)

`host/mpm_adapter.py` drives the vendored MPM and writes v2.1 episodes. The
mapping is what §9.4 verified: `F_x → tissue_pos`, `F_v → tissue_vel`,
`F → tissue_F`, no reshape.

```bash
python host/mpm_adapter.py --out data_mpm --episodes 2 --steps 100
python host/validate_dataset.py --data data_mpm/     # exit 0, no FAIL
```

11.5 MB per 100-step episode (3,000 of 24,000 particles, `delta16`), ~16-21 s
each. What you need to know before touching it:

- **`--episodes N` launches one child process per episode.** Taichi bakes `dt`
  AND `p_mass` into kernels at compile time, so two materials cannot share an
  interpreter. Not `os.fork()` — the child would inherit a Metal context whose
  owning threads did not survive the fork. `--index` marks a child and is what
  makes recursion impossible; a process holding it never dispatches.
- **`self.n_substeps` is the single source of truth for the timebase**, never
  `mpm3d.steps`. The reverse shipped once: every frame advanced 2.95 ms while
  claiming 12.5 ms, and nothing on disk contradicted anything. `dt` must equal
  `substep_dt * n_substeps` — asserted in the adapter, the writer and the
  validator.
- **`p_mass = p_vol * rho` is set before the first substep.** Until it was, the
  sampled density was decoration: the solver ran every episode at ρ = 1000
  while the file recorded something else.
- **The episode seed drives numpy only** — material and subset. `mpm3d.py` calls
  `ti.init()` with no `random_seed`, so the initial particle cloud is identical
  in every episode. Do not read "seed=N" as an independent initial condition.
  That holds *per process only* — `init_cube()` advances the RNG, so calling it
  twice in one interpreter gives two different clouds (§9.7).
- **Metrics are computed over all 24,000 particles before subsampling**, because
  `safety_strain` is a maximum and a maximum over a subset is biased low.
- **Subset metrics are bounded, not equated.** Exposure over a subset can only
  be ≥ the logged full-set value (removing particles removes occluders); peak
  stretch can only be ≤ it. Demanding equality rejected correct data.

### The substep is a STABILITY bound (§9.6), and refinement DOES converge (§9.7)

`host/substep_study.py` swept `n_substeps` over a factor of 32 at both collected
materials. `host/energy_audit.py` then re-ran the question against a closed
energy budget and overturned half of what §9.6 concluded.

- **`safety = 0.3` is kept, and is a good stability bound.** The only rows that
  diverged were above `safety ≈ 1.2`, so 0.3 carries a factor-of-four margin.
  Unaffected by §9.7.
- **§9.6's "nothing converges and nothing can" is WITHDRAWN.** It was inferred
  from kinetic energy alone, under gravity and floor contact, at a time by which
  the body has settled — so the two KE numbers compared are both residual jitter
  near zero. Under the full budget, in that same configuration, dissipation
  *falls* with refinement (2.1× across a 16× ladder at the stiff material) and
  successive positions converge at observed order ≈ 0.3. The direct refutation
  is the zero-stiffness test, where nothing but the transfers can act: total
  energy loss is constant to **0.2% across a factor of 64 in `dt`**. Refining
  `dt` at fixed `dx` is therefore a legitimate instrument here — just a slow and
  expensive one.
- **KE alone is not a proxy for dissipation, in either direction.** In §9.6's
  configuration KE at T falls with refinement while the total *rises*; in the
  clean no-contact cell KE rises while the total falls. Report the budget
  (`src/energy_ledger.py`), never one term.
- **`safety_strain` is still not a material property under this solver.** This
  survives §9.7. Dissipation does vary with substep count — weakly, and in the
  opposite direction from what §9.6 said — and the collector picks substeps
  *per material*, so `data_mpm/`'s two episodes are **not comparable to each
  other**. Carry this into any training that conditions on material.
- **The `*= 0.1` floor band removes 89–93% of the energy** in a cell that arms
  it alone (§9.7 cell 3), and does so essentially independently of `dt` — it is
  so aggressive that it saturates into a hard clamp at every substep tested. Not
  a convergence problem; an enormous arbitrary sink under every `data_mpm/`
  episode.

A coupled `dx`/`dt` study is still unattempted and is still the right instrument
for a *spatial* convergence claim. §4's PyBullet study is still unrun and now
matters more, not less: the excuse that convergence studies do not apply to this
class of solver has been removed.

**Three constants are baked into Taichi kernels at compile time, not two:**
`dt`, `p_mass` and **`gravity`** (`Boundary()` reads it as a Python global,
`mpm3d.py:215`). One process per (dt, p_mass, gravity). Also: **`init_cube()` is
not idempotent** — it draws from `ti.random()` and the RNG advances, so a second
call in one interpreter lays out a *different* cloud. "Identical cloud in every
episode" holds only because every episode is a fresh process.

### The PSM — landed, `--task grasp` episodes verified (§10)

`psm_Si_model/psm_si_surrol.urdf` (dVRK Si variant, 13 links) is driven
kinematically through `host/psm.py`'s `PSM` class and wired into
`host/mpm_adapter.py`'s `record_grasp_episode()` (`--task grasp`, `--task
settle` stays the unchanged passive default). `host/smoke_test_psm.py`
passes 19/19. What it established that you need to know:

- **A `co_obj` slot with `link_id == -1` is corruption, not a no-op.**
  `sdf.py` takes the "needle" precomputed-SDF path for it, and nothing in
  this repo ever calls `sdf.init_static_sdf(...)` — that field sits at
  Taichi's zero default, a phantom zero-distance surface across the whole
  domain. Every collider here is a one-link proxy body (`link_id = 0`),
  never a bare base. `smoke_test_psm.py` asserts this directly.
- **`sdf.py` never reads a collision shape's local frame offset**
  (`collisionFramePosition`) — only `getCollisionShapeData(...)[0][3]`, the
  extents. A box's world pose for the SDF is entirely whatever
  `i_rot_list`/`i_pos_list` the caller supplies. `host/psm.py` composes
  `T_world_box = T_world_jaw . T_jaw_box` by hand every frame; it does not
  delegate the offset to PyBullet's own collision-frame machinery.
- **The two jaws get independent collision velocity**, finite-differenced
  from each proxy's own consecutive `T_world_box`, never derived from the
  shared commanded EE-pose delta — jaw closure moves the two jaws
  differently from each other and from the EE frame.
- **`p.calculateInverseKinematics` returns one value per *movable* joint**
  (fixed joints skipped), in raw-joint-index order — not one value per raw
  joint index. `PSM` resolves a movable-index -> raw-index map once, at
  construction. `set_ee_pose()` also validates the achieved pose against the
  commanded one and raises past 2 mm / 0.02 rad residual.
- **`mpm3d.init_pos()` is owned by `PSM.__init__`, once.**
  `MPMRecorder._init_solver()` never calls it, robot or not — it populates
  `sdf.position`, which only the collision path needs.
- **Taichi's Metal backend has no f64 primitive.** Every array fed to
  `switch_reference_frame_and_update_sdf` must be explicit `float32`;
  `numpy`'s default float64 fails at the first `.from_numpy()` call with no
  warning beforehand. Found by running it, not predicted.
- **This solver has no persistent grasp.** `Boundary()`'s collision branch is
  an unconditional, non-persistent zero-slip constraint — mechanically
  `CONTACT_STICK`, never the schema's `CONTACT_GRASP` ("jaws closed, tissue
  kinematically attached"). Every grasp episode emits only
  `NONE`/`TOUCH`/`STICK`; `grasp_active` is always `False`,
  `grasp_node_ids` always empty; `check_grasp_is_consistent` keeps WARNing,
  honestly, until real attachment is built. **Do not label proximity as
  `CONTACT_GRASP` to make that WARN go away** — a first design draft tried
  exactly that and was rejected specifically for it.
- **Jaw half-extents measured at ~6–9 mm**, smaller than one grid cell
  (`dx=15.6 mm`). Collection uses `threshold=0.02`, not the vendored
  default `0.05`, which would extend the contact halo roughly two grid
  cells beyond the jaw's true surface.
- **Base placement is empirical** (`psm.DEFAULT_BASE_POSITION`/
  `ORIENTATION`), picked via `workspace_aabb()`, not derived — the chain has
  too many compounded rotated offsets to hand-derive reliably.
- **Trocar-pivot (RCM-constrained) motion is not implemented.** IK is
  free-space 6-DOF; the URDF's `RCM` link is visual-only.

Material ranges in `materials.py` are still placeholders, so no result should be
claimed from a model trained on this data yet.
