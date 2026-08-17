# BM2 — Session Transcript, 1 August 2026

**Topic:** Environment setup for soft tissue dynamics modelling; first working
data pipeline.

> **Note on fidelity.** This is a structured reconstruction of the session, not a
> byte-exact export — the Cowork session's raw transcript was not retrievable
> programmatically. Content, decisions, code, and error messages are accurate;
> exact wording of conversational prose is paraphrased. The
> substantive record is in `DECISION_LOG.md`; this file preserves the
> chronological path, including the dead ends, which the decision log compresses
> away.

> **Note added 17 August 2026.** This file covers session 1 only. **No transcript
> exists for session 2 (13–16 August)** — the work from those days was
> reconstructed from the source files and is recorded in `DECISION_LOG.md` §8.
> If you are reading this while a session is still live, writing the transcript
> now is far cheaper than reconstructing it later.

---

## Exchange 1 — Opening question

**Andric:** First task is a dynamics model that predicts soft tissue responses
during deformation and external disturbance, starting with basic tissue
retraction. Plan: set up a dVRK arm and tissue in SurRoL, then eventually refine
and run on Isaac Sim. Beginning on a new M3 Max MacBook Pro, 128 GB. What's the
best environment setup, or would external resources be needed?

**Claude:** Researched SurRoL requirements, Isaac Sim system requirements,
PyBullet on Apple Silicon, and the SurRoL soft-body extension.

Findings:

- SurRoL is pure Python + PyBullet; runs natively on arm64. Three friction
  points: pinned Python 3.7 (poor arm64 support), `tensorflow-gpu==1.14`
  (doesn't exist for arm64, but is only needed for the OpenAI Baselines agents,
  not the simulator), and flaky PyBullet OpenGL GUI on Apple Silicon.
- **Isaac Sim: hard no.** x86_64 Ubuntu/Windows + NVIDIA RTX only. No macOS
  build, no ARM. Requires external hardware — cloud container with WebRTC
  streaming, or a Linux workstation.
- Soft tissue is the real decision point: PyBullet deformables are CPU-only and
  crude; the MPM/Taichi approach (arXiv:2402.01181) is the better target and
  Taichi has a Metal backend for Apple Silicon.

Recommended phasing: Phase 0 rigid-body SurRoL locally → Phase 1 soft body →
Phase 2 Isaac Sim on external hardware. Emphasised decoupling the dynamics model
from the simulator from day one.

---

## Exchange 2 — VM versus native

**Andric:** On my old laptop I ran SurRoL in a local Ubuntu VM. Would that be
preferable to macOS?

**Claude:** The tradeoff is sharp and it's about GPU access.

- **VM costs:** Apple Silicon virtualization has no GPU passthrough. Inside a
  Linux guest there is no Metal, so Taichi drops to `ti.cpu` and PyTorch loses
  MPS — exactly the two things that matter for Phase 1.
- **VM buys:** ROS. Genuinely painful on macOS, trivial on Ubuntu. Plus
  dependency pinning and matching the eventual Isaac Sim target OS.
- **VM does not fix:** it's still arm64. x86-only wheels fail identically.

Recommended **container rather than VM** for the Linux side — same Linux
userspace, reproducible pinning, no second kernel — with GPU work staying on the
host. CPU cost of a VM is minor (~5–10%); the GPU cost is total.

**Andric:** Let's go with host-native + container. Little to no CS experience —
walk me through every step, explaining what each instruction does, so I can do
it unassisted in future.

---

## Exchange 3 — The full build

Clarified: fresh machine, nothing installed; full guide plus config files.

Verified version constraints before writing anything: Taichi 1.7.4 supports
Python 3.9–3.13 with macOS arm64 wheels; PyBullet has no macOS arm64 wheel
(compiles from source); SurRoL's `setup.py` confirmed TensorFlow is in
`extras_require`, not `install_requires`.

Produced a 12-file project:

- `SETUP_GUIDE.md` — terminal basics → Xcode CLT → Homebrew → git → Miniforge →
  host env → Docker → container → end-to-end run, with a checkpoint per part
- `docker/Dockerfile`, `docker-compose.yml`, `host/environment.yml`
- `src/trajectory_io.py` — the sim-agnostic data contract
- `host/verify_host.py`, `container/verify_container.py`
- `host/train_dynamics.py`, `container/collect_retraction.py`
- `README.md`, `CHEATSHEET.md`, `.gitignore`

Verified in sandbox: trajectory format round-trip, dataset-builder shapes, and
that the constant-velocity baseline is exact on synthetic constant-velocity data.
macOS-specific and Docker steps could not be executed remotely — flagged as such.

---

## Exchange 4 — Orientation questions

Short exchanges on: what the folder contains and why subfolder structure matters
(scripts reference each other by path); where to download it (`~/tissue-dynamics`
— *not* Desktop or Documents, which sync to iCloud and break Docker bind mounts);
whether Homebrew 6.0.14 works (yes — 6.0 released June 2026; guide's "4.x.x" was
stale and was corrected); arm64 vs x86 explained; `git add -A` explained; and
confirmation of the mental model that `docker compose` runs Linux while plain
commands run macOS — refined to *the file is the same, the interpreter differs*.

---

## Exchange 5 — Build failure: wrong branch

```
ERROR: file:///opt/SurRoL does not appear to be a Python project:
neither 'setup.py' nor 'pyproject.toml' found.
```

**Diagnosis:** GitHub API showed `"default_branch": "SR-VPPV"` — the Science
Robotics'25 release, organised as paper artifacts with no root `setup.py`.

**Fix:** `git clone --depth 1 --branch main`, plus an explicit `test -f setup.py`
guard so the failure surfaces where the mistake is.

**Bonus finding:** the `Dev` branch holds the Taichi MPM soft-body work — the
Phase 1 destination, and a branch of the same repo, so the robot API is
unchanged.

---

## Exchange 6 — OpenMP conflict

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
already initialized.
```

**Diagnosis:** conda-forge numpy links against conda's `libomp.dylib`; the PyPI
torch wheel bundles its own.

**Fix:** rewrote `environment.yml` so conda supplies only the interpreter and the
entire numerical stack comes from pip. Explicitly advised *against*
`KMP_DUPLICATE_LIB_OK=TRUE` — documented as unsupported and able to silently
produce incorrect numerical results.

---

## Exchange 7 — Container verification

Nine checks; eight passed. The SurRoL import check failed with
`TypeError: expected str, bytes or os.PathLike object, not NoneType` — a bug in
the check, not the environment. SurRoL has no `__init__.py`, making it a
namespace package with `__file__ = None`. Fixed to use `__path__`. The *next*
check — PSM1 loading with correct kinematics — had already passed, a far stronger
test.

Same bug also appeared in the guide's exploration snippet; corrected there.

Flagged at this point: the demo cloth reported only 25 simulation nodes.

---

## Exchange 8 — Visualization

**Andric:** Any way to visualize the actions being performed?

Added two, answering different questions:

- `collect_retraction.py --video` — CPU software rasteriser (`ER_TINY_RENDERER`),
  works headless, ~3× runtime. Renders the *simulator*.
- `host/visualize_trajectory.py` — 3D animation plus stability diagnostics.
  Renders the *file*.

Argued the second matters more: if the sim looks right but the model can't learn,
the bug is between them — node ordering, units, wrong differencing interval — and
a simulator video cannot reveal any of it because it never reads the file.

---

## Exchange 9 — Grasp caught zero nodes

72 identical warnings per episode.

**Two bugs.** The retry condition was "no anchors yet," which stays true when the
grasp catches nothing — needed a separate "attempted" flag. And the grasp
coordinates were hardcoded, but the sheet falls under gravity during the approach
and the asset's true extent isn't inferable from its scale factor.

**Fix:** settle first, read actual node positions, derive the grasp target from
the mesh; nearest-N fallback. Verified 200/200 random seeds grasped ≥1 node
against a synthetic mesh placed in the exact failure configuration.

---

## Exchange 10 — Dragging, not retracting

Output revealed two deeper problems:

```
settled sheet: 25 nodes, x[-0.300,+0.300] y[-0.300,+0.300]
max motion 139.6 mm
```

A 60 cm sheet with 15 cm node spacing, and peak motion almost exactly equal to
the gripper's own travel — the sheet was being *dragged*, not stretched.

**Fixes:**

- `container/make_tissue_mesh.py` — author the tissue. 10 cm, 20×20 = 400 nodes,
  5 mm spacing, one vertex per simulation node so topology maps cleanly.
- Pin the perimeter to the world. Real tissue is continuous with surrounding
  structure; pulling it stretches it.
- All motion rescaled ~5× down to match the new length scale.
- Grasp selection made geometric (middle half in each direction), excluding
  pinned nodes.
- Fixed `n_grasped` being read after release cleared it.

Verified: 76 boundary nodes pinned (exactly 4R−4); 300/300 seeds grasped 5
interior nodes, never a pinned one.

---

## Exchange 11 — Solver divergence

**Andric:** Solver divergence definition.

Explained: explicit integrators are conditionally stable; roughly `dt < 2/ω`
with `ω = sqrt(k_eff/m_node)`. Overshoot past equilibrium creates positive
feedback and exponential blow-up. Noted the central tension — tissue is stiff,
and stiffness is exactly what forces small timesteps, which is why implicit
methods and MPM exist.

Flagged a concrete concern: moving 25 → 400 nodes dropped per-node mass 32× while
stiffness was raised, pushing ω up ~8×.

**Andric:** Peak node speed 0.661 m/s, at dt = 1/1000.

Observed that's 39× the gripper's own speed, so it isn't the tissue following the
gripper. Two candidates — release recoil (real) or divergence (garbage) —
indistinguishable from one run. Built `container/timestep_study.py`: identical
episodes at 1/240…1/4000 with logging held constant, reporting peak speed, peak
deformation, when the peak occurred, and convergence between refinements.

Added `--dt` to `collect_retraction.py`; verified by AST that it genuinely
rebinds the global and no function shadows it.

**Status: unresolved. The study has not yet been run.** *(Still true on
17 August 2026.)*

---

## Exchange 12 — Validation beyond stability

**Andric:** How else can I check the program is working correctly, other than
proving no divergence? Other failure modes?

Identified four categories: constraint failures, parameter failures,
discretization failures (mesh convergence as the missing half), and logging
failures invisible in any video.

Built `container/validate_physics.py` (nine controlled experiments) and
`host/validate_dataset.py` (eight data-integrity checks).

Highlighted the symmetry test: centre grasp, straight pull, field must be
invariant under 90° rotation — a correct answer known without any reference
simulation. Verified the permutation is a true bijection of order 4, that a
scrambled ordering yields 94% error against a 2% threshold, and that deliberately
broken datasets (wrong dt on velocities; actions offset from what happened) are
both caught while clean data passes.

Closed by noting all of it checks *internal consistency* only — a simulation can
pass every check and still not resemble real tissue.

> *17 August note:* `validate_physics.py` was written in this exchange but was
> never saved to the repository — it existed only as a project artifact until it
> was restored on 17 August. It has still never been executed.

---

## Also discussed

**Editor:** Recommended VS Code, decided by the **Dev Containers** extension —
`pybullet` and `surrol` exist only inside the container, so a host-side editor
gives no autocomplete and flags every import as an error. Dev Containers attaches
the language server inside the container. Noted Cursor's stronger AI but its
friction with Microsoft-licensed extensions, which is exactly the dependency that
matters here; and Zed's speed but weaker container support.

---

## End state

Working: environment on both sides, container verified, mesh authored, retraction
episodes collecting with a real grasp and pinned boundary, visualization,
validation suite.

Blocking: timestep convergence unresolved (§4 of the decision log).

Next: run `timestep_study.py`, then `validate_physics.py`, then swap the
kinematic block for the dVRK PSM.

> *17 August note:* none of those three next steps has been done. Session 2 went
> in a different direction — see `DECISION_LOG.md` §8.
