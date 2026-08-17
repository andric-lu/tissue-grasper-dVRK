# tissue-dynamics

Learned dynamics models for soft tissue under robotic manipulation, starting with
tissue retraction. Read `DECISION_LOG.md` before changing any parameter — it
records why each one is what it is. Section references below are to that file.

## Working with Andric

- **Not a computer science major.** Explain what a thing does and why it exists,
  not just what to type. Assume no familiarity with build systems, packaging, or
  git internals; assume real fluency in the physics and the research.
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
pytest tests/ -q                                    # 200 tests
python host/validate_dataset.py --data data/        # exit != 0 on FAIL
python host/visualize_trajectory.py data/x.npz      # --stride 3 for long episodes
python src/synthetic_traj.py --out data_synth/ --kinds all

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
- **The MLP is a placeholder.** MeshGraphNets is the target; the schema already
  carries `tissue_faces` / `tissue_tets`.

### Next: MPM

SurRoL's `Dev` branch MPM lifts out as four files (`config.py`, `mpm3d.py`,
`sdf.py`, `requirements.txt`) with no `surrol.*` or `panda3d` imports. Decision
is to **vendor it at `third_party/MPM/` and record the upstream commit SHA**
(§9.3). Open, to be settled by a smoke test rather than argument:

1. `ti._lib.core.with_metal()` is a **private** Taichi API; their code pins 1.6.0
   and the host has 1.7.4. If it moved, the guard fails silently and everything
   runs on `ti.cpu` — correct physics, wrong speed, no error. Print the backend.
2. `pybullet` and `scikit-image` are not yet in the host env. PyBullet has no
   macOS arm64 wheel and compiles from source: 5–8 minutes, no output, not a hang.
3. `from MPM.config import ...` means the parent directory must be on `sys.path`.

Then the adapter: read MPM particle state, write a v2 `.npz`. It is small because
the receiving end was built first — `F_x → tissue_pos`, `F_v → tissue_vel`,
`F → tissue_F`, and `set_parameters()` takes what `materials.py` produces.
