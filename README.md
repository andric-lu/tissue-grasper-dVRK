# tissue-dynamics

Learned dynamics models for soft tissue under robotic manipulation, starting with
tissue retraction using the dVRK.

**Start here: [`SETUP_GUIDE.md`](SETUP_GUIDE.md)** — a complete setup walkthrough
assuming no command-line experience. Day-to-day commands are in
[`CHEATSHEET.md`](CHEATSHEET.md). **Why anything is the way it is:
[`DECISION_LOG.md`](DECISION_LOG.md)** — read this before changing a parameter.

---

## The architecture

Two environments on one MacBook, sharing one folder.

```
HOST (macOS-native)                    CONTAINER (Ubuntu 22.04, ARM64)
conda env "tissue-host", Python 3.11   Docker, Python 3.10
─────────────────────────────────      ────────────────────────────────
PyTorch  → Apple GPU via MPS           PyBullet physics
Taichi   → Apple GPU via Metal         SurRoL + dVRK robot models
analysis, plots, notebooks             ROS (later)
            │                                      │
            └──────────► data/*.npz ◄──────────────┘
```

**Why split.** SurRoL's dependency stack targets Ubuntu and pins to versions from
its publication era; a container reproduces that exactly. But Apple's GPU is only
reachable through Metal, which is a macOS API — no container can use it. So
physics runs in the container, learning runs on the host, and they exchange data
through a shared folder.

**Why it matters later.** Every simulator writes the same trajectory format, so
moving from PyBullet to Taichi/MPM to Isaac Sim adds a data *source* rather than
forcing a rewrite of the modelling code.

---

## Layout

| Path | Runs on | Contents |
|---|---|---|
| `src/` | **both** | The shared layer — `trajectory_io.py` (data contract, schema v2.0), `actions.py` (pose/action encoding, now built on `scipy.spatial.transform.Rotation`), `materials.py`, `tissue_metrics.py`, `synthetic_traj.py`, `checks.py` (shared PASS/WARN/SKIP/FAIL reporting), `diagnostics.py` (shared release-step / recoil-vs-divergence helpers) |
| `host/` | macOS | `environment.yml`, `verify_host.py`, `train_dynamics.py`, `visualize_trajectory.py`, `validate_dataset.py` |
| `container/` | Linux | `verify_container.py`, `make_tissue_mesh.py`, `collect_retraction.py`, `timestep_study.py`, `validate_physics.py` |
| `tests/` | macOS | 200 unit tests over the `src/` layer |
| `docker/` | — | `Dockerfile` (built via `docker-compose.yml` at the repo root) |
| `data/`, `data_synth/`, `models/` | **both** | Trajectories and checkpoints (git-ignored) |

**What lives in `src/`** — everything both sides of the split must agree on. The
trajectory schema, the action encoding, the material parameterisation, and the
two metrics the controller optimises. Defined once here so a collector and a
trainer cannot quietly disagree about what a number means. `checks.py` and
`diagnostics.py` are newer additions to that same idea: four scripts had each
grown their own copy of "run a check, print PASS/WARN/SKIP/FAIL, tally the
result" and two files had each grown their own copy of "which step did the
grasp release on, and was the fastest motion near it" — both are now written
once and imported everywhere they're used (`DECISION_LOG.md` §12).

---

## Quick start

Assuming setup is done (see `SETUP_GUIDE.md`):

```bash
cd ~/tissue-dynamics

# collect episodes in the container
docker compose run --rm surrol python container/collect_retraction.py --episodes 20

# check the data before training on it — exits non-zero on FAIL
conda activate tissue-host
python host/validate_dataset.py --data data/

# train on the Apple GPU
python host/train_dynamics.py --data data --epochs 100
```

Health checks:

```bash
python host/verify_host.py
docker compose run --rm surrol python container/verify_container.py
pytest tests/ -v
```

Generate the analytic test episodes (needed by nothing else, useful for checking
the metric/validator stack end to end — see `DECISION_LOG.md` §8.6):

```bash
python src/synthetic_traj.py --out data_synth/ --kinds all
python host/validate_dataset.py --data data_synth/
```

---

## Current status and honest limitations

**Working:** end-to-end pipeline — deformable-sheet retraction in PyBullet →
shared `.npz` trajectories → MLP dynamics baseline trained on Metal. On top of
that, a schema (v2.0) and a shared `src/` layer built for a constitutive
simulator that does not exist yet, verified against analytically-known episodes.

**Not yet meaningful as physics.** The current tissue is a mass-spring cloth. It
has no volume, so it cannot represent the incompressibility that dominates real
soft-tissue response, and its stiffness parameter has no physical units. Treat
the present dataset as a pipeline test, not as ground truth.

**Not yet the right model.** The MLP flattens the mesh into a vector, discarding
connectivity and locking the model to a single mesh resolution. A graph network
(MeshGraphNets-style) is the appropriate architecture; the trajectory format
already carries `tissue_faces` / `tissue_tets` for that purpose.

**Open and blocking: timestep convergence.** `container/timestep_study.py` has
never been run, `DT` is still 1/1000 in `container/collect_retraction.py`, and
every episode in `data/` was collected at that unvalidated timestep.
`DECISION_LOG.md` §4.

**Never run: `container/validate_physics.py`.** Nine controlled physics
experiments, written 1 August, restored to the repository 17 August. Its
PASS/WARN/SKIP/FAIL reporting was folded into the shared `src/checks.py` on
2 September, but the experiments themselves have still never been executed —
see `DECISION_LOG.md` §10, §12.4.

**The material ranges are placeholders.** `src/materials.py` ships
order-of-magnitude stand-ins for generic soft tissue. Liver, bowel and fat differ
from each other by more than the width of those ranges. No result should be
claimed from a model trained on them.

**2 September code-review pass — no physics changed.** A pass through the
codebase asking "is anything hand-built here that a well-tested library already
does, and is any logic copied across files instead of shared" produced three
changes, each verified to reproduce the prior behaviour exactly rather than
assumed to: the ~100 lines of hand-rolled quaternion math in `src/actions.py`
were replaced with `scipy.spatial.transform.Rotation` (already a project
dependency); the PASS/WARN/SKIP/FAIL check-reporting duplicated across
`verify_host.py`, `verify_container.py`, `validate_physics.py` and
`validate_dataset.py` was consolidated into `src/checks.py`; and the
release-step/peak-speed/recoil-vs-divergence logic duplicated between
`visualize_trajectory.py` and `timestep_study.py` was consolidated into
`src/diagnostics.py`. No schema, data contract, or numerical result changed.
`DECISION_LOG.md` §12.

**Roadmap.** Resolve the timestep question → swap the kinematic block for the
dVRK PSM → scale up data collection → graph network → MPM/Neo-Hookean tissue for
physically meaningful ground truth → Isaac Sim on Linux/cloud GPU.

---

## References

- Xu et al., *SurRoL: An Open-source RL Centered and dVRK Compatible Platform for
  Surgical Robot Learning*, IROS 2021 — [arXiv:2108.13035](https://arxiv.org/abs/2108.13035)
- *Efficient Physically-based Simulation of Soft Bodies in Embodied Environment
  for Surgical Robot* — [arXiv:2402.01181](https://arxiv.org/abs/2402.01181) (MPM
  soft-body extension to SurRoL)
- Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021 —
  [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
- *Autonomous Soft Tissue Retraction Using Demonstration-Guided RL* —
  [arXiv:2309.00837](https://arxiv.org/abs/2309.00837)
