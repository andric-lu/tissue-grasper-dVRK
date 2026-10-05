# tissue-dynamics

Learned dynamics models for soft tissue under robotic manipulation, starting with
tissue retraction using the dVRK.

**Start here: [`logs/SETUP_GUIDE.md`](logs/SETUP_GUIDE.md)** — a complete setup walkthrough
assuming no command-line experience. Day-to-day commands are in
[`logs/CHEATSHEET.md`](logs/CHEATSHEET.md). **Why anything is the way it is:
[`logs/DECISION_LOG.md`](logs/DECISION_LOG.md)** — read this before changing a parameter.

---

## The architecture

Two environments on one MacBook, sharing one folder. All current simulation work
runs on the host; the container holds the original PyBullet track.

```
HOST (macOS-native)                        CONTAINER (Ubuntu 22.04, ARM64)
conda env "tissue-host", Python 3.11       Docker, Python 3.10
──────────────────────────────────────     ─────────────────────────────────
Taichi MPM → Apple GPU via Metal           PyBullet mass-spring cloth
  (third_party/MPM, Neo-Hookean)             (legacy track, schema v1)
PyBullet → PSM IK + collision proxies      SurRoL main (smoke test only)
PyTorch  → Apple GPU via MPS
validation, tests, plots
          │                                            │
          ▼                                            ▼
  data_mpm*/  (schema v2.1)                      data/  (schema v1)
          └───── one format: src/trajectory_io.py ─────┘
```

**Why the simulator runs on the host.** The MPM solver is written in Taichi, and
Taichi has no Linux ARM64 build in any release, so on this Mac it runs natively
or not at all (§9.3). It also needs Apple's GPU, which is reachable only through
Metal, a macOS API that no container can use; Metal runs this solver 8.1× faster
than the CPU (§9.4). And the solver's collision code (`third_party/MPM/sdf.py`)
reads the robot's collision geometry from PyBullet in the same process, so
PyBullet, the dVRK PSM model and its IK run on the host too (§10).

**What the container is for.** The original PyBullet mass-spring retraction
track (`container/`), on SurRoL's pinned Ubuntu stack (gym 0.21, numpy 1.23.5).
It produced the v1 data and the MLP baseline. SurRoL itself is loaded only as a
smoke test in `verify_container.py`.

**Why one data format.** Every simulator writes the same trajectory format
(`src/trajectory_io.py`), so moving from PyBullet to Taichi/MPM to Isaac Sim adds
a data *source* rather than forcing a rewrite of the modelling code. `src/` is
numpy-only so that both environments can import it.

**Direction (decided 5 October 2026, not yet done).** Run all simulation natively
on macOS and retire the ARM container. Docker comes back later on an x86 + NVIDIA
machine or cluster, where Taichi's CUDA build and GPU access both work.

---

## Layout

| Path | Runs on | Contents |
|---|---|---|
| `src/` | **both** | The shared layer — `trajectory_io.py` (data contract, schema v2.1), `actions.py`, `materials.py`, `tissue_metrics.py`, `synthetic_traj.py`, `graph_design.py` (GNN node sampling / radius graphs) |
| `host/` | macOS | `environment.yml`, `verify_host.py`, `train_dynamics.py`, `visualize_trajectory.py`, `validate_dataset.py`; MPM: `mpm_adapter.py`, `psm.py`, `grasp_constraint.py`, `visualize_grasp.py`, `smoke_test_{mpm,psm,grasp}.py` |
| `container/` | Linux | `verify_container.py`, `make_tissue_mesh.py`, `collect_retraction.py`, `timestep_study.py`, `validate_physics.py` |
| `tests/` | macOS | 278 unit tests over the `src/` layer and the validator |
| `third_party/`, `psm_Si_model/` | host | Vendored Taichi MPM solver and dVRK Si PSM model (see `PROVENANCE.md` in each) |
| `logs/` | — | Setup guide, cheatsheet, decision log, session transcript |
| `docker/` | — | `Dockerfile` |
| `data/`, `data_synth/`, `data_mpm*/`, `models/` | **both** | Trajectories and checkpoints (git-ignored) |

**What lives in `src/`** — everything both sides of the split must agree on. The
trajectory schema, the action encoding, the material parameterisation, and the
two metrics the controller optimises. Defined once here so a collector and a
trainer cannot quietly disagree about what a number means.

---

## Quick start

Assuming setup is done (see `logs/SETUP_GUIDE.md`):

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
the metric/validator stack end to end — see `logs/DECISION_LOG.md` §8.6):

```bash
python src/synthetic_traj.py --out data_synth/ --kinds all
python host/validate_dataset.py --data data_synth/
```

---

## Current status and honest limitations

*As of 5 October 2026. Detail is in `CLAUDE.md` ("Current state") and
`logs/DECISION_LOG.md`.*

**Working.**

- **MPM tissue on the host** (§9.3–§9.7). SurRoL `Dev`'s Taichi MPM, vendored
  unmodified in `third_party/MPM/` (Neo-Hookean, 24,000 particles, 64³ grid),
  driven by `host/mpm_adapter.py`, which writes schema v2.1 episodes.
- **dVRK PSM grasping** (§10). `--task grasp` drives a PSM through
  approach, close, retract and release, with a persistent grasp constraint
  (`host/grasp_constraint.py`).
- **Validation.** `host/validate_dataset.py` (exits non-zero on FAIL), smoke
  tests for the MPM, the PSM and the grasp, and 278 unit tests.
- **Phase 0 of the graph-network plan, measured** (§13). Provisional design:
  600 nodes, connection radius ≈ 32 mm, recording at 240 Hz, 10
  message-passing steps.
- **Legacy PyBullet track.** Mass-spring retraction → v1 `.npz` → MLP baseline
  on the Apple GPU. A pipeline test, not physics: the cloth has no volume and
  its stiffness has no physical units.

**The material ranges are placeholders.** `src/materials.py` ships
order-of-magnitude stand-ins for generic soft tissue. Liver, bowel and fat differ
from each other by more than the width of those ranges. No result should be
claimed from a model trained on them.

**Not yet the right model.** The MLP flattens the mesh into a vector, discarding
connectivity and locking the model to a single resolution. A graph network
(MeshGraphNets-style) is the target and is not built yet. On the Apple GPU,
batched graph-network inference is about 2,100× short of the MPPI planning
budget (§13.5).

**The simulated arm is not the lab's arm.** The PSM model is the dVRK Si
(`psm_Si_model/`); the lab's hardware is the Classic. It also has no
remote-centre-of-motion (trocar) constraint, since its IK is free-space 6-DOF,
and its source and license are still unrecorded (`psm_Si_model/PROVENANCE.md`).

**Grasp resolution.** The jaw is smaller than one grid cell (dx = 15.6 mm), and
a grasp typically attaches 1–3 of the 24,000 particles (§10.3, §10.7).

**Open solver issues.**

- No node subset of up to 3,000 nodes carries the peak safety stretch, and 56%
  of the peak sits within one grid cell of the grasp (§13.3).
- float32 drops small volumetric strain increments, and Metal has no float64
  (§13.2).
- `p_vol` assumes 8 particles per cell but the cloud has 20.3, so mass, force
  and absolute energy are 2.54× off (§13.1).
- Dissipation depends on the substep count, which is chosen per material, so
  episodes at different materials are not strictly comparable. The floor band
  alone removes 89–93% of the energy (§9.7).

**Known bugs, found 25 September 2026, not yet fixed.**

- A crashed or interrupted episode is still saved, as a normal-looking file.
- Each MPM episode records a different random particle subset, so node *k* is
  not the same particle across episodes.
- `host/train_dynamics.py` fails on very small datasets (one episode, or an
  empty validation split).

**PyBullet track only.** Timestep convergence is unresolved:
`container/timestep_study.py` has never run, and every v1 episode was collected
at the unvalidated `DT = 1/1000` (§4). `container/validate_physics.py` has never
been executed.

**Roadmap.** Re-verify the baseline after the September cleanup → fix the bugs
above → swap in the Classic PSM → Phase 1 of the graph-network plan (240 Hz
recording, schema 3.0) → graph network → material ranges fitted to measured
tissue data → large runs (and Isaac Sim) on an x86 + NVIDIA machine or cluster.

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
