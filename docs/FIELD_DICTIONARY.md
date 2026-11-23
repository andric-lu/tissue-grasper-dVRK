# Episode field dictionary — schema 3.0

The authoritative description of every field in an episode `.npz`, as required by
to-do Item 5. `src/trajectory_io.py` writes and reads the format; this file says
what each field *means*. If the two ever disagree, the disagreement is a bug.

Read an episode with `trajectory_io.load_trajectory(path)`. Never `np.load` by
hand unless you have read §6 (F encoding) first.

## 1. Global conventions

| Topic | Convention |
|---|---|
| Units | SI throughout: metres, seconds, kilograms, radians, m/s, rad/s, Pa, kg/m³. Converted at the writer, never at the reader. |
| World frame | The MPM domain frame. One domain unit is one metre (`mpm_adapter.DOMAIN_SCALE_M = 1.0`); gravity is 9.8 m/s² along **−z** (`mpm3d.py`: `F_grid_v[I][2] -= dt * gravity`). Origin at the domain corner; the domain is the unit cube. |
| Quaternions | **Scalar-last** `[qx, qy, qz, qw]`, unit norm. Absolute orientations only. |
| Rotation deltas | Axis-angle rotation vectors (radians), **world frame**, left-multiplied: `q_next = exp(r) · q`. |
| Angles from quaternions | `2·atan2(‖q_v‖, q_w)`, never `2·acos(q_w)`. |
| Missing values | An **empty array** (size 0) means "not recorded". All-False / zero means "recorded, and the value is zero". The scalars `substep_dt`, `n_substeps`, `grid_dx`, `n_particles_simulated` use 0 for "not recorded" because 0 is not physically possible for them. The v3 scalars below are stored as `(1,)` arrays, `(0,)` when not recorded, because 0 *is* meaningful for them (a 0 s pre-roll). |
| Node alignment | Every per-node array's node axis is aligned to `particle_ids`: node `i` is solver particle `particle_ids[i]`. |
| Logs | Natural logarithm, of the value in Pa. |

## 2. Time and indexing

| Field | Shape | Meaning |
|---|---|---|
| `dt` | scalar f32 | **Recording interval** and the **training Δt**: seconds of simulated time between consecutive rows. 1/240 s for schema-3.0 MPM collection (decision D1). |
| `substep_dt`, `n_substeps` | scalar | **Simulation timestep**: the solver substep, and how many make one row. `dt = substep_dt × n_substeps` exactly (asserted in adapter, writer and validator). |
| `control_dt` | (1,) f64 | **Control interval**: seconds between successive actions. Equals `dt` in every current collector (one action per row). |
| `time` | (T,) f64 | **Timestamps**: seconds since row 0; `time[t] = t·dt`. |
| `preroll_s` | (1,) f64 | Seconds simulated **before** row 0 and not recorded (settle pre-roll). `0.0` means row 0 *is* the reference configuration. |

**State/action indexing.** A file holds `T` rows. Row `t` is the state at
`time[t]` together with the action **issued at** `t`, which carries the system
from row `t` to row `t+1`:

    (G_t, a_t) → G_{t+1}        for t = 0 … T−2

So a file is the to-do's convention of *T states and T−1 actions*, stored as
`T` rows with **one pad action on the last row** (`action[T−1, :6] = 0`; there
is no row `T` to move to). Training pairs therefore use rows `0…T−2` as inputs
and `1…T−1` as targets, and never the last row's action. Checked by
`validate_dataset.check_action_state_indexing`.

`phase` `(T,)` int8 labels each row with the scripted phase (index into
`collection_config.PHASES = ("settle", "approach", "close", "hold", "retract",
"release", "post_release")`).

## 3. Tissue state (per row, per node)

| Field | Shape | Units / frame | Meaning |
|---|---|---|---|
| `tissue_pos` | (T, N, 3) f32 | m, world | Particle position x. |
| `tissue_vel` | (T, N, 3) f32 | m/s, world | Particle velocity v **at the end of** the row's last substep. |
| `tissue_F` | (T, N, 3, 3) f32 on read | — | Deformation gradient, `F_ij = ∂x_i/∂X_j`. See §6 for flattening and storage. |
| `rest_pos` | (N, 3) f32 | m, world | **Reference (undeformed) position X**: where each recorded particle was when `F = I`, i.e. the solver's spawn cloud before the first substep. Asserted `F = I` there at collection. Not "frame 0" unless `preroll_s == 0`. |
| `rest_pos_simulated` | (n_sim, 3) f32 | m, world | X for **every** solver particle (24,000). `rest_pos == rest_pos_simulated[particle_ids]`. Lets a node set be re-chosen after collection, and gives the true rest positions of grasped particles the subset misses. |
| `material_field` | (N, 3) f32 | [ln Pa, ln Pa, kg/m³] | Per-node `[ln μ, ln λ, ρ]`. Homogeneous tissue: `material_params` broadcast. |
| `boundary_mask` | (N,) bool | — | Kinematically clamped particles. **All False** for the MPM collector: the solver has no clamp (decision D5), so the graph feature `b` is a constant-zero slot and boundary handling is **untested**. |
| `particle_ids` | (N,) i32 | — | Solver index of each node. Schema-3.0 collection passes one fixed list for every episode (`--particle-ids`). |
| `n_particles_simulated` | scalar i32 | — | Solver particle count (24,000). |

Material relations (Item 1): `μ = E / (2(1+ν))`, `λ = Eν / ((1+ν)(1−2ν))`.
Episodes sample `ln μ` and `ln λ` directly and log-uniformly
(`materials.sample_material`), never (E, ν). `material_params` `(3,)` keeps the
episode-level value.

## 4. Tool (per row, per jaw)

K = 2 jaws, in the order given by `jaw_names`. Every `(…, K, …)` array uses
that order.

**Which arm.** `robot_model` (str) and `robot_scale` `(1,)` say which arm
drove the jaws; a v3 file without `robot_model` predates the Classic arm and is
the Si PSM.

| `robot_model` | Arm | `jaw_names` | `robot_scale` | `jaw` means | `ee_pose` is |
|---|---|---|---|---|---|
| `dvrk_si_psm` | `host/psm.py`, our own IK | `tool_gripper_link_1/2` | 1 | **per-jaw** angle; joints `[+jaw, −jaw]` | `tool_gripper_center` |
| `dvrk_classic_psm_surrol` | SurRoL `Psm1` via `host/psm_classic.py`, SurRoL's RCM-frame IK | `psm_tool_gripper1/2_link` | 5 (SurRoL's task scaling) | **total** opening; joints `[−jaw/2, +jaw/2]` | SurRoL's tool-tip frame: the jaw pivot (`psm_tool_yaw_link`) with `Psm.tool_T_tip`'s rotation |

At scale 5 the arm is five times life size in an unscaled MPM world (1 m
domain, real gravity), so the 300 mm slab stands for ~60 mm of tissue. No
similarity scaling of material, density or time is applied (DECISION_LOG
§14.15). `joint_pos` for the Classic arm is SurRoL's 6 joints with the
insertion **unscaled** (real-arm metres), then the total jaw opening.

| Field | Shape | Units / frame | Meaning |
|---|---|---|---|
| `jaw_names` | (K,) str | — | Fixes the jaw axis order. |
| `jaw_half_extents` | (K, 3) f32 | m, box frame | Half-extents of each jaw's collision box (measured from the jaw mesh in its own frame; ≈ [2.50, 5.75, 1.58] mm). |
| `jaw_box_offset` | (K, 7) f32 | m + quat, jaw-link frame | `T_jaw_box`: box frame in the jaw-link frame. Rotation is identity. Rebuild a box from a link pose with `tool_geometry.compose_box_pose`. |
| `jaw_box_pose` | (T, K, 7) f32 | m + quat, world | **Achieved** pose of each jaw collision box (FK of the IK solution) at row t. This is the geometry the solver collided with. |
| `jaw_box_vel` | (T, K, 6) f32 | [m/s, rad/s], world | `[v, ω]` by **backward difference** `(pose_t − pose_{t−1}) / dt`; row 0 is zero (arm starts at rest). Known at time t without future information. The motion the solver applied over t→t+1 is **row t+1's** value. |
| `jaw_joint` | (T, K) f32 | rad | Each jaw's own joint angle, by the arm's convention in the table above (`tool_geometry.JAW_JOINT_FACTORS`). The per-jaw **opening**. |
| `jaw` | (T,) f32 | rad | Commanded gripper opening at row t, in the arm's convention (table above): Si per-jaw, 1.0 open; Classic total, 1.396 (80°) open. 0.0 closed for both. |
| `ee_pose` | (T, 7) f32 | m + quat, world | **Commanded** end-effector pose (`tool_gripper_center`) at row t. |
| `ee_vel` | (T, 6) f32 | — | Zeros in the MPM collector (not measured). |
| `joint_pos` | (T, 7) f32 | rad / m | **Achieved** joint state, `psm.DRIVEN_JOINTS` order. |
| `action` | (T, 7) f32 | [m, rad, rad] | **Commanded action**, `action_spec = "delta_pose_jaw"`: `[dx, dy, dz, rx, ry, rz, jaw_next]`. A **world-frame pose increment** (not a velocity, not a joint command) of the end-effector over **one recording interval** `dt`, plus the **absolute** jaw angle commanded for the end of that interval. Both jaws share it; the per-jaw commanded motion is reconstructed by FK of `decode_action(ee_pose[t], action[t])`, and its result is `jaw_box_pose[t+1]` within IK tolerance (2 mm / 0.02 rad, enforced at collection). |

Commanded vs achieved: `ee_pose`/`jaw`/`action` are commanded; `jaw_box_pose`,
`joint_pos` are achieved. They agree within the IK tolerance above.
Consistency of `jaw_box_pose` with `joint_pos` needs FK and is checked in
`host/smoke_test_psm.py`, not by the validator.

## 5. Contact and grasp

| Field | Shape | Level | Meaning |
|---|---|---|---|
| `grasp_active` | (T,) bool | **global** | The solver's grasp constraint holds at least one particle. Does **not** imply any particular node is grasped. |
| `grasp_full_ids_flat/offset` | CSR, T+1 | full solver | Ground truth: solver particle indices held at row t (typically 1–3 of 24,000). |
| `grasp_ids_flat/offset` | CSR, T+1 | recorded node | `grasp_full_ids ∩ particle_ids`, as node indices. Often empty even while `grasp_active`. |
| `node_grasp` | (T, N) bool | **node** | Decision D3 node-level grasp latch, `tool_geometry.grasp_latch_step`: a node latches when it is within `grasp_capture_radius` of **both** jaw boxes and nothing is latched; it releases when `jaw ≥ jaw_release_rad`. Derived from recorded geometry only, so a model can apply it to its own predictions. A node-level **stand-in** for the 1–3 full-resolution particles. |
| `node_contact` | (T, N) bool | **node** | Node within `contact_radius` of **either** jaw box surface. Proximity, **not** proof of contact. |
| `contact_mode` | (T,) int8 | global | `CONTACT_*` enum: none/touch/stick/slip/grasp, over all particles. |
| `grasp_capture_radius`, `jaw_release_rad`, `contact_radius` | (1,) f64 | — | The rule parameters actually used (0.02 m, 0.3 rad, 0.02 m). Readers use these, not code constants. `jaw_release_rad` compares against the recorded `jaw`, so it is a per-jaw angle for Si and a total opening for Classic. |

Distances to a jaw use the box surface (`tool_geometry`):
**signed** distance is positive outside and negative inside (minus depth to the
nearest face); the solver's rule uses the **unsigned** distance `max(d, 0)`.
The normal points out of the box, from the nearest surface point toward the
particle.

## 6. Deformation gradient storage

`f_encoding` says what is on disk; `load_trajectory` always returns float32 F.

| `f_encoding` | On disk | Use |
|---|---|---|
| `float32` | F | **Default for schema-3.0 collection.** At 240 Hz, per-row ΔF in slow segments is comparable to `delta16`'s quantisation. |
| `delta16` | float16 of F − I | The v2.1 default. Fine for strain metrics, not for ΔF targets. |
| `float16` | float16 of F | v2.0 only. |

**Flattening order.** When F is flattened for a model, use C (row-major) order:
`F.reshape(..., 9)` gives `[F00, F01, F02, F10, F11, F12, F20, F21, F22]`,
`F_ij = ∂x_i/∂X_j`. `graph_builder.node_features` does exactly this on `F − I`.

## 7. What is not stored, and why

- **Edges.** Reference, spatial and tool–tissue edges are rebuilt from
  `rest_pos`, `tissue_pos` and the jaw geometry by `src/graph_builder.py`
  (to-do Item 5: "Do not store dynamic edge lists").
- **Forces.** `contact_force` is zeros: the solver has no force channel. Not a
  measurement of zero force.
- **Stress, viscoelastic state.** Deferred (Item 1).

## 8. Episode-level metadata

`schema_version`, `simulator`, `task` (`tissue_settle` / `tissue_grasp`),
`notes` (backend, seed, substep provenance), `material_params`, `grid_dx`
(15.6 mm), `target_*` (exposure target), `exposure`/`safety_strain` (logged over
all 24,000 particles; see `trajectory_io` SUBSETS).

Peak-stretch provenance (episodes from 20 Nov 2026 on; empty before;
DECISION_LOG §14.20):

| Field | Shape | Units / frame | Meaning |
|---|---|---|---|
| `safety_strain` | (T,) f32 | — (stretch ratio, 1.0 at rest) | Max principal stretch over **all** solver particles at row t. |
| `safety_strain_id` | (T,) i32 | full solver index | The particle carrying that maximum. A **solver** index (0…n_sim−1), never a node index. On rows where it is in `particle_ids`, its stored F reproduces `safety_strain` exactly (validator `safety peak is consistent`). |
| `safety_strain_pos` | (T, 3) f32 | m, world | That particle's position at row t. Its distance to the jaws is derived from `jaw_box_pose` (`tool_geometry.jaw_distances`), not stored. |

Each grasp episode has a
sibling `mpm_XXXX.config.json` holding the exact collection config, and the
output directory has `collection_log.txt` with the command line.
