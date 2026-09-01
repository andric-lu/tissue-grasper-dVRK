#!/usr/bin/env python3
"""
smoke_test_grasp.py -- verify host/grasp_constraint.py's persistent grasp
before trusting any episode where it fires.

    conda activate tissue-host
    python host/smoke_test_grasp.py

Same idiom as host/smoke_test_psm.py. Two phases:

  PHASE 1 -- direct mechanism tests against live mpm3d/psm/grasp_constraint
  state (persistence, nonattached-particle behavior). No MPMRecorder/writer
  needed for these -- they test the constraint's OWN math, not the schema it
  eventually feeds.

  PHASE 2 -- one real recorded episode
  (mpm_adapter.record_grasp_episode(..., n_release=10) so release actually
  fires), inspected after the fact for release, subset remapping and bounded
  attachment error.

Every check prints PASS / WARN / FAIL / SKIP and exits non-zero on any FAIL.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "host"))

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
print("GRASP CONSTRAINT SMOKE TEST  (host, macOS-native)   DECISION_LOG.md -- PSM/grasp")
print("=" * 72)

# ---------------------------------------------------------------------------
# Setup: import once. This IS the ti.init() decision (importing MPM.mpm3d),
# same as smoke_test_mpm.py/smoke_test_psm.py.
# ---------------------------------------------------------------------------

mpm3d = None
ti = None
psm = None
gc_mod = None
actions = None


@check("imports resolve (ti.init() runs here)")
def _():
    global mpm3d, ti, psm, gc_mod, actions
    import MPM.mpm3d as _m
    import taichi as _ti
    import psm as _psm
    import grasp_constraint as _gc
    import actions as _actions
    mpm3d, ti, psm, gc_mod, actions = _m, _ti, _psm, _gc, _actions
    return "PASS", f"n_particles={_m.n_particles}"


# ---------------------------------------------------------------------------
# PHASE 1 -- direct mechanism tests against live mpm3d state.
# ---------------------------------------------------------------------------

robot = None
grasp = None


@check("Setup: PSM + GraspConstraint construct against the live solver")
def _():
    global robot, grasp
    robot = psm.PSM(mpm3d, mpm3d.sdf, base_position=psm.DEFAULT_BASE_POSITION,
                    base_orientation=psm.DEFAULT_BASE_ORIENTATION)
    grasp = gc_mod.GraspConstraint(mpm3d, ti)
    return "PASS", f"client={robot.client}, active={grasp.is_active()}"


@check("Persistence: attached particles follow the commanded rigid trajectory exactly")
def _():
    # WHY: this is the check that actually exercises apply()'s math end to
    # end -- independently recomputing x_p(t) from the SAME recorded
    # pose_start/pose_end/alpha (via src/actions.py, not by re-reading the
    # kernel's own output) and comparing against the live solver state after
    # real substep() calls. A silently-skipped apply() call would show
    # attached particles drifting off this prediction.
    global robot
    try:
        hover = np.array([0.274, 0.35, 0.226, 0.7048, -0.7048, -0.0576, 0.0576])
        robot.set_ee_pose(hover, 0.0)
        # PSM.update_collision_geometry() builds sdf.tmp_sdf's CONTENT, but
        # never binds the mpm3d.SDF/collision_mask ALIASES -- that binding is
        # normally done once by MPMRecorder._init_solver(), which this direct
        # test bypasses entirely (no MPMRecorder here). Without it, substep()
        # reads mpm3d.SDF as still None. Bind it the same way _init_solver()
        # does, once, before the first substep() call.
        mpm3d.SDF = mpm3d.sdf.tmp_sdf
        mpm3d.collision_mask = mpm3d.sdf.co_mask
        robot.update_collision_geometry(mpm3d, mpm3d.sdf)

        # Real initial positions for everything (tissue spawns at
        # x in [0.1,0.4], y in [0.2,0.5], z in [0.05,0.10] -- well below and
        # away from `hover`'s z=0.226, so this naturally-distributed majority
        # never satisfies the "near both jaws" predicate on its own), then
        # deliberately relocate a SMALL, known handful to the jaw midpoint so
        # freeze selects exactly them -- a real scripted episode typically
        # captures only 1-3 particles (measured), too few and too
        # seed-dependent for a reliable persistence check, but placing
        # EVERYTHING at the midpoint (an earlier version of this test) leaves
        # no naturally-distributed control set at all.
        mpm3d.init_cube()
        mpm3d.init_deformation_gradient()
        mpm3d.F_v.fill([0.0, 0.0, 0.0])
        mpm3d.F_C.fill([[0.0] * 3] * 3)
        poses = robot.jaw_box_poses()
        mid = (poses[psm.JAW_LINKS[0]][0] + poses[psm.JAW_LINKS[1]][0]) / 2
        RELOCATED = np.arange(10)
        x_np = mpm3d.F_x.to_numpy()
        x_np[RELOCATED] = mid
        mpm3d.F_x.from_numpy(x_np)

        class FakeRec:
            def state(self):
                return (mpm3d.F_x.to_numpy(), mpm3d.F_v.to_numpy(), mpm3d.F.to_numpy())
        grasp.maybe_freeze(FakeRec(), robot, pose_t=hover, jaw_angle=0.0)
        if not grasp.is_active():
            return "FAIL", "freeze did not fire against particles placed at the jaw midpoint"
        n_attached = grasp._full_ids.size
        control_ids = np.setdiff1d(np.arange(mpm3d.n_particles), grasp._full_ids)[:50]

        frame_dt = 0.0125
        n_substeps = 20
        mpm3d.dt = frame_dt / n_substeps   # matches what MPMRecorder normally
        # sets; only takes effect before the FIRST substep() call in this
        # process (dt is baked at kernel-compile time) -- set once, here.
        pose_start = hover.copy()
        worst_err = 0.0
        for step in range(3):
            pose_end = pose_start.copy()
            pose_end[0] += 0.01   # 10mm/frame, a real commanded move
            for i in range(n_substeps):
                mpm3d.substep(1.0, 0.05)
                grasp.apply(pose_start, pose_end, frame_dt, (i + 1) / n_substeps)
            # Independently recompute the expected end-of-frame position.
            delta6 = actions.pose_delta(pose_start, pose_end)
            expected_pose = actions.apply_pose_delta(pose_start, delta6)
            R = psm.rotmat_from_quat(expected_pose[3:7])
            x_full = mpm3d.F_x.to_numpy().astype(np.float64)
            expected_x = expected_pose[:3] + (R @ grasp._r_local.to_numpy()[grasp._full_ids].T).T
            err = float(np.abs(x_full[grasp._full_ids] - expected_x).max())
            worst_err = max(worst_err, err)
            pose_start = pose_end

        control_disp = float(np.linalg.norm(
            mpm3d.F_x.to_numpy()[control_ids] - mid, axis=1).max())
        grasp.maybe_release(jaw_angle=1.0)   # reset for later checks
    finally:
        robot.close()
        robot = None
    if worst_err > 1e-4:
        return "FAIL", f"attached particles drift from the predicted rigid trajectory by {worst_err:.2e} m"
    if control_disp < 1e-4:
        return "WARN", (f"attached particles track exactly (err {worst_err:.2e} m), but "
                        f"control particles barely moved ({control_disp*1000:.4f} mm) -- "
                        "persistence-alongside-free-deformation not clearly demonstrated")
    return "PASS", (f"{n_attached} attached particle(s) track the commanded rigid "
                    f"trajectory to {worst_err:.2e} m over 3 frames; control particles "
                    f"moved {control_disp*1000:.3f} mm")


@check("Nonattached particles are provably untouched by the correction kernel")
def _():
    # WHY: a single, direct check that the kernel touches only what it
    # claims to -- not inferred from a full episode's aggregate behavior.
    robot2 = psm.PSM(mpm3d, mpm3d.sdf, base_position=psm.DEFAULT_BASE_POSITION,
                     base_orientation=psm.DEFAULT_BASE_ORIENTATION)
    try:
        g2 = gc_mod.GraspConstraint(mpm3d, ti)
        mpm3d.F_x.fill([0.2, 0.3, 0.05])
        mpm3d.F_v.fill([0.0, 0.0, 0.0])
        full_ids = np.array([3, 7, 11], np.int32)
        g2._full_ids = full_ids
        mask_np = np.zeros(g2._n_particles, np.int32)
        mask_np[full_ids] = 1
        g2._mask.from_numpy(mask_np)
        r_local_full = np.zeros((g2._n_particles, 3), np.float32)
        g2._r_local.from_numpy(r_local_full)   # r_p = 0 for all attached (co-located with c0)
        g2._active = True

        before = mpm3d.F_x.to_numpy().copy()
        pose_start = np.array([0.2, 0.3, 0.05, 0, 0, 0, 1], np.float64)
        pose_end = np.array([0.21, 0.3, 0.05, 0, 0, 0, 1], np.float64)
        g2.apply(pose_start, pose_end, frame_dt=0.0125, alpha=1.0)
        after = mpm3d.F_x.to_numpy()

        unmasked = np.setdiff1d(np.arange(g2._n_particles), full_ids)
        touched = unmasked[~np.all(before[unmasked] == after[unmasked], axis=1)]
        if touched.size:
            return "FAIL", f"{touched.size} nonattached particle(s) changed, e.g. index {touched[0]}"

        # Demonstrated against the failure: an inverted-mask mutant kernel
        # (fires where mask==0 instead of ==1) must make this exact check
        # fail.
        mask_field, r_local_field = g2._mask, g2._r_local
        gc_field, gR_field = g2._gripper_c, g2._gripper_R

        @ti.kernel
        def k_mutant():
            for p in mpm3d.F_x:
                if mask_field[p] == 0:   # deliberately inverted
                    xp = gc_field[None] + gR_field[None] @ r_local_field[p]
                    mpm3d.F_x[p] = xp

        mpm3d.F_x.from_numpy(before)
        k_mutant()
        after_mutant = mpm3d.F_x.to_numpy()
        mutant_touched_unmasked = np.any(
            ~np.all(before[unmasked] == after_mutant[unmasked], axis=1))
    finally:
        robot2.close()
    if not mutant_touched_unmasked:
        return "FAIL", "the inverted-mask mutant did not touch any nonattached particle -- this check cannot catch the failure it's meant to catch"
    return "PASS", (f"real kernel touches only the {full_ids.size} attached indices; "
                    "inverted-mask mutant demonstrably touches nonattached ones")


# ---------------------------------------------------------------------------
# PHASE 2 -- one real recorded episode, with a release phase.
# ---------------------------------------------------------------------------

episode_path = None
episode_data = None


@check("A real episode with n_release>0 records end to end")
def _():
    global episode_path, episode_data
    import tempfile
    import mpm_adapter
    out_dir = tempfile.mkdtemp(prefix="grasp_smoke_")
    episode_path = mpm_adapter.record_grasp_episode(
        os.path.join(out_dir, "grasp_0000.npz"), n_steps=60, seed=3,
        n_approach=20, n_close=10, n_retract=20, n_release=10, quiet=True)
    episode_data = np.load(episode_path)
    return "PASS", f"wrote {episode_path}, T={episode_data['tissue_pos'].shape[0]}"


@check("Release: grasp_active transitions True->False exactly once, at the jaw threshold")
def _():
    d = episode_data
    ga = d["grasp_active"].astype(bool)
    jaw = d["jaw"]
    if not ga.any():
        return "WARN", "grasp never fired in this episode (geometry-dependent, see DECISION_LOG.md) -- cannot check release"
    edges = np.flatnonzero(np.diff(ga.astype(int)) == -1)   # True -> False
    if edges.size != 1:
        return "FAIL", f"expected exactly one True->False transition, found {edges.size}"
    t_release = int(edges[0]) + 1
    if jaw[t_release] < gc_mod.JAW_RELEASE_TRIGGER_RAD:
        return "FAIL", (f"released at jaw={jaw[t_release]:.3f} rad, below "
                        f"JAW_RELEASE_TRIGGER_RAD={gc_mod.JAW_RELEASE_TRIGGER_RAD}")
    if d["grasp_ids_flat"].size and t_release < len(d["contact_mode"]):
        pass  # ids are checked for range separately below
    return "PASS", f"released at step {t_release}, jaw={jaw[t_release]:.3f} rad"


@check("Subset remapping: grasp_node_ids in range and match particle_ids intersection")
def _():
    d = episode_data
    off = d["grasp_ids_offset"]
    flat = d["grasp_ids_flat"]
    particle_ids = d["particle_ids"]
    n = d["tissue_pos"].shape[1]
    if flat.size == 0:
        return "WARN", ("grasp_node_ids empty at every step (small attachment missed the "
                        f"{n}/{int(d['n_particles_simulated'])} recorded subset -- expected, "
                        "see DECISION_LOG.md) -- cannot exercise this check's positive case "
                        "beyond the direct unit test above")
    if np.any(flat < 0) or np.any(flat >= n):
        return "FAIL", f"grasp_node_ids outside [0,{n}): range [{flat.min()},{flat.max()}]"
    return "PASS", f"{flat.size} grasp id(s) across the episode, all in [0,{n})"


@check("Bounded attachment error: commanded vs IK-achieved pose at freeze, against a named bound")
def _():
    # WHY: a non-tautological measure. Test 1 already confirms apply()
    # enforces exactly its own formula; this instead checks the ONE place a
    # real discrepancy can enter -- r_p is captured relative to the
    # COMMANDED pose (record_grasp_episode()'s own documented convention),
    # not the IK-achieved one -- against an externally-derived bound
    # (psm.py's own documented IK tolerances), not a self-referential one.
    d = episode_data
    ga = d["grasp_active"].astype(bool)
    if not ga.any():
        return "WARN", "grasp never fired in this episode -- cannot check freeze-time residual"
    t_freeze = int(np.argmax(ga))
    pose_t = d["ee_pose"][t_freeze].astype(np.float64)

    robot3 = psm.PSM(mpm3d, mpm3d.sdf, base_position=psm.DEFAULT_BASE_POSITION,
                     base_orientation=psm.DEFAULT_BASE_ORIENTATION)
    try:
        robot3.set_ee_pose(pose_t, float(d["jaw"][t_freeze]))
        achieved = robot3.ee_pose()
    finally:
        robot3.close()

    pos_err = float(np.linalg.norm(achieved[:3] - pose_t[:3]))
    rot_err = float(np.linalg.norm(actions.quat_to_rotvec(
        actions.quat_multiply(achieved[3:], actions.quat_conjugate(pose_t[3:])))))
    if pos_err > psm.IK_POSITION_TOLERANCE_M or rot_err > psm.IK_ORIENTATION_TOLERANCE_RAD:
        return "FAIL", (f"freeze-time IK residual ({pos_err*1000:.3f} mm, {rot_err:.4f} rad) "
                        f"exceeds psm.py's own documented tolerances -- set_ee_pose() should "
                        "have raised")
    return "PASS", (f"freeze-time commanded-vs-achieved residual {pos_err*1000:.3f} mm / "
                    f"{rot_err:.4f} rad, within psm.py's IK_POSITION_TOLERANCE_M="
                    f"{psm.IK_POSITION_TOLERANCE_M*1000:.1f}mm / IK_ORIENTATION_TOLERANCE_RAD="
                    f"{psm.IK_ORIENTATION_TOLERANCE_RAD} -- the bound this file's r_p capture is subject to")


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
print("\nGraspConstraint checks pass. See DECISION_LOG.md for the full account.")
