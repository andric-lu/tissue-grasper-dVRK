#!/usr/bin/env python3
"""
check_surrol_interface.py -- to-do Item 6's first step, and the 5 Oct "deciding check".

    # in the environment you intend to collect with (see SETUP below)
    python host/check_surrol_interface.py

Item 6: "Use the SurRoL robot control interface coupled to the MPM tissue
simulation. First confirm this environment exposes particle x, v, F and
reference IDs; robot-only logs cannot supply the tissue training targets."

The 25 Sep / 5 Oct handoff defined the deciding check for using SurRoL's
Classic PSM (`Psm1`) on the Mac: in ONE Python process, import taichi,
pybullet, gym and surrol.robots.psm, build a Psm1, and step the MPM. If that
fails and cannot be fixed cheaply, SurRoL runs on x86 Linux rather than the
robot being hand-written (handoff section 5).

Each step prints PASS / FAIL with the reason, and later steps still run where
they can, so one run tells you everything that is and is not available. Exit
code 0 only if every step passed. Nothing here writes into the repo.

STEPS
  1  imports, one at a time, with versions and file locations
  2  Psm1 constructs on a DIRECT PyBullet client; joint/link names, the jaw
     links' collision shapes, and one small move()
  3  in the SAME process, the vendored MPM initialises, one MPMRecorder row is
     captured, and the schema-3.0 file holds x, v, F, particle_ids and rest_pos
     -- the "exposes particle x, v, F and reference IDs" condition
  4  where SurRoL's Psm1 URDF lives (is it the Classic model?)

SETUP (why a separate environment may be needed): SurRoL `main` targets gym
0.21 and numpy 1.23; gym 0.21 does not install with modern setuptools. The
container pins pip 23.3.2, setuptools 65.5.0, wheel 0.38.4 and installs gym with
--no-deps (docker/Dockerfile), and PyBullet on macOS needs
CFLAGS="-Dfdopen=fdopen" before it builds (DECISION_LOG section 9.4). Without
those pins step 1 fails for packaging reasons, not compatibility ones -- read
the error before concluding anything.
"""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "host"))

RESULTS = []
STEPS = []


def step(name):
    """Register a step; main() runs them in order (importing this file runs nothing)."""
    def wrap(fn):
        STEPS.append((name, fn))
        return fn
    return wrap


def run_steps():
    for name, fn in STEPS:
        print(f"\n== {name}")
        try:
            msg = fn()
            RESULTS.append(("PASS", name))
            print(f"PASS  {msg or ''}")
        except Exception as exc:                          # noqa: BLE001
            RESULTS.append(("FAIL", name))
            print(f"FAIL  {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)


STATE = {}


@step("1. imports (taichi, pybullet, gym, surrol.robots.psm)")
def _():
    import importlib
    lines, missing = [], []
    for mod in ("numpy", "taichi", "pybullet", "gym", "surrol", "surrol.robots.psm"):
        try:
            m = importlib.import_module(mod)
            lines.append(f"{mod} {getattr(m, '__version__', '?')} at "
                         f"{os.path.dirname(getattr(m, '__file__', '') or '?')}")
        except Exception as exc:                          # noqa: BLE001
            missing.append(f"{mod}: {type(exc).__name__}: {exc}")
    print("\n".join("   " + s for s in lines))
    if missing:
        raise ImportError("; ".join(missing))
    return f"python {sys.version.split()[0]}"


@step("2. Psm1 builds and moves on a DIRECT PyBullet client")
def _():
    import numpy as np
    import pybullet as p
    from surrol.robots.psm import Psm1
    cid = p.connect(p.DIRECT)
    if cid != 0:
        raise RuntimeError(f"PyBullet client {cid}, not 0: third_party/MPM/sdf.py's "
                           "calls carry no physicsClientId and would miss it")
    print(f"   Psm1{inspect.signature(Psm1.__init__)}")
    robot = Psm1()
    STATE["robot"] = robot
    body = getattr(robot, "body", None)
    names = [p.getJointInfo(body, i)[1].decode() for i in range(p.getNumJoints(body))]
    links = [p.getJointInfo(body, i)[12].decode() for i in range(p.getNumJoints(body))]
    print(f"   {len(names)} joints: {names}")
    print(f"   links: {links}")
    jaw = [n for n in links if "jaw" in n.lower() or "gripper" in n.lower()]
    for n in jaw:
        shapes = p.getCollisionShapeData(body, links.index(n))
        print(f"   collision of {n}: {[(s[2], s[3]) for s in shapes]} "
              "(geometry type, dimensions/mesh scale)")
    api = [m for m in ("move", "move_jaw", "get_current_position", "get_current_jaw_position",
                       "get_current_joint_position", "reset_joint") if hasattr(robot, m)]
    print(f"   dVRK-style API present: {api}")
    if hasattr(robot, "get_current_position") and hasattr(robot, "move"):
        pose = np.asarray(robot.get_current_position())
        target = pose.copy()
        target[2, 3] -= 0.005                              # 5 mm along z, RCM frame
        robot.move(target)
        after = np.asarray(robot.get_current_position())
        err = float(np.linalg.norm(after[:3, 3] - target[:3, 3]))
        print(f"   move(): commanded -5 mm z, residual {err*1e3:.3f} mm "
              "(SurRoL's IK has no residual check of its own)")
    p.disconnect(cid)        # step 3b needs client 0 for the MPM's sdf.py
    return f"{len(names)} joints, jaw links {jaw}"


@step("3. MPM in the same process: one row with x, v, F, particle_ids, rest_pos")
def _():
    import numpy as np
    import mpm_adapter
    from trajectory_io import load_trajectory
    out = os.path.join(tempfile.mkdtemp(prefix="surrol_check_"), "probe.npz")
    with mpm_adapter.MPMRecorder(out, mu=3000.0, lam=3e5, rho=1050.0, n_record=300,
                                 frame_dt=1 / 240, f_encoding="float32") as rec:
        ee = np.array([0.25, 0.35, 0.2, 0, 0, 0, 1], np.float32)
        rec.capture(ee_pose=ee, action=np.zeros(7))
        rec.advance(1)
        rec.capture(ee_pose=ee, action=np.zeros(7))
        rec.close()
        backend = rec.backend
    tr = load_trajectory(out)
    have = {k: getattr(tr, k).shape for k in ("tissue_pos", "tissue_vel", "tissue_F",
                                              "particle_ids", "rest_pos")}
    if not tr.has_F or tr.rest_pos.size == 0 or tr.particle_ids.size == 0:
        raise RuntimeError(f"missing fields: {have}")
    return f"backend {backend}; {have}"


@step("3b. Classic PSM at x5 (host/psm_classic.py) on the live solver: pose, grasp, one frame")
def _():
    import numpy as np
    import collection_config as cc
    import mpm_adapter
    cfg = cc.load_config(os.path.join(REPO, "configs", "grasp_pilot_classic.json"))
    out = os.path.join(tempfile.mkdtemp(prefix="surrol_check_"), "probe_classic.npz")
    with mpm_adapter.MPMRecorder(out, mu=3000.0, lam=3e5, rho=1050.0, n_record=300,
                                 frame_dt=1 / 240, f_encoding="float32") as rec:
        robot = mpm_adapter.make_robot(cfg, rec)
        rec.robot = robot
        wp = cc.build_waypoints(cfg)
        k = 1 + wp["frames"]["approach"] + wp["frames"]["close"]     # jaws closed at the grasp point
        robot.set_ee_pose(wp["poses"][k], float(wp["jaw"][k]))
        robot.update_collision_geometry(rec.m, rec.m.sdf)
        robot.update_collision_velocity(rec.m, rec.frame_dt)
        rec.advance(1, threshold=mpm_adapter.GRASP_CONTACT_THRESHOLD_M)
        half = [robot.jaw_half_extents(n) * 1e3 for n in robot.jaw_links]
        quat = robot.tip_down_quat()
        rec.capture(ee_pose=wp["poses"][k].astype(np.float32), action=np.zeros(7))
        rec.close()
        robot.close()
    cfg_q = np.asarray(cfg["orientation_xyzw"])
    if min(np.abs(quat - cfg_q).max(), np.abs(quat + cfg_q).max()) > 1e-6:
        raise RuntimeError(f"configs/grasp_pilot_classic.json orientation {cfg_q} != "
                           f"measured tool-down {quat}")
    return (f"{robot.model_name} x{robot.scaling:g}, RCM {np.round(robot.rcm_position, 3)}, "
            f"jaw half-extents {np.round(half[0], 2)} mm, config orientation = measured "
            "tool-down")


@step("4. which PSM model SurRoL loads")
def _():
    import surrol.robots.psm as sp
    src = inspect.getsource(sp)
    urdf = [ln.strip() for ln in src.splitlines() if ".urdf" in ln]
    root = os.path.dirname(sp.__file__)
    return f"{root}; URDF references: {urdf[:4]}"


if __name__ == "__main__":
    run_steps()
    print("\nsummary")
    for status, name in RESULTS:
        print(f"  {status}  {name}")
    sys.exit(0 if all(s == "PASS" for s, _ in RESULTS) else 1)
