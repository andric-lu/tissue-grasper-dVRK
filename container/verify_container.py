#!/usr/bin/env python3
"""
verify_container.py -- confirm the Linux container can run physics and SurRoL.

Run INSIDE the container:

    docker compose run --rm surrol python container/verify_container.py

Mirrors host/verify_host.py, but checks the things that matter on this side:
ARM-native Linux, a working PyBullet, deformable-body support (the whole point
of the project), SurRoL importing, and the shared /work mount being writable.
"""

import platform
import sys

results = []


def check(name):
    def wrap(fn):
        try:
            status, detail = fn()
        except ImportError as e:
            status, detail = "FAIL", f"not installed ({e.name}); rebuild the image"
        except Exception as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        results.append((status, name, detail))
        icon = {"PASS": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}[status]
        print(f"[{icon}] {name}\n         {detail}")
        return fn
    return wrap


print("=" * 72)
print("CONTAINER ENVIRONMENT CHECK  (Linux, simulation side)")
print("=" * 72)


@check("Running on ARM64 Linux, not emulated x86")
def _():
    m = platform.machine()
    # aarch64 is what ARM64 is called on Linux. If this says x86_64, Docker is
    # emulating an Intel machine through QEMU and physics will be ~5x slower.
    # Fix: the `platform: linux/arm64` line in docker-compose.yml.
    if m == "aarch64":
        return "PASS", f"{platform.system()} {m}, python {platform.python_version()}"
    return "FAIL", f"machine={m} -- image is being emulated, expect ~5x slowdown"


@check("numpy is pinned below 1.24")
def _():
    import numpy as np
    major, minor = (int(x) for x in np.__version__.split(".")[:2])
    # gym 0.21 and SurRoL use np.bool / np.float, removed in numpy 1.24.
    # A surprise upgrade here is one of the most common ways this image breaks.
    if (major, minor) < (1, 24):
        return "PASS", f"numpy {np.__version__}"
    return "FAIL", (f"numpy {np.__version__} removed np.bool/np.float aliases "
                    "that gym 0.21 and SurRoL still use")


@check("gym is the old-API version SurRoL expects")
def _():
    import gym
    if gym.__version__.startswith("0.21"):
        return "PASS", f"gym {gym.__version__}"
    return "FAIL", (f"gym {gym.__version__} -- SurRoL needs the 4-value step() "
                    "API from 0.21.x. Something upgraded it.")


@check("PyBullet imports and a physics server starts")
def _():
    import pybullet as p
    # DIRECT = headless, no window. This is the only mode that makes sense in a
    # container, and it is also the fast mode: no frames are rendered unless
    # you explicitly ask for a camera image.
    cid = p.connect(p.DIRECT)
    if cid < 0:
        return "FAIL", "could not start a physics server"
    p.disconnect(cid)
    return "PASS", f"pybullet {p.getAPIVersion()}, DIRECT (headless) mode works"


@check("Rigid-body simulation produces correct physics")
def _():
    import pybullet as p
    import pybullet_data
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    dt = 1 / 240
    p.setTimeStep(dt)
    # Drop a sphere from 1 m in free fall for 0.25 s and compare against
    # the analytic answer z = 1 - g*t^2/2. If the integrator or the units are
    # wrong, this catches it immediately.
    body = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.05),
        basePosition=[0, 0, 1.0],
    )
    n = 60
    for _ in range(n):
        p.stepSimulation()
    z = p.getBasePositionAndOrientation(body)[0][2]
    t = n * dt
    expected = 1.0 - 0.5 * 9.81 * t * t
    p.disconnect(cid)
    if abs(z - expected) > 0.02:
        return "FAIL", f"z={z:.4f} after {t:.3f}s, expected ~{expected:.4f}"
    return "PASS", f"free fall matches analytic solution ({z:.4f} vs {expected:.4f})"


@check("Deformable (soft body) simulation is available")
def _():
    import pybullet as p
    import pybullet_data
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Deformables need a different world type, and you must request it by
    # resetting the simulation with this flag. Forgetting it is the #1 cause of
    # "loadSoftBody says the method is not supported".
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0, 0, -9.81)
    soft = p.loadSoftBody(
        "cloth_z_up.obj", basePosition=[0, 0, 1], scale=0.4, mass=0.1,
        useNeoHookean=0, useBendingSprings=1, useMassSpring=1,
        springElasticStiffness=40, springDampingStiffness=0.1,
        useSelfCollision=0, frictionCoeff=0.5, useFaceContact=1,
    )
    n_verts, verts = p.getMeshData(soft, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    for _ in range(10):
        p.stepSimulation()
    p.disconnect(cid)
    if n_verts < 4:
        return "FAIL", f"soft body loaded with only {n_verts} vertices"
    return "PASS", f"cloth loaded and stepped, {n_verts} simulation nodes"


@check("SurRoL imports and its assets are present")
def _():
    import os
    import surrol
    root = os.path.dirname(surrol.__file__)
    assets = os.path.join(root, "assets")
    if not os.path.isdir(assets):
        return "WARN", f"surrol imported from {root} but no assets/ directory found"
    n = sum(len(f) for _, _, f in os.walk(assets))
    return "PASS", f"surrol at {root}, {n} asset files"


@check("The dVRK PSM robot model loads")
def _():
    import pybullet as p
    from surrol.robots.psm import Psm1
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    # Psm1 is SurRoL's wrapper around the da Vinci Patient Side Manipulator.
    # Loading it exercises the URDF, the asset paths, and the kinematics code
    # all at once -- if this passes, SurRoL is genuinely working.
    psm = Psm1((0, 0, 0.1), (0, 0, 0, 1), scaling=1.0)
    pose = psm.get_current_position()
    p.disconnect(cid)
    # SurRoL returns a 4x4 homogeneous transform; the last column's top three
    # entries are the tool-tip position. Handled defensively because this API
    # has changed across SurRoL branches.
    import numpy as np
    pose = np.asarray(pose)
    tip = pose[:3, 3] if pose.shape == (4, 4) else pose.ravel()[:3]
    return "PASS", f"PSM1 loaded, tip at {np.round(tip, 4).tolist()}"


@check("The shared /work mount is writable from inside the container")
def _():
    import os
    # Anything written here must appear on the Mac. If this fails, the volume
    # mount in docker-compose.yml is wrong and all your collected data would
    # vanish when the container exits.
    path = "/work/data/.container_write_test"
    os.makedirs("/work/data", exist_ok=True)
    with open(path, "w") as f:
        f.write("ok")
    os.remove(path)
    return "PASS", "/work/data is writable; files land in your Mac project folder"


@check("The shared trajectory format round-trips")
def _():
    import subprocess
    r = subprocess.run([sys.executable, "/work/src/trajectory_io.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return "FAIL", (r.stderr.strip().splitlines() or ["self-test failed"])[-1]
    return "PASS", "src/trajectory_io.py self-test passed"


print("=" * 72)
n_fail = sum(1 for s, _, _ in results if s == "FAIL")
n_warn = sum(1 for s, _, _ in results if s == "WARN")
if n_fail:
    print(f"{n_fail} check(s) FAILED, {n_warn} warning(s).")
    print("See SETUP_GUIDE.md, Troubleshooting.")
    sys.exit(1)
print(f"All checks passed ({n_warn} warning(s)). Container is ready.")
