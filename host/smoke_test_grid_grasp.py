#!/usr/bin/env python3
"""
smoke_test_grid_grasp.py -- verify host/grasp_grid_bc.py before trusting an
episode collected with `"grasp_model": "grid_bc"` (DECISION_LOG 14.22).

    conda activate tissue-host        # or tissue-sim
    python host/smoke_test_grid_grasp.py

No arm, no PyBullet scene: the SDF is filled large and positive so
Boundary()'s collision branch never fires (CLAUDE.md's documented way to
drive substep() bare), and membership is set directly. Checks:

  1  inactive == vendored    with nothing held, the host-owned substep must
                             reproduce mpm3d.substep(). Judged against the
                             vendored substep's OWN run-to-run spread, because
                             the Metal solver is not bit-reproducible (14.20).
  2  rigid tracking          held particles follow v_g + w x (x - c).
  3  F follows motion        under a pure translation a held particle's F
                             must not change: the grid field is uniform, so
                             F_C = 0 comes out of G2P() by itself, without the
                             teleport's forced zeroing.
  4  tissue is dragged       particles just outside the held set move with
                             the grasp. The teleport version (14.21) left
                             them behind; this is the property the change
                             exists for, so it is checked, not assumed.

Exits non-zero on any FAIL.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "host"))

import MPM.mpm3d as m          # noqa: E402  -- this import IS ti.init()
import taichi as ti            # noqa: E402
import grasp_grid_bc           # noqa: E402

FRAME_DT, N_SUB = 1 / 240, 20
m.dt = FRAME_DT / N_SUB        # baked into kernels at first compile
results = []


def report(status, name, detail):
    results.append(status)
    print(f"[{status:^6}] {name}\n         {detail}")


# Bare solver: SDF far from everything, so only P2G / gravity / domain walls act.
m.SDF = m.sdf.tmp_sdf
m.collision_mask = m.sdf.co_mask
m.SDF.fill(1e3)
m.mu[None], m.la[None] = 3000.0, 3e5
m.init_cube()
m.init_deformation_gradient()
m.F_v.fill([0.0, 0.0, 0.0])
m.F_C.fill([[0.0] * 3] * 3)
snap = [f.to_numpy() for f in (m.F_x, m.F_v, m.F_C, m.F)]


def restore():
    for f, a in zip((m.F_x, m.F_v, m.F_C, m.F), snap):
        f.from_numpy(a)


def run(fn, n):
    restore()
    for _ in range(n):
        fn()
    ti.sync()
    return m.F_x.to_numpy().astype(np.float64)


g = grasp_grid_bc.GridGraspConstraint(m, ti)
pose = np.array([0.25, 0.35, 0.12, 0, 0, 0, 1.0])
N = 60

# 1 -----------------------------------------------------------------------
a = run(lambda: m.substep(1.0, 0.02), N)
b = run(lambda: m.substep(1.0, 0.02), N)
c = run(lambda: g.substep(1.0, 0.02, pose, pose, FRAME_DT, 0.0), N)
spread, diff = float(np.abs(a - b).max()), float(np.abs(a - c).max())
if diff == 0.0:
    report("PASS", "inactive == vendored", f"bitwise identical over {N} substeps "
           f"(vendored repeat spread {spread:.1e} m)")
elif diff <= max(spread, 0.0) * 10 and spread > 0:
    report("PASS", "inactive == vendored", f"max |dx| {diff:.1e} m vs the vendored "
           f"substep's own repeat spread {spread:.1e} m over {N} substeps")
else:
    report("FAIL", "inactive == vendored", f"max |dx| {diff:.1e} m, vendored repeat "
           f"spread {spread:.1e} m: the grasp step changes an inactive substep")

# Held set: a 5 mm ball of particles at the top centre of the cloud.
restore()
x0 = snap[0].astype(np.float64)
top = x0[np.argmax(x0[:, 2])]
centre = np.array([x0[:, 0].mean(), x0[:, 1].mean(), top[2] - 0.006])
d0 = np.linalg.norm(x0 - centre, axis=1)
held = np.flatnonzero(d0 < 0.006).astype(np.int32)
# Rings OUTSIDE the BC footprint. The stencil reaches ~1.5 cells (23 mm) from
# a held particle and G2P another 1.5, so anything inside ~2 cells moves
# rigidly by construction; a ring there would pass trivially (the first draft
# of this check did exactly that). Drag is measured beyond it.
dxm = float(m.dx)
shell = np.flatnonzero((d0 > 2 * dxm) & (d0 < 3 * dxm))
shell2 = np.flatnonzero((d0 > 3 * dxm) & (d0 < 4 * dxm))
far = np.flatnonzero(d0 > 0.10)


def hold(ids):
    mask = np.zeros(m.n_particles, np.int32)
    mask[ids] = 1
    g._mask.from_numpy(mask)
    g._full_ids = ids
    g._active = True


# 2, 3, 4: lift the held set 4 mm/frame straight up for 10 frames (0.96 m/s
# is fast; it makes drag, or its absence, unmistakable in a short test).
hold(held)
p0 = np.concatenate([centre, [0, 0, 0, 1.0]])
F_before = snap[3][held].astype(np.float64)
worst = 0.0
for k in range(10):
    p1 = p0.copy()
    p1[2] += 0.004
    for i in range(N_SUB):
        g.substep(1.0, 0.02, p0, p1, FRAME_DT, i / N_SUB)
    ti.sync()
    x = m.F_x.to_numpy().astype(np.float64)
    worst = max(worst, float(np.abs((x[held] - x0[held]) - (p1[:3] - centre)).max()))
    p0 = p1
lift = 0.04
x = m.F_x.to_numpy().astype(np.float64)
F_after = m.F.to_numpy()[held].astype(np.float64)
report("PASS" if worst < 2e-4 else "FAIL", "rigid tracking",
       f"{held.size} held particles: max deviation from the gripper's rigid path "
       f"{worst*1e3:.3f} mm over a {lift*1e3:.0f} mm lift (limit 0.2 mm)")
dF = float(np.abs(F_after - F_before).max())
report("PASS" if dF < 1e-3 else "FAIL", "F follows motion",
       f"held particles' F changed by at most {dF:.2e} under pure translation "
       "(limit 1e-3; the teleport version only met this by zeroing F_C)")
shell_lift = float(np.median(x[shell, 2] - x0[shell, 2]))
shell2_lift = float(np.median(x[shell2, 2] - x0[shell2, 2]))
far_lift = float(np.median(x[far, 2] - x0[far, 2]))
status = "PASS" if shell_lift > 0.25 * lift else "WARN"
report(status, "tissue is dragged",
       f"while the grasp rose {lift*1e3:.0f} mm: ring 2-3 cells out "
       f"({shell.size} particles) rose {shell_lift*1e3:.1f} mm median, ring 3-4 cells "
       f"({shell2.size}) {shell2_lift*1e3:.1f} mm, far field (>100 mm) "
       f"{far_lift*1e3:+.1f} mm (gravity alone)")

print(f"\n{results.count('PASS')} PASS, {results.count('WARN')} WARN, "
      f"{results.count('FAIL')} FAIL")
sys.exit(1 if "FAIL" in results else 0)
