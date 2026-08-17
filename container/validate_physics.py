#!/usr/bin/env python3
"""
validate_physics.py -- controlled experiments that check the simulation is
right, not merely stable.

Run INSIDE the container:

    docker compose run --rm surrol python container/validate_physics.py

!! STATUS AS OF 17 AUGUST 2026: THIS FILE HAS NEVER BEEN EXECUTED. !!
It was written on 1 August 2026, was never saved to the repository at the time,
and was restored from the session artifact on 17 August. It targets the
session-1 world -- PyBullet mass-spring cloth, schema v1 -- so treat every
result it reports as unverified until you have actually run it once and
confirmed the harness itself works. In particular the `sim()` helper below calls
into `collect_retraction` and `make_tissue_mesh`, both of which have changed
since this was written; check those imports still resolve before trusting a
FAIL.

WHY THIS EXISTS
---------------
A timestep convergence study proves the solver is not diverging. It says nothing
about whether the thing that converged is the thing you meant to simulate. A
sheet with broken anchors, a grasp that silently slips, a stiffness parameter
wired to nothing, a mesh whose node ordering is scrambled -- all of these
produce smooth, stable, entirely plausible-looking trajectories.

The defence is to run experiments whose correct answer you know in advance from
symmetry or from physical reasoning, and check the simulator reproduces it.
That is a much stronger statement than "it didn't blow up".

Each test below states what it asserts and why that assertion must hold.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_retraction as cr  # noqa: E402
from make_tissue_mesh import build_grid, write_obj  # noqa: E402

results = []


def check(name, detail_on_pass=""):
    def wrap(fn):
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        results.append((status, name, detail))
        icon = {"PASS": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}[status]
        print(f"[{icon}] {name}\n         {detail}")
        return fn
    return wrap


# --------------------------------------------------------------------------
# A minimal, controlled harness -- deliberately NOT the full episode script.
# Fewer moving parts means a failure here points at one thing.
# --------------------------------------------------------------------------

def make_mesh(res: int, size: float, height: float, tmp: str) -> str:
    path = os.path.join(tmp, f"grid_{res}.obj")
    v, t = build_grid(size, res, height)
    write_obj(path, v, t)
    return path


def sim(mesh, stiffness=80.0, damping=0.15, dt=1 / 1000, mass=0.05,
        grasp_node=None, pull=(0.0, 0.0, 0.02), hold_s=0.4, pull_s=0.8,
        anchor_boundary=True, settle_s=0.5):
    """Pin the boundary, optionally grasp one node and displace it, return the
    node positions before and after, plus peak speed.

    The grasp here is a direct world-anchor moved by hand rather than a rigid
    body, which removes contact resolution from the picture entirely. If a test
    fails here, it is the deformable solver or the constraints -- not collision.
    """
    cid = p.connect(p.DIRECT)
    try:
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)

        sheet = p.loadSoftBody(
            mesh, basePosition=[0, 0, 0], scale=1.0, mass=mass,
            useNeoHookean=0, useMassSpring=1, useBendingSprings=1,
            springElasticStiffness=stiffness, springDampingStiffness=damping,
            springDampingAllDirections=1, useSelfCollision=0,
            frictionCoeff=0.5, useFaceContact=1)

        pos0 = cr.node_positions(sheet)
        boundary = cr.boundary_nodes(pos0)
        if anchor_boundary:
            for nid in boundary:
                p.createSoftBodyAnchor(sheet, int(nid), -1, -1)

        for _ in range(int(settle_s / dt)):
            p.stepSimulation()
        settled = cr.node_positions(sheet)

        peak_speed = 0.0
        gripper = None
        if grasp_node is not None:
            start = settled[grasp_node].copy()
            gripper = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_SPHERE, radius=0.001),
                basePosition=start.tolist())
            p.createSoftBodyAnchor(sheet, int(grasp_node), gripper, -1)

            n_pull = int(pull_s / dt)
            prev = settled
            for k in range(n_pull):
                s = (k + 1) / n_pull
                tgt = start + np.asarray(pull, np.float32) * s
                p.resetBasePositionAndOrientation(gripper, tgt.tolist(), [0, 0, 0, 1])
                p.stepSimulation()
                if k % 20 == 0:
                    cur = cr.node_positions(sheet)
                    peak_speed = max(peak_speed,
                                     float(np.linalg.norm(cur - prev, axis=1).max()
                                           / (20 * dt)))
                    prev = cur

        for _ in range(int(hold_s / dt)):
            p.stepSimulation()

        return dict(settled=settled, final=cr.node_positions(sheet),
                    boundary=boundary, peak_speed=peak_speed)
    finally:
        p.disconnect(cid)


def rot90_index(res: int) -> np.ndarray:
    """Index permutation for a 90-degree rotation of an res x res grid.

    With index = i*res + j (i along y, j along x), rotating the grid maps
    (i, j) -> (j, res-1-i).
    """
    idx = np.arange(res * res).reshape(res, res)
    return np.rot90(idx).ravel()


def mirror_index(res: int) -> np.ndarray:
    """Index permutation for reflection about the vertical centre line."""
    idx = np.arange(res * res).reshape(res, res)
    return np.fliplr(idx).ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=int, default=21,
                    help="odd, so there is a true centre node for symmetry tests")
    ap.add_argument("--size", type=float, default=0.10)
    ap.add_argument("--height", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=1 / 1000)
    ap.add_argument("--quick", action="store_true",
                    help="skip the mesh-convergence test, which is the slow one")
    args = ap.parse_args()

    if args.res % 2 == 0:
        raise SystemExit("--res must be odd so the mesh has a centre node")

    tmp = tempfile.mkdtemp(prefix="validate_")
    R = args.res
    mesh = make_mesh(R, args.size, args.height, tmp)
    centre = (R // 2) * R + (R // 2)

    print("=" * 72)
    print(f"PHYSICS VALIDATION  ({R}x{R} mesh, {args.size*100:.0f} cm, "
          f"dt = 1/{1/args.dt:.0f})")
    print("=" * 72)

    # ---------------------------------------------------------------- 1
    @check("Undisturbed sheet reaches rest and stays there")
    def _():
        # ASSERTS: with no grasp, after settling, nothing keeps moving.
        # WHY: a sheet that drifts or breathes at equilibrium has an energy
        # source it should not have -- usually a constraint fighting gravity or
        # a damping term with the wrong sign. Every later measurement would sit
        # on top of that motion.
        a = sim(mesh, dt=args.dt, settle_s=0.5)
        b = sim(mesh, dt=args.dt, settle_s=1.5)
        drift = np.abs(b["settled"] - a["settled"]).max() * 1000
        if drift > 0.5:
            return "FAIL", (f"nodes moved {drift:.3f} mm between 0.5 s and 1.5 s "
                            "of settling -- the sheet has not reached equilibrium")
        return "PASS", f"max drift {drift:.4f} mm over an extra second of settling"

    # ---------------------------------------------------------------- 2
    @check("Pinned boundary nodes do not move")
    def _():
        # ASSERTS: anchored-to-world nodes stay exactly put, even while the
        # interior is being pulled hard.
        # WHY: this is the constraint that turns dragging into retraction. If
        # the anchors silently fail, the deformation field is meaningless and
        # every downstream number is wrong -- but nothing crashes.
        r = sim(mesh, dt=args.dt, grasp_node=centre, pull=(0, 0, 0.02))
        moved = np.linalg.norm(
            r["final"][r["boundary"]] - r["settled"][r["boundary"]], axis=1).max()
        if moved * 1000 > 0.1:
            return "FAIL", (f"a pinned node moved {moved*1000:.3f} mm -- "
                            "world anchors are not holding")
        return "PASS", (f"{len(r['boundary'])} pinned nodes, max movement "
                        f"{moved*1e6:.1f} um")

    # ---------------------------------------------------------------- 3
    @check("Grasped node tracks the gripper")
    def _():
        # ASSERTS: the grasped node ends up where the gripper put it.
        # WHY: PyBullet anchors can be broken by large forces. A grasp that
        # slips halfway through produces a trajectory where the action no
        # longer explains the outcome -- which is precisely the relationship a
        # dynamics model is trying to learn.
        lift = 0.02
        r = sim(mesh, dt=args.dt, grasp_node=centre, pull=(0, 0, lift))
        expected = r["settled"][centre] + np.array([0, 0, lift], np.float32)
        err = np.linalg.norm(r["final"][centre] - expected) * 1000
        if err > 1.0:
            return "FAIL", (f"grasped node is {err:.2f} mm from where the gripper "
                            "left it -- the anchor slipped or broke")
        return "PASS", f"grasped node within {err:.3f} mm of the commanded position"

    # ---------------------------------------------------------------- 4
    @check("Deformation obeys the mesh's 4-fold rotational symmetry")
    def _():
        # ASSERTS: grasp the exact centre, pull straight up, and the resulting
        # displacement field is unchanged by a 90-degree rotation.
        # WHY: the geometry, the boundary condition and the load are all
        # symmetric under that rotation, so the solution must be too. This is a
        # correct answer known in advance without any reference simulation, and
        # it is remarkably sensitive: scrambled node ordering, an asymmetric
        # mesh, a directional bug in the solver, or an anchor applied to the
        # wrong node all break it while leaving the trajectory looking normal.
        r = sim(mesh, dt=args.dt, grasp_node=centre, pull=(0, 0, 0.02))
        d = np.linalg.norm(r["final"] - r["settled"], axis=1)
        rot = d[rot90_index(R)]
        scale = max(d.max(), 1e-12)
        err = np.abs(d - rot).max() / scale
        if err > 0.02:
            return "FAIL", (f"displacement field differs by {err*100:.1f}% of peak "
                            "under 90-degree rotation -- the setup is not symmetric")
        return "PASS", f"rotationally symmetric to {err*100:.2f}% of peak displacement"

    # ---------------------------------------------------------------- 5
    @check("Deformation obeys mirror symmetry")
    def _():
        # ASSERTS: same idea, reflection instead of rotation. Catches
        # asymmetries that happen to survive a 90-degree rotation.
        r = sim(mesh, dt=args.dt, grasp_node=centre, pull=(0, 0, 0.02))
        d = np.linalg.norm(r["final"] - r["settled"], axis=1)
        err = np.abs(d - d[mirror_index(R)]).max() / max(d.max(), 1e-12)
        if err > 0.02:
            return "FAIL", f"mirror asymmetry {err*100:.1f}% of peak displacement"
        return "PASS", f"mirror symmetric to {err*100:.2f}% of peak displacement"

    # ---------------------------------------------------------------- 6
    @check("Deformation decays with distance from the grasp point")
    def _():
        # ASSERTS: nodes near the grasp move more than nodes far from it.
        # WHY: a localised load produces a localised response that dies away
        # (Saint-Venant). If far nodes move as much as near ones, the sheet is
        # behaving like a rigid plate -- stiffness far too high for the
        # timestep, or the interior accidentally over-constrained.
        r = sim(mesh, dt=args.dt, grasp_node=centre, pull=(0, 0, 0.02))
        d = np.linalg.norm(r["final"] - r["settled"], axis=1)
        rad = np.linalg.norm(r["settled"][:, :2] - r["settled"][centre, :2], axis=1)
        near = d[rad < 0.2 * args.size].mean()
        far = d[rad > 0.4 * args.size].mean()
        if not near > far:
            return "FAIL", (f"near-field mean {near*1000:.3f} mm is not greater "
                            f"than far-field {far*1000:.3f} mm")
        return "PASS", (f"near {near*1000:.3f} mm > far {far*1000:.3f} mm "
                        f"(ratio {near/max(far,1e-12):.1f}x)")

    # ---------------------------------------------------------------- 7
    @check("Stiffness parameter changes the result monotonically")
    def _():
        # ASSERTS: a stiffer sheet transmits the pull further, so the far field
        # moves more.
        # WHY: confirms the parameter is actually wired to the solver. A
        # parameter that is being ignored -- misspelled keyword, overwritten
        # default -- gives identical results across values, and you would spend
        # weeks "tuning" something inert.
        far_means = []
        for k in (40.0, 120.0, 360.0):
            r = sim(mesh, stiffness=k, dt=args.dt, grasp_node=centre, pull=(0, 0, 0.02))
            d = np.linalg.norm(r["final"] - r["settled"], axis=1)
            rad = np.linalg.norm(r["settled"][:, :2] - r["settled"][centre, :2], axis=1)
            far_means.append(float(d[rad > 0.35 * args.size].mean()) * 1000)
        spread = (max(far_means) - min(far_means)) / max(max(far_means), 1e-12)
        txt = " -> ".join(f"{v:.3f}" for v in far_means)
        if spread < 0.02:
            return "FAIL", (f"far-field displacement barely changes across a 9x "
                            f"stiffness range ({txt} mm) -- the parameter may be "
                            "having no effect")
        if not (far_means[0] < far_means[1] < far_means[2]):
            return "WARN", (f"not monotonic in stiffness ({txt} mm). Physically "
                            "expected to increase; worth investigating.")
        return "PASS", f"far-field displacement rises with stiffness: {txt} mm"

    # ---------------------------------------------------------------- 8
    @check("Simulation is deterministic for a fixed configuration")
    def _():
        # ASSERTS: identical inputs give bit-identical outputs.
        # WHY: without determinism you cannot reproduce a bug, cannot bisect a
        # regression, and cannot attribute a change in results to a change you
        # made. It is the precondition for every other kind of debugging.
        a = sim(mesh, dt=args.dt, grasp_node=centre, pull=(0, 0, 0.02))
        b = sim(mesh, dt=args.dt, grasp_node=centre, pull=(0, 0, 0.02))
        diff = np.abs(a["final"] - b["final"]).max()
        if diff > 0:
            return "WARN", (f"two identical runs differ by {diff*1e9:.1f} nm. "
                            "Small differences usually mean threaded ordering; "
                            "large ones mean uncontrolled state.")
        return "PASS", "two identical runs produced bit-identical output"

    # ---------------------------------------------------------------- 9
    if not args.quick:
        @check("Result converges as the mesh is refined")
        def _():
            # ASSERTS: refining the mesh stops changing the answer.
            # WHY: this is the other half of the convergence story. Timestep
            # convergence shows you resolved time; mesh convergence shows you
            # resolved space. A result that keeps changing with resolution is a
            # property of your discretisation, not of the tissue -- and it also
            # means a model trained at one resolution will not transfer.
            probe = 0.3 * args.size    # sample the field at a fixed physical point
            vals = []
            for res in (11, 21, 31):
                m = make_mesh(res, args.size, args.height, tmp)
                c = (res // 2) * res + (res // 2)
                r = sim(m, dt=args.dt, grasp_node=c, pull=(0, 0, 0.02))
                d = np.linalg.norm(r["final"] - r["settled"], axis=1)
                rad = np.linalg.norm(r["settled"][:, :2] - r["settled"][c, :2], axis=1)
                # Displacement at the node nearest the probe radius -- comparing
                # by physical position, since node INDICES mean different things
                # at different resolutions.
                vals.append(float(d[np.argmin(np.abs(rad - probe))]) * 1000)
            txt = " -> ".join(f"{v:.3f}" for v in vals)
            rel = abs(vals[-1] - vals[-2]) / max(abs(vals[-1]), 1e-12)
            if rel > 0.15:
                return "WARN", (f"displacement at r={probe*100:.0f} cm still moving "
                                f"{rel*100:.0f}% between 21x21 and 31x31 ({txt} mm). "
                                "Not mesh-converged; results depend on resolution.")
            return "PASS", (f"displacement at r={probe*100:.0f} cm: {txt} mm, "
                            f"changing {rel*100:.1f}% at the finest step")

    print("=" * 72)
    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_warn = sum(1 for s, _, _ in results if s == "WARN")
    if n_fail:
        print(f"{n_fail} FAILED, {n_warn} warning(s). The simulation is not "
              "behaving correctly;\ncollected data is not trustworthy yet.")
        sys.exit(1)
    print(f"All physics checks passed ({n_warn} warning(s)).")


if __name__ == "__main__":
    main()
