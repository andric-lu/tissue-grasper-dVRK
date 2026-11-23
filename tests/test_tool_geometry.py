"""
test_tool_geometry.py -- the jaw-box distance and the D3 node-level rules.

These rules label training data AND will run on a model's own predictions at
inference (to-do Item 10), so they are tested against hand-computable cases and
against an independent brute-force reference, not just for running.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

import tool_geometry as tg  # noqa: E402

IDENT = np.array([0, 0, 0, 0, 0, 0, 1.0])
HALF = np.array([0.01, 0.02, 0.005])


def _quat_about(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


class TestRotation:
    def test_matches_a_known_rotation(self):
        R = tg.rotmat_from_quat(_quat_about([0, 0, 1], np.pi / 2))
        assert np.allclose(R @ [1, 0, 0], [0, 1, 0])

    def test_scalar_last_is_enforced_by_the_identity(self):
        # [0,0,0,1] is identity ONLY under scalar-last. Under scalar-first it
        # would be a 180 degree rotation about z.
        assert np.allclose(tg.rotmat_from_quat([0, 0, 0, 1.0]), np.eye(3))


class TestBoxDistance:
    def test_outside_face_edge_and_corner(self):
        pts = np.array([[0.03, 0, 0],            # 2 cm outside the +x face
                        [0.03, 0.05, 0],         # off an edge: (2 cm, 3 cm)
                        [0.03, 0.05, 0.015]])    # off a corner: (2, 3, 1) cm
        out = tg.box_nearest_point(pts, IDENT, HALF)
        assert np.allclose(out["signed_distance"],
                           [0.02, np.hypot(0.02, 0.03), np.linalg.norm([0.02, 0.03, 0.01])])
        assert np.allclose(out["nearest"][0], [0.01, 0, 0])
        assert np.allclose(out["normal"][0], [1, 0, 0])
        assert np.allclose(np.linalg.norm(out["normal"], axis=1), 1)

    def test_inside_is_negative_depth_through_the_nearest_face(self):
        out = tg.box_nearest_point(np.array([[0.008, 0, 0]]), IDENT, HALF)
        assert out["signed_distance"][0] == pytest.approx(-0.002)
        assert np.allclose(out["nearest"][0], [0.01, 0, 0])
        assert np.allclose(out["normal"][0], [1, 0, 0])

    def test_unsigned_matches_the_solver_formula_and_the_signed_one(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(-0.05, 0.05, (500, 3))
        pose = np.concatenate([[0.01, -0.02, 0.03], _quat_about([1, 2, 3], 0.7)])
        signed = tg.box_nearest_point(pts, pose, HALF)["signed_distance"]
        unsigned = tg.unsigned_box_distance(pts, pose, HALF)
        assert np.allclose(np.maximum(signed, 0), unsigned)

    def test_against_a_brute_force_surface_sample(self):
        """Independent reference: densely sample the box surface in its frame,
        transform to world, take the min distance. Agreement to the sampling
        pitch for points outside the box."""
        pose = np.concatenate([[0.1, 0.2, 0.3], _quat_about([0, 1, 1], 1.1)])
        g = np.linspace(-1, 1, 41)
        faces = []
        for ax in range(3):
            for s in (-1, 1):
                a, b = np.meshgrid(g, g)
                f = np.zeros((a.size, 3))
                others = [i for i in range(3) if i != ax]
                f[:, ax] = s
                f[:, others[0]], f[:, others[1]] = a.ravel(), b.ravel()
                faces.append(f * HALF)
        surf = np.concatenate(faces) @ tg.rotmat_from_quat(pose[3:]).T + pose[:3]
        rng = np.random.default_rng(1)
        pts = pose[:3] + rng.normal(0, 0.04, (200, 3))
        out = tg.box_nearest_point(pts, pose, HALF)
        brute = np.linalg.norm(pts[:, None] - surf[None], axis=-1).min(1)
        outside = out["signed_distance"] > 0
        assert outside.sum() > 100
        assert np.abs(out["signed_distance"][outside] - brute[outside]).max() < 1.5e-3

    def test_nearest_point_lies_on_the_surface(self):
        rng = np.random.default_rng(2)
        pose = np.concatenate([[0, 0, 0], _quat_about([1, 0, 1], 0.4)])
        pts = rng.uniform(-0.04, 0.04, (300, 3))
        near = tg.box_nearest_point(pts, pose, HALF)["nearest"]
        local, _ = tg.box_local(near, pose)
        on_face = np.isclose(np.abs(local), HALF, atol=1e-12).any(axis=1)
        assert on_face.all() and (np.abs(local) <= HALF + 1e-12).all()


def _two_jaws(gap):
    """Two boxes either side of the origin along y, `gap` between inner faces."""
    off = gap / 2 + HALF[1]
    return np.stack([np.concatenate([[0, +off, 0], [0, 0, 0, 1]]),
                     np.concatenate([[0, -off, 0], [0, 0, 0, 1]])]), np.stack([HALF, HALF])


class TestGraspRule:
    def test_capture_needs_both_jaws(self):
        boxes, half = _two_jaws(gap=0.01)
        pts = np.array([[0, 0, 0],          # between the jaws: 5 mm from each
                        [0, 0.05, 0]])      # beside one jaw only
        sel = tg.grasp_capture_candidates(pts, boxes, half, capture_radius=0.02)
        assert sel.tolist() == [True, False]

    def test_latch_freezes_membership_and_releases_on_the_jaw_angle(self):
        boxes, half = _two_jaws(gap=0.01)
        pts = np.array([[0, 0, 0], [0.5, 0, 0]])
        s = tg.grasp_latch_step(np.zeros(2, bool), pts, boxes, half, 0.0)
        assert s.tolist() == [True, False]
        # Node 1 walks between the jaws while held: membership does NOT grow.
        moved = np.array([[0.5, 0, 0], [0, 0, 0]])
        s = tg.grasp_latch_step(s, moved, boxes, half, 0.1)
        assert s.tolist() == [True, False]
        # Opening past the release angle clears it.
        s = tg.grasp_latch_step(s, moved, boxes, half, tg.JAW_RELEASE_TRIGGER_RAD)
        assert not s.any()

    def test_freeze_then_release_in_one_row_matches_the_solver_order(self):
        """The solver calls maybe_freeze then maybe_release on the same row; a
        row whose jaw is already open therefore never ends latched."""
        boxes, half = _two_jaws(gap=0.01)
        s = tg.grasp_latch_step(np.zeros(1, bool), np.zeros((1, 3)), boxes, half, 0.9)
        assert not s.any()

    def test_contact_flag_is_proximity_to_either_jaw(self):
        boxes, half = _two_jaws(gap=0.2)
        pts = np.array([[0, 0.1 + 0.015, 0], [0, 0, 0]])   # 15 mm from jaw 0 / far
        assert tg.node_contact_flags(pts, boxes, half, 0.02).tolist() == [True, False]

    def test_derive_over_an_episode_is_the_step_rule_applied_in_order(self):
        boxes, half = _two_jaws(gap=0.01)
        T = 6
        pos = np.zeros((T, 1, 3))
        jaw = np.array([0.0, 0.0, 0.2, 0.5, 0.0, 0.0])
        g = tg.derive_node_grasp(pos, np.repeat(boxes[None], T, 0), half, jaw)
        # latched, held, held (0.2 < 0.3), released at 0.5, re-latched after
        assert g[:, 0].tolist() == [True, True, True, False, True, True]


class TestVelocity:
    def test_backward_difference_and_zero_first_row(self):
        dt = 0.01
        T = 4
        pose = np.zeros((T, 2, 7))
        pose[..., 6] = 1
        pose[:, 0, 0] = 0.002 * np.arange(T)              # jaw 0 moves +x 2 mm/row
        q = [_quat_about([0, 0, 1], 0.05 * t) for t in range(T)]
        pose[:, 1, 3:] = q                                # jaw 1 spins about z
        v = tg.box_velocity_backward(pose, dt)
        assert np.allclose(v[0], 0)
        assert np.allclose(v[1:, 0, :3], [0.2, 0, 0])
        assert np.allclose(v[1:, 1, 3:], [0, 0, 5.0])

    def test_compose_box_pose(self):
        link = np.concatenate([[1, 0, 0], _quat_about([0, 0, 1], np.pi / 2)])
        off = np.array([0.1, 0, 0, 0, 0, 0, 1])
        box = tg.compose_box_pose(link, off)
        assert np.allclose(box[:3], [1, 0.1, 0])
        assert np.allclose(np.abs(box[3:]), np.abs(link[3:]))


class TestQuatFromRotmat:
    def test_inverse_of_rotmat_from_quat_including_near_180_degrees(self):
        rng = np.random.default_rng(0)
        qs = list(rng.normal(size=(500, 4)))
        # Near-180-degree rotations are where naive conversions break.
        qs += [np.array([1, 0, 0, 1e-9]), np.array([0, 1, 1, 1e-7]), np.array([1, 1, 1, 0.0])]
        for q in qs:
            q = np.asarray(q, float) / np.linalg.norm(q)
            q = q if q[3] >= 0 else -q
            got = tg.quat_from_rotmat(tg.rotmat_from_quat(q))
            assert np.allclose(tg.rotmat_from_quat(got), tg.rotmat_from_quat(q), atol=1e-9)
            assert got[3] >= 0
