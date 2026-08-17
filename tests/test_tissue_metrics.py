"""
test_tissue_metrics.py -- the metrics are only worth having if they are right.

Every case here has an answer that is known in closed form, so a failure points
at the code rather than at a tolerance someone chose. The important one is
`TestRotationInvariance`: a strain measure that reports deformation under a pure
rigid rotation is wrong, and that is the single most common bug in
large-deformation code. It is what you get from using F directly, or from the
linearised strain (F + F^T)/2 - I, which is only valid for small rotations and
is silently wrong for large ones.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tissue_metrics import (  # noqa: E402
    compute_exposure,
    compute_safety_strain,
    hard_exposure,
    invariants,
    max_principal_stretch,
    principal_stretches,
    soft_max,
    strain_energy_neohookean,
)

MU, LAM = 1500.0, 2.0e5      # representative soft-tissue-ish Lame pair


def random_rotations(n, seed=0):
    """n Haar-distributed 3x3 rotation matrices, numpy only.

    QR of a Gaussian matrix gives an orthogonal Q; fixing the signs of R's
    diagonal makes the distribution uniform over O(3), and flipping one column
    where det < 0 restricts it to SO(3). Without the determinant fix roughly
    half the samples are reflections, which are not rotations and would make
    this test pass for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, 3))
    for i in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        out[i] = q
    return out


# --------------------------------------------------------------------------
# Closed-form deformation gradients
# --------------------------------------------------------------------------

class TestRestState:
    def test_identity(self):
        F = np.eye(3)
        J, I1 = invariants(F)
        assert J == pytest.approx(1.0)
        assert I1 == pytest.approx(3.0)
        assert principal_stretches(F) == pytest.approx([1.0, 1.0, 1.0])
        assert max_principal_stretch(F) == pytest.approx(1.0)
        assert strain_energy_neohookean(F, MU, LAM) == pytest.approx(0.0, abs=1e-9)

    def test_batched_identity(self):
        F = np.tile(np.eye(3), (4, 7, 1, 1))
        J, I1 = invariants(F)
        assert J.shape == (4, 7) and I1.shape == (4, 7)
        assert np.allclose(J, 1.0) and np.allclose(I1, 3.0)
        assert principal_stretches(F).shape == (4, 7, 3)
        assert np.allclose(strain_energy_neohookean(F, MU, LAM), 0.0, atol=1e-9)


class TestRotationInvariance:
    """The critical case. Rigid rotation is not deformation."""

    def test_energy_is_zero_under_pure_rotation(self):
        R = random_rotations(32, seed=1)
        psi = strain_energy_neohookean(R, MU, LAM)
        # Absolute tolerance scaled by mu: the energy is a difference of terms
        # of order mu, so "zero" means small relative to mu, not to 1.0.
        assert np.allclose(psi, 0.0, atol=1e-9 * MU)

    def test_stretches_are_unity_under_pure_rotation(self):
        R = random_rotations(32, seed=2)
        assert np.allclose(principal_stretches(R), 1.0, atol=1e-12)
        assert np.allclose(max_principal_stretch(R), 1.0, atol=1e-12)

    def test_invariants_under_pure_rotation(self):
        R = random_rotations(32, seed=3)
        J, I1 = invariants(R)
        assert np.allclose(J, 1.0, atol=1e-12)
        assert np.allclose(I1, 3.0, atol=1e-12)

    def test_linearised_strain_would_have_failed(self):
        """Guards the guard: confirm these rotations are large enough that the
        naive measure really does break, so a passing test above means the code
        is right rather than that the rotations were negligible."""
        R = random_rotations(32, seed=4)
        naive = 0.5 * (R + np.swapaxes(R, -1, -2)) - np.eye(3)
        assert np.abs(naive).max() > 0.1

    def test_rotation_composed_with_stretch(self):
        """Stretch then rotate: R @ U must report U's stretches exactly."""
        R = random_rotations(16, seed=5)
        U = np.diag([1.3, 0.9, 1.0 / (1.3 * 0.9)])
        F = R @ U
        assert np.allclose(np.sort(principal_stretches(F), axis=-1),
                           np.sort(np.diag(U)))
        assert np.allclose(invariants(F)[0], 1.0)


class TestIsotropicDilation:
    @pytest.mark.parametrize("c", [0.8, 1.0, 1.25, 2.0])
    def test_dilation(self, c):
        F = np.diag([c, c, c])
        J, I1 = invariants(F)
        assert J == pytest.approx(c ** 3)
        assert I1 == pytest.approx(3 * c * c)
        assert principal_stretches(F) == pytest.approx([c, c, c])
        assert max_principal_stretch(F) == pytest.approx(c)


class TestIsochoricUniaxial:
    @pytest.mark.parametrize("s", [1.05, 1.2, 1.4, 2.0])
    def test_volume_preserved_and_max_stretch(self, s):
        F = np.diag([s, 1 / np.sqrt(s), 1 / np.sqrt(s)])
        J, _ = invariants(F)
        assert J == pytest.approx(1.0)
        assert max_principal_stretch(F) == pytest.approx(s)
        assert principal_stretches(F) == pytest.approx(
            sorted([s, 1 / np.sqrt(s), 1 / np.sqrt(s)]))

    def test_energy_grows_with_stretch(self):
        s = np.array([1.0, 1.1, 1.2, 1.4, 1.8])
        F = np.zeros((s.size, 3, 3))
        F[:, 0, 0] = s
        F[:, 1, 1] = F[:, 2, 2] = 1 / np.sqrt(s)
        psi = strain_energy_neohookean(F, MU, LAM)
        assert psi[0] == pytest.approx(0.0, abs=1e-9 * MU)
        assert np.all(np.diff(psi) > 0)


class TestSimpleShear:
    @pytest.mark.parametrize("g", [0.05, 0.3, 1.0, 2.0])
    def test_shear(self, g):
        F = np.array([[1.0, g, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        J, I1 = invariants(F)
        assert J == pytest.approx(1.0)
        assert I1 == pytest.approx(3.0 + g * g)
        expected = g / 2 + np.sqrt(1 + g * g / 4)
        assert max_principal_stretch(F) == pytest.approx(expected)
        # The other two: the reciprocal of the largest, and 1 out of plane.
        assert principal_stretches(F) == pytest.approx(
            sorted([expected, 1.0 / expected, 1.0]))


class TestInvertedElement:
    def test_negative_determinant_raises(self):
        """J <= 0 is an inverted element, not a deformation. Returning NaN and
        letting it reach the loss wastes the run."""
        F = np.diag([1.0, 1.0, -1.0])
        with pytest.raises(ValueError, match="inverted element"):
            strain_energy_neohookean(F, MU, LAM)


class TestMaterialBroadcasting:
    def test_per_particle_material_matches_global(self):
        """§2.1: material is global today and per-particle later. A function
        that only accepts a scalar makes that a rewrite instead of a data change."""
        F = np.tile(np.diag([1.2, 1.0, 1.0 / 1.2]), (5, 1, 1))
        glob = strain_energy_neohookean(F, MU, LAM)
        per = strain_energy_neohookean(F, np.full(5, MU), np.full(5, LAM))
        assert per.shape == (5,)
        assert np.allclose(glob, per)

    def test_per_particle_material_actually_varies(self):
        F = np.tile(np.diag([1.2, 1.0, 1.0 / 1.2]), (3, 1, 1))
        psi = strain_energy_neohookean(F, np.array([100.0, 1000.0, 10000.0]), LAM)
        assert np.all(np.diff(psi) > 0)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

class TestSoftMax:
    """Assert the invariants, not a value: the point of a soft max is where it
    sits relative to mean and max, and how it moves with p."""

    @pytest.mark.parametrize("p", [1.0, 2.0, 4.0, 8.0, 16.0, 64.0])
    def test_bracketed_by_mean_and_max(self, p):
        rng = np.random.default_rng(7)
        x = np.abs(rng.normal(size=(20, 500))) + 1e-3
        s = soft_max(x, p=p, axis=-1)
        assert s.shape == (20,)
        assert np.all(s >= x.mean(axis=-1) - 1e-12)
        assert np.all(s <= x.max(axis=-1) + 1e-12)

    def test_monotone_non_decreasing_in_p(self):
        rng = np.random.default_rng(8)
        x = np.abs(rng.normal(size=(10, 300))) + 1e-3
        ps = [1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        vals = np.stack([soft_max(x, p=p, axis=-1) for p in ps])
        assert np.all(np.diff(vals, axis=0) >= -1e-12)

    def test_p_equals_one_is_the_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert soft_max(x, p=1.0) == pytest.approx(x.mean())

    def test_converges_towards_max(self):
        x = np.array([1.0, 1.0, 1.0, 3.0])
        assert soft_max(x, p=256.0) == pytest.approx(3.0, rel=1e-2)

    def test_constant_input_equals_that_constant(self):
        x = np.full(50, 1.7)
        assert soft_max(x, p=8.0) == pytest.approx(1.7)

    def test_all_zeros_does_not_divide_by_zero(self):
        assert soft_max(np.zeros(10), p=8.0) == pytest.approx(0.0)

    def test_large_values_do_not_overflow(self):
        """A diverged solver produces stretches in the thousands. x**8 there
        overflows float64 unless the max is factored out first."""
        x = np.array([1.0, 5.0e4, 2.0])
        s = soft_max(x, p=8.0)
        assert np.isfinite(s) and s <= x.max() + 1e-6

    def test_axis_argument(self):
        x = np.arange(1.0, 13.0).reshape(3, 4)
        assert soft_max(x, p=8.0, axis=0).shape == (4,)
        assert soft_max(x, p=8.0, axis=1).shape == (3,)

    def test_negative_input_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            soft_max(np.array([-1.0, 2.0]), p=8.0)


class TestSafetyStrain:
    def test_rest_state(self):
        F = np.tile(np.eye(3), (200, 1, 1))
        out = compute_safety_strain(F)
        assert out["soft"] == pytest.approx(1.0)
        assert out["max"] == pytest.approx(1.0)
        assert out["n_above_threshold"] == 0

    def test_soft_is_bracketed_and_max_is_exact(self):
        rng = np.random.default_rng(11)
        s = 1.0 + 0.5 * rng.random(400)
        F = np.zeros((400, 3, 3))
        F[:, 0, 0] = s
        F[:, 1, 1] = F[:, 2, 2] = 1 / np.sqrt(s)
        out = compute_safety_strain(F)
        assert out["max"] == pytest.approx(s.max())
        assert s.mean() <= out["soft"] <= out["max"] + 1e-12

    def test_counts_particles_over_threshold(self):
        s = np.array([1.0, 1.2, 1.6, 2.0])
        F = np.zeros((4, 3, 3))
        F[:, 0, 0] = s
        F[:, 1, 1] = F[:, 2, 2] = 1 / np.sqrt(s)
        assert compute_safety_strain(F, threshold=1.5)["n_above_threshold"] == 2

    def test_time_axis_is_vectorised(self):
        F = np.tile(np.eye(3), (30, 400, 1, 1))
        out = compute_safety_strain(F)
        assert out["soft"].shape == (30,)
        assert out["max"].shape == (30,)
        assert out["n_above_threshold"].shape == (30,)

    def test_rotation_does_not_register_as_strain(self):
        """The end-to-end version of the rotation test, through the metric the
        planner will actually read."""
        R = random_rotations(64, seed=12)
        out = compute_safety_strain(R[None])
        assert out["soft"] == pytest.approx(1.0, abs=1e-9)
        assert out["max"] == pytest.approx(1.0, abs=1e-9)
        assert out["n_above_threshold"] == 0


# --------------------------------------------------------------------------
# Exposure
# --------------------------------------------------------------------------

ORIGIN = np.zeros(3)
NORMAL = np.array([0.0, 0.0, 1.0])
EXTENT = np.array([0.01, 0.01])


def slab(centre_xy, n=400, half=0.015, z=0.005):
    """A flat square sheet of particles at height `z`, centred on `centre_xy`."""
    k = int(np.sqrt(n))
    a = np.linspace(-half, half, k)
    xx, yy = np.meshgrid(a, a, indexing="ij")
    return np.stack([xx.ravel() + centre_xy[0],
                     yy.ravel() + centre_xy[1],
                     np.full(k * k, z)], axis=-1)


class TestExposure:
    """Assert ordering and monotonicity, not absolute values. Monotonicity is
    what an MPC cost actually consumes -- a planner needs the gradient to point
    the right way, not the number to be calibrated."""

    def test_covered_target_is_near_zero(self):
        e = compute_exposure(slab([0.0, 0.0]), ORIGIN, NORMAL, EXTENT)
        assert e < 0.05

    def test_distant_tissue_is_near_one(self):
        e = compute_exposure(slab([0.5, 0.5]), ORIGIN, NORMAL, EXTENT)
        assert e > 0.95

    def test_ordering(self):
        near = compute_exposure(slab([0.0, 0.0]), ORIGIN, NORMAL, EXTENT)
        mid = compute_exposure(slab([0.02, 0.0]), ORIGIN, NORMAL, EXTENT)
        far = compute_exposure(slab([0.5, 0.0]), ORIGIN, NORMAL, EXTENT)
        assert near < mid < far

    def test_monotone_over_a_translation_sweep(self):
        offsets = np.linspace(0.0, 0.08, 14)
        vals = np.array([compute_exposure(slab([d, 0.0]), ORIGIN, NORMAL, EXTENT)
                         for d in offsets])
        assert np.all(np.diff(vals) > -1e-9), vals
        assert vals[-1] - vals[0] > 0.9

    def test_particles_behind_the_target_do_not_occlude(self):
        """A particle below the target plane is not between the camera and the
        target. If this fails, exposure cannot distinguish lifting tissue clear
        from pushing it through the table."""
        above = compute_exposure(slab([0.0, 0.0], z=+0.005), ORIGIN, NORMAL, EXTENT)
        below = compute_exposure(slab([0.0, 0.0], z=-0.005), ORIGIN, NORMAL, EXTENT)
        assert above < 0.05
        assert below > 0.95

    def test_distance_along_normal_does_not_matter(self):
        """Only the side matters: occlusion is a line-of-sight question."""
        a = compute_exposure(slab([0.0, 0.0], z=0.002), ORIGIN, NORMAL, EXTENT)
        b = compute_exposure(slab([0.0, 0.0], z=0.200), ORIGIN, NORMAL, EXTENT)
        assert a == pytest.approx(b)

    def test_empty_tissue_is_fully_exposed(self):
        """Zero particles is a real case -- every particle on the far side of
        the target is total success, not an error."""
        e = compute_exposure(np.zeros((0, 3)), ORIGIN, NORMAL, EXTENT)
        # A sigmoid never reaches its asymptote, so the smooth version tops out
        # just under 1. The hard count has no such excuse.
        assert e > 0.99
        assert hard_exposure(np.zeros((0, 3)), ORIGIN, NORMAL, EXTENT) == 1.0

    def test_time_axis_matches_per_frame_calls(self):
        frames = np.stack([slab([d, 0.0]) for d in np.linspace(0, 0.06, 9)])
        batched = compute_exposure(frames, ORIGIN, NORMAL, EXTENT)
        assert batched.shape == (9,)
        single = np.array([compute_exposure(f, ORIGIN, NORMAL, EXTENT) for f in frames])
        assert np.allclose(batched, single)

    def test_chunking_does_not_change_the_answer(self):
        frames = np.stack([slab([d, 0.0]) for d in np.linspace(0, 0.06, 9)])
        a = compute_exposure(frames, ORIGIN, NORMAL, EXTENT, chunk=1)
        b = compute_exposure(frames, ORIGIN, NORMAL, EXTENT, chunk=64)
        assert np.allclose(a, b)

    def test_normal_need_not_be_unit_length(self):
        a = compute_exposure(slab([0.02, 0.0]), ORIGIN, NORMAL, EXTENT)
        b = compute_exposure(slab([0.02, 0.0]), ORIGIN, NORMAL * 37.0, EXTENT)
        assert a == pytest.approx(b)

    def test_result_is_a_fraction(self):
        for d in np.linspace(0.0, 0.1, 11):
            e = compute_exposure(slab([d, 0.0]), ORIGIN, NORMAL, EXTENT)
            assert 0.0 <= e <= 1.0

    def test_tilted_target_plane(self):
        """The metric must not be secretly hardcoded to a z-normal target.

        Note the tissue is moved WITHIN the tilted plane. Moving it along the
        normal would prove nothing -- distance along the normal is deliberately
        ignored, so that translation would pass even on a z-hardcoded metric.
        """
        n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        e1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        e2 = np.cross(n, e1)
        flat = slab([0.0, 0.0], z=0.0)                    # (N,3), z == 0
        # Re-express the sheet in the tilted plane, lifted onto the occluding side.
        tilted = flat[:, 0:1] * e1 + flat[:, 1:2] * e2 + 0.005 * n
        assert compute_exposure(tilted, ORIGIN, n, EXTENT) < 0.05
        assert compute_exposure(tilted + 0.5 * e1, ORIGIN, n, EXTENT) > 0.95

    def test_scene_rotation_invariance(self):
        """Exposure is geometric: rotating tissue, target origin and target
        normal together must leave it unchanged. Only the extremes are asserted
        -- for a partially covered target, rotating the normal also rotates the
        target rectangle within its own plane, so the covered fraction really
        does change and equality would be the wrong expectation."""
        R = random_rotations(4, seed=21)
        for r in R:
            covered = compute_exposure(slab([0.0, 0.0]) @ r.T, r @ ORIGIN,
                                       r @ NORMAL, EXTENT)
            clear = compute_exposure(slab([0.5, 0.5]) @ r.T, r @ ORIGIN,
                                     r @ NORMAL, EXTENT)
            assert covered < 0.05
            assert clear > 0.95


class TestHardExposure:
    def test_agrees_with_smooth_version_at_the_extremes(self):
        for centre, lo, hi in ([0.0, 0.0], 0.0, 0.05), ([0.5, 0.5], 0.95, 1.0):
            h = hard_exposure(slab(centre), ORIGIN, NORMAL, EXTENT)
            assert lo <= h <= hi

    def test_takes_only_the_grid_resolution_of_values(self):
        """The hard count is k/grid^2 exactly -- no smoothing parameter in it."""
        h = hard_exposure(slab([0.015, 0.0]), ORIGIN, NORMAL, EXTENT, grid=16)
        assert (h * 256) == pytest.approx(round(h * 256))

    def test_smooth_tracks_hard_across_a_sweep(self):
        offsets = np.linspace(0.0, 0.08, 14)
        soft = np.array([compute_exposure(slab([d, 0]), ORIGIN, NORMAL, EXTENT)
                         for d in offsets])
        hard = np.array([hard_exposure(slab([d, 0]), ORIGIN, NORMAL, EXTENT)
                         for d in offsets])
        assert np.abs(soft - hard).max() < 0.15
        assert np.all(np.diff(hard) > -1e-9)


class TestExposureInputValidation:
    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(\.\.\., N, 3\)"):
            compute_exposure(np.zeros((10, 2)), ORIGIN, NORMAL, EXTENT)

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError, match="zero length"):
            compute_exposure(slab([0, 0]), ORIGIN, np.zeros(3), EXTENT)

    def test_non_positive_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            compute_exposure(slab([0, 0]), ORIGIN, NORMAL, EXTENT, sigma=0.0)
