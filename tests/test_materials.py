"""
test_materials.py -- the parameterisation, and the trap next to it.

The sweep tests here matter more than the round-trips. The reason (E, nu) is
not the sampling parameterisation is that lambda is singular at nu = 0.5 and
tissue sits at nu ~ 0.49; `TestPoissonSingularity` pins down both that the guard
fires and that the sensitivity it is guarding against is real.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from materials import (  # noqa: E402
    DEFAULT_LAM_RANGE,
    DEFAULT_MU_RANGE,
    DEFAULT_RHO_RANGE,
    E_nu_from_lame,
    lame_from_E_nu,
    sample_material,
    suggested_substep_dt,
    unpack_material,
)


class TestLameRoundTrip:
    @pytest.mark.parametrize("nu", [0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.49, 0.499])
    @pytest.mark.parametrize("E", [1.0e3, 1.0e4, 1.0e5])
    def test_round_trip(self, E, nu):
        mu, lam = lame_from_E_nu(E, nu)
        E_back, nu_back = E_nu_from_lame(mu, lam)
        assert E_back == pytest.approx(E, rel=1e-12)
        assert nu_back == pytest.approx(nu, rel=1e-12)

    def test_sweep_is_vectorised(self):
        nu = np.linspace(0.1, 0.49, 40)
        E = np.full_like(nu, 5.0e3)
        mu, lam = lame_from_E_nu(E, nu)
        E_back, nu_back = E_nu_from_lame(mu, lam)
        assert np.allclose(E_back, E) and np.allclose(nu_back, nu)

    def test_incompressible_limit_gives_E_near_three_mu(self):
        """E = 2*mu*(1 + nu) -> 3*mu as nu -> 0.5. This is the substitution
        suggested_substep_dt relies on."""
        mu, lam = lame_from_E_nu(1.0e4, 0.499)
        E, _ = E_nu_from_lame(mu, lam)
        assert E == pytest.approx(3.0 * mu, rel=1e-2)

    def test_reverse_round_trip_from_lame(self):
        """Round-trips only over pairs (E, nu) can actually represent -- see
        `test_default_ranges_reach_past_what_E_nu_can_represent` for why that
        is a restriction rather than an oversight."""
        rng = np.random.default_rng(0)
        mu = np.exp(rng.uniform(np.log(200), np.log(20000), 200))
        # nu = lam / (2*(lam + mu)) crosses NU_SINGULARITY_LIMIT at
        # lam/mu = 4999, so cap the ratio below that.
        lam = mu * np.exp(rng.uniform(np.log(10.0), np.log(4000.0), 200))
        E, nu = E_nu_from_lame(mu, lam)
        assert nu.max() < 0.4999
        mu_back, lam_back = lame_from_E_nu(E, nu)
        assert np.allclose(mu_back, mu, rtol=1e-9)
        assert np.allclose(lam_back, lam, rtol=1e-9)

    def test_default_ranges_reach_past_what_E_nu_can_represent(self):
        """A property of the design worth pinning down rather than discovering.

        Sampling mu and lambda independently reaches lambda/mu ratios up to
        10,000, which is nu = 0.49995 -- past the guard. Roughly 1% of draws
        land there. Two consequences:

          - Reporting a sampled material as (E, nu) is fine; the FORWARD
            direction has no singularity.
          - Converting that (E, nu) back raises, correctly. Any reporting code
            that round-trips through (E, nu) must expect it.

        This is the sampling scheme working as intended -- it reaches
        incompressibility that the (E, nu) parameterisation cannot express
        stably, which is the entire reason §2.1 rejects sampling (E, nu).
        """
        rng = np.random.default_rng(4)
        draws = np.stack([sample_material(rng) for _ in range(5000)])
        mu, lam, _ = unpack_material(draws)
        _, nu = E_nu_from_lame(mu, lam)          # forward direction: always fine
        assert np.all(np.isfinite(nu)) and np.all(nu < 0.5)
        over = nu >= 0.4999
        assert 0.001 < over.mean() < 0.05, f"{over.mean():.4f} of draws past the guard"
        with pytest.raises(ValueError, match="singular"):
            lame_from_E_nu(1.0e4, float(nu[over][0]))


class TestPoissonSingularity:
    def test_raises_at_the_limit(self):
        with pytest.raises(ValueError, match="singular"):
            lame_from_E_nu(1.0e4, 0.4999)

    def test_raises_above_the_limit(self):
        with pytest.raises(ValueError, match="singular"):
            lame_from_E_nu(1.0e4, 0.5)

    def test_message_names_the_cause(self):
        with pytest.raises(ValueError) as exc:
            lame_from_E_nu(1.0e4, 0.51)
        assert "1 - 2nu" in str(exc.value) and "0.5" in str(exc.value)

    def test_raises_on_a_vector_containing_a_bad_value(self):
        with pytest.raises(ValueError, match="singular"):
            lame_from_E_nu(np.full(5, 1e4), np.array([0.3, 0.4, 0.49, 0.4999, 0.2]))

    def test_below_minus_one_raises(self):
        with pytest.raises(ValueError, match="outside the isotropic range"):
            lame_from_E_nu(1.0e4, -1.5)

    def test_the_sensitivity_being_guarded_against_is_real(self):
        """§2.1's justification, asserted rather than asserted-in-a-comment:
        a 0.2% change in nu near the tissue regime moves lambda by ~10x."""
        _, lam_a = lame_from_E_nu(1.0e4, 0.49)
        _, lam_b = lame_from_E_nu(1.0e4, 0.499)
        assert (0.499 - 0.49) / 0.49 < 0.02          # nu barely moved
        assert lam_b / lam_a > 8.0                   # lambda moved an order of magnitude


class TestSampleMaterial:
    def test_shape_and_dtype(self):
        m = sample_material(np.random.default_rng(0))
        assert m.shape == (3,) and m.dtype == np.float32

    def test_ten_thousand_draws_stay_in_range(self):
        rng = np.random.default_rng(1)
        draws = np.stack([sample_material(rng) for _ in range(10_000)])
        mu, lam, rho = unpack_material(draws)
        assert mu.min() >= DEFAULT_MU_RANGE[0] * (1 - 1e-4)
        assert mu.max() <= DEFAULT_MU_RANGE[1] * (1 + 1e-4)
        assert lam.min() >= DEFAULT_LAM_RANGE[0] * (1 - 1e-4)
        assert lam.max() <= DEFAULT_LAM_RANGE[1] * (1 + 1e-4)
        assert rho.min() >= DEFAULT_RHO_RANGE[0] - 1e-3
        assert rho.max() <= DEFAULT_RHO_RANGE[1] + 1e-3

    def test_flat_in_log_space(self):
        """Each decade of stiffness must get the same number of episodes. Drawn
        linearly, ~90% of a two-decade range lands in the stiffest decade and
        the model never sees soft tissue."""
        rng = np.random.default_rng(2)
        draws = np.stack([sample_material(rng) for _ in range(10_000)])
        for col, (lo, hi) in ((0, DEFAULT_MU_RANGE), (1, DEFAULT_LAM_RANGE)):
            u = (draws[:, col] - np.log(lo)) / (np.log(hi) - np.log(lo))
            counts, _ = np.histogram(u, bins=10, range=(0, 1))
            # 10,000 draws in 10 bins: 1000 +/- ~32 by binomial sd, so 15% is
            # a wide band that still catches a linear (or otherwise skewed) draw.
            assert counts.min() > 850 and counts.max() < 1150, counts

    def test_a_linear_draw_would_fail_that_test(self):
        """Guards the guard: confirm the flatness test can actually detect the
        mistake it exists to detect."""
        rng = np.random.default_rng(3)
        linear = rng.uniform(*DEFAULT_MU_RANGE, size=10_000)
        u = (np.log(linear) - np.log(DEFAULT_MU_RANGE[0])) / \
            (np.log(DEFAULT_MU_RANGE[1]) - np.log(DEFAULT_MU_RANGE[0]))
        counts, _ = np.histogram(u, bins=10, range=(0, 1))
        assert counts.min() < 850 or counts.max() > 1150

    def test_reproducible_from_a_seed(self):
        a = sample_material(np.random.default_rng(7))
        b = sample_material(np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_successive_draws_differ(self):
        rng = np.random.default_rng(8)
        assert not np.array_equal(sample_material(rng), sample_material(rng))

    def test_custom_ranges_are_honoured(self):
        rng = np.random.default_rng(9)
        draws = np.stack([sample_material(rng, mu_range=(500.0, 600.0))
                          for _ in range(200)])
        mu, _, _ = unpack_material(draws)
        assert mu.min() >= 500.0 - 1e-6 and mu.max() <= 600.0 + 1e-6

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="mu_range"):
            sample_material(np.random.default_rng(0), mu_range=(0.0, 100.0))
        with pytest.raises(ValueError, match="lam_range"):
            sample_material(np.random.default_rng(0), lam_range=(100.0, 10.0))

    def test_sampled_materials_are_physically_sensible(self):
        """Every draw must convert to an (E, nu) a reviewer would recognise as
        soft tissue: near-incompressible, kilopascal-scale."""
        rng = np.random.default_rng(10)
        draws = np.stack([sample_material(rng) for _ in range(1000)])
        mu, lam, _ = unpack_material(draws)
        E, nu = E_nu_from_lame(mu, lam)
        assert np.all(nu > 0.0) and np.all(nu < 0.5)
        assert np.median(nu) > 0.4
        assert np.all(E > 0.0)


class TestUnpackMaterial:
    def test_global_material_returns_scalars(self):
        mu, lam, rho = unpack_material(np.array([np.log(1500.0), np.log(2e5), 1050.0]))
        assert mu == pytest.approx(1500.0)
        assert lam == pytest.approx(2.0e5)
        assert rho == pytest.approx(1050.0)
        assert np.ndim(mu) == 0

    def test_per_particle_material_returns_vectors(self):
        """§2.1: material is global today, per-particle later. Consumers must
        already accept (N, 3) so that becomes a data change, not a rewrite."""
        p = np.tile(np.array([np.log(1500.0), np.log(2e5), 1050.0]), (7, 1))
        mu, lam, rho = unpack_material(p)
        assert mu.shape == (7,) and lam.shape == (7,) and rho.shape == (7,)
        assert np.allclose(mu, 1500.0)

    def test_broadcasts_against_per_particle_quantities(self):
        """The shape contract that matters: (N,) material against a (T, N)
        per-particle field."""
        p = np.tile(np.array([np.log(1500.0), np.log(2e5), 1050.0]), (5, 1))
        mu, _, _ = unpack_material(p)
        assert (mu * np.ones((3, 5))).shape == (3, 5)

    def test_round_trip_through_sample(self):
        rng = np.random.default_rng(11)
        m = sample_material(rng)
        mu, lam, rho = unpack_material(m)
        assert np.log(mu) == pytest.approx(float(m[0]), rel=1e-6)
        assert np.log(lam) == pytest.approx(float(m[1]), rel=1e-6)
        assert rho == pytest.approx(float(m[2]), rel=1e-6)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(3,\) or \(N, 3\)"):
            unpack_material(np.zeros(4))

    def test_error_names_the_log_convention(self):
        """The most likely misuse is passing raw Pascals. Say so in the error."""
        with pytest.raises(ValueError) as exc:
            unpack_material(np.zeros((2, 2)))
        assert "log_mu" in str(exc.value)


class TestSuggestedSubstepDt:
    def test_scales_as_one_over_sqrt_stiffness(self):
        a = suggested_substep_dt(1000.0, 1050.0, 1e-3)
        b = suggested_substep_dt(4000.0, 1050.0, 1e-3)
        assert a / b == pytest.approx(2.0, rel=1e-9)

    def test_scales_linearly_with_cell_size(self):
        a = suggested_substep_dt(1000.0, 1050.0, 1e-3)
        b = suggested_substep_dt(1000.0, 1050.0, 2e-3)
        assert b / a == pytest.approx(2.0, rel=1e-9)

    def test_scales_as_sqrt_density(self):
        a = suggested_substep_dt(1000.0, 1000.0, 1e-3)
        b = suggested_substep_dt(1000.0, 4000.0, 1e-3)
        assert b / a == pytest.approx(2.0, rel=1e-9)

    def test_safety_factor_is_linear(self):
        a = suggested_substep_dt(1000.0, 1050.0, 1e-3, safety=0.3)
        b = suggested_substep_dt(1000.0, 1050.0, 1e-3, safety=0.6)
        assert b / a == pytest.approx(2.0, rel=1e-9)

    def test_the_stiffness_spread_the_plan_predicts(self):
        """§2.6: over a 100x stiffness range the stable step varies by 10x, so
        a single fixed substep cannot serve both ends. This is the whole reason
        substep_dt is a per-episode field."""
        soft = suggested_substep_dt(DEFAULT_MU_RANGE[0], 1050.0, 1e-3)
        stiff = suggested_substep_dt(DEFAULT_MU_RANGE[1], 1050.0, 1e-3)
        ratio = soft / stiff
        assert ratio == pytest.approx(
            np.sqrt(DEFAULT_MU_RANGE[1] / DEFAULT_MU_RANGE[0]), rel=1e-9)
        assert ratio > 3.0

    def test_vectorised_over_a_batch_of_materials(self):
        rng = np.random.default_rng(12)
        draws = np.stack([sample_material(rng) for _ in range(50)])
        mu, _, rho = unpack_material(draws)
        dt = suggested_substep_dt(mu, rho, 1e-3)
        assert dt.shape == (50,) and np.all(dt > 0) and np.all(np.isfinite(dt))

    def test_agrees_with_the_stated_formula(self):
        mu, rho, dx, safety = 1500.0, 1050.0, 1.2e-3, 0.3
        expected = safety * dx / np.sqrt(3.0 * mu / rho)
        assert suggested_substep_dt(mu, rho, dx, safety) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", [
        {"mu": 0.0, "rho": 1050.0, "dx": 1e-3},
        {"mu": 1000.0, "rho": 0.0, "dx": 1e-3},
        {"mu": 1000.0, "rho": 1050.0, "dx": -1e-3},
    ])
    def test_non_positive_inputs_raise(self, bad):
        with pytest.raises(ValueError, match="must all be positive"):
            suggested_substep_dt(**bad)

    def test_bad_safety_raises(self):
        with pytest.raises(ValueError, match="safety"):
            suggested_substep_dt(1000.0, 1050.0, 1e-3, safety=0.0)
        with pytest.raises(ValueError, match="safety"):
            suggested_substep_dt(1000.0, 1050.0, 1e-3, safety=1.5)
