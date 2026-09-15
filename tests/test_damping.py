"""test_damping.py -- the decay-rate algebra behind the locality study's damping levels.

src/damping.py is numpy only. These check the DERIVATIONS (that the stated mode
equations have the stated decay rates), not the MPM discretisation of them --
that is host/mpm_locality_study.py --selftest's job, on the GPU.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from damping import (  # noqa: E402
    S_DIFFUSION,
    WAVE_P,
    WAVE_S,
    kv_eta_for_p_decay,
    kv_velocity_decay_rate,
    mass_alpha_for_decay,
    p_wave_speed,
    s_wave_speed,
    viscous_stable_substep,
)

RHO, MU, LAM = 1050.0, 2000.0, 2.0e5
L = 4.0 / 64.0          # 4 grid cells: the shortest well-resolved wavelength


def _envelope_rate(a, b, c):
    """Decay rate of a u'' + b u' + c u = 0: minus the largest real part of its roots."""
    return -float(np.max(np.roots([a, b, c]).real))


class TestDecayRates:
    def test_p_wave_rate_is_the_root_of_its_mode_equation(self):
        # WHY: catches the factor-of-two slip between longitudinal (2 eta) and
        # transverse (eta) viscosity -- the kind that would silently halve or
        # double every damping level in section 13's table.
        eta = 8.0
        k2 = (2 * np.pi / L) ** 2
        truth = _envelope_rate(RHO, 2 * eta * k2, (LAM + 2 * MU) * k2)
        assert kv_velocity_decay_rate(eta, L, RHO, WAVE_P) == pytest.approx(truth, rel=1e-9)

    def test_s_wave_rate_is_half_the_p_rate(self):
        eta = 8.0
        k2 = (2 * np.pi / L) ** 2
        truth = _envelope_rate(RHO, eta * k2, MU * k2)
        assert kv_velocity_decay_rate(eta, L, RHO, WAVE_S) == pytest.approx(truth, rel=1e-9)
        assert kv_velocity_decay_rate(eta, L, RHO, WAVE_S) == pytest.approx(
            0.5 * kv_velocity_decay_rate(eta, L, RHO, WAVE_P))

    def test_zero_stiffness_shear_is_diffusion_not_half(self):
        # WHY: the GPU selftest runs at zero stiffness, where the S equation is
        # first order. Using the underdamped S rate there would predict half the
        # true decay and the check would be off by 2x before measuring anything.
        eta = 8.0
        k2 = (2 * np.pi / L) ** 2
        # rho v' + eta k^2 v = 0  ->  v ~ exp(-eta k^2 t / rho)
        assert kv_velocity_decay_rate(eta, L, RHO, S_DIFFUSION) == pytest.approx(eta * k2 / RHO)

    def test_the_overdamped_regime_is_not_what_the_formula_claims(self):
        # WHY: a boundary on the formula, shown rather than assumed. gamma = eta k^2
        # / rho is the ENVELOPE rate of an UNDERDAMPED mode. Far past critical
        # damping the slow root decays much more slowly, so the formula must not
        # be read as the decay rate there.
        eta = 1.0e4
        k2 = (2 * np.pi / L) ** 2
        slow = _envelope_rate(RHO, 2 * eta * k2, (LAM + 2 * MU) * k2)
        assert slow < 0.5 * kv_velocity_decay_rate(eta, L, RHO, WAVE_P)

    def test_eta_for_decay_inverts_the_rate(self):
        tau = 1.0 / 800.0
        eta = kv_eta_for_p_decay(tau, L, RHO)
        assert kv_velocity_decay_rate(eta, L, RHO, WAVE_P) == pytest.approx(1.0 / tau)

    def test_mass_alpha(self):
        assert mass_alpha_for_decay(0.1) == pytest.approx(10.0)
        with pytest.raises(ValueError):
            mass_alpha_for_decay(0.0)

    def test_bad_mode_is_rejected(self):
        with pytest.raises(ValueError):
            kv_velocity_decay_rate(1.0, L, RHO, "p")


class TestStability:
    def test_limit_scales_as_dx_squared_over_eta(self):
        a = viscous_stable_substep(10.0, RHO, 0.01)
        assert viscous_stable_substep(20.0, RHO, 0.01) == pytest.approx(a / 2)
        assert viscous_stable_substep(10.0, RHO, 0.02) == pytest.approx(a * 4)

    def test_explicit_diffusion_just_past_the_unsafe_bound_diverges(self):
        # WHY: the bound is derived, so demonstrate it on the thing it is derived
        # for: forward Euler of v' = D lap v on a 3-D periodic grid, D = 2 eta / rho.
        # At safety 0.9 it must stay bounded; at 1.1 the checkerboard mode grows.
        eta, dx, n = 10.0, 1.0, 8
        D = 2 * eta / RHO

        def run(safety, steps=400):
            dt = viscous_stable_substep(eta, RHO, dx, safety=safety)
            rng = np.random.default_rng(0)
            v = rng.standard_normal((n, n, n))
            for _ in range(steps):
                lap = sum(np.roll(v, s, a) for a in range(3) for s in (1, -1)) - 6 * v
                v = v + dt * D * lap / dx ** 2
            return np.abs(v).max()

        assert run(0.9) < 10.0
        assert run(1.1) > 1e6

    def test_zero_viscosity_is_unbounded(self):
        assert viscous_stable_substep(0.0, RHO, 0.01) == float("inf")


def test_wave_speeds():
    assert p_wave_speed(MU, LAM, RHO) == pytest.approx(np.sqrt((LAM + 2 * MU) / RHO))
    assert s_wave_speed(MU, RHO) == pytest.approx(np.sqrt(MU / RHO))
