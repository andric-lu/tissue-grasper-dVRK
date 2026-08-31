"""test_mpm_energy.py -- the energy ledger is only worth having if it is closed.

These test src/energy_ledger.py, which is numpy only, so they run in the main
environment alongside the other 249 and never import taichi. The Taichi half of
the instrument (host/mpm_energy.py) cannot be unit tested this way -- importing
it runs ti.init() and needs a Metal device -- so its checks live in
`host/energy_audit.py --selftest` instead, which is the same split
host/smoke_test_mpm.py already uses.

THE ONE THAT MATTERS IS `TestStressIsTheDerivativeOfPsi`. The whole section 9.7
audit rests on the claim that `tissue_metrics.strain_energy_neohookean` is the
potential whose Kirchhoff stress is the expression at mpm3d.py:165. If that is
false, every joule the ledger reports is measuring a different material than the
solver is simulating, and the resulting verdict about dissipation would be an
artifact of the mismatch. The claim was made by reading the two expressions and
noticing they were conjugate; this differentiates one numerically and checks.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from energy_ledger import (  # noqa: E402
    QUADRATIC_D_COEFF,
    affine_kinetic_energy,
    elastic_energy,
    energy_ledger,
    fit_scaling_exponent,
    gravitational_pe,
    kinetic_energy,
    kirchhoff_stress,
    normalisation_collapse,
    verdict,
)
from tissue_metrics import strain_energy_neohookean  # noqa: E402

MU, LAM = 3758.0, 69279.0          # the soft-lambda material from data_mpm/
DX = 1.0 / 64.0                    # MPM/config.py
P_VOL = (DX * 0.5) ** 3            # mpm3d.py:40
P_MASS = P_VOL * 1004.1


def _random_F(rng, n=8, scale=0.08):
    """Deformation gradients near identity, all with det > 0."""
    F = np.eye(3) + scale * rng.standard_normal((n, 3, 3))
    assert np.all(np.linalg.det(F) > 0)
    return F


class TestStressIsTheDerivativeOfPsi:
    def test_kirchhoff_stress_matches_dpsi_dF_Ft(self):
        # WHY: catches a constitutive-model mismatch between the ledger and the
        # solver. mpm3d.py:165 uses tau = mu(FF^T - I) + lambda*log(J)*I; the
        # ledger integrates Psi = (mu/2)(I1-3) - mu*log(J) + (lambda/2)(log J)^2.
        # If those are not conjugate, the ledger measures a different material
        # than the one being simulated and every dissipation number in section 9.7
        # is meaningless. Nothing else in the repository checks this.
        rng = np.random.default_rng(7)
        F = _random_F(rng, n=6)
        h = 1e-6
        for f in F:
            dPsi = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    fp, fm = f.copy(), f.copy()
                    fp[i, j] += h
                    fm[i, j] -= h
                    dPsi[i, j] = (strain_energy_neohookean(fp[None], MU, LAM)[0]
                                  - strain_energy_neohookean(fm[None], MU, LAM)[0]) / (2 * h)
            tau_fd = dPsi @ f.T
            tau_solver = kirchhoff_stress(f[None], MU, LAM)[0]
            assert np.allclose(tau_fd, tau_solver, rtol=1e-6, atol=1e-6 * MU)

    def test_a_wrong_lambda_is_detected(self):
        # WHY: a check only ever shown passing has not been shown to work
        # (CLAUDE.md, validation discipline). Differentiate Psi at one lambda,
        # compare against the stress at another, and the identity must FAIL --
        # otherwise the test above would pass for a ledger that had drifted.
        rng = np.random.default_rng(8)
        f = _random_F(rng, n=1)[0]
        h = 1e-6
        dPsi = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                fp, fm = f.copy(), f.copy()
                fp[i, j] += h
                fm[i, j] -= h
                dPsi[i, j] = (strain_energy_neohookean(fp[None], MU, LAM)[0]
                              - strain_energy_neohookean(fm[None], MU, LAM)[0]) / (2 * h)
        wrong = kirchhoff_stress(f[None], MU, LAM * 2.0)[0]
        assert not np.allclose(dPsi @ f.T, wrong, rtol=1e-6, atol=1e-6 * MU)

    def test_stress_vanishes_at_rest_and_under_rotation(self):
        # WHY: a stress that is non-zero at F = I would inject energy from the
        # first substep, and one that is non-zero under a pure rotation would
        # make the audit's rigid-motion null test fail for a reason that has
        # nothing to do with the transfers.
        theta = 0.9
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for F in (np.eye(3), R):
            assert np.allclose(kirchhoff_stress(F[None], MU, LAM)[0], 0.0,
                               atol=1e-9 * MU)


class TestElasticEnergy:
    def test_isochoric_stretch_matches_closed_form(self):
        # WHY: this is the initial condition of the audit's primary cell, and
        # its energy is quoted in DECISION_LOG as a closed form. If the code and
        # the closed form disagree, E(0) is wrong and every dissipation is
        # measured from the wrong origin.
        s = 1.2
        A = np.diag([s, s ** -0.5, s ** -0.5])
        assert np.isclose(np.linalg.det(A), 1.0)
        n = 24000
        got = elastic_energy(np.broadcast_to(A, (n, 3, 3)), MU, LAM, P_VOL)
        want = n * P_VOL * (MU / 2.0) * (s ** 2 + 2.0 / s - 3.0)
        assert np.isclose(got, want, rtol=1e-12)

    def test_zero_at_rest(self):
        # WHY: a non-zero energy at F = I makes every ledger total carry a
        # constant offset, which survives differencing but breaks any claim
        # about the ABSOLUTE energy of the initial condition.
        assert elastic_energy(np.broadcast_to(np.eye(3), (10, 3, 3)),
                              MU, LAM, P_VOL) == 0.0

    def test_scales_with_reference_volume(self):
        # WHY: Psi is an energy DENSITY per unit REFERENCE volume. Multiplying
        # by the deformed volume, or forgetting p_vol entirely, is a silent
        # error of a factor of ~2.5 here and it would not look wrong.
        rng = np.random.default_rng(3)
        F = _random_F(rng)
        assert np.isclose(elastic_energy(F, MU, LAM, 2 * P_VOL),
                          2 * elastic_energy(F, MU, LAM, P_VOL))


class TestAffineKineticEnergy:
    def test_matches_direct_quadrature(self):
        # WHY: the APIC affine term is the one the ledger could most plausibly
        # get wrong, and getting it wrong makes energy parked in C read as
        # dissipation -- exactly the false positive section 9.7 exists to exclude.
        # Six-point quadrature at +-a e_i with a^2 = 3 dx^2 / 4 reproduces the
        # second moment m * (dx^2/4) * I exactly, so the two must agree to
        # roundoff, not to a tolerance.
        rng = np.random.default_rng(11)
        C = rng.standard_normal((4, 3, 3))
        a = DX * np.sqrt(3.0) / 2.0
        pts = np.concatenate([np.eye(3) * a, -np.eye(3) * a])
        direct = 0.0
        for c in C:
            for y in pts:
                direct += 0.5 * (P_MASS / 6.0) * float(np.dot(c @ y, c @ y))
        assert np.isclose(direct, affine_kinetic_energy(C, P_MASS, DX), rtol=1e-12)

    def test_quadratic_bspline_second_moment_is_dx2_over_4(self):
        # WHY: QUADRATIC_D_COEFF = 1/4 is asserted from theory. This derives it
        # from the solver's OWN weight expression (mpm3d.py:153), at many
        # fractional positions, so a change of interpolation order upstream
        # cannot silently invalidate the affine term. The solver never writes
        # the constant down -- G2P only ever uses its inverse, 4*inv_dx.
        for fx0 in np.linspace(1.0, 2.0, 21):
            fx = np.array([fx0, fx0, fx0])
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            wts = np.array([w[o][0] for o in range(3)])
            offs = np.array([0.0, 1.0, 2.0]) - fx0
            assert np.isclose(wts.sum(), 1.0, atol=1e-12)          # partition of unity
            assert np.isclose((wts * offs ** 2).sum(), QUADRATIC_D_COEFF, atol=1e-12)

    def test_zero_for_zero_affine_field(self):
        # WHY: rigid translation has C = 0, and the audit's null test requires
        # the affine term to contribute exactly nothing there.
        assert affine_kinetic_energy(np.zeros((5, 3, 3)), P_MASS, DX) == 0.0


class TestLedger:
    def test_terms_sum_to_total(self):
        # WHY: "closed" is the entire claim. If total is computed any way other
        # than as the sum of the reported terms, an attribution can be wrong
        # while the total looks right.
        rng = np.random.default_rng(5)
        led = energy_ledger(rng.random((7, 3)), rng.standard_normal((7, 3)),
                            rng.standard_normal((7, 3, 3)), _random_F(rng, 7),
                            p_mass=P_MASS, p_vol=P_VOL, mu=MU, lam=LAM,
                            dx=DX, gravity=9.8)
        assert np.isclose(led["total"], led["ke"] + led["ke_affine"]
                          + led["elastic"] + led["gravity_pe"], rtol=1e-14)

    def test_gravity_off_means_no_pe_term(self):
        # WHY: the clean cell's whole design is that PE is identically constant.
        # If a stray g leaks in, a falling body would register as "energy
        # created" and the exponent fit would be nonsense.
        rng = np.random.default_rng(6)
        led = energy_ledger(rng.random((7, 3)) + 10.0, rng.standard_normal((7, 3)),
                            rng.standard_normal((7, 3, 3)), _random_F(rng, 7),
                            p_mass=P_MASS, p_vol=P_VOL, mu=MU, lam=LAM,
                            dx=DX, gravity=0.0)
        assert led["gravity_pe"] == 0.0

    def test_pe_datum_cancels_in_differences(self):
        # WHY: the datum is arbitrary and the ledger is only ever used through
        # differences. A datum that failed to cancel would put a constant into
        # every dissipation number.
        rng = np.random.default_rng(9)
        x0, x1 = rng.random((7, 3)), rng.random((7, 3))
        for datum in (0.0, 0.5, -3.0):
            d = (gravitational_pe(x1, P_MASS, 9.8, datum=datum)
                 - gravitational_pe(x0, P_MASS, 9.8, datum=datum))
            assert np.isclose(d, gravitational_pe(x1, P_MASS, 9.8)
                              - gravitational_pe(x0, P_MASS, 9.8))

    def test_pe_uses_z_not_y(self):
        # WHY: mpm3d.py:215 applies gravity to component [2] -- the pybullet
        # branch; the taichi Y-up line above it is commented out. Reading PE off
        # the wrong axis is the exact class of convention bug CLAUDE.md warns
        # about, and it would be invisible in a body that is wide in both.
        x = np.zeros((1, 3))
        x[0, 1] = 5.0
        assert gravitational_pe(x, P_MASS, 9.8) == 0.0
        x[0, 1], x[0, 2] = 0.0, 5.0
        assert np.isclose(gravitational_pe(x, P_MASS, 9.8), P_MASS * 9.8 * 5.0)

    def test_momentum_is_reported(self):
        # WHY: the clean cell starts at zero linear and angular momentum and no
        # term in the solver can create either, so drift is a free instrument
        # fault detector -- but only if the ledger actually carries it.
        v = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        x = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        led = energy_ledger(x, v, np.zeros((2, 3, 3)),
                            np.broadcast_to(np.eye(3), (2, 3, 3)),
                            p_mass=P_MASS, p_vol=P_VOL, mu=MU, lam=LAM, dx=DX)
        assert np.isclose(led["p_x"], 0.0)
        assert not np.isclose(led["l_z"], 0.0)      # this pair does spin

    def test_kinetic_energy_rejects_wrong_shape(self):
        # WHY: an (N,3,3) passed where (N,3) was meant would sum silently to a
        # plausible number. Shapes are checked so a mis-wiring is loud.
        with pytest.raises(ValueError):
            kinetic_energy(np.zeros((4, 3, 3)), P_MASS)


class TestVerdict:
    """The verdict logic, demonstrated against fabricated data of each kind.

    A verdict function shown only on the real run has not been shown to work --
    and by then the real run's answer is the thing in question.
    """

    @pytest.mark.parametrize("q_true,expected", [
        (-1.0, "PER_TRANSFER"),      # dissipation counts transfers
        (0.0, "RATE_LIKE"),          # dissipation is a rate in simulated time
        (1.0, "CONVERGENT"),         # ordinary truncation error
        (-0.35, "INTERMEDIATE"),     # the real cell-1 answer: neither one
        (0.27, "INTERMEDIATE"),      # the real cell-4 answer: neither one
    ])
    def test_recovers_each_mechanism(self, q_true, expected):
        # WHY: section 9.6 concluded PER_TRANSFER. If the verdict function cannot
        # also produce CONVERGENT from data that says so, the audit can only
        # confirm and never falsify, which would make it worthless.
        dt = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5])
        loss = 3.7 * dt ** q_true
        fit = fit_scaling_exponent(dt, loss)
        assert np.isclose(fit["q"], q_true, atol=1e-9)
        assert np.isclose(fit["r2"], 1.0, atol=1e-9)
        code, _ = verdict(fit["q"], fit["r2"])
        assert code == expected

    def test_intermediate_is_not_rounded_into_rate_like(self):
        # WHY: this is the mistake the first version of the verdict made, on the
        # real data. With the old +-0.5 band, a measured q = -0.34 was reported
        # as "dissipation is essentially independent of dt" -- while across that
        # ladder the decay rate had moved by a factor of 2.5. Asserting
        # independence from a number that is not independent is the same
        # overstatement section 9.6 made, in the document written to correct it.
        code, why = verdict(-0.337, 0.999, "per_time", collapse_ratio=5.0,
                            span=16.0)
        assert code == "INTERMEDIATE"
        assert "2.5x" in why and "16x" in why      # quotes the measured factor
        assert "essentially independent" not in why

    def test_collapse_note_is_silent_where_it_has_no_opinion(self):
        # WHY: the collapse diagnostic only chooses between per-transfer and
        # per-time. Letting it contradict an INTERMEDIATE verdict would
        # manufacture a disagreement out of a question it was never asked, and
        # every such spurious caveat makes the real ones easier to ignore.
        _, why = verdict(0.27, 0.99, "per_time", collapse_ratio=5.0, span=16.0)
        assert "does not agree" not in why

    def test_non_power_law_is_ambiguous_not_rounded(self):
        # WHY: an exponent read off points that are not on a power law is a
        # number, not a finding. Without this gate a noisy ladder would still
        # produce a confident-sounding verdict.
        dt = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5])
        loss = np.array([1.0, 8.0, 0.9, 30.0, 1.1])
        fit = fit_scaling_exponent(dt, loss)
        code, why = verdict(fit["q"], fit["r2"])
        assert code == "AMBIGUOUS"
        assert "power law" in why

    def test_collapse_diagnostic_agrees_with_the_exponent(self):
        # WHY: the collapse test is scale-free and needs no threshold, so it is
        # the independent check on the banded exponent. Section 9.6's postmortem
        # records that a magnitude cutoff gave one material the wrong advice.
        dt = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4])
        col = normalisation_collapse(dt, 3.0 * dt ** -1.0, total_time=0.75)
        assert col["collapses"] == "per_transfer"
        assert np.isclose(col["spread_per_transfer"], 1.0)
        col = normalisation_collapse(dt, np.full_like(dt, 3.0), total_time=0.75)
        assert col["collapses"] == "per_time"

    def test_flat_series_is_read_as_rate_like_not_ambiguous(self):
        # WHY: the real zero-stiffness ladder produces a loss that is constant
        # to 0.2% across a factor of 64 in dt. That is the STRONGEST possible
        # rate-like signal, and it drives r2 to nonsense (a constant series has
        # no variance to explain), so the r2 gate alone would file the clearest
        # result in the study as "ambiguous". The fallback reads the scale-free
        # collapse instead -- but only when the collapse is decisive.
        dt = np.array([3.125e-3, 7.8125e-4, 1.953e-4, 4.883e-5])
        loss = np.array([3.77957e-3, 3.78365e-3, 3.78514e-3, 3.78560e-3])
        fit = fit_scaling_exponent(dt, loss)
        col = normalisation_collapse(dt, loss, total_time=0.05)
        assert fit["r2"] < 0.9                      # the gate really does trip
        code, why = verdict(fit["q"], fit["r2"], col["collapses"],
                            collapse_ratio=col["collapse_ratio"])
        assert code == "RATE_LIKE"
        assert "CAUTION" in why          # stands, but says it is thin evidence

    def test_degenerate_fit_annotates_but_does_not_relabel(self):
        # WHY: the real cell-3 stiff ladder is non-monotone (r2 = 0.67) with
        # q = +0.24. The first fallback returned RATE_LIKE for it purely because
        # the collapse diagnostic preferred "per time" -- relabelling a
        # measurably dt-dependent result as "independent of dt to within a few
        # percent". The gate may now caveat a verdict or withdraw it; it may not
        # swap in a different one.
        code, why = verdict(0.238, 0.674, "per_time", collapse_ratio=16.0,
                            span=16.0)
        assert code == "INTERMEDIATE"
        assert "CAUTION" in why and "r2 = 0.674" in why
        assert "independent of dt" not in why

    def test_indecisive_collapse_stays_ambiguous(self):
        # WHY: the fallback must not become a way to always get an answer. When
        # neither normalisation is clearly tighter, "we did not settle it" is
        # the honest output -- and a check only shown passing has not been shown
        # to work, so the refusal is demonstrated too.
        code, _ = verdict(-0.3, 0.2, "per_time", collapse_ratio=1.2)
        assert code == "AMBIGUOUS"

    def test_collapse_ratio_reports_decisiveness(self):
        # WHY: the ratio is what gates the fallback above, so it has to mean
        # what the fallback assumes -- how many times tighter the winner is.
        dt = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4])
        col = normalisation_collapse(dt, np.full_like(dt, 3.0), total_time=0.75)
        assert np.isclose(col["spread_per_time"], 1.0)
        assert np.isclose(col["collapse_ratio"], col["spread_per_transfer"])

    def test_disagreement_is_surfaced_not_hidden(self):
        # WHY: if the two independent diagnostics disagree, the run has not
        # settled the question and saying so is the honest output.
        _, why = verdict(-1.0, 1.0, collapse="per_time")
        assert "provisional" in why

    def test_energy_creation_is_not_averaged_away(self):
        # WHY: a negative measured dissipation means the solver CREATED energy,
        # which would be a real finding about the vendored code. Dropping such
        # rows silently would hide it; they are dropped from the log-log fit
        # (log of a negative is undefined) but counted so the caller can see.
        dt = np.array([1e-3, 5e-4, 2.5e-4])
        fit = fit_scaling_exponent(dt, np.array([1.0, -0.5, 0.25]))
        assert fit["n_dropped"] == 1
        assert fit["n"] == 2
