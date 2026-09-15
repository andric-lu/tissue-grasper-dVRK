"""damping.py -- the arithmetic behind the two damping forms host/mpm_energy.py adds.

numpy only, so it imports on both sides (CLAUDE.md, `src/` rule) and its
derivations are unit tested without a Metal device. The kernels themselves are
`Instrument.set_damping()` in host/mpm_energy.py; this file answers "what eta
produces a given decay time" and "what substep keeps that eta stable".

WHY DAMPING IS ON THE TABLE AT ALL (DECISION_LOG.md section 13)
---------------------------------------------------------------
An undamped Neo-Hookean solid rings. A pressure wave from a local kick crosses
the 300 mm slab within one recorded step at every material in materials.py's
range, so a graph network with a finite receptive field cannot represent the
response. Real tissue is viscoelastic and heavily damped; if damping makes the
observable response at the recording rate LOCAL, it improves the physics and the
learnability at once. host/mpm_locality_study.py measures whether it does.

THE TWO FORMS
-------------
Kelvin-Voigt viscous stress, the form tissue actually has:

    tau_v = eta * (grad v + grad v^T)            eta in Pa*s

Its divergence is eta * (lap v + grad div v), so a LONGITUDINAL mode sees an
effective viscosity 2*eta and a TRANSVERSE mode sees eta. For one Fourier mode
of wavenumber k = 2*pi / wavelength, the displacement amplitude obeys

    P wave:  rho u'' + 2 eta k^2 u' + (lambda + 2 mu) k^2 u = 0
    S wave:  rho u'' +   eta k^2 u' +            mu k^2 u = 0

and in the underdamped case the envelope decays at gamma = (coefficient of u')
/ (2 rho):

    gamma_P = eta k^2 / rho          gamma_S = eta k^2 / (2 rho)

With the elastic moduli set to zero the S equation is first order in velocity,
pure diffusion, and v decays at eta k^2 / rho -- twice the underdamped S rate.
host/mpm_locality_study.py --selftest uses that zero-stiffness case, so all
three rates are named separately here rather than one being reused for another.

Short wavelengths decay fastest (k^2), which is what makes Kelvin-Voigt
localising: the sharp front of a local kick is carried by high k.

Mass-proportional damping, the comparison:

    v *= exp(-alpha * dt) every substep          alpha in 1/s

decays every mode, including rigid motion, at alpha. Not physical for tissue --
a grasp lifting the slab would feel drag toward zero velocity -- but the
simplest possible term, and a useful contrast.

STABILITY
---------
Explicit viscosity is a diffusion of velocity with diffusivity up to
D = 2 eta / rho (the longitudinal case). Forward Euler on a 3-D grid of spacing
dx is stable for dt <= dx^2 / (6 D) = rho dx^2 / (12 eta). The same `safety`
factor the elastic CFL bound uses (materials.suggested_substep_dt, 0.3) is
applied on top, because MPM's B-spline transfers are not the textbook
7-point Laplacian this bound is derived for.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray

WAVE_P = "p_wave"             # underdamped longitudinal
WAVE_S = "s_wave"             # underdamped transverse
S_DIFFUSION = "s_diffusion"   # transverse with zero stiffness: pure diffusion
MODES = (WAVE_P, WAVE_S, S_DIFFUSION)


def p_wave_speed(mu: ArrayLike, lam: ArrayLike, rho: ArrayLike) -> np.ndarray:
    """sqrt((lambda + 2 mu) / rho) -- the wave that sets both the CFL bound (section 4)
    and how far a kick travels in one recorded step (section 13)."""
    return np.sqrt((np.asarray(lam, np.float64) + 2.0 * np.asarray(mu, np.float64))
                   / np.asarray(rho, np.float64))


def s_wave_speed(mu: ArrayLike, rho: ArrayLike) -> np.ndarray:
    return np.sqrt(np.asarray(mu, np.float64) / np.asarray(rho, np.float64))


def wavenumber(wavelength: float) -> float:
    if wavelength <= 0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    return 2.0 * np.pi / wavelength


def kv_velocity_decay_rate(eta: float, wavelength: float, rho: float,
                           mode: str = WAVE_P) -> float:
    """Decay rate (1/s) of a single mode's amplitude under Kelvin-Voigt viscosity.
    Kinetic energy decays at twice this."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if eta < 0 or rho <= 0:
        raise ValueError(f"need eta >= 0 and rho > 0, got eta={eta}, rho={rho}")
    k2 = wavenumber(wavelength) ** 2
    if mode == WAVE_S:
        return eta * k2 / (2.0 * rho)
    return eta * k2 / rho                     # P wave, and S with zero stiffness


def kv_eta_for_p_decay(tau: float, wavelength: float, rho: float) -> float:
    """eta (Pa*s) at which a P-wave of this wavelength decays with e-folding time tau."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return rho / (tau * wavenumber(wavelength) ** 2)


def mass_alpha_for_decay(tau: float) -> float:
    """alpha (1/s) for an e-folding time tau. Wavelength-independent by construction."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return 1.0 / tau


def viscous_stable_substep(eta: float, rho: float, dx: float,
                           safety: float = 0.3) -> float:
    """Largest stable substep (s) for explicit Kelvin-Voigt viscosity. inf at eta = 0."""
    if rho <= 0 or dx <= 0 or safety <= 0:
        raise ValueError(f"need rho, dx, safety > 0, got {rho}, {dx}, {safety}")
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    if eta == 0:
        return float("inf")
    return safety * rho * dx * dx / (12.0 * eta)
