"""
materials.py -- how a material becomes three numbers the model can consume.

WHY THIS FILE EXISTS
--------------------
Domain randomisation over tissue stiffness is what makes a learned dynamics
model transfer to tissue it has not seen. Doing that badly is easy, and the two
ways it goes wrong both look fine until the dataset is already collected.

1. SAMPLING (E, nu) UNIFORMLY IS A TRAP.

       lambda = E*nu / ((1 + nu)(1 - 2nu))

   is singular at nu = 0.5, and soft tissue sits at nu ~ 0.49 -- right next to
   the singularity. nu = 0.49 and nu = 0.499 differ by 0.2% and give lambda
   values roughly 10x apart. A uniform sweep over nu is therefore a wildly
   non-uniform, heavy-tailed sweep over the quantity the solver actually uses.
   So mu and lambda are sampled DIRECTLY, log-uniformly. Conversions to and
   from (E, nu) still exist, because that is the parameterisation every paper
   and every indentation test reports.

2. PASCALS ARE THE WRONG UNITS FOR A NETWORK INPUT.

   Soft tissue shear modulus spans orders of magnitude. Fed raw Pascals, a
   network spends its capacity encoding the exponent. The stored representation
   is therefore [log(mu), log(lambda), rho] -- logs for the two that span
   decades, linear for density, which varies by a few percent and whose log
   would be a near-constant.

A NOTE ON SHAPE
---------------
Material is one global triple per episode today. It will be per-particle when
heterogeneous tissue arrives. Every function here that CONSUMES material accepts
(3,) or (N, 3) and broadcasts, so that change is a data-generation change rather
than a rewrite. `sample_material` returns (3,) because that is what the schema
stores; nothing else assumes it.

numpy only, and it must import under both numpy 1.23 (the container pin) and
numpy 2.x (the host env).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, float]

# Poisson's ratio is bounded above by 0.5 for an isotropic material, and lambda
# diverges there. Refusing to convert above this cutoff is not pedantry: 0.4999
# already gives lambda ~ 3300x mu, and the next digit changes it by another
# order of magnitude. If you want that regime, sample lambda directly.
NU_SINGULARITY_LIMIT = 0.4999

# PLACEHOLDER RANGES. These are order-of-magnitude stand-ins for generic soft
# tissue and nothing more. Before any result is claimed from a model trained on
# them, replace them with values fitted to indentation or uniaxial tension
# curves for the specific tissue type being retracted -- liver, bowel and fat
# differ from each other by more than the width of these ranges.
DEFAULT_MU_RANGE = (200.0, 20000.0)      # Pa, shear modulus
DEFAULT_LAM_RANGE = (2.0e4, 2.0e6)       # Pa, first Lame parameter
DEFAULT_RHO_RANGE = (1000.0, 1100.0)     # kg/m^3, near water, as tissue is


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------

def lame_from_E_nu(E: ArrayLike, nu: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """(Young's modulus, Poisson ratio) -> (mu, lambda), in Pascals.

    For reading literature values into the solver's parameterisation. Do NOT
    build a randomisation sweep on top of this -- see the module docstring.

    Raises ValueError near the incompressible limit, where lambda diverges.
    """
    E = np.asarray(E, np.float64)
    nu = np.asarray(nu, np.float64)
    if np.any(nu >= NU_SINGULARITY_LIMIT):
        worst = float(np.max(nu))
        raise ValueError(
            f"nu = {worst:.6g} is at or past {NU_SINGULARITY_LIMIT}, where "
            "lambda = E*nu/((1+nu)(1-2nu)) is singular: the (1 - 2nu) "
            "denominator goes to zero at nu = 0.5. Sample lambda directly "
            "instead of pushing nu towards the incompressible limit.")
    if np.any(nu <= -1.0):
        raise ValueError(f"nu = {float(np.min(nu)):.6g} is at or below -1, "
                         "outside the isotropic range (-1, 0.5)")
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def E_nu_from_lame(mu: ArrayLike, lam: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """(mu, lambda) -> (Young's modulus, Poisson ratio). The inverse.

    For REPORTING. A sampled (mu, lambda) pair means nothing to a reader until
    it is quoted as "E = 4.4 kPa, nu = 0.497", which is comparable against
    published tissue measurements.
    """
    mu = np.asarray(mu, np.float64)
    lam = np.asarray(lam, np.float64)
    if np.any(mu <= 0.0):
        raise ValueError(f"mu must be positive, got min {float(np.min(mu)):.6g}")
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    return E, nu


# --------------------------------------------------------------------------
# Sampling and packing
# --------------------------------------------------------------------------

def sample_material(rng: np.random.Generator, *,
                    mu_range: Tuple[float, float] = DEFAULT_MU_RANGE,
                    lam_range: Tuple[float, float] = DEFAULT_LAM_RANGE,
                    rho_range: Tuple[float, float] = DEFAULT_RHO_RANGE) -> np.ndarray:
    """One episode's material as (3,) float32 [log(mu), log(lambda), rho].

    mu and lambda are drawn LOG-uniformly, so each decade of stiffness gets the
    same number of episodes. Drawn linearly, a range spanning two decades would
    put ~90% of the dataset in the stiffest decade and the model would never
    see soft tissue.

    Density is drawn linearly: it varies by a few percent around water, so its
    log is a near-constant and log-sampling it would buy nothing.
    """
    for name, (lo, hi) in (("mu_range", mu_range), ("lam_range", lam_range),
                           ("rho_range", rho_range)):
        if not (0.0 < lo <= hi):
            raise ValueError(f"{name} must satisfy 0 < lo <= hi, got ({lo}, {hi})")

    log_mu = rng.uniform(np.log(mu_range[0]), np.log(mu_range[1]))
    log_lam = rng.uniform(np.log(lam_range[0]), np.log(lam_range[1]))
    rho = rng.uniform(rho_range[0], rho_range[1])
    return np.array([log_mu, log_lam, rho], np.float32)


def unpack_material(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[log(mu), log(lambda), rho] -> (mu, lambda, rho) in Pascals and kg/m^3.

    Accepts (3,) for a global material or (N, 3) for a per-particle one, and
    returns scalars or (N,) arrays to match. Both broadcast correctly against a
    per-particle quantity of shape (..., N), which is what every consumer in
    tissue_metrics expects -- so heterogeneous tissue needs no code change here.
    """
    p = np.asarray(params, np.float64)
    if p.shape == (3,):
        return np.exp(p[0]), np.exp(p[1]), p[2]
    if p.ndim == 2 and p.shape[-1] == 3:
        return np.exp(p[:, 0]), np.exp(p[:, 1]), p[:, 2]
    raise ValueError(
        f"material_params must be (3,) or (N, 3), got {p.shape}. Note this is "
        "[log_mu, log_lambda, rho], not [mu, lambda, rho].")


# --------------------------------------------------------------------------
# Solver stability
# --------------------------------------------------------------------------

def suggested_substep_dt(mu: ArrayLike, rho: ArrayLike, dx: ArrayLike,
                         safety: float = 0.3,
                         lam: Optional[ArrayLike] = None) -> np.ndarray:
    """Advisory MPM substep, seconds, for a given material and grid spacing.

    PASS `lam` AND YOU GET THE BOUND THAT ACTUALLY HOLDS. Without it this uses
    the bar-wave speed sqrt(3*mu/rho), which ignores the pressure wave and is
    therefore an upper bound only -- see KNOWN LIMITATION below. With it the
    wave speed is sqrt((lambda + 2*mu)/rho), the real P-wave speed, and the
    result is the step a nearly incompressible material genuinely needs.

    This is not academic. Sampling this module's own default ranges and running
    the vendored MPM at its fixed 500 us substep: the bar-wave bound called
    every material safe (1400-5200 us), the P-wave bound called three of six
    unstable, and the three it flagged were exactly the three that diverged --
    one of them badly enough that det(F) went non-finite and an SVD inside
    `max_principal_stretch` refused to converge. lambda/mu reaches ~750 here,
    so the two bounds differ by a factor of ~16.

        c  = sqrt(E / rho)        elastic wave speed
        dt = safety * dx / c      CFL-style bound: no wave crosses a cell in
                                  one step

    WHY THIS IS NOT A CONSTANT: MPM stability follows the wave speed, so the
    stable step shrinks like 1/sqrt(E). Randomising stiffness over an order of
    magnitude means the stiffest episodes need a step ~3x smaller than the
    softest. A single fixed "default" substep tuned on a soft episode diverges
    silently on a stiff one, and the resulting dataset looks plausible until a
    model trained on it refuses to learn.

    WITHOUT `lam`, E is taken as 3*mu, the near-incompressible limit
    (E = 2*mu*(1 + nu) with nu -> 0.5). That substitution is what lets a caller
    who has only mu get an answer at all, and it is why the lam-less result is
    a bar-wave upper bound rather than the real thing. WITH `lam`, no such
    substitution happens: the P-wave speed sqrt((lambda + 2*mu)/rho) is used
    directly, which is the bound tissue actually obeys. New callers should pass
    it; the default stays lam-less only because v1 and PyBullet callers predate
    the argument and must keep getting what they always got.

    KNOWN LIMITATION (applies only when `lam` is omitted): for a nearly
    incompressible material lambda >> mu, and the pressure (P-wave) speed
    sqrt((lambda + 2*mu) / rho) is several times the bar speed used here. With
    the default ranges lambda/mu reaches ~750, so the true stable step can be an
    order of magnitude smaller than this returns. Treat the lam-less result as
    an upper bound to start a convergence study from, not as a guarantee --
    exactly as container/timestep_study.py exists to do for the PyBullet side.
    """
    mu = np.asarray(mu, np.float64)
    rho = np.asarray(rho, np.float64)
    dx = np.asarray(dx, np.float64)
    if np.any(mu <= 0) or np.any(rho <= 0) or np.any(dx <= 0):
        raise ValueError(
            f"mu, rho and dx must all be positive; got mu={mu}, rho={rho}, dx={dx}")
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"safety must be in (0, 1], got {safety}")
    if lam is None:
        c = np.sqrt(3.0 * mu / rho)          # bar wave, E = 3*mu
    else:
        lam = np.asarray(lam, np.float64)
        if np.any(lam <= 0):
            raise ValueError(f"lam must be positive when given; got lam={lam}")
        c = np.sqrt((lam + 2.0 * mu) / rho)  # P wave -- the one that bites
    return safety * dx / c


if __name__ == "__main__":
    # A quick look at what the default ranges actually mean in the units the
    # literature reports, plus the substep spread they imply.
    rng = np.random.default_rng(0)
    print("sampled materials, quoted as (E, nu) for comparison with literature:")
    for _ in range(5):
        mu, lam, rho = unpack_material(sample_material(rng))
        E, nu = E_nu_from_lame(mu, lam)
        dt = suggested_substep_dt(mu, rho, dx=1e-3)
        print(f"  mu={mu:9.1f} Pa  lambda={lam:11.1f} Pa  rho={rho:6.1f}  "
              f"-> E={E/1000:7.2f} kPa  nu={nu:.4f}  substep<={dt*1e6:7.2f} us")

    soft = suggested_substep_dt(DEFAULT_MU_RANGE[0], 1050.0, 1e-3)
    stiff = suggested_substep_dt(DEFAULT_MU_RANGE[1], 1050.0, 1e-3)
    print(f"\nsubstep across the mu range: {soft*1e6:.2f} us (softest) -> "
          f"{stiff*1e6:.2f} us (stiffest), a factor of {soft/stiff:.1f}")
    print("That factor is why substep_dt is recorded per episode rather than fixed.")
