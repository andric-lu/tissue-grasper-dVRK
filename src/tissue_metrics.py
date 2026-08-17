"""
tissue_metrics.py -- the two numbers the controller actually optimises.

WHY THIS FILE EXISTS
--------------------
A learned dynamics model that predicts particle positions is not yet useful. An
MPC loop cannot score a rollout on positions; it needs a scalar cost. This file
defines that cost, and it defines it ONCE so that the simulator, the validator,
the training targets and the planner cannot drift into disagreeing about what
"success" and "unsafe" mean.

Two metrics, both scalars per timestep:

    exposure       success. Fraction of a designated target region that the
                   tissue is no longer occluding. Higher is better.
    safety_strain  safety. A soft maximum over per-particle stretch. Lower is
                   better; this is the quantity that stands in for tearing.

Both are deterministic functions of particle state, so they are COMPUTED rather
than stored as an independent source of truth. The schema logs them anyway --
`validate_dataset.py` recomputes and compares, which is how you find out that a
collector and a trainer have quietly diverged.

Each lives behind a named function on purpose. Later, when the privileged inputs
these need (every particle position, the full deformation gradient) are no
longer available at control time, the body of `compute_exposure` becomes a
learned head and every caller stays exactly as it is.

REPORT THE MAX, OPTIMISE THE SURROGATE
--------------------------------------
Two splits run through this file, and both exist for the same reason: the
physically meaningful quantity has a bad gradient.

1. Eigenvalues vs invariants. Maximum principal stretch is what maps to a
   tissue-tear threshold in the literature, so it is what gets REPORTED. But it
   requires an eigendecomposition of F^T F, and eigenvalue gradients diverge
   when eigenvalues coincide -- and at rest F = I, where all three coincide
   exactly. The undeformed state sits precisely on the degenerate point. So
   anything in a LOSS path uses `strain_energy_neohookean` or the raw
   invariants J = det(F) and I1 = tr(F^T F), which are polynomial in F and
   smooth everywhere J > 0.

2. Hard max vs soft max. Injury is a maximum phenomenon: tissue tears at the
   single worst point, not on average. But a hard max over ~10,000 particles is
   non-smooth and hands the entire gradient to one particle per batch. So
   aggregation uses a power mean, and `compute_safety_strain` returns the true
   max alongside it. Report `max`, optimise `soft`.

A NOTE ON THE WORD "STRAIN"
---------------------------
`safety_strain` is a STRETCH RATIO, not a strain. Undeformed is 1.0, not 0.0.
It is named for the role it plays in the cost function, not for its units.

numpy only, and it must import under both numpy 1.23 (the container pin) and
numpy 2.x (the host env).
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, float]

# Defaults for the exposure geometry. These live here, not in the schema,
# because the writer and the validator MUST use identical values or the
# "logged vs recomputed" check fails for a reason that has nothing to do with
# the data. If exposure ever needs to vary per episode, these three have to
# move into the trajectory file and this comment has to come with them.
DEFAULT_SIGMA = 0.004        # m, occlusion radius of one particle
DEFAULT_THRESHOLD = 0.5      # summed Gaussian weight below which a point is clear
DEFAULT_GRID = 16            # query points per side over the target rectangle
DEFAULT_SHARPNESS = 0.10     # softness of the coverage->clear transition. Note
                             # that a sigmoid never reaches its asymptote: with
                             # these values a completely unoccluded target
                             # scores 0.993, not 1.0. That is harmless for a
                             # cost (it is monotone and bounded) but it means
                             # `compute_exposure` and `hard_exposure` differ by
                             # a fraction of a percent even in the clear case.

# Above this stretch, treat a particle as injured. PLACEHOLDER: 1.5 is an
# order-of-magnitude stand-in, not a measured value. Replace it with a failure
# stretch from uniaxial tension data for the tissue in question before any
# safety claim is made from it.
DEFAULT_STRAIN_THRESHOLD = 1.5


# --------------------------------------------------------------------------
# Kinematics of the deformation gradient
# --------------------------------------------------------------------------

def invariants(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (J, I1) for a stack of deformation gradients.

        J  = det(F)          volume ratio; 1 is incompressible, <= 0 is inverted
        I1 = tr(F^T F)       first invariant of the right Cauchy-Green tensor

    `F` is (..., 3, 3); both outputs are (...).

    These are the quantities to differentiate through. Unlike principal
    stretches they are polynomial in the entries of F, so they are smooth at
    F = I where the three stretches coincide and an eigendecomposition's
    gradient blows up.
    """
    F = _as_F(F)
    J = np.linalg.det(F)
    # tr(F^T F) is the sum of the squares of every entry of F -- the squared
    # Frobenius norm. Computing it that way avoids forming F^T F at all, which
    # for 10,000 particles is 9 multiply-adds saved per particle and, more to
    # the point, removes a matmul that can introduce asymmetry in float32.
    I1 = np.sum(F * F, axis=(-2, -1))
    return J, I1


def principal_stretches(F: np.ndarray) -> np.ndarray:
    """Singular values of F, ascending. Shape (..., 3) for F of (..., 3, 3).

    These are the principal stretch ratios: 1.0 means undeformed along that
    principal direction, 1.2 means stretched 20%.

    For REPORTING only. See the module docstring for why these must not appear
    in a loss.
    """
    F = _as_F(F)
    # compute_uv=False returns singular values only, and returns them in
    # DESCENDING order. The convention here is ascending, matching the usual
    # lambda_1 <= lambda_2 <= lambda_3 ordering in continuum mechanics texts.
    s = np.linalg.svd(F, compute_uv=False)
    return s[..., ::-1]


def max_principal_stretch(F: np.ndarray) -> np.ndarray:
    """Largest principal stretch. Shape (...) for F of (..., 3, 3).

    The single number that maps most directly onto a tissue-tear threshold.
    """
    # svd returns descending, so the largest is already first -- take it
    # directly rather than paying for the reversal in principal_stretches.
    return np.linalg.svd(_as_F(F), compute_uv=False)[..., 0]


def strain_energy_neohookean(F: np.ndarray, mu: ArrayLike, lam: ArrayLike) -> np.ndarray:
    """Compressible Neo-Hookean strain energy density, joules per cubic metre.

        Psi(F) = (mu/2)(I1 - 3) - mu*ln(J) + (lam/2)(ln J)^2

    `F` is (..., 3, 3). `mu` and `lam` are scalars, or arrays broadcastable
    against the leading (...) shape -- so a global material and a per-particle
    material both work, unchanged. That is deliberate: heterogeneous tissue has
    to be a data-generation change, not a rewrite of this file.

    Zero at F = I AND at every pure rotation, which is the property that makes
    it a valid strain measure. Anything built from F directly, or from the
    linearised strain (F + F^T)/2 - I, reports a rigid rotation as deformation.

    Raises ValueError on inverted elements (J <= 0), where ln(J) is undefined.
    An inverted element means the solver failed; returning NaN and letting it
    propagate into a loss wastes the run.
    """
    J, I1 = invariants(F)
    if np.any(J <= 0.0):
        n_bad = int(np.count_nonzero(J <= 0.0))
        raise ValueError(
            f"{n_bad} of {J.size} elements have det(F) <= 0 (min {float(J.min()):.4g}). "
            "That is an inverted element, not a deformation -- the solver "
            "diverged and this episode should be discarded, not trained on.")
    mu = np.asarray(mu, np.float64)
    lam = np.asarray(lam, np.float64)
    lnJ = np.log(J)
    return (mu / 2.0) * (I1 - 3.0) - mu * lnJ + (lam / 2.0) * lnJ * lnJ


# --------------------------------------------------------------------------
# Aggregation over particles
# --------------------------------------------------------------------------

def soft_max(x: np.ndarray, p: float = 8.0, axis: int = -1) -> np.ndarray:
    """Power mean ( mean(x^p) )^(1/p) over `axis`. Non-negative `x` only.

    A differentiable stand-in for `max`. As p -> inf it converges to the true
    max; at p = 1 it is the plain mean. p = 8 sits close enough to the max to
    be dominated by the worst particles while still spreading gradient over the
    worst few percent of them rather than exactly one.

    Guaranteed for p >= 1:  mean(x) <= soft_max(x, p) <= max(x),
    and non-decreasing in p.

    Negative inputs raise: x^p for fractional p is NaN there, and silently
    returning NaN from a safety metric is the worst available outcome.
    """
    x = np.asarray(x, np.float64)
    if p < 1.0:
        raise ValueError(f"p must be >= 1 for the mean/max bracket to hold, got {p}")
    if x.shape[axis] == 0:
        raise ValueError("soft_max over an empty axis is undefined")
    if np.any(x < 0.0):
        raise ValueError(
            f"soft_max requires non-negative input, got min {float(x.min()):.4g}. "
            "Stretch ratios are positive by construction; a negative one means "
            "the caller passed a strain measure rather than a stretch.")

    # Factor the largest element out before raising to the power. x^8 with a
    # handful of large entries overflows float64 at x ~ 1e38**(1/8) ~ 5e4,
    # which a diverged solver reaches easily. Scaling first makes the result
    # exact and the intermediate bounded by 1.
    m = np.max(x, axis=axis, keepdims=True)
    safe = np.where(m > 0.0, m, 1.0)
    scaled = np.mean((x / safe) ** p, axis=axis, keepdims=True) ** (1.0 / p)
    out = np.squeeze(safe * scaled, axis=axis)
    # All-zero slices: m == 0 makes the scaled mean 0, so the product is
    # already 0 and no special case is needed -- asserted by the tests.
    return out


def compute_safety_strain(F: np.ndarray, p: float = 8.0, *,
                          threshold: float = DEFAULT_STRAIN_THRESHOLD) -> dict:
    """The safety metric. `F` is (..., N, 3, 3); every value returned is (...).

    Returns a dict:
        soft                 power-mean stretch over particles -- OPTIMISE THIS
        max                  true maximum stretch -- REPORT THIS
        n_above_threshold    how many particles exceed `threshold`

    `n_above_threshold` is the interpretable one: "41 particles past the tear
    stretch" is actionable in a way that "soft max 1.37" is not. It is a hard
    count on purpose, since nothing differentiates it.
    """
    stretch = max_principal_stretch(F)          # (..., N)
    return {
        "soft": soft_max(stretch, p=p, axis=-1),
        "max": np.max(stretch, axis=-1),
        "n_above_threshold": np.count_nonzero(stretch > threshold, axis=-1),
    }


# --------------------------------------------------------------------------
# Exposure
# --------------------------------------------------------------------------

def compute_exposure(positions: np.ndarray, target_origin: np.ndarray,
                     target_normal: np.ndarray, target_extent: np.ndarray, *,
                     sigma: float = DEFAULT_SIGMA,
                     threshold: float = DEFAULT_THRESHOLD,
                     grid: int = DEFAULT_GRID,
                     sharpness: float = DEFAULT_SHARPNESS,
                     chunk: int = 64) -> np.ndarray:
    """Smooth fraction of the target region that tissue is not occluding.

    `positions` is (..., N, 3); the return is (...), each value in [0, 1].
    1.0 = the target is fully visible, 0.0 = fully covered.

    Method: lay a grid x grid sheet of query points over the target rectangle,
    project onto the target plane every particle on the OCCLUDING side of it
    (positive signed distance along `target_normal`), and give each query point
    a coverage equal to the summed Gaussian weight exp(-d^2 / 2 sigma^2) of the
    projected particles. A query point is clear when its coverage falls below
    `threshold`.

    "Clear" is counted with a sigmoid rather than a comparison, so the result
    is differentiable -- MPC needs a gradient, and a step function has none
    anywhere useful. `hard_exposure` gives the honest count for reporting.

    Only the SIDE matters, not the distance: a particle 1 mm above the target
    and one 10 cm above it both occlude the line of sight equally. Particles on
    the far side are excluded entirely, which is what makes exposure increase
    when tissue is lifted clear rather than merely moved.
    """
    cov, lead = _coverage(positions, target_origin, target_normal, target_extent,
                          sigma=sigma, grid=grid, chunk=chunk)
    # sigmoid((threshold - coverage) / sharpness): ~1 well under threshold,
    # ~0 well over it. `sharpness` in units of coverage sets the width of the
    # transition and trades differentiability against fidelity to the hard count.
    clear = _sigmoid((threshold - cov) / sharpness)
    return np.mean(clear, axis=-1).reshape(lead)


def hard_exposure(positions: np.ndarray, target_origin: np.ndarray,
                  target_normal: np.ndarray, target_extent: np.ndarray, *,
                  sigma: float = DEFAULT_SIGMA,
                  threshold: float = DEFAULT_THRESHOLD,
                  grid: int = DEFAULT_GRID,
                  chunk: int = 64) -> np.ndarray:
    """`compute_exposure` with a hard comparison instead of a sigmoid.

    For REPORTING. This is the number that means what it says -- "37 of 256
    query points are clear" -- with no smoothing parameter mixed in. It has no
    usable gradient, so it must not appear in a cost.
    """
    cov, lead = _coverage(positions, target_origin, target_normal, target_extent,
                          sigma=sigma, grid=grid, chunk=chunk)
    return np.mean(cov < threshold, axis=-1).reshape(lead)


def _coverage(positions: np.ndarray, target_origin: np.ndarray,
              target_normal: np.ndarray, target_extent: np.ndarray, *,
              sigma: float, grid: int, chunk: int):
    """Summed Gaussian occlusion at every query point. Shared by both exposures.

    Returns (coverage, leading_shape) where coverage is (B, grid*grid) with B
    the flattened leading dimensions.
    """
    pos = np.asarray(positions, np.float64)
    if pos.ndim < 2 or pos.shape[-1] != 3:
        raise ValueError(f"positions must be (..., N, 3), got {pos.shape}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    origin = np.asarray(target_origin, np.float64).reshape(3)
    normal = np.asarray(target_normal, np.float64).reshape(3)
    extent = np.asarray(target_extent, np.float64).reshape(2)
    nrm = np.linalg.norm(normal)
    if nrm < 1e-12:
        raise ValueError("target_normal has zero length; it must be a direction")
    normal = normal / nrm     # normalise rather than trust the caller: an
                              # unnormalised normal silently rescales the
                              # signed distance and flips which particles occlude

    e1, e2 = _plane_basis(normal)

    # Query points over the rectangle, as (G*G, 2) in-plane coordinates.
    u = np.linspace(-extent[0], extent[0], grid)
    v = np.linspace(-extent[1], extent[1], grid)
    qu, qv = np.meshgrid(u, v, indexing="ij")
    q = np.stack([qu.ravel(), qv.ravel()], axis=-1)          # (G*G, 2)

    lead = pos.shape[:-2]
    # B is computed rather than inferred with -1: numpy cannot infer a -1 axis
    # when the array is empty, and N == 0 is a real case (every particle on the
    # far side of the target, which is total success rather than an error).
    B = int(np.prod(lead)) if lead else 1
    flat = pos.reshape(B, pos.shape[-2], 3)                  # (B, N, 3)
    n_q = q.shape[0]
    out = np.empty((B, n_q), np.float64)

    # WHY CHUNKED: the pairwise term is (batch, G*G, N). At MPM scale with a
    # 400-step episode that is 400 * 256 * 10000 doubles -- 8 GB -- for a
    # result of 400 * 256. Chunking the LEADING axis keeps the vectorisation
    # over particles and query points (the axes that matter) while bounding the
    # working set. There is no Python loop over particles anywhere here.
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    for lo in range(0, B, max(1, chunk)):
        blk = flat[lo:lo + max(1, chunk)]                    # (b, N, 3)
        rel = blk - origin                                   # (b, N, 3)
        # Occluding side only. Kept as a 0/1 WEIGHT rather than a boolean index
        # because the count varies per frame, and boolean indexing would make
        # the array ragged across the batch.
        side = (rel @ normal) > 0.0                          # (b, N)
        proj = np.stack([rel @ e1, rel @ e2], axis=-1)       # (b, N, 2)
        d2 = np.sum((proj[:, None, :, :] - q[None, :, None, :]) ** 2, axis=-1)
        w = np.exp(-d2 * inv_two_sigma_sq) * side[:, None, :]
        out[lo:lo + max(1, chunk)] = w.sum(axis=-1)          # (b, G*G)
    return out, lead


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _as_F(F: np.ndarray) -> np.ndarray:
    """Validate and promote a stack of 3x3 deformation gradients to float64.

    Promotion is not optional. F is stored float32 (or float16), and det(F) for
    a near-incompressible material is 1 +/- 1e-4 -- differences that float32
    resolves to barely three digits. Every check downstream that asks "is J
    close to 1" is then measuring the storage dtype, not the physics.
    """
    F = np.asarray(F, np.float64)
    if F.ndim < 2 or F.shape[-2:] != (3, 3):
        raise ValueError(f"expected deformation gradients of shape (..., 3, 3), got {F.shape}")
    return F


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to `normal`.

    The choice of in-plane axes is arbitrary and only has to be consistent:
    exposure is a fraction of query points, and rotating the grid within its
    own plane leaves that fraction essentially unchanged.
    """
    # Seed from whichever world axis is least aligned with the normal. Using a
    # fixed seed axis instead breaks exactly when the normal happens to equal
    # it, and +z is the normal you would actually pick for a table-top target.
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(normal)))] = 1.0
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return e1, e2


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, written so large |z| cannot overflow exp()."""
    out = np.empty_like(z, np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
