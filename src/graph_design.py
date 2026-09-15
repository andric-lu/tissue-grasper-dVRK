"""graph_design.py -- the arithmetic that chooses a graph before any model exists.

numpy only, so it imports on both sides (CLAUDE.md, `src/` rule) and is unit
tested without a Metal device. The Taichi and torch halves of Phase 0 live in
host/ (mpm_locality_study.py, mps_scatter_check.py); everything here operates on
arrays that have already left the solver.

WHY THIS FILE EXISTS
--------------------
A graph network is a local model: M message-passing steps over a radius-R graph
see at most M*R of the object. Near-incompressible tissue is not local over one
recorded step -- the P-wave crosses the 300 mm slab in ~7 ms at the stiff corner
of materials.py. DECISION_LOG.md section 13 turns that mismatch into three
numbers (node count, radius, recording rate), and each number has to come from a
measurement rather than a preference. This module is the measurement's
arithmetic:

    influence_profile / influence_radius    how far a local kick is felt (13.2)
    farthest_point_sample / lattice_sample
      / surface_quota_fps                   which n particles become nodes (13.3)
    radius_graph / degree_stats
      / connected_components / hop_diameter what graph those nodes make (13.4)

WHY NO SCIPY
------------
scipy.spatial.cKDTree would be the obvious neighbour search. `src/` is numpy
only by rule, and every node set Phase 0 builds has n <= a few thousand, where
brute-force pairwise distances (chunked) are a few million entries. Whether
Phase 4's graph builder earns a scipy dependency is left to Phase 4.

RADII ARE REPORTED WITH A STATUS, NEVER AS A BARE NUMBER
--------------------------------------------------------
An influence radius is only meaningful if the threshold it was measured at sits
above the run-to-run noise, and it is only a *value* if the influence stops
before the object does -- otherwise it is a lower bound. So
influence_radius() returns a status alongside the number, and the number is NaN
whenever the status says it cannot be trusted. Same rule as validate_dataset's
SKIP: a measurement that cannot apply says so and says why.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

ArrayLike = np.ndarray

# influence_radius statuses. Strings rather than an enum so they print into a
# table and a JSON file without translation.
RESOLVED = "resolved"          # influence decays below threshold inside the object
SATURATED = "saturated"        # still above threshold at the farthest bin: radius is a LOWER BOUND
BELOW_NOISE = "below_noise"    # threshold is under the noise floor: cannot be measured at all
NO_SIGNAL = "no_signal"        # nothing clears the threshold, not even the source

# A bin must beat the noise floor by this factor to count. 3x, not 1x: the floor
# is itself a single sample of run-to-run difference, and a response equal to it
# is indistinguishable from it.
NOISE_FACTOR = 3.0

# Pairwise distance blocks are chunked to bound memory: (chunk, N) float64.
_CHUNK = 512


# --------------------------------------------------------------------------
# Influence (section 13.2)
# --------------------------------------------------------------------------

def influence_profile(dist: ArrayLike, response: ArrayLike, bin_width: float,
                      r_max: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Bin a per-particle response by distance from the source.

    `dist` and `response` are (N,). Returns the per-bin max, 95th percentile and
    count. Empty bins carry NaN, not zero: a bin with no particles has no
    measurement, and zero would read as "measured, and nothing arrived".

    MAX, NOT MEAN. The question is whether ANY node at that distance must be
    predicted from the source -- a network that gets the mean right and the
    peak wrong has still failed to represent the coupling.
    """
    d = np.asarray(dist, np.float64).ravel()
    y = np.asarray(response, np.float64).ravel()
    if d.shape != y.shape:
        raise ValueError(f"dist {d.shape} and response {y.shape} must match")
    if bin_width <= 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")
    top = float(d.max()) if r_max is None else float(r_max)
    n_bins = max(1, int(np.ceil(top / bin_width)))
    edges = np.arange(n_bins + 1) * bin_width
    which = np.minimum((d / bin_width).astype(np.int64), n_bins - 1)
    keep = d <= edges[-1]

    mx = np.full(n_bins, np.nan)
    p95 = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, np.int64)
    for b in range(n_bins):
        vals = y[keep & (which == b)]
        count[b] = vals.size
        if vals.size:
            mx[b] = vals.max()
            p95[b] = np.percentile(vals, 95)
    return {"edges": edges, "centres": 0.5 * (edges[:-1] + edges[1:]),
            "max": mx, "p95": p95, "count": count}


def influence_radius(profile: Dict[str, np.ndarray], *, ref: float, frac: float,
                     noise_floor: float = 0.0, stat: str = "max",
                     noise_factor: float = NOISE_FACTOR) -> Dict[str, object]:
    """Farthest distance at which the response exceeds `frac * ref`.

    `ref` is the response AT THE SOURCE (the kicked patch's own displacement),
    so `frac = 0.01` is "the 1%-influence radius". A bin counts only if it beats
    both `frac * ref` and `noise_factor * noise_floor`.

    Returns {"radius", "status", "threshold"}. `radius` is the upper edge of the
    farthest significant bin, and is NaN unless status is RESOLVED or SATURATED.
    SATURATED means the farthest POPULATED bin is still significant: the
    influence reaches the edge of the object, so the true radius is at least
    `radius` and possibly unbounded.

    FARTHEST, NOT FIRST-BELOW. Reflections off free surfaces can make a far bin
    exceed a nearer one. That far bin is genuinely influenced and a network has
    to represent it, so it counts.
    """
    if ref <= 0:
        raise ValueError(f"ref must be positive, got {ref}")
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    threshold = frac * ref
    floor = noise_factor * max(float(noise_floor), 0.0)
    out = {"threshold": threshold, "noise_threshold": floor}

    # WHY: a threshold under the noise floor would "measure" the noise. Refuse
    # before looking at the data, so no profile can talk its way past it.
    if threshold <= floor:
        return {**out, "radius": float("nan"), "status": BELOW_NOISE}

    y = profile[stat]
    populated = profile["count"] > 0
    significant = populated & (y > threshold) & (y > floor)
    if not significant.any():
        return {**out, "radius": float("nan"), "status": NO_SIGNAL}

    last_sig = int(np.flatnonzero(significant)[-1])
    last_pop = int(np.flatnonzero(populated)[-1])
    radius = float(profile["edges"][last_sig + 1])
    status = SATURATED if last_sig == last_pop else RESOLVED
    return {**out, "radius": radius, "status": status}


def required_message_passing_steps(influence_radius_m: float,
                                   graph_radius_m: float) -> int:
    """Smallest M with M * R >= the influence radius. One hop moves at most R."""
    if graph_radius_m <= 0:
        raise ValueError(f"graph_radius_m must be positive, got {graph_radius_m}")
    if not np.isfinite(influence_radius_m):
        raise ValueError("influence radius is not finite -- check its status first")
    return int(np.ceil(influence_radius_m / graph_radius_m - 1e-12))


# --------------------------------------------------------------------------
# Node sampling (section 13.3)
# --------------------------------------------------------------------------

def _sq_dist_to(pos: np.ndarray, point: np.ndarray) -> np.ndarray:
    diff = pos - point
    return np.einsum("ij,ij->i", diff, diff)


def _fps(pos: np.ndarray, n: int, min_sq: np.ndarray, first: int,
         candidates: np.ndarray) -> np.ndarray:
    """FPS over `candidates`, with distances already reduced by any preselection."""
    chosen = np.empty(n, np.int64)
    min_sq = min_sq.copy()
    cand_pos = pos[candidates]
    nxt = first
    for k in range(n):
        chosen[k] = candidates[nxt]
        min_sq = np.minimum(min_sq, _sq_dist_to(cand_pos, cand_pos[nxt]))
        min_sq[nxt] = -1.0          # never pick the same candidate twice
        nxt = int(np.argmax(min_sq))
    return chosen


def farthest_point_sample(pos: ArrayLike, n: int, seed: int = 0) -> np.ndarray:
    """n indices, each the farthest remaining point from those already chosen.

    O(n * N). The start point is drawn from `seed` so different seeds give
    genuinely different node sets -- the coarsening study reports spread over
    seeds, and a fixed start would report zero spread for a reason unrelated to
    the method. Returned sorted, so node order matches particle order.
    """
    p = np.asarray(pos, np.float64)
    N = p.shape[0]
    if not 0 < n <= N:
        raise ValueError(f"n must be in [1, {N}], got {n}")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(N))
    ids = _fps(p, n, np.full(N, np.inf), first, np.arange(N))
    return np.sort(ids)


def lattice_sample(pos: ArrayLike, n: int) -> np.ndarray:
    """Nearest particle to each point of a regular lattice over the measured bbox.

    The lattice pitch is solved so the number of lattice points is as close to n
    as the bbox's aspect ratio allows; on a 300 x 300 x 50 mm slab the thin axis
    quantises that heavily, which is why the returned count is reported by the
    caller rather than assumed. Two lattice points can share a nearest particle,
    so duplicates collapse and the result can be slightly smaller than the
    lattice. Returned sorted and unique.
    """
    p = np.asarray(pos, np.float64)
    N = p.shape[0]
    if not 0 < n <= N:
        raise ValueError(f"n must be in [1, {N}], got {n}")
    lo, hi = p.min(axis=0), p.max(axis=0)
    ext = np.maximum(hi - lo, 1e-12)

    def counts(s):
        return np.maximum(1, np.round(ext / s)).astype(np.int64)

    # Bisection on pitch: count is monotone non-increasing in s.
    s_lo, s_hi = 1e-9, float(ext.max()) * 2
    for _ in range(80):
        mid = 0.5 * (s_lo + s_hi)
        if np.prod(counts(mid)) > n:
            s_lo = mid
        else:
            s_hi = mid
    best = min((s_lo, s_hi), key=lambda s: abs(int(np.prod(counts(s))) - n))
    k = counts(best)
    axes = [lo[d] + (np.arange(k[d]) + 0.5) * ext[d] / k[d] for d in range(3)]
    lattice = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    nearest = np.empty(lattice.shape[0], np.int64)
    p_sq = np.einsum("ij,ij->i", p, p)
    for a in range(0, lattice.shape[0], _CHUNK):
        q = lattice[a:a + _CHUNK]
        d2 = np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * q @ p.T + p_sq[None, :]
        nearest[a:a + _CHUNK] = np.argmin(d2, axis=1)
    return np.unique(nearest)


def surface_mask_from_bbox(pos: ArrayLike, depth: float) -> np.ndarray:
    """Particles within `depth` of any face of the MEASURED rest bounding box.

    Measured, not hardcoded (CLAUDE.md: code should measure the world it operates
    in) -- init_cube's 0.1/0.2/0.05 offsets are the vendored defaults today and
    nothing here should break when they change. Exact for a box-shaped body, which
    is what init_cube lays out; a different rest shape needs a different mask.
    """
    p = np.asarray(pos, np.float64)
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    lo, hi = p.min(axis=0), p.max(axis=0)
    to_face = np.minimum(p - lo, hi - p).min(axis=1)
    return to_face < depth


def surface_quota_fps(pos: ArrayLike, n: int, surface_mask: ArrayLike,
                      surface_frac: float, seed: int = 0) -> np.ndarray:
    """FPS with round(n * surface_frac) nodes guaranteed to lie on the surface.

    The surface picks are made first, by FPS within the surface set; the
    interior is then filled by FPS CONTINUED from those picks (distances already
    reduced by them), so interior nodes keep their distance from surface nodes
    rather than being laid out as if the surface nodes did not exist.

    Why a quota and not "keep every surface particle": on this slab ~7,000
    particles sit within one spacing of a face, more than any candidate node
    count. Exposure is computed from occluding particles, which are surface
    particles, and a uniform subsample gives them only their volume share.
    """
    p = np.asarray(pos, np.float64)
    m = np.asarray(surface_mask, bool)
    N = p.shape[0]
    if m.shape != (N,):
        raise ValueError(f"surface_mask must be ({N},), got {m.shape}")
    if not 0 < n <= N:
        raise ValueError(f"n must be in [1, {N}], got {n}")
    if not 0 <= surface_frac <= 1:
        raise ValueError(f"surface_frac must be in [0, 1], got {surface_frac}")
    surf, inter = np.flatnonzero(m), np.flatnonzero(~m)
    n_s = min(int(round(n * surface_frac)), surf.size)
    n_i = n - n_s
    if n_i > inter.size:
        raise ValueError(f"need {n_i} interior nodes, only {inter.size} exist")
    rng = np.random.default_rng(seed)

    chosen = np.empty(0, np.int64)
    if n_s:
        chosen = _fps(p, n_s, np.full(surf.size, np.inf),
                      int(rng.integers(surf.size)), surf)
    if n_i:
        min_sq = np.full(inter.size, np.inf)
        for c in chosen:
            min_sq = np.minimum(min_sq, _sq_dist_to(p[inter], p[c]))
        first = int(np.argmax(min_sq)) if n_s else int(rng.integers(inter.size))
        chosen = np.concatenate([chosen, _fps(p, n_i, min_sq, first, inter)])
    return np.sort(chosen)


# --------------------------------------------------------------------------
# Graph statistics (section 13.4)
# --------------------------------------------------------------------------

def radius_graph(pos: ArrayLike, radius: float) -> np.ndarray:
    """Directed edge_index (2, E), both directions, no self loops, sorted.

    Brute force, chunked. `d < radius`, strictly: a node exactly one pitch away
    on a lattice built at R = pitch is a floating-point coin toss, and the
    caller's choice of R should not be.
    """
    p = np.asarray(pos, np.float64)
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    n = p.shape[0]
    r2 = radius * radius
    p_sq = np.einsum("ij,ij->i", p, p)
    senders, receivers = [], []
    for a in range(0, n, _CHUNK):
        q = p[a:a + _CHUNK]
        d2 = p_sq[a:a + _CHUNK, None] - 2.0 * q @ p.T + p_sq[None, :]
        i, j = np.nonzero(d2 < r2)
        i = i + a
        keep = i != j
        senders.append(i[keep])
        receivers.append(j[keep])
    s = np.concatenate(senders) if senders else np.empty(0, np.int64)
    r = np.concatenate(receivers) if receivers else np.empty(0, np.int64)
    order = np.lexsort((r, s))
    return np.stack([s[order], r[order]]).astype(np.int64)


def degrees(edge_index: ArrayLike, n: int) -> np.ndarray:
    """In-degree per node. The graph is symmetric, so in == out."""
    return np.bincount(np.asarray(edge_index)[1], minlength=n)


def degree_stats(deg: ArrayLike) -> Dict[str, object]:
    d = np.asarray(deg)
    return {"mean": float(d.mean()), "min": int(d.min()), "max": int(d.max()),
            "median": float(np.median(d)),
            "hist": np.bincount(d).tolist()}


def connected_components(edge_index: ArrayLike, n: int) -> Tuple[int, np.ndarray]:
    """(number of components, label per node). Label propagation + pointer jumping.

    A node in its own component never receives a message about the tool, so its
    prediction can only ever be "keep doing what you were doing". This is the
    check that catches it before a training run does.
    """
    ei = np.asarray(edge_index)
    labels = np.arange(n)
    if ei.shape[1] == 0:
        return n, labels
    s, r = ei[0], ei[1]
    while True:
        new = labels.copy()
        np.minimum.at(new, r, labels[s])
        new = new[new]                      # pointer jumping: halves chain length
        if np.array_equal(new, labels):
            break
        labels = new
    _, compact = np.unique(labels, return_inverse=True)
    return int(compact.max()) + 1, compact


def hop_diameter(edge_index: ArrayLike, n: int) -> Dict[str, object]:
    """Exact hop diameter: BFS from every node at once, as dense boolean steps.

    Returns {"diameter", "connected"}. If the graph is disconnected the true
    diameter is infinite; "diameter" is then the largest FINITE eccentricity and
    "connected" is False, and a caller must not read the number without the flag.

    Dense n x n reachability is 13 MB at n = 1800 and each hop is one matrix
    product -- well inside budget for Phase 0's node sets, and exact, which a
    sampled-source estimate is not.
    """
    ei = np.asarray(edge_index)
    A = np.zeros((n, n), np.float32)
    if ei.shape[1]:
        A[ei[0], ei[1]] = 1.0
    reached = np.eye(n, dtype=bool)
    frontier = reached.astype(np.float32)
    hops = 0
    while True:
        nxt = (frontier @ A > 0) & ~reached
        if not nxt.any():
            break
        reached |= nxt
        frontier = nxt.astype(np.float32)
        hops += 1
    return {"diameter": hops, "connected": bool(reached.all())}


def median_nn_distance(pos: ArrayLike) -> float:
    """Median distance to the nearest other point: the node set's MEASURED pitch."""
    p = np.asarray(pos, np.float64)
    n = p.shape[0]
    if n < 2:
        raise ValueError("need at least two points")
    p_sq = np.einsum("ij,ij->i", p, p)
    nn = np.empty(n)
    for a in range(0, n, _CHUNK):
        q = p[a:a + _CHUNK]
        d2 = p_sq[a:a + _CHUNK, None] - 2.0 * q @ p.T + p_sq[None, :]
        d2[np.arange(q.shape[0]), np.arange(a, a + q.shape[0])] = np.inf
        nn[a:a + _CHUNK] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return float(np.median(nn))
