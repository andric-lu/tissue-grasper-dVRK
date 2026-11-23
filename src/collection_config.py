"""collection_config.py -- the scripted grasp episode as data, not code.

numpy + json only (CLAUDE.md `src/` rule), so every waypoint the robot will be
driven through can be generated and unit-tested without Taichi, PyBullet or a
Metal device. host/mpm_adapter.py only executes what this produces.

WHY THIS FILE EXISTS (to-do Item 6)
-----------------------------------
"Put grasp location, approach/retraction distance, speed, direction, jaw
settings, and durations in a configuration file using explicit units." Until
now they lived as literals inside mpm_adapter._scripted_waypoints, counted in
FRAMES. A frame count is not a unit: the same 20 approach frames take 0.25 s at
80 Hz and 0.083 s at 240 Hz (decision D1), so moving to 240 Hz would have
silently tripled every speed. Here everything is metres, seconds, radians and
metres per second, and frame counts are derived from the recording interval.

THE PHASES (to-do Item 6: approach -> close/grasp -> retract -> release)
------------------------------------------------------------------------
    approach      EE moves from start to the grasp point, jaw open
    close         EE still at the grasp point, jaw interpolates open -> closed
    hold          EE still, jaw closed (optional pause before pulling)
    retract       EE moves from the grasp point along the retraction direction
    release       EE still at the retract end, jaw closed -> open
    post_release  EE still, jaw open (optional: watch the tissue recoil)

The tissue settle pre-roll (`preroll_s`) is simulated BEFORE row 0 and not
recorded; it exists because the tissue spawns in the air and was still falling
when the 80 Hz script's approach began (TODO_ASSESSMENT section 4.3).

The default config reproduces the geometry of the section 10 script that
smoke_test_grasp.py verified (hover point, grasp point, orientation), at the
same physical durations it had at 80 Hz. Its approach/retract speed is about
0.56 m/s, fast for surgical retraction; that is inherited, not chosen, and is
one of the parameters the config now makes visible.
"""

from __future__ import annotations

import copy
import json
from typing import Dict, Optional, Tuple

import numpy as np

# Phase labels. The integer is what lands in the episode's `phase` field;
# defined once here so collector, validator and visualiser agree.
PHASES = ("settle", "approach", "close", "hold", "retract", "release", "post_release")
PHASE_ID = {name: i for i, name in enumerate(PHASES)}

# The section 10 script, in physical units. hover/grasp/quat are copied from
# mpm_adapter._scripted_waypoints; durations are its 20/10/20/10 frames at the
# 80 Hz it was verified at (1/80 s per frame).
_HOVER = [0.274, 0.35, 0.226]
_GRASP = [0.252, 0.35, 0.088]
_APPROACH_DIST = float(np.linalg.norm(np.subtract(_HOVER, _GRASP)))   # 0.1397 m
_APPROACH_DIR = (np.subtract(_HOVER, _GRASP) / _APPROACH_DIST).tolist()

DEFAULT_GRASP_CONFIG: Dict = {
    "schema": "grasp_config/1",
    "units": "SI: metres, seconds, radians, m/s. Directions are unit vectors, world frame.",
    "frame_dt_s": 1.0 / 240.0,
    "preroll_s": 1.0,
    # Which arm drives the jaws (DECISION_LOG 14.15). "si": host/psm.py, scale 1.
    # "classic": SurRoL's Psm1 via host/psm_classic.py; `scaling` is SurRoL's
    # globalScaling (5 in SurRoL's own tasks), `rcm_position_m` the remote
    # centre's world position. `jaw` angles below are in the ARM's convention:
    # per-jaw for "si", total opening for "classic".
    "arm": {"model": "si", "scaling": 1.0, "rcm_position_m": None},
    # How a grasp holds tissue (DECISION_LOG 14.21-14.22). "teleport": the
    # 10.7 particle constraint (host/grasp_constraint.py) -- held particles
    # overwritten after each substep, F frozen. "grid_bc": host/grasp_grid_bc.py
    # -- the held particles' grid stencil moved rigidly with the gripper inside
    # the substep, so every particle's F follows the velocity it moves with.
    "grasp_model": "teleport",
    "orientation_xyzw": [0.7048, -0.7048, -0.0576, 0.0576],
    "grasp_position_m": _GRASP,
    "approach": {
        # The EE STARTS at grasp_position + distance * direction and moves in.
        "direction_from_grasp": _APPROACH_DIR,
        "distance_m": _APPROACH_DIST,
        "duration_s": 20 / 80,
    },
    "jaw": {"open_rad": 1.0, "closed_rad": 0.0},
    "close": {"duration_s": 10 / 80},
    "hold": {"duration_s": 0.0},
    "retract": {
        # Default: straight back up the approach path, as the section 10 script did.
        "direction": _APPROACH_DIR,
        "distance_m": _APPROACH_DIST,
        "duration_s": 20 / 80,
    },
    "release": {"duration_s": 10 / 80},
    "post_release": {"duration_s": 0.0},
    # Ranges used by sample_episode_config(). Empty dict = no randomisation.
    "randomize": {},
}

# Modest variation for the Item 6 debug set. Placeholders sized to stay inside
# the verified workspace; they are recorded into every episode's config so the
# ranges are never lost.
DEBUG_SET_RANDOMIZE: Dict = {
    "grasp_offset_xy_m": [-0.02, 0.02],          # uniform per axis, added to grasp_position
    "retract_cone_deg": 20.0,                    # direction drawn within this cone of nominal
    "retract_distance_m": [0.01, 0.03],          # short pulls (DECISION_LOG 14.15)
    "material": "sample",                        # draw (mu, lambda, rho) from the episode seed
}


def load_config(path: Optional[str]) -> Dict:
    """Read a JSON config over the defaults. Unknown keys are an error.

    WHY STRICT: a misspelt key ("duraton_s") in a permissive loader silently
    keeps the default, and the episode runs with a parameter nobody chose.
    """
    cfg = copy.deepcopy(DEFAULT_GRASP_CONFIG)
    if path is None:
        return cfg
    with open(path) as fh:
        user = json.load(fh)
    _merge_strict(cfg, user, "")
    validate_config(cfg)
    return cfg


def _merge_strict(base: Dict, over: Dict, where: str) -> None:
    for k, v in over.items():
        if k.startswith("_"):
            continue            # "_comment" etc.: human notes, not parameters
        if k not in base:
            raise KeyError(f"unknown config key {where + k!r}; known: {sorted(base)}")
        if isinstance(base[k], dict) and isinstance(v, dict) and k != "randomize":
            _merge_strict(base[k], v, where + k + ".")
        else:
            base[k] = v


ARM_MODELS = ("si", "classic")


def validate_config(cfg: Dict) -> None:
    """Raise on a config that cannot describe a physical episode."""
    arm = cfg.get("arm", {"model": "si"})
    if arm.get("model") not in ARM_MODELS:
        raise ValueError(f"arm.model must be one of {ARM_MODELS}, got {arm.get('model')!r}")
    if not float(arm.get("scaling", 1.0)) > 0:
        raise ValueError("arm.scaling must be positive")
    if arm["model"] == "classic" and arm.get("rcm_position_m") is None:
        raise ValueError("arm.model 'classic' needs arm.rcm_position_m")
    if not cfg["frame_dt_s"] > 0:
        raise ValueError("frame_dt_s must be positive")
    if cfg["preroll_s"] < 0:
        raise ValueError("preroll_s must be >= 0")
    for key in ("approach", "close", "hold", "retract", "release", "post_release"):
        if cfg[key]["duration_s"] < 0:
            raise ValueError(f"{key}.duration_s must be >= 0")
    for key in ("approach", "retract"):
        d = np.asarray(cfg[key]["direction" if key == "retract" else "direction_from_grasp"], float)
        if d.shape != (3,) or abs(np.linalg.norm(d) - 1.0) > 1e-6:
            raise ValueError(f"{key} direction must be a unit 3-vector, got {d}")
        if cfg[key]["distance_m"] < 0:
            raise ValueError(f"{key}.distance_m must be >= 0")
    q = np.asarray(cfg["orientation_xyzw"], float)
    if q.shape != (4,) or abs(np.linalg.norm(q) - 1.0) > 1e-3:
        raise ValueError(f"orientation_xyzw must be a unit quaternion [qx,qy,qz,qw], got {q}")
    if cfg["approach"]["duration_s"] <= 0 or cfg["close"]["duration_s"] <= 0:
        raise ValueError("approach and close need a positive duration")


def n_frames(duration_s: float, frame_dt: float) -> int:
    """Whole recording intervals in a phase. Rounded, and the RESULTING
    duration is what the episode records -- see `build_waypoints`."""
    return int(round(duration_s / frame_dt))


def build_waypoints(cfg: Dict) -> Dict[str, np.ndarray]:
    """The episode's commanded trajectory, one row per recorded state.

    Returns {"poses" (T,7), "jaw" (T,), "phase" (T,) int8, "speeds"}:
    row t is the commanded state AT recorded step t, so T = transitions + 1 and
    the action of row t takes row t to row t+1 (the to-do's (T+1 states,
    T actions) convention; the writer adds one zero pad action on the last row).

    Speeds are reported back in m/s as they come out after rounding durations
    to whole frames, because that -- not the config's nominal number -- is
    what the robot will actually do.
    """
    validate_config(cfg)
    dt = float(cfg["frame_dt_s"])
    quat = np.asarray(cfg["orientation_xyzw"], np.float64)
    quat = quat / np.linalg.norm(quat)
    grasp = np.asarray(cfg["grasp_position_m"], np.float64)
    start = grasp + cfg["approach"]["distance_m"] * np.asarray(
        cfg["approach"]["direction_from_grasp"], np.float64)
    end = grasp + cfg["retract"]["distance_m"] * np.asarray(
        cfg["retract"]["direction"], np.float64)
    j_open, j_closed = float(cfg["jaw"]["open_rad"]), float(cfg["jaw"]["closed_rad"])

    pos, jaw, phase = [start], [j_open], [PHASE_ID["approach"]]

    def segment(name, p0, p1, j0, j1):
        k = n_frames(cfg[name]["duration_s"], dt)
        for i in range(1, k + 1):
            s = i / k
            pos.append(p0 * (1 - s) + p1 * s)
            jaw.append(j0 * (1 - s) + j1 * s)
            phase.append(PHASE_ID[name])
        return k

    k = {}
    k["approach"] = segment("approach", start, grasp, j_open, j_open)
    k["close"] = segment("close", grasp, grasp, j_open, j_closed)
    k["hold"] = segment("hold", grasp, grasp, j_closed, j_closed)
    k["retract"] = segment("retract", grasp, end, j_closed, j_closed)
    k["release"] = segment("release", end, end, j_closed, j_open)
    k["post_release"] = segment("post_release", end, end, j_open, j_open)

    poses = np.concatenate([np.asarray(pos), np.tile(quat, (len(pos), 1))], axis=1)
    speeds = {
        "approach_m_s": cfg["approach"]["distance_m"] / (k["approach"] * dt)
        if k["approach"] else 0.0,
        "retract_m_s": cfg["retract"]["distance_m"] / (k["retract"] * dt)
        if k["retract"] else 0.0,
    }
    return {"poses": poses, "jaw": np.asarray(jaw, np.float64),
            "phase": np.asarray(phase, np.int8), "frames": k, "speeds": speeds}


def _random_unit_in_cone(rng: np.random.Generator, axis: np.ndarray,
                         half_angle_rad: float) -> np.ndarray:
    """Uniform on the spherical cap of half-angle `half_angle_rad` about `axis`."""
    axis = axis / np.linalg.norm(axis)
    cos_t = rng.uniform(np.cos(half_angle_rad), 1.0)
    sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    phi = rng.uniform(0.0, 2.0 * np.pi)
    # Any vector not parallel to axis gives a basis for the cap.
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return cos_t * axis + sin_t * (np.cos(phi) * u + np.sin(phi) * v)


def sample_episode_config(base: Dict, seed: int) -> Dict:
    """A concrete config for one episode: `base` plus draws from base["randomize"].

    Deterministic in `seed`. The returned config is complete and is what gets
    saved alongside the episode, so the draw is recoverable without re-running
    this function.
    """
    cfg = copy.deepcopy(base)
    r = cfg.get("randomize") or {}
    # A separate stream from the material draw (which mpm_adapter takes from
    # default_rng(seed)), so adding a randomised field here never shifts the
    # material an existing seed produces.
    rng = np.random.default_rng([seed, 0x6A57])
    if "grasp_offset_xy_m" in r:
        lo, hi = r["grasp_offset_xy_m"]
        g = np.asarray(cfg["grasp_position_m"], float)
        g[:2] += rng.uniform(lo, hi, size=2)
        cfg["grasp_position_m"] = g.tolist()
    if "retract_cone_deg" in r:
        d = _random_unit_in_cone(rng, np.asarray(cfg["retract"]["direction"], float),
                                 np.deg2rad(r["retract_cone_deg"]))
        cfg["retract"]["direction"] = d.tolist()
    if "retract_distance_m" in r:
        lo, hi = r["retract_distance_m"]
        nominal_speed = base["retract"]["distance_m"] / base["retract"]["duration_s"]
        dist = float(rng.uniform(lo, hi))
        cfg["retract"]["distance_m"] = dist
        # Hold the nominal SPEED, not the duration: a shorter pull is a
        # shorter phase, not a slower one.
        cfg["retract"]["duration_s"] = dist / nominal_speed
    cfg["seed"] = int(seed)
    validate_config(cfg)
    return cfg


def matched_variants(base_episode_cfg: Dict, directions, distances) -> list:
    """Item 12: the same initial state and grasp, different retractions.

    Everything that determines the state at the start of retraction --
    material (via the same seed), pre-roll, grasp point, approach, close, hold
    -- is copied verbatim; only the retraction direction and distance change.
    Each returned config keeps `seed`, so the collector reproduces the same
    material and the same (Taichi-fixed) particle cloud.
    """
    out = []
    speed = base_episode_cfg["retract"]["distance_m"] / base_episode_cfg["retract"]["duration_s"]
    for d in directions:
        for dist in distances:
            c = copy.deepcopy(base_episode_cfg)
            c["randomize"] = {}
            dv = np.asarray(d, float)
            c["retract"]["direction"] = (dv / np.linalg.norm(dv)).tolist()
            c["retract"]["distance_m"] = float(dist)
            c["retract"]["duration_s"] = float(dist) / speed
            validate_config(c)
            out.append(c)
    return out


def config_to_json(cfg: Dict) -> str:
    return json.dumps(cfg, indent=2, sort_keys=True)


def nominal_start_and_end(cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """(EE start position, EE retract end position) -- for workspace checks."""
    grasp = np.asarray(cfg["grasp_position_m"], float)
    start = grasp + cfg["approach"]["distance_m"] * np.asarray(
        cfg["approach"]["direction_from_grasp"], float)
    end = grasp + cfg["retract"]["distance_m"] * np.asarray(cfg["retract"]["direction"], float)
    return start, end
