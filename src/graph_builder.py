"""graph_builder.py -- turn one recorded row into the hybrid particle graph.

numpy only (CLAUDE.md `src/` rule). This is the reference implementation: the
tests in tests/test_graph_builder.py pin its behaviour, and the torch path a
training loop will need for per-step rebuilds on MPS (to-do Item 7) must be
checked against it, not the other way round.

WHAT THE GRAPH IS (to-do Items 1, 2, 6)
---------------------------------------
Nodes
    tissue nodes  one per recorded MPM particle, ordered as `particle_ids`
    tool nodes    one per jaw (K = 2), ordered as `jaw_names`

Edges, all DIRECTED, sender -> receiver, stored as edge_index (2, E) with
row 0 = sender i, row 1 = receiver j. Every tissue rule adds both i->j and
j->i; no self-loops. For an edge i -> j every relative quantity is
"receiver minus sender": X_j - X_i, x_j - x_i, v_j - v_i.

    reference   ||X_j - X_i|| < r_ref           connectivity FIXED per episode
                features [X_j-X_i, x_j-x_i, |X_j-X_i|, |x_j-x_i|]      (8)
    spatial     ||x_j - x_i|| < r_spatial       connectivity REBUILT every row
                features [x_j-x_i, |x_j-x_i|, v_j-v_i]                 (7)
    tool        dist(node i, jaw box k) < r_c   jaw k -> tissue node i only
                features in TOOL_EDGE_FEATURES                         (14)

A pair satisfying both tissue rules appears in BOTH edge sets: the two types
carry different information (material connectivity vs current proximity) and
go to separate encoders, so merging them would throw one away.

WHY TOOL EDGES ARE ONE-WAY: the decoder predicts tissue state only and tool
motion is supplied from the recording (Items 6, 10), so a message INTO a tool
node would update a latent that is never read. Reverse edges can be added
later if a tool-state predictor is ever trained.

NODE FEATURES (Item 1), width 21, or 19 with g omitted:
    x - X (3), v (3), F - I (9, row-major), ln mu, ln lambda, rho, b, g (2)
g is two SEPARATE indicators [g_contact, g_grasp] (Item 1: "separate
meanings"), defined in src/tool_geometry.py. b is the fixed-boundary flag,
constant zero for the MPM data (decision D5): kept as a slot so a clamped
dataset needs no architecture change, never claimed to have been tested.
Absolute position x is deliberately NOT a feature: dynamics do not depend on
where in the world the tissue sits, and an absolute coordinate lets a network
memorise location instead of learning physics (CLAUDE.md: predict deltas).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

import tool_geometry as tg

NODE_FEATURES = (
    "dx_x", "dx_y", "dx_z",                       # x - X
    "v_x", "v_y", "v_z",
    "dF00", "dF01", "dF02", "dF10", "dF11", "dF12", "dF20", "dF21", "dF22",  # F - I
    "log_mu", "log_lambda", "rho",
    "b",
    "g_contact", "g_grasp",
)
REF_EDGE_FEATURES = ("dX_x", "dX_y", "dX_z", "dx_x", "dx_y", "dx_z", "|dX|", "|dx|")
SPATIAL_EDGE_FEATURES = ("dx_x", "dx_y", "dx_z", "|dx|", "dv_x", "dv_y", "dv_z")
TOOL_EDGE_FEATURES = (
    "rel_x", "rel_y", "rel_z",        # x_i - nearest point on jaw k surface
    "signed_dist",                    # + outside, - inside (tool_geometry)
    "n_x", "n_y", "n_z",              # outward surface normal at that point
    "dv_x", "dv_y", "dv_z",           # v_i - velocity of the jaw at that point
    "jaw0", "jaw1",                   # jaw identity, one-hot
    "g_contact", "g_grasp",           # the receiving node's indicators
)
TOOL_NODE_FEATURES = (
    "jaw0", "jaw1",                   # identity, one-hot
    "opening",                        # this jaw's joint angle, rad
    "v_x", "v_y", "v_z", "w_x", "w_y", "w_z",          # jaw_box_vel (backward)
    "R00", "R10", "R20", "R01", "R11", "R21",          # orientation, 6D (2 columns)
    "a_dx", "a_dy", "a_dz", "a_rx", "a_ry", "a_rz", "a_jaw",  # commanded action
    "hx", "hy", "hz",                 # box half-extents
)
assert len(REF_EDGE_FEATURES) == 8 and len(SPATIAL_EDGE_FEATURES) == 7

# Brute-force pairwise blocks are chunked: (chunk, N) float64 = 24 MB at N=3000.
_CHUNK = 1024


# --------------------------------------------------------------------------
# Radius search
# --------------------------------------------------------------------------

def radius_pairs(points: np.ndarray, radius: float) -> np.ndarray:
    """All ordered pairs (i, j), i != j, with ||p_j - p_i|| < radius. (2, E), sorted.

    EXACTLY SYMMETRIC BY CONSTRUCTION. The fast expanded form
    |a|^2 - 2a.b + |b|^2 is used only to find CANDIDATES (with slack); the
    decision is then made on sum((p_j - p_i)^2), which is bitwise identical
    for (i, j) and (j, i) because (a - b)^2 == (b - a)^2 in IEEE arithmetic.
    The expanded form alone can include i->j and exclude j->i for a pair
    sitting within rounding of the radius, which would silently break the
    "both directions" rule the to-do states.

    Brute force: O(N^2), ~0.1 s at N = 3,000 in numpy. A spatial hash or a
    KD-tree is the obvious upgrade, deferred until it is the bottleneck
    (graph_design.py, "WHY NO SCIPY").
    """
    p = np.asarray(points, np.float64)
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    n = p.shape[0]
    r2 = radius * radius
    slack = r2 * 1e-6 + 1e-15
    p_sq = np.einsum("ij,ij->i", p, p)
    send, recv = [], []
    for a in range(0, n, _CHUNK):
        q = p[a:a + _CHUNK]
        approx = p_sq[a:a + _CHUNK, None] - 2.0 * q @ p.T + p_sq[None, :]
        i, j = np.nonzero(approx < r2 + slack)
        i = i + a
        keep = i != j
        i, j = i[keep], j[keep]
        d = p[j] - p[i]
        exact = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
        ok = exact < r2
        send.append(i[ok])
        recv.append(j[ok])
    s = np.concatenate(send) if send else np.empty(0, np.int64)
    r = np.concatenate(recv) if recv else np.empty(0, np.int64)
    order = np.lexsort((r, s))
    return np.stack([s[order], r[order]]).astype(np.int64)


# --------------------------------------------------------------------------
# Tissue edges
# --------------------------------------------------------------------------

def reference_edges(rest_pos: np.ndarray, r_ref: float) -> np.ndarray:
    """Fixed connectivity from the REFERENCE configuration X. Build once per
    episode; features are recomputed every row from the current x."""
    return radius_pairs(rest_pos, r_ref)


def reference_edge_features(edge_index: np.ndarray, rest_pos: np.ndarray,
                            pos: np.ndarray) -> np.ndarray:
    """(E, 8) [X_j - X_i, x_j - x_i, |X_j - X_i|, |x_j - x_i|] for i -> j."""
    s, r = edge_index
    X = np.asarray(rest_pos, np.float64)
    x = np.asarray(pos, np.float64)
    dX = X[r] - X[s]
    dx = x[r] - x[s]
    return np.concatenate([dX, dx, np.linalg.norm(dX, axis=1, keepdims=True),
                           np.linalg.norm(dx, axis=1, keepdims=True)], axis=1)


def spatial_edges(pos: np.ndarray, r_spatial: float) -> np.ndarray:
    """Connectivity from the CURRENT positions. Rebuild every row."""
    return radius_pairs(pos, r_spatial)


def spatial_edge_features(edge_index: np.ndarray, pos: np.ndarray,
                          vel: np.ndarray) -> np.ndarray:
    """(E, 7) [x_j - x_i, |x_j - x_i|, v_j - v_i] for i -> j."""
    s, r = edge_index
    x = np.asarray(pos, np.float64)
    v = np.asarray(vel, np.float64)
    dx = x[r] - x[s]
    return np.concatenate([dx, np.linalg.norm(dx, axis=1, keepdims=True),
                           v[r] - v[s]], axis=1)


# --------------------------------------------------------------------------
# Node features
# --------------------------------------------------------------------------

def node_features(pos: np.ndarray, vel: np.ndarray, F: np.ndarray,
                  rest_pos: np.ndarray, material_field: np.ndarray,
                  boundary: Optional[np.ndarray] = None,
                  g_contact: Optional[np.ndarray] = None,
                  g_grasp: Optional[np.ndarray] = None,
                  include_g: bool = True) -> np.ndarray:
    """(N, 21), or (N, 19) with include_g=False. Column order: NODE_FEATURES.

    `material_field` is [ln mu, ln lambda, rho] per node -- natural logs of
    Pascals, kg/m^3 (docs/FIELD_DICTIONARY.md). F - I is flattened row-major.
    Missing b / g are ZEROS here only because the caller said so: pass them
    explicitly from the episode. With include_g=False the two g columns are
    dropped rather than zeroed, so "omitted" and "zero" stay distinguishable
    to the model (the Item 10 ablation).
    """
    x = np.asarray(pos, np.float64)
    N = x.shape[0]
    Fm = np.asarray(F, np.float64).reshape(N, 3, 3)
    mf = np.asarray(material_field, np.float64)
    if mf.shape != (N, 3):
        raise ValueError(f"material_field must be ({N}, 3), got {mf.shape}")
    b = np.zeros(N) if boundary is None else np.asarray(boundary, np.float64).ravel()
    cols = [x - np.asarray(rest_pos, np.float64), np.asarray(vel, np.float64),
            (Fm - np.eye(3)).reshape(N, 9), mf, b[:, None]]
    if include_g:
        gc = np.zeros(N) if g_contact is None else np.asarray(g_contact, np.float64)
        gg = np.zeros(N) if g_grasp is None else np.asarray(g_grasp, np.float64)
        cols += [gc[:, None], gg[:, None]]
    return np.concatenate(cols, axis=1)


# --------------------------------------------------------------------------
# Tool nodes and tool -> tissue edges (Item 6)
# --------------------------------------------------------------------------

def tool_node_features(jaw_box_pose: np.ndarray, jaw_box_vel: np.ndarray,
                       jaw_joint: np.ndarray, action: np.ndarray,
                       half_extents: np.ndarray) -> np.ndarray:
    """(K, 25), column order TOOL_NODE_FEATURES.

    The COMMANDED action is included (Item 6: "Include commanded actions in
    tool-node features"), the same 7-vector for both jaws: it is one command
    to one arm. No absolute position, for the same reason tissue nodes have
    none; where the jaw is relative to the tissue reaches the model through
    the tool edges. Orientation is absolute (as 6D, the first two rotation
    columns -- continuous, unlike a quaternion's double cover) because
    gravity makes "which way is down" physically meaningful.
    """
    bp = np.asarray(jaw_box_pose, np.float64)
    K = bp.shape[0]
    R = tg.rotmat_from_quat(bp[:, 3:7])
    six = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)
    onehot = np.eye(K)
    a = np.broadcast_to(np.asarray(action, np.float64), (K, 7))
    return np.concatenate([onehot, np.asarray(jaw_joint, np.float64).reshape(K, 1),
                           np.asarray(jaw_box_vel, np.float64).reshape(K, 6), six, a,
                           np.asarray(half_extents, np.float64).reshape(K, 3)], axis=1)


def tool_edges(pos: np.ndarray, vel: np.ndarray, jaw_box_pose: np.ndarray,
               jaw_box_vel: np.ndarray, half_extents: np.ndarray, r_c: float,
               g_contact: Optional[np.ndarray] = None,
               g_grasp: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Edges jaw k -> tissue node i for every node within `r_c` of box k.

    Returns {"edge_index" (2, E): row 0 = jaw index k, row 1 = node i,
             "features" (E, 14) in TOOL_EDGE_FEATURES order}.

    The criterion is the UNSIGNED surface distance (0 inside), the same
    distance the solver's contact and grasp rules use. The features carry the
    SIGNED distance, so penetration depth is not lost.

    Relative velocity is the node's velocity minus the jaw's velocity AT the
    nearest surface point, v_box + omega x (p_near - c): a rotating jaw's
    surface moves differently at its tip and its hinge.

    `r_c` is a GRAPH parameter, not physics: Item 6 asks for it to be chosen
    by inspecting contact coverage and edge counts on the pilot. Proximity is
    not contact; these edges say "close enough that contact is possible".
    """
    x = np.asarray(pos, np.float64)
    v = np.asarray(vel, np.float64)
    bp = np.asarray(jaw_box_pose, np.float64)
    bv = np.asarray(jaw_box_vel, np.float64)
    he = np.asarray(half_extents, np.float64)
    N, K = x.shape[0], bp.shape[0]
    gc = np.zeros(N) if g_contact is None else np.asarray(g_contact, np.float64)
    gg = np.zeros(N) if g_grasp is None else np.asarray(g_grasp, np.float64)
    senders, receivers, feats = [], [], []
    for k in range(K):
        near = tg.box_nearest_point(x, bp[k], he[k])
        sel = np.flatnonzero(np.maximum(near["signed_distance"], 0.0) < r_c)
        if sel.size == 0:
            continue
        pn = near["nearest"][sel]
        v_surf = bv[k, :3] + np.cross(bv[k, 3:], pn - bp[k, :3])
        onehot = np.zeros((sel.size, K))
        onehot[:, k] = 1.0
        feats.append(np.concatenate([
            x[sel] - pn, near["signed_distance"][sel, None], near["normal"][sel],
            v[sel] - v_surf, onehot, gc[sel, None], gg[sel, None]], axis=1))
        senders.append(np.full(sel.size, k))
        receivers.append(sel)
    if not feats:
        return {"edge_index": np.zeros((2, 0), np.int64),
                "features": np.zeros((0, 2 * 3 + 1 + 3 + 3 + K + 2))}
    return {"edge_index": np.stack([np.concatenate(senders),
                                    np.concatenate(receivers)]).astype(np.int64),
            "features": np.concatenate(feats)}


# --------------------------------------------------------------------------
# One row -> one graph
# --------------------------------------------------------------------------

class EpisodeGraphBuilder:
    """Build graphs for every row of one episode.

    Reference edges come from `rest_pos` once, at construction. The builder
    takes rest_pos from the episode rather than from a global file: today the
    Taichi cloud is identical in every episode, but if its RNG is ever seeded
    the clouds will differ and a stored global reference graph would be wrong
    without any error (TODO_ASSESSMENT 4.4).
    """

    def __init__(self, traj, *, r_ref: float, r_spatial: float,
                 r_c: Optional[float] = None, include_g: bool = True,
                 include_tool: bool = True):
        if traj.rest_pos.size == 0:
            raise ValueError(f"{traj.path} has no rest_pos (schema "
                             f"{traj.schema_version}); the to-do forbids "
                             "substituting frame 0 for the reference X")
        if traj.material_field.size == 0:
            raise ValueError(f"{traj.path} has no material_field")
        self.tr = traj
        self.r_ref, self.r_spatial = float(r_ref), float(r_spatial)
        self.include_g = include_g
        self.include_tool = include_tool and traj.has_jaws
        if self.include_tool and r_c is None:
            raise ValueError("tool edges need r_c (Item 6: choose it on the pilot)")
        self.r_c = None if r_c is None else float(r_c)
        self.ref_edge_index = reference_edges(traj.rest_pos, self.r_ref)

    def __len__(self) -> int:
        return len(self.tr)

    def graph(self, t: int, *, pos: Optional[np.ndarray] = None,
              vel: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None,
              g_contact: Optional[np.ndarray] = None,
              g_grasp: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """The graph for row t. pos/vel/F/g default to the recorded row; a
        rollout passes its own PREDICTED state instead (Item 10), and the
        tool geometry still comes from the recording."""
        tr = self.tr
        x = tr.tissue_pos[t] if pos is None else pos
        v = tr.tissue_vel[t] if vel is None else vel
        Fm = tr.tissue_F[t] if F is None else F
        gc = g_contact if g_contact is not None else (
            tr.node_contact[t] if tr.node_contact.size else None)
        gg = g_grasp if g_grasp is not None else (
            tr.node_grasp[t] if tr.node_grasp.size else None)
        b = tr.boundary_mask if tr.boundary_mask.size else None
        out = {
            "node_features": node_features(x, v, Fm, tr.rest_pos, tr.material_field,
                                           b, gc, gg, self.include_g),
            "ref_edge_index": self.ref_edge_index,
            "ref_edge_features": reference_edge_features(self.ref_edge_index,
                                                         tr.rest_pos, x),
        }
        sp = spatial_edges(x, self.r_spatial)
        out["spatial_edge_index"] = sp
        out["spatial_edge_features"] = spatial_edge_features(sp, x, v)
        if self.include_tool:
            K = tr.jaw_names.size
            out["tool_node_features"] = tool_node_features(
                tr.jaw_box_pose[t], tr.jaw_box_vel[t], tr.jaw_joint[t],
                tr.action[t], tr.jaw_half_extents)
            te = tool_edges(x, v, tr.jaw_box_pose[t], tr.jaw_box_vel[t],
                            tr.jaw_half_extents, self.r_c, gc, gg)
            out["tool_edge_index"] = te["edge_index"]
            out["tool_edge_features"] = te["features"]
            assert out["tool_node_features"].shape[0] == K
        return out


def combined_pairs(*edge_indices: np.ndarray) -> np.ndarray:
    """Union of tissue edge sets as plain (i, j) pairs, for the to-do's
    "combined graph" statistics. The model never sees this -- it keeps the
    types separate -- but connectivity of E_ref U E_spatial is what message
    passing can actually traverse."""
    stacked = np.concatenate([np.asarray(e) for e in edge_indices if e.size], axis=1) \
        if any(e.size for e in edge_indices) else np.zeros((2, 0), np.int64)
    if stacked.size == 0:
        return stacked.astype(np.int64)
    return np.unique(stacked, axis=1).astype(np.int64)


def feature_widths(include_g: bool = True) -> Dict[str, int]:
    return {"node": len(NODE_FEATURES) - (0 if include_g else 2),
            "ref_edge": len(REF_EDGE_FEATURES),
            "spatial_edge": len(SPATIAL_EDGE_FEATURES),
            "tool_edge": len(TOOL_EDGE_FEATURES),
            "tool_node": len(TOOL_NODE_FEATURES)}


def check_edge_index(edge_index: np.ndarray, n: int,
                     symmetric: bool = True) -> Sequence[str]:
    """Structural problems with an edge set, as a list of messages (empty = OK).

    Used by the tests and by host/visualize_graph.py's checklist, so the same
    definition of "well-formed" applies in both.
    """
    e = np.asarray(edge_index)
    problems = []
    if e.ndim != 2 or e.shape[0] != 2:
        return [f"edge_index must be (2, E), got {e.shape}"]
    if e.size == 0:
        return problems
    if e.min() < 0 or e.max() >= n:
        problems.append(f"indices outside [0, {n}): {int(e.min())}..{int(e.max())}")
    if (e[0] == e[1]).any():
        problems.append(f"{int((e[0] == e[1]).sum())} self-loop(s)")
    if np.unique(e, axis=1).shape[1] != e.shape[1]:
        problems.append("duplicate edges")
    if symmetric:
        fwd = set(map(tuple, e.T.tolist()))
        missing = sum(1 for (i, j) in fwd if (j, i) not in fwd)
        if missing:
            problems.append(f"{missing} edge(s) without their reverse")
    return problems
