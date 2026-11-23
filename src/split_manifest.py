"""split_manifest.py -- decide train / held-out BEFORE any window is extracted.

numpy + json + hashlib only (CLAUDE.md `src/` rule).

WHY THIS FILE EXISTS (to-do Items 6 and 9)
------------------------------------------
"Reserve held-out episodes before extracting training windows" and "split by
episode, not random frames". A split decided after windows exist tends to be
decided while looking at results. Writing it to a file first, with a hash of
every episode, makes the split a fact on disk that a training script reads,
not a choice it makes.

THREE RULES
  1. Split by EPISODE. Never by frame (CLAUDE.md: consecutive frames are
     near-duplicates; a frame split leaks them into validation).
  2. At least one held-out episode per task type that has two or more
     episodes (Item 9: "Keep a held-out trajectory per available task type").
     A task with a single episode gets none, and the manifest says so: any
     result on it is overfit/debug, not generalisation.
  3. GROUPS STAY TOGETHER. Item 12's matched-initial-state variants share a
     seed and therefore a material and an initial state; one in train and its
     sibling held out would let the model see the held-out episode's first
     half. Grouping key: (task, seed).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_manifest(episodes: List[Dict], *, held_out_per_task: int = 1,
                  seed: int = 0, note: str = "") -> Dict:
    """`episodes`: dicts with at least "path", "task", "seed" (and optionally
    "group" to override the (task, seed) grouping, "config").

    Returns the manifest dict; `write_manifest` saves it. Held-out groups are
    drawn uniformly from each task's groups with a fixed RNG seed, so the
    split is reproducible and not hand-picked.
    """
    import numpy as np
    by_task: Dict[str, Dict[str, List[Dict]]] = {}
    for e in episodes:
        g = e.get("group") or f"{e['task']}:seed{e['seed']}"
        by_task.setdefault(e["task"], {}).setdefault(g, []).append(e)

    rng = np.random.default_rng(seed)
    rows, notes = [], []
    for task in sorted(by_task):
        groups = sorted(by_task[task])
        k = min(held_out_per_task, len(groups) - 1) if len(groups) > 1 else 0
        held = set(rng.choice(groups, size=k, replace=False).tolist()) if k else set()
        if not held:
            notes.append(f"task {task!r} has {len(groups)} group(s): nothing held out; "
                         "results on it are overfit/debug, not generalisation")
        for g in groups:
            for e in by_task[task][g]:
                rows.append({**e, "group": g, "split": "heldout" if g in held else "train",
                             "sha256": sha256(e["path"]) if os.path.exists(e["path"]) else None})
    return {"schema": "split_manifest/1",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rule": "by episode; >=1 held-out group per task with >=2 groups; "
                    "groups = (task, seed) so matched variants never straddle the split",
            "rng_seed": seed, "note": note, "warnings": notes, "episodes": rows}


def write_manifest(manifest: Dict, path: str) -> str:
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def load_split(path: str, split: str, verify: bool = True) -> List[str]:
    """Episode paths in `split` ("train" / "heldout"). With verify=True a file
    whose bytes changed since the manifest was written is an error: the split
    was decided about THAT file."""
    with open(path) as fh:
        m = json.load(fh)
    out = []
    for e in m["episodes"]:
        if e["split"] != split:
            continue
        if verify and e.get("sha256") and sha256(e["path"]) != e["sha256"]:
            raise ValueError(f"{e['path']} changed after the split was written")
        out.append(e["path"])
    return out


def episodes_in_dir(directory: str, task: Optional[str] = None) -> List[Dict]:
    """Discover episodes and their seeds from the .config.json written next to
    each grasp episode, or from the npz's own metadata for settle episodes."""
    import numpy as np
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".npz"):
            continue
        path = os.path.join(directory, name)
        cfg_path = path[:-4] + ".config.json"
        with np.load(path) as z:
            t = str(z["task"])
            notes = str(z["notes"])
        if task and t != task:
            continue
        seed = None
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                seed = json.load(fh).get("seed")
        if seed is None and "seed=" in notes:
            seed = int(notes.split("seed=")[1].split()[0].rstrip(","))
        out.append({"path": path, "task": t, "seed": seed,
                    "config": cfg_path if os.path.exists(cfg_path) else None})
    return out
