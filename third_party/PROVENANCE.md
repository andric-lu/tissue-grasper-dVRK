# third_party/ — vendored code and where it came from

Code in here was written by someone else. It is copied into this repository
rather than installed, so it needs a record of exactly which copy it is.

---

## MPM/

Taichi MPM soft-body solver from SurRoL's `Dev` branch.

| | |
|---|---|
| Upstream | https://github.com/med-air/SurRoL |
| Branch | `Dev` |
| Commit | `cb797f360bb16ea629b449cb902a1dae60c46e81` |
| Commit date | 2024-05-21 |
| Vendored on | 2026-08-17 |
| Paper | [arXiv:2402.01181](https://arxiv.org/abs/2402.01181) |
| Upstream path | `MPM/` |

Retrieved with a sparse checkout, which avoids cloning a repository whose other
branches carry large paper artifacts:

```bash
git clone --filter=blob:none --no-checkout --depth 1 --branch Dev \
    https://github.com/med-air/SurRoL.git surrol_dev
cd surrol_dev
git sparse-checkout init --cone
git sparse-checkout set MPM
git checkout
git rev-parse HEAD          # cb797f360bb16ea629b449cb902a1dae60c46e81
```

### Files, as vendored

`sha256`, so a later "did I change this?" is answerable without the network:

```
f91ff08592d7be16bb804e73f86c13f1a61b1486344d36aee2fcc78cf72bfbc0  config.py
cffb3b7a6b1ab5f49a872a286371cd71253d08aeb1d44f37c7a329458f192447  mpm3d.py
1cf634fedaa82b272673f2d433f9253e2cb201b4a6d88f959b4ac3f0b040935a  sdf.py
c044d73a1b9f07d9b347590aad723d8f4da8e0ce23e9d5bf9c69a19234643bf1  requirements.txt
```

### Local modifications

**None yet.** These four files are byte-identical to upstream at that commit.

When that changes, say so here — what was changed and why — because from that
point the SHA above stops describing the code in this directory and starts
describing only its ancestor.

### Why vendored rather than a submodule

DECISION_LOG.md §9.3. These files are expected to be edited: to expose particle
state to the adapter and to script the tool. A submodule you cannot edit is
friction with no benefit. The commit SHA is what keeps the fork honest.

### Why it is not installed as a dependency

`MPM/requirements.txt` pins `taichi==1.6.0` and asks for `pymeshlab` and
`plyfile`. Neither of the latter two is imported by any of the four files —
they belong to upstream scripts that were not vendored. The host environment
runs taichi **1.7.4**, not 1.6.0; `host/smoke_test_mpm.py` exists largely to
check what that version gap breaks. So `requirements.txt` is kept for the
record, and `host/environment.yml` remains the file that is actually installed.

### Known divergence from DECISION_LOG.md §9.3

§9.3 states "Neo-Hookean … with μ/λ from `set_parameters()`". The actual
signature is `set_parameters(s_E=8000, s_nu=0.2)` — it takes **(E, ν)** and
derives μ and λ internally through

```
la = E*nu / ((1 + nu)(1 - 2nu))
```

which is precisely the singular conversion `src/materials.py` exists to keep
out of the sampling path. μ and λ are plain Taichi fields and can be written
directly; the adapter should do that rather than round-trip through (E, ν).
Measured cost of the round trip is small (worst relative error 2.7e-4 over
`materials.py`'s ranges) but it is avoidable for free. See §9.4.

### Imports, confirmed at this commit

`taichi`, `numpy`, `pybullet`, `skimage.measure`. No `surrol.*`, no `panda3d` —
which is what made lifting these four files out possible.

`from MPM.config import ...` is an absolute import of a top-level package named
`MPM`, so **`third_party/` goes on `sys.path`**, not `third_party/MPM/`. There
is no `__init__.py`; it resolves as a PEP 420 namespace package.
