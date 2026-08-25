"""
test_trajectory_io.py -- the data contract, under pytest.

`python src/trajectory_io.py` has always run a self-test covering most of this,
and it stays: SETUP_GUIDE.md tells the user to run it, and verify_host.py and
verify_container.py both run it as a subprocess to prove the format round-trips
on a fresh machine. That is a different job from this file. A self-test behind
`if __name__ == "__main__"` runs only when someone remembers to run it, reports
the first failure and stops, and cannot be collected, selected or counted by
CI. Anything it is the *only* cover for is effectively untested.

So the properties that would cost real data if they broke live here as well:
the delta16 encoding, the v1/v2/v2.1 compatibility path, rejection of unknown
encodings, particle-id coherence, and the subset bookkeeping that schema 2.1
introduced.

Numpy only, like everything in tests/ -- this must pass in both environments.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))

from trajectory_io import (  # noqa: E402
    F_ENCODINGS,
    KNOWN_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    TIMEBASE_TOLERANCE,
    TrajectoryWriter,
    load_trajectory,
)

N, T = 40, 8


def _write(path, **kw):
    """A minimal valid episode. Any writer argument is overridable."""
    steps = kw.pop("_steps", T)
    F = kw.pop("_F", None)
    with TrajectoryWriter(path, "test_mpm", "tissue_retraction", 0.0125, **kw) as w:
        for t in range(steps):
            frame = {} if F is None else {"tissue_F": F[t]}
            w.append(tissue_pos=np.full((N, 3), 0.001 * t, np.float32),
                     ee_pose=np.array([0, 0, 0.1, 0, 0, 0, 1.0]),
                     action=np.zeros(7), **frame)
    return load_trajectory(path)


# --------------------------------------------------------------------------
# delta16 -- the encoding schema 2.1 exists for
# --------------------------------------------------------------------------

class TestDelta16RoundTrip:
    """tissue_F is 864 KB/frame at 24k particles, so it gets stored at half
    precision. The naive way destroys the data: float16's spacing at 1.0 is
    ~9.8e-4, and F sits AT 1.0 at rest, so a small strain rounds away entirely.
    delta16 stores F - I, moving the quantisation to where float16 is dense.
    """

    SMALL = 1.0e-4          # 0.01% stretch -- below float16's step above 1.0

    def test_delta16_keeps_a_strain_that_plain_float16_erases(self, tmp_path):
        F = np.tile(np.diag([1.0 + self.SMALL, 1.0, 1.0]), (T, N, 1, 1))
        tr = _write(str(tmp_path / "d16.npz"), f_encoding="delta16", _F=F)
        got = float(tr.tissue_F[0, 0, 0, 0])
        assert got == pytest.approx(1.0 + self.SMALL, abs=1e-6)

        # Demonstrated against the alternative, not merely asserted: the same
        # value through plain float16 collapses to exactly 1.0.
        naive = float(np.float32(np.float16(np.float32(1.0 + self.SMALL))))
        assert naive == 1.0

    def test_plain_float16_really_does_lose_it(self, tmp_path):
        """The counterpart, run through the actual writer rather than through
        numpy by hand -- otherwise this file asserts a fact about float16 and
        not a fact about the encoder."""
        F = np.tile(np.diag([1.0 + self.SMALL, 1.0, 1.0]), (T, N, 1, 1))
        tr = _write(str(tmp_path / "f16.npz"), f_encoding="float16", _F=F)
        assert float(tr.tissue_F[0, 0, 0, 0]) == 1.0

    def test_float32_encoding_keeps_the_strain_to_float32_precision(self, tmp_path):
        """"Lossless" here means lossless AT float32, which is the dtype the
        reader hands back for every encoding. A float64 input still gets
        rounded once on the way in -- 1.0001 stores as 1.000100016593933 -- so
        the bound is float32 epsilon, not zero. The point of the test is that
        the error is ~1e-8 rather than delta16's ~1e-7 or float16's total loss.
        """
        F = np.tile(np.diag([1.0 + self.SMALL, 1.0, 1.0]), (T, N, 1, 1))
        tr = _write(str(tmp_path / "f32.npz"), f_encoding="float32", _F=F)
        got = float(tr.tissue_F[0, 0, 0, 0])
        assert got == pytest.approx(1.0 + self.SMALL, rel=1e-7)
        assert got != 1.0

    def test_every_encoding_reads_back_as_float32(self, tmp_path):
        """The storage choice must not leak into the dtype downstream metrics
        see. compute_safety_strain does an SVD; handing it float16 would change
        results for reasons no caller asked about."""
        F = np.tile(np.eye(3) * 1.05, (T, N, 1, 1))
        for enc in F_ENCODINGS:
            tr = _write(str(tmp_path / f"enc_{enc}.npz"), f_encoding=enc, _F=F)
            assert tr.tissue_F.dtype == np.float32, enc
            assert tr.tissue_F.shape == (T, N, 3, 3), enc

    def test_delta16_is_exact_at_rest(self, tmp_path):
        """F = I is the single most common value in the whole dataset, and
        delta16 stores it as exactly zero, so it must survive bit-for-bit."""
        F = np.tile(np.eye(3, dtype=np.float32), (T, N, 1, 1))
        tr = _write(str(tmp_path / "rest.npz"), f_encoding="delta16", _F=F)
        assert np.array_equal(tr.tissue_F[0, 0], np.eye(3, dtype=np.float32))

    def test_delta16_beats_float16_on_a_realistic_strain_field(self, tmp_path):
        """Not just the adversarial small case: over a spread of strains around
        rest, delta16's worst-case error must be the smaller of the two."""
        rng = np.random.default_rng(0)
        F = np.tile(np.eye(3), (T, N, 1, 1)) + rng.normal(0, 2e-3, (T, N, 3, 3))
        a = _write(str(tmp_path / "a.npz"), f_encoding="delta16", _F=F)
        b = _write(str(tmp_path / "b.npz"), f_encoding="float16", _F=F)
        err_delta = np.abs(a.tissue_F - F).max()
        err_naive = np.abs(b.tissue_F - F).max()
        assert err_delta < err_naive


class TestEncodingIsValidated:
    def test_unknown_encoding_is_refused(self, tmp_path):
        """An unrecognised encoding must not fall through to a default. The
        reader would then decode with the wrong scheme and return plausible
        numbers -- F off by exactly I, which looks like a physics bug."""
        with pytest.raises(ValueError, match="f_encoding"):
            TrajectoryWriter(str(tmp_path / "x.npz"), "s", "t", 0.01,
                             f_encoding="float64")

    def test_the_two_spellings_of_float16_may_not_disagree(self, tmp_path):
        """store_F_as_float16 predates f_encoding and both still work. If they
        contradict each other the writer must refuse rather than pick one."""
        with pytest.raises(ValueError, match="conflicts"):
            TrajectoryWriter(str(tmp_path / "y.npz"), "s", "t", 0.01,
                             store_F_as_float16=True, f_encoding="delta16")

    def test_the_legacy_flag_still_selects_float16(self, tmp_path):
        """v2.0 files and src/synthetic_traj.py pass the old flag. It has to
        keep meaning what it meant."""
        F = np.tile(np.eye(3) * 1.5, (T, N, 1, 1))
        tr = _write(str(tmp_path / "legacy.npz"), store_F_as_float16=True, _F=F)
        assert tr.f_encoding == "float16"


# --------------------------------------------------------------------------
# Schema compatibility -- old files must keep opening
# --------------------------------------------------------------------------

class TestSchemaCompatibility:
    def test_current_version_is_known(self):
        assert SCHEMA_VERSION == "2.1"
        assert SCHEMA_VERSION in KNOWN_SCHEMA_VERSIONS

    def test_older_versions_are_still_readable(self):
        """data/ is v1.0 and the synthetic episodes are v2.0. Dropping either
        from the reader strands data that is still on disk."""
        for older in ("1.0", "2.0"):
            assert older in KNOWN_SCHEMA_VERSIONS

    def test_a_v1_style_caller_still_works(self, tmp_path):
        """Exactly how container/collect_retraction.py calls the writer: no v2
        argument anywhere. If this breaks, the PyBullet collector is broken."""
        p = str(tmp_path / "v1.npz")
        with TrajectoryWriter(p, "pybullet", "tissue_retraction", 1 / 240) as w:
            for t in range(T):
                w.append(tissue_pos=np.zeros((N, 3)),
                         ee_pose=np.array([0, 0, 0.1 * t, 0, 0, 0, 1]),
                         action=np.zeros(4), jaw=0.3)
        tr = load_trajectory(p)
        assert len(tr) == T and tr.n_nodes == N

    def test_unsupplied_v2_fields_are_empty_not_zero(self, tmp_path):
        """The schema's sharpest rule: an empty array means "not recorded",
        all-zero means "recorded, and it was zero". 0.0 is a real exposure
        value and a real contact mode, so the two must stay distinguishable."""
        tr = _write(str(tmp_path / "sparse.npz"))
        assert tr.exposure.size == 0
        assert tr.safety_strain.size == 0
        assert tr.contact_mode.size == 0
        assert tr.material_params.size == 0
        assert tr.boundary_mask.size == 0
        assert not tr.has_F

    def test_a_file_from_the_future_is_refused(self, tmp_path):
        """Better to refuse than to read a newer layout with older rules."""
        p = str(tmp_path / "future.npz")
        _write(p)
        d = dict(np.load(p, allow_pickle=False))
        d["schema_version"] = np.array("99.0")
        np.savez_compressed(p, **d)
        with pytest.raises(ValueError, match="schema"):
            load_trajectory(p)


# --------------------------------------------------------------------------
# Subsets -- the v2.1 bookkeeping
# --------------------------------------------------------------------------

class TestSubsetBookkeeping:
    """24,000 particles is what the solver needs; 3,000 is what the model can
    take. The file records which 3,000, and those three numbers -- n_nodes,
    particle_ids, n_particles_simulated -- can silently disagree.
    """

    def test_a_subset_reports_itself_as_one(self, tmp_path):
        ids = np.linspace(0, 23999, N).astype(np.int32)
        tr = _write(str(tmp_path / "sub.npz"), particle_ids=ids,
                    n_particles_simulated=24000)
        assert tr.is_subset
        assert tr.n_nodes == N
        assert int(tr.n_particles_simulated) == 24000
        assert tr.subset_fraction == pytest.approx(N / 24000)

    def test_a_complete_record_is_not_a_subset(self, tmp_path):
        """The normal case for a mesh solver, and for an MPM run that records
        everything. is_subset must be False so the metric checks demand
        equality rather than the looser subset bounds."""
        tr = _write(str(tmp_path / "full.npz"))
        assert not tr.is_subset

    def test_duplicate_particle_ids_are_refused(self, tmp_path):
        """One particle recorded as two nodes: two perfectly correlated columns
        masquerading as independent data, which a model will happily exploit."""
        with pytest.raises(ValueError, match="duplicate"):
            TrajectoryWriter(str(tmp_path / "dup.npz"), "s", "t", 0.01,
                             particle_ids=np.array([0, 1, 1], np.int32))

    def test_ids_beyond_the_simulated_count_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="only 10 particles"):
            TrajectoryWriter(str(tmp_path / "oob.npz"), "s", "t", 0.01,
                             particle_ids=np.array([0, 1, 99], np.int32),
                             n_particles_simulated=10)

    def test_negative_ids_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="negative"):
            TrajectoryWriter(str(tmp_path / "neg.npz"), "s", "t", 0.01,
                             particle_ids=np.array([0, -1, 2], np.int32))

    def test_a_subset_cannot_be_larger_than_its_superset(self, tmp_path):
        with pytest.raises(ValueError, match="cannot be larger"):
            with TrajectoryWriter(str(tmp_path / "big.npz"), "s", "t", 0.01,
                                  n_particles_simulated=3) as w:
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7))

    def test_particle_ids_length_must_match_the_nodes_stored(self, tmp_path):
        """particle_ids names the stored nodes, so a mismatch means the file
        cannot say which particle any row belongs to."""
        with pytest.raises(ValueError):
            with TrajectoryWriter(str(tmp_path / "mismatch.npz"), "s", "t", 0.01,
                                  particle_ids=np.arange(N + 5, dtype=np.int32),
                                  n_particles_simulated=24000) as w:
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7))


# --------------------------------------------------------------------------
# The writer's other refusals
# --------------------------------------------------------------------------

class TestWriterRefusals:
    def test_a_half_populated_optional_field_is_refused(self, tmp_path):
        """Supplying an optional field on some steps but not others produces a
        column shorter than tissue_pos, which nothing downstream detects."""
        with pytest.raises(ValueError, match="all-or-nothing"):
            with TrajectoryWriter(str(tmp_path / "half.npz"), "s", "t", 0.01) as w:
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7), exposure=0.5)
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7))

    def test_a_mis_sized_boundary_mask_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="boundary_mask"):
            with TrajectoryWriter(str(tmp_path / "mask.npz"), "s", "t", 0.01,
                                  boundary_mask=np.zeros(N + 1, bool)) as w:
                w.append(tissue_pos=np.zeros((N, 3)), ee_pose=np.zeros(7),
                         action=np.zeros(7))

    def test_an_unknown_action_spec_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="action_spec"):
            TrajectoryWriter(str(tmp_path / "spec.npz"), "s", "t", 0.01,
                             action_spec="wiggle")

    def test_a_negative_grid_dx_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="grid_dx"):
            TrajectoryWriter(str(tmp_path / "dx.npz"), "s", "t", 0.01,
                             grid_dx=-1.0)


class TestTimebaseInvariant:
    """dt must equal substep_dt * n_substeps. See DECISION_LOG 9.5 -- the MPM
    adapter shipped a version where it did not, and the file was internally
    consistent while every frame advanced 4.2x less time than it claimed.
    """

    def test_a_closing_timebase_is_accepted(self, tmp_path):
        tr = _write(str(tmp_path / "ok.npz"), substep_dt=0.0125 / 100,
                    n_substeps=100)
        assert float(tr.substep_dt) * int(tr.n_substeps) == pytest.approx(0.0125)

    def test_the_real_bug_is_refused(self, tmp_path):
        """The exact numbers the adapter produced: the P-wave substep for the
        stiffest sampled material, against the vendored module's 25 steps."""
        with pytest.raises(ValueError, match="timebase does not close"):
            TrajectoryWriter(str(tmp_path / "bug.npz"), "taichi_mpm", "t", 0.0125,
                             substep_dt=1.179e-4, n_substeps=25)

    def test_recording_neither_field_is_allowed(self, tmp_path):
        """v1 files and the PyBullet collector record no substep information,
        and a solver that does not substep is not lying about anything."""
        tr = _write(str(tmp_path / "none.npz"))
        assert float(tr.substep_dt) == 0.0 and int(tr.n_substeps) == 0

    def test_float32_reconstruction_error_is_tolerated(self, tmp_path):
        """substep_dt is stored as float32, so dt cannot be reconstructed from
        it exactly. Measured at 4.7e-10 s on a real 24-substep episode; the
        bound is relative precisely so this passes and a real error does not."""
        n = 24
        tr = _write(str(tmp_path / "f32.npz"), substep_dt=0.0125 / n, n_substeps=n)
        implied = float(tr.substep_dt) * int(tr.n_substeps)
        assert abs(implied - 0.0125) / 0.0125 < TIMEBASE_TOLERANCE
