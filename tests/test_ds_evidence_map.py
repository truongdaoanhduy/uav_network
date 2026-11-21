"""Paper-faithful DS occupancy evidence: no truth-aware target confirmation."""
import numpy as np
import pytest

from uav_network.environment.ds_evidence import (
    DSEvidenceMap, combine_evidence, mass_from_counts,
)


def test_paper_table_single_and_repeated_positive():
    assert combine_evidence((0, 0, 1), True, 0.8) == pytest.approx((0.8, 0, 0.2))
    assert combine_evidence((0.8, 0, 0.2), True, 0.8) == pytest.approx((0.96, 0, 0.04))


def test_paper_conflicting_observations():
    pos = combine_evidence((0, 0, 1), True, 0.8)
    both = combine_evidence(pos, False, 0.8)
    assert both == pytest.approx((4 / 9, 4 / 9, 1 / 9))
    assert combine_evidence(combine_evidence((0, 0, 1), False, 0.8), True, 0.8) == pytest.approx(both)


def test_paper_closed_form_matches_recursive_updates():
    for n_pos in range(5):
        for n_neg in range(5):
            m = (0, 0, 1)
            for _ in range(n_pos):
                m = combine_evidence(m, True, 0.8)
            for _ in range(n_neg):
                m = combine_evidence(m, False, 0.8)
            assert mass_from_counts(n_pos, n_neg, 0.8, 0.8) == pytest.approx(m, abs=1e-8)


def test_counts_with_many_conflicting_events_are_finite():
    result = mass_from_counts(1000, 1000, 0.8, 0.8)
    assert np.isfinite(result).all()
    assert sum(result) == pytest.approx(1.0)
    assert result[0] == pytest.approx(0.5, abs=1e-7)
    assert result[1] == pytest.approx(0.5, abs=1e-7)


def test_local_map_predicts_only_from_observations_not_true_target_ids():
    mapper = DSEvidenceMap(num_uavs=2, grid_cells=3)
    assert mapper.local_masses().shape == (2, 3, 3, 3)
    assert mapper.local_masses()[0, 1, 2] == pytest.approx((0, 0, 1))
    mapper.observe(uav_id=0, row=1, col=2, detected=True)
    assert not mapper.target_candidates(uav_id=0).any()
    mapper.observe(uav_id=0, row=1, col=2, detected=True)
    assert mapper.target_candidates(uav_id=0)[1, 2]
    assert not mapper.target_candidates(uav_id=1).any()
    mapper.observe(uav_id=1, row=0, col=0, detected=False)
    assert mapper.local_masses()[1, 0, 0] == pytest.approx((0, 0.8, 0.2))


def test_fusion_uses_max_count_exactly_as_paper_not_sum():
    mapper = DSEvidenceMap(num_uavs=2, grid_cells=1)
    mapper.observe(0, 0, 0, True)
    mapper.observe(1, 0, 0, True)
    # Zhang Eq.(16) uses max per-sign counts; NOT the total of both sensors.
    assert mapper.paper_global_masses()[0, 0] == pytest.approx((0.8, 0, 0.2))


def test_heterogeneous_evidence_cannot_be_reconstructed_from_simple_counts():
    mapper = DSEvidenceMap(num_uavs=2, grid_cells=1)
    mapper.observe(0, 0, 0, True, confidence=0.6)
    assert mapper.local_masses()[0, 0, 0] == pytest.approx((0.6, 0, 0.4))
    with pytest.raises(ValueError, match='fixed'):
        mapper.paper_global_masses()


def test_bad_input_is_rejected_before_map_changes():
    mapper = DSEvidenceMap(num_uavs=2, grid_cells=2)
    before = mapper.local_masses().copy()
    with pytest.raises(ValueError):
        mapper.observe(0, 0, 0, True, confidence=1.0)
    with pytest.raises(IndexError):
        mapper.observe(0, 2, 0, True)
    assert np.array_equal(before, mapper.local_masses())


def test_map_reset_and_readout_are_independent():
    mapper = DSEvidenceMap(num_uavs=1, grid_cells=2)
    mapper.observe(0, 0, 0, True)
    copy = mapper.local_masses()
    copy[0, 0, 0, 0] = -999.0
    assert mapper.local_masses()[0, 0, 0, 0] == pytest.approx(0.8)
    mapper.reset()
    assert mapper.local_masses()[0, 0, 0] == pytest.approx((0, 0, 1))
