"""Hand-computed reliability scores and first-terminal snapshot semantics."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/gpu"))


def test_first_terminal_snapshot_is_not_overwritten_by_later_world_steps():
    from v93_probe_metrics import TerminalSnapshots
    s = TerminalSnapshots(2)
    s.capture(np.array([True, False]), {"tp": np.array([3, 4])})
    s.capture(np.array([True, True]), {"tp": np.array([99, 5])})
    assert s.values["tp"].tolist() == [3, 5]
    assert s.seen.tolist() == [True, True]


def test_brier_and_ece_against_four_hand_computed_predictions():
    from v93_probe_metrics import binary_reliability
    r = binary_reliability([0., .25, .75, 1.], [0, 0, 1, 1], bins=2)
    assert r["brier"] == pytest.approx(.03125)
    assert r["ece_equal_width"] == pytest.approx(.125)
    assert r["adaptive_ece_positive_class"] == pytest.approx(.125)


def test_equal_probability_ties_cannot_change_calibration_when_rows_reordered():
    from v93_probe_metrics import binary_reliability
    p = np.full(6, .5)
    a = binary_reliability(p, [0, 0, 0, 1, 1, 1], bins=4)
    b = binary_reliability(p, [0, 1, 0, 1, 0, 1], bins=4)
    assert a == b
    assert a["adaptive_ece_positive_class"] == 0.


def test_reliability_does_not_turn_an_empty_visited_set_into_perfect_score():
    from v93_probe_metrics import binary_reliability
    result = binary_reliability([], [], bins=15)
    assert result["n"] == 0 and result["brier"] is None


def test_nonfinite_or_out_of_range_predictions_are_rejected():
    from v93_probe_metrics import binary_reliability
    for probabilities in ([float("nan")], [-.1], [1.1]):
        with pytest.raises(ValueError):
            binary_reliability(probabilities, [1])


def test_paired_belief_evaluation_excludes_unvisited_and_uses_same_samples():
    from v93_probe_metrics import compare_counts
    r = compare_counts([2, 0, 0], [0, 2, 0], [1, 0, 1],
                       prior=.1, pd=.8, pf=.1, mf=.8, me=.8)
    assert r["visited_cells"] == 2
    assert r["bayes"]["n"] == r["ds_pignistic"]["n"] == 2
    # P(T|++) = .1*.8^2 / (.1*.8^2 + .9*.1^2) = 64/73
    assert r["bayes_posterior"][0] == pytest.approx(64/73)

