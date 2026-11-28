"""Regression for sequential DS false-alarm calculation."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/maintenance"))
from v107_ds_first_passage import passage

def test_two_positive_samples_at_old_threshold():
    assert passage(.03,.90,2)["2"] == pytest.approx(.03**2)

def test_99_threshold_improves_sequential_false_alarm():
    old = passage(.03,.90)["900"]
    new = passage(.03,.99)["900"]
    assert new < old/30
    assert passage(.85,.99)["10"] > .97
    assert passage(.03,.99,2)["2"] == 0
