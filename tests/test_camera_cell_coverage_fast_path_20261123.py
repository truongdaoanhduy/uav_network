"""Fast exact-zero path for disjoint camera cells avoids circle integrals."""
from unittest.mock import patch
import numpy as np
import pytest
from uav_network.environment import bayesian


def test_outside_square_diagonally_is_exact_zero_without_sqrt_or_arc():
    # Cell [575,600]^2 has X/Y bbox overlap with radius75 but its NEAR
    # corner is ~88.4m away; no positive overlap is geometrically possible.
    # Old code unnecessarily evaluates sqrt/arc sorting despite area=0.
    with patch.object(bayesian.math,'sqrt',side_effect=AssertionError('slow circle integration')):
        assert bayesian.cell_coverage(23,23,np.array([512.5,512.5]),75,25,1000)==0.0


@pytest.mark.parametrize('gx,gy',[ (0,0),(19,19),(20,20),(21,20),(30,30)])
def test_nontrivial_cells_still_return_physical_probability_in_unit_interval(gx,gy):
    value=bayesian.cell_coverage(gx,gy,[512.5,512.5],75,25,1000)
    assert 0<=value<=1
