"""Numeric and scientific labels must match actual mathematics."""
import math
import pytest
from zhang_xue.belief.dempster_shafer import DENG_MAX_BITS, deng_entropy
from zhang_xue.environment.configuration import make_v62_config


def test_deng_vacuous_entropy_is_not_global_maximum():
    from zhang_xue.belief.dempster_shafer import DENG_THEORETICAL_MAX_BITS
    assert deng_entropy(0.0,0.0,1.0)==pytest.approx(DENG_MAX_BITS)
    assert deng_entropy(0.2,0.2,0.6)==pytest.approx(DENG_THEORETICAL_MAX_BITS)
    assert DENG_THEORETICAL_MAX_BITS==pytest.approx(math.log2(5.0))
    assert DENG_THEORETICAL_MAX_BITS>DENG_MAX_BITS


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),0,-1])
def test_power_bounds_reject_nonfinite_or_nonpositive(bad):
    with pytest.raises(ValueError):
        make_v62_config(tx_power_min_w=bad)
    with pytest.raises(ValueError):
        make_v62_config(tx_power_max_w=bad)
