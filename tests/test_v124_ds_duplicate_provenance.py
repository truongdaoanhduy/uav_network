"""C95: repeated evidence same source-cell-tick must not be counted twice."""
import pytest
from zhang_xue.belief.ds_correlation_mitigation import DSCorrelationMitigator

@pytest.mark.parametrize('mode',DSCorrelationMitigator.VALID_MODES)
def test_duplicate_tick_idempotent_across_all_modes(mode):
    d=DSCorrelationMitigator(num_uavs=6,num_cells=400,mode=mode,cooldown_s=3)
    assert d.record(0,42,True,1)>0
    assert d.record(0,42,True,1)==0
    assert d.record(0,42,False,1)==0
    assert float(d.eff_pos[0,42])==1.0
    assert float(d.eff_neg[0,42])==0.0

@pytest.mark.parametrize('kwargs',({'num_uavs':9},{'num_uavs':0},{'num_cells':0},{'cooldown_s':32768},{'rho_hat':1.0}))
def test_invalid_shape_or_numeric_rejected(kwargs):
    with pytest.raises(ValueError): DSCorrelationMitigator(**kwargs)

@pytest.mark.parametrize('u,c,t',[(-1,1,1),(0,-1,1),(6,1,1),(0,400,1),(0,1,-1),(0,1,32768)])
def test_invalid_origin_cell_or_tick_rejected(u,c,t):
    d=DSCorrelationMitigator()
    with pytest.raises(ValueError):d.record(u,c,True,t)

def test_time_rewind_forbidden():
    d=DSCorrelationMitigator()
    assert d.record(0,3,True,9)>0
    with pytest.raises(ValueError):d.record(0,3,False,8)
