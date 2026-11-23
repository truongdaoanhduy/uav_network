"""Mission metric identity must be target, not NoOracle REPORT slot.

Truth alignment is allowed only OUTSIDE Actor/critic/radio, on CPU evaluator.
"""
import numpy as np
import pytest
from uav_network.environment.gpu.evaluation_truth import truth_aligned_report_status


def test_metric_projection_uses_cell_id_not_report_slot_order():
    targets=np.array([[11,11],[91,91],[74,11]],np.float64)
    # 10x10 grid with 10m cells; target cell IDs [11,99,17].
    cells=np.array([50,17,99,11,-1,-1])
    known=np.array([[1,1,0,1,0,0],[0,0,1,0,0,0]],dtype=bool)
    delivered=np.array([1,1,1,0,0,0],dtype=bool)
    per_uav,received=truth_aligned_report_status(targets,cells,known,delivered,10,10.)
    assert per_uav.shape==(2,3)
    assert per_uav.tolist()==[[True,False,True],[False,True,False]]
    assert received.tolist()==[False,True,True]
    # Slot 0 was a false hypothesis, so it doesn't count as discovery/delivery.
    assert per_uav.any(axis=0).sum()==3


def test_metric_projection_invalid_shapes_fail_closed():
    with pytest.raises(ValueError):
        truth_aligned_report_status(np.zeros((3,2)),np.array([1,2]),np.zeros((1,3),bool),np.zeros(2,bool),10,10.)
