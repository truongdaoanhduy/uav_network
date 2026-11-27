"""Do not commit the observed seed44/world7 step143 inevitable-collision state.

At the resulting state, UAV3/5 cannot possibly end the next 1s step 50m apart:
even ignoring speed limits their maximum endpoint separation is only 44.9934m.
The bound follows from the triangle inequality and two 5m/s velocity changes.
"""
import numpy as np
import pytest
import warp as wp
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80

POSITIONS = np.array([
    [583.2967529296875,765.168701171875,50.],
    [651.9452514648438,599.2255249023438,50.],
    [658.078125,652.883056640625,50.],
    [593.8975830078125,716.0562133789062,50.],
    [598.0559692382812,600.1155395507812,50.],
    [602.9916381835938,650.2520751953125,50.],
    [50.,50.,15.],
],dtype=np.float32)
VELOCITIES = np.array([
    [-10.03382682800293,-4.632073879241943],[-11.037171363830566,2.5178470611572266],
    [2.6073436737060547,-5.278759479522705],[.25875312089920044,-2.2298288345336914],
    [-5.239447593688965,13.932930946350098],[-3.2840373516082764,11.961078643798828],
],dtype=np.float32)
ACTION = np.array([
    [-.5977882742881775,-.8016539812088013],[-.9861544370651245,.16582965850830078],
    [-.7101337909698486,-.7040667533874512],[.9771653413772583,.21248023211956024],
    [-.30531126260757446,.9522526264190674],[-.46915915608406067,.8831136226654053],
],dtype=np.float32)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_inevitable_next_step_collision_is_rejected_before_commit(device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    p=GPUWorldV80Params.from_yaml("research/zhang_xue/configs/scenario_v91_bayes_1km.yaml",envs=1,device=device)
    w=WarpParallelMissionV80(p)
    for name,value in [("positions",POSITIONS[None]),("velocities",VELOCITIES[None]),("pos_xy",POSITIONS[None,:6,:2])]:
        wp.copy(w.buffers[name],wp.array(value,dtype=wp.float32,device=device))
    w._update_phy()
    w.step(wp.array(ACTION[None],dtype=wp.float32,device=device),wp.zeros((1,6),dtype=wp.int32,device=device))
    assert int(w.buffers["failed"].numpy()[0]) == 1
    assert int(w.buffers["safety_reason"].numpy()[0]) & 32
    np.testing.assert_array_equal(w.buffers["positions"].numpy()[0],POSITIONS)

