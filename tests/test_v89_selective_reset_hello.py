"""V89 reset invariants: HELLO cache and static ledgers do not bleed across environments."""
import numpy as np
import pytest
import torch
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params,WarpParallelMissionV80
from zhang_xue.gpu.state_v67 import wp

@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_selective_reset_does_not_mutate_other_world_or_leave_old_hello_cache(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p=GPUWorldV80Params.from_yaml("research/zhang_xue/configs/scenario_v89_static_1km.yaml",
                                  envs=2,device=device,seed=44)
    w=WarpParallelMissionV80(p)
    a=wp.zeros((2,6,2),dtype=wp.float32,device=device)
    d=wp.ones((2,6),dtype=wp.int32,device=device)
    for _ in range(5):
        w.step(a,d)
    before={
       "time":w.network.data["now"].numpy().copy(),
       "rx":w.network.data["hello_rx_at_us"].numpy().copy(),
       "ledger":w.network.data["ledger_pos"].numpy().copy(),
       "reward":w.buffers["reward_components"].numpy().copy(),
       "steps":w.buffers["episode_step"].numpy().copy(),
    }
    assert before["steps"].tolist()==[5,5]
    w.reset(np.array([1,0],dtype=np.uint8))
    assert w.buffers["episode_step"].numpy().tolist()==[0,5]
    assert w.network.data["now"].numpy()[0] == 600_000
    assert w.network.data["now"].numpy()[1] == before["time"][1]
    np.testing.assert_array_equal(w.network.data["hello_rx_at_us"].numpy()[1],before["rx"][1])
    np.testing.assert_array_equal(w.network.data["ledger_pos"].numpy()[1],before["ledger"][1])
    np.testing.assert_array_equal(w.buffers["reward_components"].numpy()[1],before["reward"][1])
    np.testing.assert_array_equal(w.buffers["reward_components"].numpy()[0],np.zeros(8))
    new_rx=w.network.data["hello_rx_at_us"].numpy()[0]
    assert np.all((new_rx<600_000) & (new_rx>=-1))
