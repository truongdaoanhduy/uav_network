"""FP16 Replay is opt-in and quantized: check dtype, memory, and checkpointing."""
import math
import pytest
import torch
from uav_network.configuration import load_config
from uav_network.algorithms.replay import TensorReplayBuffer
from uav_network.experiments.pql_replay import MeasuredReplay, make_replay

BASE="configs/matd3-nooracle-map1000-tx06-prior0021875-reward02-actorstable-20261123.yaml"
FP16="configs/matd3-nooracle-map1000-t4x2-gpu-fp16-replay-ablation-20261123.yaml"


def test_optin_config_keeps_science_network_marl_budget():
    orig,new=load_config(BASE),load_config(FP16)
    assert orig.runtime.replay_storage_precision=="fp32"
    assert orig.world == new.world
    assert orig.sensor == new.sensor
    assert orig.network == new.network
    assert orig.energy == new.energy
    assert orig.reward == new.reward
    assert orig.matd3 == new.matd3
    assert orig.runtime.episodes == new.runtime.episodes == 20000
    assert new.runtime.replay_cpu_fraction==0
    assert new.runtime.replay_device=="cuda"
    assert new.runtime.replay_storage_precision=="fp16"
    new.validate()
    for bad in ("fp8","int16"):
        new.runtime.replay_storage_precision=bad
        with pytest.raises(ValueError,match="replay_storage_precision"):
            new.validate()
    new.runtime.replay_storage_precision="fp16"
    new.runtime.replay_cpu_fraction=.75
    with pytest.raises(ValueError,match="hybrid replay requires"):
        new.validate()
    new.runtime.replay_cpu_fraction=0.
    new.runtime.replay_device="cpu"
    with pytest.raises(ValueError,match="FP16 replay requires"):
        new.validate()


def _transitions(count=6):
    torch.manual_seed(44)
    observation=torch.randn(count,6,12)
    state=torch.randn(count,18)
    action=type("Action", (), {
        "continuous":torch.randn(count,6,4).clamp(-1,1),
        "destination":torch.arange(count).view(count,1).expand(count,6),
    })()
    reward=torch.arange(count,dtype=torch.float32) / 3.
    terminated=torch.zeros(count,dtype=torch.float32)
    return observation,state,action,reward,observation+.001,state+.001,terminated


def test_compact_replay_storage_bytes_and_sample_training_dtypes():
    dense=MeasuredReplay(16,(6,12),18,seed=44,device="cpu",continuous_dim=4)
    compact=MeasuredReplay(16,(6,12),18,seed=44,device="cpu",
                           continuous_dim=4,storage_precision="fp16")
    values=_transitions()
    dense.add_batch(*values)
    compact.add_batch(*values)
    assert compact.allocated_bytes < dense.allocated_bytes*.6
    assert compact.data["destination"].dtype==torch.int8
    assert compact.data["state"].dtype==torch.float16
    assert compact.data["reward"].dtype==torch.float32
    for _ in range(3):
        baseline=dense.sample(4)
        sampled=compact.sample(4)
        for name in baseline:
            assert baseline[name].dtype == sampled[name].dtype
            if baseline[name].dtype.is_floating_point:
                torch.testing.assert_close(sampled[name],baseline[name],
                                           atol=.0015,rtol=.0015)
            else:
                torch.testing.assert_close(sampled[name],baseline[name])
    assert compact.stats()["replay/sample_uses"]==3*4
    assert compact.stats()["replay/measurement_unknown_rows"]==0


def test_compact_replay_accurately_roundtrips_its_quantized_checkpoint():
    a=MeasuredReplay(16,(6,12),18,seed=55,device="cpu",
                     continuous_dim=4,storage_precision="fp16")
    a.add_batch(*_transitions())
    a.sample(4)
    state=a.state_dict()
    b=MeasuredReplay(16,(6,12),18,seed=99,device="cpu",
                     continuous_dim=4,storage_precision="fp16")
    b.load_state_dict(state)
    for k in a.data:
        torch.testing.assert_close(a.data[k][:a.size],b.data[k][:b.size])
    for _ in range(2):
        x,y=a.sample(4),b.sample(4)
        for key in x:
            torch.testing.assert_close(x[key],y[key])


def test_dense_storage_formula_vs_float16_gpu_estimate():
    obs=(6,228)
    dense=TensorReplayBuffer(1,obs,830,device="cpu",continuous_dim=4)
    compact=TensorReplayBuffer(1,obs,830,device="cpu",
                               continuous_dim=4,storage_precision="fp16")
    assert dense.allocated_bytes == 17736
    assert compact.allocated_bytes == 8854
    gib=8854*1_000_000/1024**3
    assert 8.2 < gib < 8.3
