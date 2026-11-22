"""Twin GPU collector/learner Actor sync must copy params AND BN buffers."""
import inspect
from unittest.mock import patch
import torch
from uav_network.configuration import Config
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.algorithms.checkpoints import make_learner
from uav_network.experiments.training import refresh_base_collector


def test_direct_gpu_collector_copy_is_explicit_opt_in():
    source=inspect.getsource(refresh_base_collector)
    assert 'UAV_DIRECT_GPU_COLLECTOR_COPY' in source
    assert 'learner.actors.state_dict()' in source


def test_direct_copy_actor_batchnorm_buffers_and_weights_equal_reference():
    c=Config(); c.runtime.algorithm='matd3'; c.matd3.hidden_sizes=[16,16]; c.matd3.actor_batch_norm=True
    c.matd3.actor_layer_norm=True;c.validate()
    source=make_learner(c,observation_dim(c),state_dim(c),'cpu')
    old=make_learner(c,observation_dim(c),state_dim(c),'cpu')
    direct=make_learner(c,observation_dim(c),state_dim(c),'cpu')
    torch.manual_seed(44)
    with torch.no_grad():
        for v in source.actors.state_dict().values():
            if v.dtype.is_floating_point:v.copy_(torch.randn_like(v))
            else:v.add_(9)
    with patch.dict('os.environ', {'UAV_DIRECT_GPU_COLLECTOR_COPY':'false'}):
        refresh_base_collector(old,source)
    with patch.dict('os.environ', {'UAV_DIRECT_GPU_COLLECTOR_COPY':'true'}):
        refresh_base_collector(direct,source)
    for k,v in source.actors.state_dict().items():
        assert torch.equal(old.actors.state_dict()[k],v),k
        assert torch.equal(direct.actors.state_dict()[k],v),k
    assert all(torch.equal(old.actors.state_dict()[k],direct.actors.state_dict()[k]) for k in source.actors.state_dict())
