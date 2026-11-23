"""Original 18d time-integrated 2D union coverage on 3D flight, no proxy."""
import pytest
import warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.communication.gpu.metrics import world_info


def test_nooracle_training_terminal_coverage_auc_is_exact_first_visit_integral():
    cfg=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    cfg.runtime.num_envs=1
    cfg.runtime.device='cpu'
    env=WarpVectorEnv(cfg,device='cpu')
    try:
        env.reset(seed=[44])
        times=wp.to_torch(env.s.coverage_first_seen_step)[0].clone()
        horizon=cfg.world.max_steps
        assert times.numel()==env.p.grid_n**2
        observed=(times<=horizon)
        assert observed.any()
        weighted=int(((horizon-times)[observed]).sum().item())
        accumulator=int(wp.to_torch(env.s.coverage_auc_weight_sum)[0].item())
        assert accumulator==weighted
        actual=world_info(env.s,env.p,0)['coverage_auc']
        assert actual==pytest.approx(weighted/(horizon*times.numel()),abs=1e-12)
        assert 0<actual<1
    finally:env.close()


def test_multi_uav_overlap_contributes_only_once_not_twice():
    import torch
    from uav_network.environment.gpu.bayesian import sense_bayesian
    cfg=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    cfg.runtime.num_envs=1
    cfg.runtime.device='cpu'
    cfg.world.num_uavs=2
    env=WarpVectorEnv(cfg,device='cpu')
    try:
        env.reset(seed=[44])
        wp.to_torch(env.s.positions)[0]=torch.tensor([[500.,500.],[500.,500.]])
        wp.to_torch(env.s.altitudes)[0]=50.
        wp.to_torch(env.s.visited).zero_()
        wp.to_torch(env.s.coverage_first_seen_step).fill_(cfg.world.max_steps+1)
        wp.to_torch(env.s.coverage_auc_weight_sum).zero_()
        env._launch(sense_bayesian,(1,2),env.s,env.p,0,wp.ones((1,),dtype=wp.int32,device='cpu'))
        times=wp.to_torch(env.s.coverage_first_seen_step)[0]
        count=int((times<=cfg.world.max_steps).sum().item())
        assert count>0
        expected=int((cfg.world.max_steps-times[times<=cfg.world.max_steps]).sum().item())
        assert int(wp.to_torch(env.s.coverage_auc_weight_sum)[0])==expected
        assert expected <= cfg.world.max_steps*count
    finally:env.close()
