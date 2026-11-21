"""Required no-oracle MATD3 train/validation/test integration preflight."""
import numpy as np
import pytest
import torch
import warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv


def profile(**changes):
    opts=['runtime.device=cpu','runtime.num_envs=2','tracking.enabled=false','network.enabled=false']
    opts += [f'{key}={value}' for key,value in changes.items()]
    return load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml',opts)


def test_heldout_gpu_evaluator_may_advance_only_live_worlds_without_restarting_finished_worlds():
    c=profile()
    env=WarpVectorEnv(c,device='cpu')
    try:
        env.reset(seed=[1000,1001])
        old=wp.to_torch(env.s.positions)[0].clone()
        wp.to_torch(env.s.terminated)[0]=1
        cont=np.zeros((2,c.world.num_uavs,4),dtype=np.float32)
        dst=np.zeros((2,c.world.num_uavs),dtype=np.int64)
        obs,reward,terminated,truncated,infos=env.step({'continuous':cont,'destination':dst})
        assert bool(terminated[0])
        torch.testing.assert_close(wp.to_torch(env.s.positions)[0],old)
        assert int(wp.to_torch(env.s.steps)[1])==1
    finally:env.close()


def test_historical_stale_world_action_rejection_remains_unchanged():
    c=profile(**{'sensor.report_cell_hypotheses':'false'})
    env=WarpVectorEnv(c,device='cpu')
    try:
        env.reset(seed=[1000,1001])
        wp.to_torch(env.s.terminated)[0]=1
        cont=np.zeros((2,c.world.num_uavs,4),np.float32)
        dst=np.zeros((2,c.world.num_uavs),np.int64)
        with pytest.raises(RuntimeError,match='92'):
            env.step({'continuous':cont,'destination':dst})
    finally:env.close()


def test_finished_world_rejects_nonzero_action_in_evaluator_mode():
    c=profile()
    env=WarpVectorEnv(c,device='cpu')
    try:
        env.reset(seed=[1000,1001])
        wp.to_torch(env.s.terminated)[0]=1
        cont=np.zeros((2,c.world.num_uavs,4),np.float32)
        dst=np.zeros((2,c.world.num_uavs),np.int64)
        cont[0,0,0]=.5
        with pytest.raises(RuntimeError,match='92'):
            env.step({'continuous':cont,'destination':dst})
    finally:env.close()


def test_training_progress_scores_true_target_cells_not_internal_report_indices():
    from uav_network.experiments.vector_step_mission_progress import instantaneous_mission_progress
    import warp as wp
    c=profile()
    env=WarpVectorEnv(c,device='cpu')
    try:
        obs,_=env.reset(seed=[44,45])
        targets=wp.to_torch(env.s.targets)
        first=targets[0,0]
        true_cell=int(first[0]/env.p.cell_width)+env.p.grid_n*int(first[1]/env.p.cell_width)
        wrong=(true_cell+1)%(env.p.grid_n*env.p.grid_n)
        wp.to_torch(env.s.report_cell)[0,:2]=torch.tensor([true_cell,wrong])
        wp.to_torch(env.s.known)[0,0,:2]=1
        wp.to_torch(env.s.delivered)[0,:2]=1
        data=instantaneous_mission_progress(obs,wp.to_torch(env.s.visited),
            num_uavs=c.world.num_uavs,num_targets=c.world.num_targets,
            num_obstacles=c.world.num_obstacles,
            world_state=env.s,grid_n=env.p.grid_n,cell_width=env.p.cell_width)
        expected=100./c.world.num_targets
        assert data['discovery_rate_percent'][0].item()==pytest.approx(expected)
        assert data['gcs_delivery_rate_percent'][0].item()==pytest.approx(expected)
        assert data['gcs_delivery_rate_percent'][1].item()==0
        from uav_network.experiments.mission_cause_monitor import MissionCauseMonitor
        m=MissionCauseMonitor(c,'/tmp/matd3_v7_mission_telemetry_test')
        r=m.sample(env,0,phase='test')
        assert r['mission_probe/target_discovery_fraction']==pytest.approx(1./(2*c.world.num_targets))
        assert r['mission_probe/target_gcs_delivery_fraction']==pytest.approx(1./(2*c.world.num_targets))
        assert r['mission_probe/false_confirmed_cells_per_world']==pytest.approx(.5)
        assert r['mission_probe/false_gcs_reports_per_world']==pytest.approx(.5)
    finally:env.close()


def test_eval_works_with_real_immutable_actor_snapshot_and_heldout_seeds():
    from pathlib import Path
    import json
    from uav_network.experiments.deferred_gpu_evaluation import GPUActorOnlyPolicy,gpu_validate_one
    base=Path('/tmp/matd3_v7_train_update_smoke_20261122/deferred-gpu-eval')
    if not (base/'manifest.json').is_file():
        pytest.skip('Independent local model snapshot not available')
    metadata=json.loads((base/'manifest.json').read_text())
    payload=torch.load(base/metadata['snapshots'][-1]['file'],map_location='cpu',weights_only=True)
    from uav_network.configuration import Config,_apply
    c=Config();_apply(c,payload['config']);c.validate()
    policy=GPUActorOnlyPolicy(c,'cpu');policy.load(payload)
    values=gpu_validate_one(policy,payload,gpu='cpu',seeds=[2000000044,2000000045])
    assert values['episodes']==2
    assert len(values['records'])==2
    assert len({x['seed'] for x in values['records']})==2
    assert all(x['targets_delivered']<=x['targets_found']<=35 for x in values['records'])
