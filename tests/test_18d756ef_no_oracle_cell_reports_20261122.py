"""Regression tests for a NEW cell-based Bayesian report mission, not the historical 18d756ef checkpoint."""
import numpy as np
import pytest
import torch
import warp as wp

from uav_network.configuration import Config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.communication.gpu.metrics import world_info


def config(*, targets=1, agents=1, obstacles=0):
    c=Config()
    c.world.motion_dimensions=3
    c.world.motion_safety='reject'
    c.world.num_uavs=agents
    c.world.num_targets=targets
    c.world.num_obstacles=obstacles
    c.sensor.model='bayesian'
    c.sensor.report_cell_hypotheses=True
    c.runtime.num_envs=1
    c.runtime.device='cpu'
    c.runtime.environment_backend='warp_packet'
    c.network.enabled=True
    return c


def motion(agents=1, destination=0):
    return {'continuous':np.zeros((1,agents,4),np.float32),
            'destination':np.full((1,agents),destination,np.int64)}


def armed_world(*, target_xy, seed=44, candidate_xy=(512.5,512.5)):
    from uav_network.environment.bayesian import sensor_uniform
    cell=20*40+20
    seed=next(s for s in range(1,20000) if sensor_uniform(s,1,0,cell)<.045)
    v=WarpVectorEnv(config(),device='cpu')
    v.reset(seed=seed)
    wp.to_torch(v.s.positions)[0]=torch.tensor([list(candidate_xy)])
    wp.to_torch(v.s.altitudes)[0]=25
    wp.to_torch(v.s.targets)[0]=torch.tensor([list(target_xy)])
    wp.to_torch(v.s.belief).fill_(.5)
    wp.to_torch(v.s.belief)[0,0,cell]=.99999
    return v


def test_empty_cell_can_generate_false_report_without_truth_reset():
    v=armed_world(target_xy=(950.,950.))
    try:
        v.step(motion())
        info=world_info(v.s,v.p,0)
        assert int(wp.to_torch(v.s.report_cell)[0,0])==820
        assert int(wp.to_torch(v.s.known)[0,0,0])==1
        assert int(wp.to_torch(v.s.report_present)[0,0,0])==1
        assert float(wp.to_torch(v.s.belief)[0,0,820])>.99
        assert info['targets_found']==0
        assert info['false_confirmed_cells']>=1
        assert info['sensing']['false_confirmation']>=1
    finally:v.close()


def test_true_and_false_cells_both_create_reports_from_same_sensor_threshold():
    observed=[]
    for target in ((512.5,512.5),(950.,950.)):
        v=armed_world(target_xy=target)
        try:
            v.step(motion())
            observed.append((int(wp.to_torch(v.s.report_cell)[0,0]),
                             int(wp.to_torch(v.s.known)[0,0,0]),
                             int(wp.to_torch(v.s.report_present)[0,0,0])))
        finally:v.close()
    assert observed==[(820,1,1),(820,1,1)]


def test_critic_has_no_hidden_target_xy_in_cell_report_mode():
    v=WarpVectorEnv(config(),device='cpu')
    try:
        v.reset(seed=44)
        state0=v._publish(v._buffers[v._buffer])['state'].clone()
        wp.to_torch(v.s.targets)[0]=torch.tensor([[900.,900.]])
        state1=v._publish(v._buffers[v._buffer])['state'].clone()
        torch.testing.assert_close(state0,state1)
    finally:v.close()


def test_generated_target_cells_are_unique():
    v=WarpVectorEnv(config(targets=35,agents=2,obstacles=25),device='cpu')
    try:
        for seed in (44,45,46):
            v.reset(seed=seed)
            targets=wp.to_torch(v.s.targets)[0].numpy()
            ids={(int(x//25),int(y//25)) for x,y in targets}
            assert len(ids)==35
    finally:v.close()


def test_remote_receiver_private_buffer_does_not_change_actor_mask():
    v=WarpVectorEnv(config(agents=2),device='cpu')
    try:
        v.reset(seed=44)
        wp.to_torch(v.s.report_cell)[0,0]=820
        wp.to_torch(v.s.known)[0,0,0]=1
        wp.to_torch(v.s.report_present)[0,0,0]=1
        wp.to_torch(v.s.positions)[0]=torch.tensor([[500.,200.],[500.,210.]])
        a=v._publish(v._buffers[v._buffer])['observation'].clone()
        wp.to_torch(v.s.known)[0,1,0]=1
        b=v._publish(v._buffers[v._buffer])['observation'].clone()
        # Exact remote knowledge is unavailable without a received message.
        torch.testing.assert_close(a[0,0],b[0,0])
    finally:v.close()


def test_report_capacity_and_cell_namespace_are_separate_from_true_target_count():
    v=WarpVectorEnv(config(targets=2),device='cpu')
    try:
        v.reset(seed=44)
        assert v.p.report_slots>v.p.targets_count
        assert wp.to_torch(v.s.report_cell).shape==(1,v.p.report_slots)
        assert wp.to_torch(v.s.targets).shape==(1,2,2)
    finally:v.close()

@pytest.mark.parametrize('actual_target_here', [True,False])
def test_real_packet_to_gcs_retains_false_report_and_scores_truth_only(actual_target_here):
    target=(512.5,512.5) if actual_target_here else (950.,950.)
    v=armed_world(target_xy=target)
    try:
        v.step(motion())
        slot=int(np.flatnonzero(wp.to_torch(v.s.report_cell)[0].numpy()==820)[0])
        assert int(wp.to_torch(v.s.report_present)[0,0,slot])==1
        # Keep the verified evidence; bring the UAV into communication range.
        wp.to_torch(v.s.positions)[0]=torch.tensor([[500.,150.]])
        before=v.info() if hasattr(v,'info') else world_info(v.s,v.p,0)
        _,_,done,_,_=v.step(motion(destination=1))
        assert bool(done[0])==actual_target_here
        after=world_info(v.s,v.p,0)
        assert int(wp.to_torch(v.s.delivered)[0,slot])==1,after['network']
        assert after['network']['raw_reports_delivered']==1
        # Over-the-air packet MUST carry the actual 2D cell ID, not an internal
        # slot number known only to the omniscient simulator.
        assert 820 in wp.to_torch(v.s.tx_cell)[0].numpy()
        assert after['targets_found']==int(actual_target_here)
        assert after['targets_delivered']==int(actual_target_here)
        assert after['false_reports_delivered']==int(not actual_target_here)
        assert after['success']==actual_target_here
        # False packets must not accidentally earn +50 GCS delivery credit.
        if actual_target_here:
            assert after['reward_components']['delivery']>=50
        else:
            assert after['reward_components']['delivery']<0
    finally:v.close()


def test_new_report_cell_mode_has_no_oracle_reset_with_false_positive():
    v=armed_world(target_xy=(950.,950.))
    try:
        v.step(motion())
        assert wp.to_torch(v.s.belief)[0,0,820].item()>.99
        assert world_info(v.s,v.p,0)['false_confirmed_cells']>=1
    finally:v.close()


def test_masked_world_reset_never_processes_stale_evidence_in_other_world():
    c=config()
    c.runtime.num_envs=2
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=44)
        cell=820
        wp.to_torch(v.s.positive)[1,0,cell]=1
        wp.to_torch(v.s.belief)[1,0,cell]=.99999
        wp.to_torch(v.s.altitudes)[1,0]=25
        before=wp.to_torch(v.s.report_cell)[1].clone()
        v.reset(seed=[99,None],options={'reset_mask':[True,False]})
        torch.testing.assert_close(before,wp.to_torch(v.s.report_cell)[1])
    finally:v.close()


def test_peer_private_position_velocity_battery_not_visible_without_radio_beacon():
    c=config(agents=2)
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=44)
        wp.to_torch(v.s.positions)[0]=torch.tensor([[500.,150.],[500.,160.]])
        wp.to_torch(v.s.report_cell)[0,0]=820
        wp.to_torch(v.s.known)[0,0,0]=1
        wp.to_torch(v.s.report_present)[0,0,0]=1
        before=v._publish(v._buffers[v._buffer])['observation'].clone()
        wp.to_torch(v.s.positions)[0,1]=torch.tensor([950.,950.])
        wp.to_torch(v.s.velocities)[0,1]=torch.tensor([17.,-8.])
        wp.to_torch(v.s.battery)[0,1]=0.0
        after=v._publish(v._buffers[v._buffer])['observation'].clone()
        torch.testing.assert_close(before[0,0],after[0,0])
    finally:v.close()


def test_global_hypothesis_slot_assignment_does_not_leak_into_local_actor():
    v=WarpVectorEnv(config(),device='cpu')
    try:
        v.reset(seed=44)
        for slot in (0,44):
            wp.to_torch(v.s.known).zero_()
            wp.to_torch(v.s.report_present).zero_()
            wp.to_torch(v.s.report_cell).fill_(-1)
            wp.to_torch(v.s.known)[0,0,slot]=1
            wp.to_torch(v.s.report_present)[0,0,slot]=1
            wp.to_torch(v.s.report_cell)[0,slot]=820
            current=v._publish(v._buffers[v._buffer])['observation'].clone()
            if slot==0:reference=current
            else:torch.testing.assert_close(reference,current)
    finally:v.close()


def test_cell_report_checkpoint_snapshot_has_new_version():
    v=WarpVectorEnv(config(),device='cpu')
    try:
        v.reset(seed=44)
        snap=v.snapshot_state()
        assert snap['version']==4
        bad=dict(snap,version=2)
        with pytest.raises(ValueError, match='Incompatible'):
            v.restore_state(bad)
    finally:v.close()


def test_training_progress_does_not_mislabel_hypotheses_as_truth():
    from uav_network.experiments.tracking import progress_metrics
    c=config()
    v=WarpVectorEnv(c,device='cpu')
    try:
        obs,_=v.reset(seed=44)
        data=progress_metrics(obs,c)
        assert 'progress/search_rate_percent' not in data
        assert 'progress/delivery_rate_percent' not in data
        assert 'progress/local_confirmed_hypotheses_per_target' in data
    finally:v.close()


def test_true_delivery_delay_excludes_delivered_false_alerts():
    v=WarpVectorEnv(config(),device='cpu')
    try:
        v.reset(seed=44)
        target=wp.to_torch(v.s.targets)[0,0].numpy()
        true_cell=int(target[0]//25)+40*int(target[1]//25)
        false_cell=next(x for x in range(1600) if x!=true_cell)
        wp.to_torch(v.s.report_cell)[0,:2]=torch.tensor([true_cell,false_cell])
        wp.to_torch(v.s.delivered)[0,:2]=1
        wp.to_torch(v.s.first_discovery)[0,0]=1
        wp.to_torch(v.s.first_delivery)[0,0]=4
        wp.to_torch(v.s.report_delay_sum)[0]=13 # True 5s + false 8s
        wp.to_torch(v.s.report_hop_sum)[0]=3
        wp.to_torch(v.s.report_delivery_count)[0]=2
        wp.to_torch(v.s.true_report_delay_sum)[0]=5
        wp.to_torch(v.s.true_report_hop_sum)[0]=1
        info=world_info(v.s,v.p,0)
        assert info['targets_delivered']==1
        assert info['false_reports_delivered']==1
        assert info['report_diagnostics']['delivered_report_age_mean_s']==pytest.approx(5.)
    finally:v.close()


def test_new_observation_semantics_have_distinct_version():
    from uav_network.environment.observations import observation_version
    from uav_network.configuration import load_config
    historical=load_config('configs/reference-18d756ef-wandb-resolved-20261121.yaml')
    current=config()
    current.sensor.belief_coarse_cells=5
    assert observation_version(historical)==5
    assert observation_version(current)==8


def test_false_alert_metrics_remain_visible_in_training_tracker():
    from uav_network.experiments.tracking import episode_metrics
    v=WarpVectorEnv(config(),device='cpu')
    try:
        v.reset(seed=44)
        info=world_info(v.s,v.p,0)
        record={**info,'episode':1,'global_step':1,'episode_length':1,
                'return':0.,'success':False}
        metrics=episode_metrics(record,1,motion_dim=3,motion_safety='reject')
        assert metrics['outcome/false_reports_delivered']==0
        assert metrics['outcome/false_confirmed_cells']==0
    finally:v.close()


def test_belief_candidate_ranking_prefers_persistent_high_confidence():
    from uav_network.environment.observations import observation_dim
    c=config()
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=44)
        wp.to_torch(v.s.positions)[0,0]=torch.tensor([500.,500.])
        b=wp.to_torch(v.s.belief)
        p=wp.to_torch(v.s.positive)
        seen=wp.to_torch(v.s.visited)
        b.fill_(.5)
        p.zero_()
        seen.zero_()
        b[0,0,820]=.81
        b[0,0,3]=.96
        seen[0,0,820]=seen[0,0,3]=1
        p[0,0,820]=p[0,0,3]=1
        obs=v._publish(v._buffers[v._buffer])['observation'][0,0].numpy()
        # Candidate slots start after the target-ID summary, basic own/GCS info.
        offset=10+2*c.world.num_targets
        assert obs[offset+2]==pytest.approx(.96)
    finally:v.close()
