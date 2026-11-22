"""The original 18d W&B chart keys must contain actually observed data."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch

import pytest
from uav_network.configuration import Config
from uav_network.experiments.tracking import OnlineTracker


def _record(episode):
    # Complete but minimal truth-scored *terminal* report (never policy input).
    return {
        'episode':episode,'global_step':episode*4096,
        'episode_length':80,'episode_horizon_s':100.,'episode_dt_s':1.,
        'return':11.5,'search_rate_percent':50.,'delivery_rate_percent':50.,
        'targets_found':1,'targets_delivered':1,'success':False,
        'coverage_percent':50.,'coverage_auc':.3,
        'report_diagnostics':{
            'first_discovery_s':[20.,None],
            'first_delivery_s':[70.,None],
            'first_discovery_to_delivery_mean_s':50.,
            'delivery_count':1},
        'energy':{'propulsion_energy_j':100.,'battery_fraction':[.6,.6],
                  'active_uavs':2},
        'network':{'enabled':True,'generated_packets':10,'received_packets':8,
                   'uav_communication_energy_j':20.,'selection_counts':{}},
        'blocked_moves':0,'boundary_clips':0,'expired_reports':0,'dropped_reports':0,
        'reward_components':{'discovery':20},'motion_counts':{},'action_counts':{},
    }


def test_episode_metrics_provide_original_paper_names_with_real_aucs():
    from uav_network.experiments.training_mission_metrics import terminal_paper_metrics
    metrics=terminal_paper_metrics(_record(1))
    assert metrics['training/discovery_rate_percent']==50.
    assert metrics['training/gcs_delivery_rate_percent']==50.
    assert metrics['training/targets_found']==1
    assert metrics['training/targets_delivered']==1
    assert metrics['training/discovery_auc']==pytest.approx(.4)
    assert metrics['training/delivery_auc']==pytest.approx(.15)
    assert metrics['training/coverage_auc']==pytest.approx(.3)
    assert metrics['training/discovery_time_capped_mean_s']==pytest.approx(60.)
    assert metrics['training/gcs_delivery_time_capped_mean_s']==pytest.approx(85.)
    assert metrics['training/energy_per_delivered_target_j']==pytest.approx(120.)
    assert metrics['training/packet_pdr_percent']==pytest.approx(80.)
    assert metrics['training/mission_success_rate_percent']==0.


def test_missing_timestamps_are_never_fabricated_as_auc():
    from uav_network.experiments.training_mission_metrics import terminal_paper_metrics
    r=_record(1)
    r.pop('report_diagnostics')
    r['coverage_auc']=None
    results=terminal_paper_metrics(r)
    assert 'training/discovery_auc' not in results
    assert 'training/delivery_auc' not in results
    assert 'training/coverage_auc' not in results


def test_dense_step_reward_and_optimizer_have_true_global_step_not_fabricated_episode_axis():
    t=OnlineTracker.__new__(OnlineTracker)
    t.run=SimpleNamespace(log=Mock())
    t.step_reward({'global_step':4096,'train/step_reward_mean':2.5,
                   'train_progress/discovery_rate_percent':13.,
                   'train_loss/actor_loss':.18})
    history=t.run.log.call_args.args[0]
    assert history['global_step']==4096
    assert history['train/step_reward_mean']==2.5
    assert history['train_progress/discovery_rate_percent']==13.
    assert history['train_loss/actor_loss']==.18
    assert 'episode/index' not in history


def test_64_completed_episodes_produce_nonempty_original_wandb_mission_cohort():
    from uav_network.experiments.convergence_logging import EpisodeReturnWindow,CompletedMissionWindow
    t=OnlineTracker.__new__(OnlineTracker)
    t.run=SimpleNamespace(log=Mock())
    t.records=[]
    t.latest_evidence={}
    t.live_runtime=None
    t.config=Config()
    t.config.world.num_uavs=2
    t.config.world.num_targets=2
    t.config.world.max_steps=100
    t.config.world.dt_s=1
    t._episode_return_window=EpisodeReturnWindow(every_episodes=64)
    t._mission_outcome_window=CompletedMissionWindow(every_episodes=64)
    with patch.dict('os.environ',{'UAV_WANDB_COMPACT_LOGS':'true','UAV_WANDB_EPISODE_LOG_EVERY':'32'}):
        for i in range(1,65):t.episode(_record(i))
    logs=[call.args[0] for call in t.run.log.call_args_list]
    joined=[l for l in logs if 'train_cohort/discovery_auc_mean' in l]
    assert len(joined)==1
    cohort=joined[0]
    assert cohort['episode/index']==64
    assert cohort['train_cohort/discovery_auc_mean']==pytest.approx(.4)
    assert cohort['train_cohort/delivery_auc_mean']==pytest.approx(.15)
    assert cohort['train_cohort/map_coverage_percent_mean']==50.
    assert cohort['train_cohort/episode_count']==64
    assert len([x for x in logs if 'episode/return' in x])==3  # first, 32, 64
