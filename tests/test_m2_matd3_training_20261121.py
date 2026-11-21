"""Scientific guardrails for Native packet-accurate M2 and CUDA MATD3 pilot."""
from pathlib import Path
import csv
import numpy as np
import pytest
import torch

from uav_network.environment.m2_mission import CooperativeM2Env
from scripts.research.train_m2_matd3_native_gpu_20261121 import (
    PilotSettings, make_config, train, evaluate_policy, require_training_device,
    load_m2_policy_checkpoint,
)


def test_exact_1000m_m2_scenario_and_learner_action_contract():
    cfg=make_config(seed=44,episode_horizon=1500)
    assert cfg.world.map_size_m==1000.
    assert cfg.sensor.grid_cell_m==25.
    assert cfg.world.num_uavs==6
    assert cfg.world.num_targets==35
    assert cfg.world.num_obstacles==25
    assert cfg.world.motion_dimensions==3
    assert cfg.sensor.model=='bayesian'
    assert cfg.world.motion_safety=='cbf'
    assert cfg.matd3.hidden_sizes==[256,256]
    assert cfg.matd3.share_actor_parameters
    assert cfg.matd3.share_critic_parameters
    assert cfg.matd3.actor_layer_norm
    assert cfg.runtime.m2_enabled
    assert cfg.continuous_dim==5
    assert cfg.runtime.environment_backend=='native'


def test_kaggle_training_refuses_falsely_claiming_cuda_on_a_cpu_device():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='CUDA'):
            require_training_device('cuda:0')
    assert str(require_training_device('cpu'))=='cpu'


def test_train_two_real_m2_episodes_updates_matd3_and_logs_true_gcs_metrics(tmp_path):
    cfg=make_config(seed=44,episode_horizon=12)
    pilot=PilotSettings(episodes=2,scan_cells=4,batch_size=4,
                        replay_capacity=64,warmup_steps=4,update_every=2,
                        eval_every=1,eval_episodes=1,max_train_seconds=0.,
                        seed=44,device='cpu',output_dir=tmp_path)
    summary=train(cfg,pilot)
    assert summary['algorithm']=='MATD3'
    assert summary['environment_backend']=='native_uavnetsim'
    assert summary['warp_full_m2_environment'] is False
    assert summary['world_size_m']==1000
    assert summary['episode_horizon']==12
    assert summary['episodes_completed']==2
    assert summary['environment_transitions']==24
    assert summary['learner_critic_updates']>0
    assert summary['learner_actor_updates']>0
    assert summary['evaluation_count']==3 # initial + per episode
    assert (tmp_path/'train_episodes.csv').exists()
    assert (tmp_path/'evaluation.csv').exists()
    assert (tmp_path/'final_policy.pt').exists()
    # The legacy MATD3.load checks a different observation ABI; M2 needs an
    # explicitly versioned standalone loader for its own 255D observation.
    restored=load_m2_policy_checkpoint(tmp_path/'final_policy.pt',device='cpu')
    assert restored.observation_dim==255
    assert restored.state_dim==255*6
    assert restored.continuous_dim==5
    payload=torch.load(tmp_path/'final_policy.pt',map_location='cpu',weights_only=True)
    assert payload['format']=='m2-target-belief-2d-matd3-v2'
    for name,value in payload['model_state_dict'].items():
        assert torch.equal(value,restored.state_dict()[name])
    train_rows=list(csv.DictReader((tmp_path/'train_episodes.csv').open()))
    eval_rows=list(csv.DictReader((tmp_path/'evaluation.csv').open()))
    assert len(train_rows)==2 and len(eval_rows)==3
    assert all(0<=float(r['correct_gcs_delivery_recall'])<=1 for r in train_rows)
    assert all(0<=float(r['mean_uncertainty'])<=1 for r in train_rows)
    assert all(r['backend']=='native_uavnetsim' for r in train_rows)
    assert all('motion_saturation_rate' in r and 'action_z_mean' in r for r in train_rows)
    assert all('silent_action_fraction' in r and 'gcs_destination_fraction' in r for r in train_rows)
    assert all('radio_selected' in r and 'radio_accepted' in r for r in train_rows)
    assert all('radio_no_buffer' in r and 'radio_receiver_rejected' in r for r in train_rows)
    assert all(int(r['num_targets'])==35 for r in train_rows)
    assert all(np.isfinite(float(r['mean_return'])) for r in eval_rows)
    assert all('mean_false_reports' in r and 'mean_delivery_auc' in r for r in eval_rows)
    assert all('mean_target_brier' in r and 'mean_target_auprc' in r for r in eval_rows)
    assert not any('oracle' in key or 'truth' in key for key in train_rows[0])


def test_evaluation_runs_real_radio_and_does_not_mutate_source_config():
    cfg=make_config(seed=45,episode_horizon=10)
    env=CooperativeM2Env(cfg,max_scan_cells=4)
    try:
        obs,_=env.reset(seed=45)
        from uav_network.algorithms.matd3 import MATD3
        agent=MATD3(env.config,obs.shape[-1],env.state().size,device='cpu')
    finally:
        env.close()
    results=evaluate_policy(cfg,agent,seeds=[2000000044],scan_cells=4)
    assert len(results)==1
    row=results[0]
    assert row['steps']==10
    assert row['num_targets']==35
    assert row['backend']=='native_uavnetsim'
    assert np.isfinite(row['episode_return'])
    assert cfg.continuous_dim==5 # explicitly configured M2 caller remains M2


def test_evidence_packetizer_handles_exact_1024_boundary_with_complete_envelope():
    from uav_network.environment.correlated_mapping import Event
    from uav_network.communication.m2_protocol import CooperativeWorldMap,_wire_bytes
    from uav_network.communication.m2_network import M2PacketStore,_envelope_wire_size
    # A true boundary case: event-only size reservation is EXACTLY 1024,
    # but the complete JSON envelope includes one more byte than reserved.
    events=[Event(f't:44:0:420:{i}',0,420,i,'target',True,.9,.1,.55) for i in range(6)]
    extra=1024-_wire_bytes(events)
    assert extra>0
    events[0]=Event(events[0].event_id+'X'*extra,0,420,0,'target',True,.9,.1,.55)
    assert _wire_bytes(events)==1024
    nodes=[CooperativeWorldMap(12) for _ in range(7)]
    store=M2PacketStore(nodes,buffer_bytes=65536)
    # A busy 1500-step mission will already have 4-digit packet serial IDs.
    import itertools
    store._serial=itertools.count(5000)
    messages=store.publish(0,events,created_s=420.,payload_bytes=1024)
    if not isinstance(messages,list):messages=[messages]
    assert sum(len(msg.events) for msg in messages)==6
    assert all(msg.size_bytes<=1024 for msg in messages)
    assert all(msg.size_bytes>=_envelope_wire_size(msg) for msg in messages)
    assert nodes[0].events_seen==6


def test_diagnostic_native_pilot_never_invents_session_time_limit_by_default():
    # Existing pilot remains diagnostic only; zero means no code-level timeout.
    assert PilotSettings().max_train_seconds == 0.0
