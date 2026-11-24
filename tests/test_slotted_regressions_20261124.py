"""Regression-first checks for asynchronous episode resets and simulator fidelity claims."""
import numpy as np
import pytest
from uav_network.environment.gpu.slotted_2d_env import Slotted2DUavEnv


def zero_action(n):
    return np.zeros((n, 6, 2), dtype=np.float32), np.zeros((n, 6), dtype=np.int32)


def test_reset_selected_env_does_not_reset_other_live_world():
    env = Slotted2DUavEnv(num_envs=2, max_steps=10)
    env.reset(seeds=[44, 45]); ac, ad = zero_action(2)
    env.step(ac, ad)
    pos = env.uav_pos[1].copy()
    count = int(env.step_count[1])
    original_seed = env.snapshots[1].seed
    env.reset_indices([0], seeds=[500])
    assert env.snapshots[0].seed == 500
    assert env.step_count[0] == 0
    assert env.step_count[1] == count
    assert env.snapshots[1].seed == original_seed
    np.testing.assert_array_equal(env.uav_pos[1], pos)


def test_inactive_eval_worlds_do_not_accumulate_return_or_advance_time():
    env = Slotted2DUavEnv(num_envs=2, max_steps=4)
    env.reset(); ac, ad = zero_action(2)
    env.step(ac, ad, active_mask=np.array([True, True]))
    before_t = env.step_count.copy()
    before_r = env.ep_return.copy()
    env.step(ac, ad, active_mask=np.array([False, True]))
    assert env.step_count[0] == before_t[0]
    assert env.ep_return[0] == before_r[0]
    assert env.step_count[1] == before_t[1] + 1


def test_warp_backend_never_falls_back_to_numpy_silently():
    # Warp has a real CPU backend; it must never silently create a NumPy env.
    pytest.importorskip('warp')
    env = Slotted2DUavEnv(num_envs=1, backend='warp', device='cpu', use_cuda_graph=False)
    assert env._gpu_backend is not None
    assert env._gpu_backend.wp_device == 'cpu'


def test_action_shapes_checked_before_update():
    env=Slotted2DUavEnv(num_envs=2)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros((2, 6, 3), dtype=np.float32), np.zeros((2, 6), dtype=np.int32))


def test_no_target_knowledge_in_actor_observation_when_truth_changes():
    env=Slotted2DUavEnv(num_envs=1)
    obs,_,mask=env.reset(seeds=[44])
    env.snapshots[0].cell_has_target[:] = -1
    env.snapshots[0].target_cells[:] = 0
    obs_after,_,mask_after=env._build_outputs()
    np.testing.assert_array_equal(obs,obs_after)
    np.testing.assert_array_equal(mask,mask_after)


def test_ds_closed_form_stays_normalized_after_many_observations():
    from uav_network.environment.gpu.ds_aoi_belief_fp32 import ds_closed_form
    for p, n in [(1000, 1000), (2000, 2000), (2400, 20), (20, 2400), (0, 0), (3, 5)]:
        masses = ds_closed_form(p, n)
        assert all(np.isfinite(masses))
        assert all(0 <= v <= 1 for v in masses)
        assert abs(sum(masses) - 1) < 1e-5


def test_reference_env_does_not_step_after_550_when_stopped():
    env = Slotted2DUavEnv(num_envs=1, max_steps=1)
    env.reset()
    ac, ad = zero_action(1)
    _, _, _, _, done, _ = env.step(ac, ad)
    assert done[0] == 1.0
    assert env.step_count[0] == 1
    env.step(ac, ad, active_mask=np.array([False]))
    assert env.step_count[0] == 1


def test_motion_safety_separates_uavs_instead_of_penalty_only():
    env = Slotted2DUavEnv(num_envs=1)
    env.reset()
    env.snapshots[0].obstacles[:] = [-9999, -9999, 0, 0]
    env.uav_pos[0, 0, :2] = (50.0, 200.0)
    env.uav_pos[0, 1, :2] = (66.0, 200.0)
    env.uav_vel[0, 0] = (1.0, 0.0)
    env.uav_vel[0, 1] = (-1.0, 0.0)
    ac, ad = zero_action(1)
    env.step(ac, ad)
    dist = np.linalg.norm(env.uav_pos[0, 0, :2] - env.uav_pos[0, 1, :2])
    assert dist >= 15.0
    assert env.ep_safety_viol[0] >= 1


def test_cca_defers_at_last_subslot_without_forced_transmission():
    class RiggedRng:
        def random(self): return 1.0
        def integers(self, low, high=None): return 3

    env = Slotted2DUavEnv(num_envs=1)
    env.reset()
    env.snapshots[0].obstacles[:] = [-9999, -9999, 0, 0]
    env._rngs[0] = RiggedRng()
    env.uav_pos[0, 0, :2] = (90., 90.)
    env.uav_pos[0, 1, :2] = (110., 90.)
    env.q_count[0, :2] = 1
    env.q_cell_id[0, :2, 0] = [10, 20]
    env.q_visited_mask[0, 0, 0] = 1
    env.q_visited_mask[0, 1, 0] = 2
    ac, ad = zero_action(1)
    ad[0, :2] = 6
    env.step(ac, ad)
    assert env.ep_tx_attempts[0] == 1
    assert env.q_count[0, 1] == 1


@pytest.mark.parametrize('algorithm', ['matd3','masac'])
def test_trainer_short_full_flow_with_gradients_and_final_best(tmp_path, algorithm):
    import yaml
    from pathlib import Path
    from uav_network.experiments.train_slotted_vanilla import run_end_to_end
    config = yaml.safe_load((Path(__file__).parents[1]/'configs'/f'vanilla_{algorithm}_shared_2d_ds_aoi.yaml').read_text())
    config[algorithm].update(replay_capacity=32, learning_starts=4, batch_size=8, updates_per_step=1)
    config['runtime'].update(num_envs=2, total_env_steps=8, eval_interval_steps=1000,
                             eval_episodes_slotted=2,eval_episodes_full_uavnetsim=2,
                             learner_device='cpu', replay_device='cpu')
    config['world']['max_steps']=3
    conf = tmp_path / f'{algorithm}.yaml'
    conf.write_text(yaml.safe_dump(config))
    out = tmp_path / algorithm
    result=run_end_to_end(str(conf),str(out),wandb_mode='disabled',reference_smoke=True)
    assert np.isfinite(result['best_eval_return_iqm']) and result['best_eval_return_iqm'] > -1e17
    assert (out/'checkpoints'/'best_actor.pt').exists()
    assert (out/'checkpoints'/'final_actor.pt').exists()
    assert (out/'summary_metrics.json').exists()


def test_production_runner_refuses_silent_numpy_fallback(tmp_path):
    import yaml
    from pathlib import Path
    from uav_network.experiments.train_slotted_vanilla import run_end_to_end
    p = Path(__file__).parents[1]/'configs/vanilla_matd3_shared_2d_ds_aoi.yaml'
    import torch
    if torch.cuda.is_available():
        pytest.skip('Production is valid on CUDA and is covered by Kaggle acceptance tests')
    with pytest.raises(RuntimeError, match='CUDA|GPU'):
        run_end_to_end(str(p), str(tmp_path/'production'),wandb_mode='disabled',
                       override_total_steps=2,override_num_envs=1,override_eval_episodes=1,override_max_steps=1)


def test_radio_transmission_draws_uav_battery_besides_propulsion():
    from uav_network.environment.gpu.slotted_2d_env import BATTERY_CAPACITY_J
    env=Slotted2DUavEnv(num_envs=2)
    env.reset(seeds=[44,44])
    env.snapshots[0].obstacles[:] = [-9999,-9999,0,0]
    env.snapshots[1].obstacles[:] = [-9999,-9999,0,0]
    env.q_count[0,0] = 1
    env.q_cell_id[0,0,0] = int(env.snapshots[0].target_cells[0])
    env.q_visited_mask[0,0,0] = 1
    env.uav_pos[0,0,:2] = env.uav_pos[1,0,:2] = (50,50)
    ac,ad=zero_action(2)
    ad[0,0] = 6
    env.step(ac,ad)
    assert env.ep_tx_attempts[0] > 0
    assert env.ep_network_j[0] > 0
    assert env.uav_battery_j[0,0] < env.uav_battery_j[1,0]


def test_telemetry_streams_training_history_and_unique_run_ids(tmp_path):
    import gzip, json
    from uav_network.experiments.telemetry_wandb_hf import SlottedTelemetryLogger
    l1=SlottedTelemetryLogger(tmp_path/'run1', {}, 'matd3', 44,wandb_mode='disabled')
    l2=SlottedTelemetryLogger(tmp_path/'run2', {}, 'matd3', 44,wandb_mode='disabled')
    assert l1.run_id != l2.run_id
    l1.log_train_step(10, {'train/step_reward_mean':1.5})
    l1.close();l2.close()
    with gzip.open(tmp_path/'run1'/'train_history.jsonl.gz','rt') as f:
        data=[json.loads(line) for line in f]
    assert data==[{'global_step':10, 'train/step_reward_mean':1.5}]
    manifest=l1.write_sha256_manifest_and_upload_hf()
    assert manifest['file_count']>=3


def test_existing_run_files_are_not_overwritten(tmp_path):
    from uav_network.experiments.train_slotted_vanilla import run_end_to_end
    from pathlib import Path
    out=tmp_path/'already_used';out.mkdir()
    (out/'important_checkpoint.pt').write_bytes(b'KEEP')
    cfg=str(Path(__file__).parents[1]/'configs/vanilla_matd3_shared_2d_ds_aoi.yaml')
    with pytest.raises(FileExistsError):
        run_end_to_end(cfg,str(out),wandb_mode='disabled',reference_smoke=True,
                       override_num_envs=1,override_total_steps=1,override_eval_episodes=1,override_max_steps=1)
    assert (out/'important_checkpoint.pt').read_bytes()==b'KEEP'


def test_reference_csma_does_not_early_ack_before_overlapping_hidden_tx():
    from uav_network.experiments.uavnetsim_csma_evaluator import FullCSMA2DUavEnv
    class IdenticalBackoff:
        def integers(self, low, high=None): return 0
    env=FullCSMA2DUavEnv(num_envs=1)
    env.reset()
    for u in (0,1):
        env.q_count[0,u]=1
        env.q_cell_id[0,u,0]=int(env.snapshots[0].target_cells[u])
        env.q_visited_mask[0,u,0] = (1 << u)
    env.cca_hear_mat[0] = 0  # hidden to one another
    env.rx_pwr_mat[0,0,6] = env.rx_pwr_mat[0,1,6] = 1.0e-8
    dest=np.array([6,6,0,0,0,0],dtype=np.int32)
    env._mac_slotted_step_env(0,dest,1.0,IdenticalBackoff())
    assert env.ep_true_deliv[0] == 0
    assert env.ep_collisions[0] >= 2


def test_real_uavnetsim_entrypoint_calls_vendored_simpy():
    pytest.importorskip('simpy')
    from uav_network.experiments.uavnetsim_csma_evaluator import evaluate_on_full_csma_uavnetsim
    rows = evaluate_on_full_csma_uavnetsim(lambda obs, mask: zero_action(len(obs)), [101], max_steps=2)
    assert len(rows) == 1 and np.isfinite(rows[0].episode_return)
