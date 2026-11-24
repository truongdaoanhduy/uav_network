"""Restored regression coverage from previously audited Streamfix v2; v3 patch removed these checks."""
import gzip, json, math, pathlib, tempfile
import numpy as np
from uav_network.environment.gpu.slotted_2d_env import Slotted2DUavEnv, generate_scenario_snapshot, DISC_ACTION_DIM, OBS_DIM, STATE_DIM
from uav_network.experiments.uavnetsim_csma_evaluator import evaluate_on_slotted_warp, evaluate_on_reference_csma
from uav_network.experiments.telemetry_wandb_hf import SlottedTelemetryLogger, iqm_stratified_bootstrap_ci
from uav_network.experiments.aggregate_five_seed_rliable import probability_of_improvement, aggregate_runs

def test_scenario_and_obs_state_mask_and_reward_sum_identity():
    snap = generate_scenario_snapshot(seed=44)
    assert snap.obstacles.shape == (22, 4)
    assert snap.target_cells.shape == (25,)
    assert tuple(snap.gcs_pos.tolist()) == (40.0, 40.0, 10.0)

    env = Slotted2DUavEnv(num_envs=2, max_steps=10, base_seed=44, belief_sharing_mode="local_decpomdp")
    obs, state, mask = env.reset()
    assert obs.shape == (2, 6, OBS_DIM) and OBS_DIM == 96
    assert state.shape == (2, STATE_DIM) and STATE_DIM == 256
    assert mask.shape == (2, 6, DISC_ACTION_DIM) and DISC_ACTION_DIM == 7
    assert (mask[:, :, 0] == 1).all(), "Silent action must ALWAYS be valid"
    assert np.isfinite(obs).all() and np.isfinite(state).all()

    rng = np.random.default_rng(7)
    for _ in range(5):
        ac = rng.uniform(-1.0, 1.0, size=(2, 6, 2)).astype(np.float32)
        ad = np.zeros((2, 6), dtype=np.int32)
        obs, state, mask, rewards, dones, infos = env.step(ac, ad)
        assert np.isfinite(rewards).all()
        assert set(infos.keys()) == {"r_ds", "r_gcs_aoi", "r_propulsion", "r_safety", "r_drop_ttl"}
        comp_sum = sum(infos[k] for k in infos)
        assert np.allclose(rewards, comp_sum, atol=1e-5), "Reward component sum mismatch"

def test_local_decpomdp_zero_telepathy_privacy():
    env = Slotted2DUavEnv(num_envs=1, max_steps=10, base_seed=44, belief_sharing_mode="local_decpomdp")
    env.reset(seeds=[44])
    env.uav_pos[0, 0] = [400.0, 400.0, 50.0]
    env.uav_pos[0, 1] = [100.0, 100.0, 50.0]
    for u in range(2, 6):
        env.uav_pos[0, u] = [750.0, 750.0, 50.0]

    env.step(np.zeros((1, 6, 2), dtype=np.float32), np.zeros((1, 6), dtype=np.int32))
    cid = 10 * 20 + 10
    u1_obs_cnt = int(env.local_ds_pos[0, 1, cid] + env.local_ds_neg[0, 1, cid])
    assert u1_obs_cnt == 0, "UAV 1 must NOT telepathically receive UAV 0's local DS evidence"
    assert not np.array_equal(env.local_ds_m_u[0, 0], env.local_ds_m_u[0, 1])

def test_dual_fidelity_resume_continuity_and_five_seed_aggregator():
    def greedy_explore_policy(obs: np.ndarray, mask: np.ndarray):
        e, n, _ = obs.shape
        ac = np.full((e, n, 2), 0.35, dtype=np.float32)
        ad = np.zeros((e, n), dtype=np.int32)
        for ei in range(e):
            for ui in range(n):
                if mask[ei, ui, 6] == 1:
                    ad[ei, ui] = 6
                else:
                    peers = np.where(mask[ei, ui, 1:6] == 1)[0]
                    ad[ei, ui] = int(peers[0] + 1) if len(peers) > 0 else 0
        return ac, ad

    seeds = [101, 102, 103, 104]
    slotted_eps = evaluate_on_slotted_warp(greedy_explore_policy, seeds, max_steps=30, backend="numpy")
    csma_eps = evaluate_on_reference_csma(greedy_explore_policy, seeds, max_steps=30)
    assert len(slotted_eps) == 4 and len(csma_eps) == 4

    iqm, lo, hi = iqm_stratified_bootstrap_ci(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]))
    assert math.isclose(iqm, 45.0, abs_tol=1e-6)
    assert lo <= iqm <= hi

    poi, plo, phi = probability_of_improvement(np.array([10.0, 12.0, 11.0, 13.0, 14.0]), np.array([5.0, 6.0, 7.0, 8.0, 9.0]))
    assert poi == 1.0 and plo <= poi <= phi

    # Verify resume=True preserves train_history.jsonl.gz (append mode) and marl_eval_results.json!
    with tempfile.TemporaryDirectory() as tmp:
        run1 = pathlib.Path(tmp) / "matd3_s44"
        logger1 = SlottedTelemetryLogger(
            output_dir=run1, config={"world": {"map_size_m": 800.0}},
            algorithm="matd3", seed=44, wandb_mode="disabled",
        )
        orig_run_id = logger1.run_id
        orig_wb_id = logger1.wandb_run_id
        logger1.log_train_step(1000, {"train/step_reward_mean": 1.25})
        logger1.log_periodic_eval(1000, slotted_eps)
        logger1.close()

        # Resume same directory: must append to train_history.jsonl.gz & preserve step_1000 in marl_eval
        logger2 = SlottedTelemetryLogger(
            output_dir=run1, config={"world": {"map_size_m": 800.0}},
            algorithm="matd3", seed=44, wandb_mode="disabled", resume=True,
        )
        assert logger2.run_id == orig_run_id and logger2.wandb_run_id == orig_wb_id
        logger2.log_train_step(2000, {"train/step_reward_mean": 2.50})
        report = logger2.log_final_dual_fidelity_test(2000, slotted_eps, csma_eps)
        assert report["gpu_warp_verified"] is False
        receipt = logger2.write_sha256_manifest_and_upload_hf()
        logger2.close()
        assert receipt["file_count"] >= 6

        with gzip.open(run1 / "train_history.jsonl.gz", "rt", encoding="utf-8") as gz:
            hist_rows = [json.loads(line) for line in gz]
        assert len(hist_rows) == 2 and hist_rows[0]["global_step"] == 1000 and hist_rows[1]["global_step"] == 2000

        me = json.loads((run1 / "marl_eval_results.json").read_text(encoding="utf-8"))
        node = me["numpy_slotted_reference"]["6uav_22bldg_25tgt"]["vanilla_matd3"]["seed_44"]
        assert "step_1000" in node and "absolute_metrics" in node

        # Test 5-seed aggregator
        agg = aggregate_runs([run1], pathlib.Path(tmp) / "agg_out.json")
        assert "numpy_slotted_reference" in agg["seed_level_metrics"]
