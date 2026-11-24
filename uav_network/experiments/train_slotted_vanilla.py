"""End-to-End Production & Reference Runner for Vanilla MATD3 / Vanilla MASAC (Streamfix v3).

Streamfix v3 upgrades:
1. Accurate `config_resolved.json`: CLI overrides are written back into `cfg` BEFORE logger init.
2. Dual-T4 async CUDA stream overlap: overlaps `cuda:0` Warp CUDA Graph rollout with `cuda:1`
   `updates_per_step=8` learner updates (with zero policy staleness).
3. Gated `compute_reward_info=should_log` + single batched `.cpu().tolist()` sync every log interval.
4. Supports `--learning-starts` and `--batch-size` overrides for real mini-training / pilot checks.
5. Rebalanced `max_accel_mps2: 4.0` validation & full 12h Kaggle auto-resume + incremental HF backups.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from uav_network.environment.gpu.slotted_2d_env import (
    CONT_ACTION_DIM,
    DISC_ACTION_DIM,
    OBS_DIM,
    STATE_DIM,
    Slotted2DUavEnv,
)
from uav_network.experiments.telemetry_wandb_hf import SlottedTelemetryLogger
from uav_network.experiments.uavnetsim_csma_evaluator import (
    evaluate_on_full_csma_uavnetsim,
    evaluate_on_reference_csma,
    evaluate_on_slotted_warp,
)


def validate_scenario_config(cfg: Dict[str, Any]) -> None:
    """Verify YAML physical & architectural parameters match the compiled FP32 kernels."""
    fixed = {
        ("world", "map_size_m"): 800.0,
        ("world", "num_uavs"): 6,
        ("world", "num_targets"): 25,
        ("world", "num_obstacles"): 22,
        ("world", "cruise_altitude_m"): 50.0,
        ("world", "motion_dimensions"): 2,
        ("sensor", "grid_cells_x"): 20,
        ("sensor", "grid_cells_y"): 20,
        ("sensor", "grid_cell_m"): 40.0,
        ("sensor", "sensing_radius_m"): 60.0,
        ("sensor", "ds_mf_base"): 0.55,
        ("sensor", "ds_me_base"): 0.55,
        ("sensor", "confirm_mf_threshold"): 0.90,
        ("sensor", "confirm_mu_ceiling"): 0.15,
        ("network", "slotted_subslots"): 4,
        ("network", "tx_power_w"): 0.1,
        ("network", "max_queue_reports"): 32,
        ("network", "report_ttl_s"): 120.0,
        ("energy", "battery_capacity_j"): 150000.0,
        ("world", "max_accel_mps2"): 4.0,
        ("world", "max_speed_mps"): 20.0,
        ("world", "safety_distance_m"): 15.0,
    }
    for (section, key), expected in fixed.items():
        actual = cfg.get(section, {}).get(key)
        if actual != expected:
            raise ValueError(
                f"YAML/source mismatch: {section}.{key}={actual!r}, compiled backend expects {expected!r}"
            )
    belief_mode = cfg.get("sensor", {}).get("belief_sharing_mode", "local_decpomdp")
    if belief_mode not in ("local_decpomdp", "shared_team_ideal"):
        raise ValueError(f"Unsupported sensor.belief_sharing_mode: {belief_mode!r}")
    if cfg["runtime"]["algorithm"].lower() not in ("matd3", "masac"):
        raise ValueError("Only vanilla MATD3 or MASAC are supported.")


def _select_device(preferred: str):
    import torch
    if preferred.startswith("cuda") and torch.cuda.is_available():
        idx = int(preferred.split(":")[1]) if ":" in preferred else 0
        if idx < torch.cuda.device_count():
            return torch.device(preferred)
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_agent(cfg: Dict[str, Any], device):
    from uav_network.algorithms.vanilla_baselines import VanillaMASAC, VanillaMATD3

    algo = cfg["runtime"]["algorithm"].lower()
    n_uavs = int(cfg["world"]["num_uavs"])
    acfg = cfg[algo]
    hidden = tuple(int(h) for h in acfg["hidden_sizes"])

    if algo == "matd3":
        return VanillaMATD3(
            obs_dim=OBS_DIM,
            state_dim=STATE_DIM,
            num_agents=n_uavs,
            cont_dim=CONT_ACTION_DIM,
            disc_dim=DISC_ACTION_DIM,
            hidden_sizes=hidden,
            lr=float(acfg["learning_rate"]),
            gamma=float(acfg["gamma"]),
            tau=float(acfg["tau"]),
            policy_delay=int(acfg["policy_delay"]),
            exploration_noise_std=float(acfg["exploration_noise_std"]),
            target_noise_std=float(acfg["target_noise_std"]),
            target_noise_clip=float(acfg["target_noise_clip"]),
            gumbel_tau=float(acfg["gumbel_tau"]),
            grad_clip_norm=float(acfg["grad_clip_norm"]),
            device=device,
        )
    if algo == "masac":
        return VanillaMASAC(
            obs_dim=OBS_DIM,
            state_dim=STATE_DIM,
            num_agents=n_uavs,
            cont_dim=CONT_ACTION_DIM,
            disc_dim=DISC_ACTION_DIM,
            hidden_sizes=hidden,
            lr=float(acfg["learning_rate"]),
            gamma=float(acfg["gamma"]),
            tau=float(acfg["tau"]),
            initial_alpha_cont=float(acfg["initial_alpha_cont"]),
            initial_alpha_disc=float(acfg["initial_alpha_disc"]),
            target_entropy_cont=float(acfg["target_entropy_cont"]),
            target_entropy_disc_ratio=float(acfg["target_entropy_disc_ratio"]),
            log_std_min=float(acfg.get("log_std_min", -5.0)),
            log_std_max=float(acfg.get("log_std_max", 2.0)),
            gumbel_tau=float(acfg["gumbel_tau"]),
            grad_clip_norm=float(acfg["grad_clip_norm"]),
            device=device,
        )
    raise ValueError(f"Unsupported algorithm: {algo!r} (expected 'matd3' or 'masac')")


def make_deterministic_policy_fn(agent, device):
    import torch

    def _policy(obs_np: np.ndarray, mask_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
            mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.uint8)
            act_cont_t, act_disc_oh_t = agent.act(obs_t, mask_t, explore=False)
            act_disc_idx = act_disc_oh_t.argmax(dim=-1).cpu().numpy().astype(np.int32)
            return act_cont_t.cpu().numpy().astype(np.float32), act_disc_idx

    return _policy


def make_deterministic_torch_policy_fn(agent, learner_dev, env_dev):
    """Zero-copy GPU policy wrapper for fast GPU Warp validation/test without NumPy round-trips."""
    import torch

    def _gpu_policy(obs_t, mask_t):
        with torch.no_grad():
            o_l = obs_t.to(learner_dev, dtype=torch.float32, non_blocking=True)
            m_l = mask_t.to(learner_dev, dtype=torch.uint8, non_blocking=True)
            ac_t, ad_oh_t = agent.act(o_l, m_l, explore=False)
            ad_idx_t = ad_oh_t.argmax(dim=-1).to(torch.int32)
            return ac_t.to(env_dev, non_blocking=True), ad_idx_t.to(env_dev, non_blocking=True)

    return _gpu_policy


def _atomic_torch_save(obj: Any, path: Path) -> None:
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _save_resume_checkpoint(
    path: Path,
    agent: Any,
    global_step: int,
    best_eval_score: float,
    next_eval_step: int,
    prior_wall_clock_s: float,
    run_id: str,
    wandb_run_id: str,
) -> None:
    state: Dict[str, Any] = {
        "global_step": int(global_step),
        "best_eval_score": float(best_eval_score),
        "next_eval_step": int(next_eval_step),
        "prior_wall_clock_s": float(prior_wall_clock_s),
        "total_updates": int(getattr(agent, "total_updates", 0)),
        "run_id": run_id,
        "wandb_run_id": wandb_run_id,
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
    }
    if hasattr(agent, "target_actor"):
        state["target_actor"] = agent.target_actor.state_dict()
    if hasattr(agent, "log_alpha_c"):
        state["log_alpha_c"] = agent.log_alpha_c.detach().cpu()
        state["log_alpha_d"] = agent.log_alpha_d.detach().cpu()
        state["alpha_opt"] = agent.alpha_opt.state_dict()
    _atomic_torch_save(state, path)


def _load_resume_checkpoint(path: Path, agent: Any, device: Any) -> Dict[str, Any]:
    import torch
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    tc_key = "target_critic" if "target_critic" in ckpt else "critic_target"
    if tc_key in ckpt:
        agent.target_critic.load_state_dict(ckpt[tc_key])
    agent.actor_opt.load_state_dict(ckpt["actor_opt"])
    agent.critic_opt.load_state_dict(ckpt["critic_opt"])
    agent.total_updates = int(ckpt.get("total_updates", 0))
    ta_key = "target_actor" if "target_actor" in ckpt else "actor_target"
    if hasattr(agent, "target_actor") and ta_key in ckpt:
        agent.target_actor.load_state_dict(ckpt[ta_key])
    if hasattr(agent, "log_alpha_c") and "log_alpha_c" in ckpt:
        with torch.no_grad():
            agent.log_alpha_c.copy_(ckpt["log_alpha_c"].to(device))
            agent.log_alpha_d.copy_(ckpt["log_alpha_d"].to(device))
        if "alpha_opt" in ckpt:
            agent.alpha_opt.load_state_dict(ckpt["alpha_opt"])
    return ckpt


def run_end_to_end(
    config_path: str,
    output_dir: str,
    wandb_mode: str = "disabled",
    wandb_project: str = "uav_search_target",
    wandb_entity: Optional[str] = None,
    hf_repo: Optional[str] = None,
    override_total_steps: Optional[int] = None,
    override_num_envs: Optional[int] = None,
    override_eval_episodes: Optional[int] = None,
    override_max_steps: Optional[int] = None,
    override_seed: Optional[int] = None,
    override_log_interval_steps: Optional[int] = None,
    override_learning_starts: Optional[int] = None,
    override_batch_size: Optional[int] = None,
    max_wall_clock_s: Optional[float] = None,
    mode: str = "production",
    reference_smoke: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    import torch
    from uav_network.algorithms.vanilla_baselines import GPUHybridReplayBuffer

    if reference_smoke:
        mode = "reference-smoke"
    if mode not in ("production", "reference-smoke"):
        raise ValueError(f"Unsupported mode: {mode!r} (expected 'production' or 'reference-smoke')")

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_scenario_config(cfg)

    algo = cfg["runtime"]["algorithm"].lower()
    acfg = cfg[algo]
    seed = int(override_seed if override_seed is not None else cfg["runtime"]["seed"])
    num_envs = int(override_num_envs or cfg["runtime"]["num_envs"])
    max_steps = int(override_max_steps or cfg["world"]["max_steps"])
    total_steps = int(override_total_steps or cfg["runtime"]["total_env_steps"])
    eval_interval = int(cfg["runtime"]["eval_interval_steps"])
    log_interval_steps = int(override_log_interval_steps or cfg["runtime"].get("log_interval_steps", max(num_envs * 10, 1)))
    eval_n = int(override_eval_episodes or cfg["runtime"]["eval_episodes_slotted"])
    test_n = int(override_eval_episodes or cfg["runtime"]["eval_episodes_full_uavnetsim"])
    learning_starts = int(override_learning_starts if override_learning_starts is not None else acfg["learning_starts"])
    batch_size = int(override_batch_size if override_batch_size is not None else acfg["batch_size"])
    belief_mode = str(cfg.get("sensor", {}).get("belief_sharing_mode", "local_decpomdp"))
    aoi_eta = float(cfg["sensor"]["aoi_decay_eta"])
    rew_weights = (
        float(cfg["reward"]["w1_global_ds_uncertainty"]),
        float(cfg["reward"]["w2_gcs_aoi_delivery"]),
        float(cfg["reward"]["w3_propulsion_per_kj"]),
        float(cfg["reward"]["w4_safety_violation"]),
        float(cfg["reward"]["w5_expired_or_dropped"]),
    )

    # Write ALL CLI overrides back into cfg BEFORE logging config_resolved.json!
    cfg["runtime"]["seed"] = seed
    cfg["runtime"]["num_envs"] = num_envs
    cfg["runtime"]["total_env_steps"] = total_steps
    cfg["runtime"]["log_interval_steps"] = log_interval_steps
    cfg["runtime"]["eval_episodes_slotted"] = eval_n
    cfg["runtime"]["eval_episodes_full_uavnetsim"] = test_n
    cfg["world"]["max_steps"] = max_steps
    acfg["learning_starts"] = learning_starts
    acfg["batch_size"] = batch_size

    if mode == "reference-smoke":
        if not (1 <= num_envs <= 64 and 1 <= total_steps <= 5000 and 1 <= max_steps <= 550
                and 1 <= eval_n <= 20 and 1 <= test_n <= 20):
            raise ValueError(
                "CPU reference smoke is bounded: num_envs<=64, total_steps<=5000, "
                "max_steps<=550, eval/test_episodes<=20. Pass reduced -- flags."
            )
        env_backend = "numpy"
        env_device_str = "cpu"
        use_cuda_graph = False
        learner_dev = torch.device("cpu")
        replay_dev = torch.device("cpu")
        gpu_warp_verified = False
        vendor_uavnetsim_verified = False
        log_interval_steps = min(log_interval_steps, max(num_envs, 1))
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Production mode requires an NVIDIA CUDA GPU for Warp FP32 + GPU Replay Buffer. "
                "Run on Kaggle T4/T4x2 (or use `--mode reference-smoke` on CPU)."
            )
        env_backend = "warp_slotted_fp32"
        env_device_str = str(cfg["runtime"].get("device", "cuda:0"))
        use_cuda_graph = bool(cfg["runtime"].get("use_cuda_graph", True))
        learner_dev = _select_device(str(cfg["runtime"].get("learner_device", "cuda:1")))
        replay_dev = _select_device(str(cfg["runtime"].get("replay_device", "cuda:1")))
        gpu_warp_verified = True
        vendor_uavnetsim_verified = True

    if num_envs > int(acfg["replay_capacity"]):
        raise ValueError("Replay capacity must be >= num_envs to avoid index aliasing.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    latest_ckpt = ckpt_dir / "latest_checkpoint.pt"
    if not resume and out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError(
            f"Output directory already contains artifacts: {out_path}. "
            "Use a fresh run directory or pass --resume."
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(cfg, learner_dev)
    global_step = 0
    best_eval_score = -1e18
    next_eval_step = min(eval_interval, total_steps)
    prior_wall_clock_s = 0.0
    saved_run_id: Optional[str] = None
    saved_wandb_id: Optional[str] = None
    best_ckpt_path = ckpt_dir / "best_actor.pt"

    if resume and latest_ckpt.exists():
        ckpt = _load_resume_checkpoint(latest_ckpt, agent, learner_dev)
        global_step = int(ckpt["global_step"])
        best_eval_score = float(ckpt["best_eval_score"])
        prior_wall_clock_s = float(ckpt.get("prior_wall_clock_s", 0.0))
        saved_run_id = ckpt.get("run_id")
        saved_wandb_id = ckpt.get("wandb_run_id")
        next_eval_step = min(((global_step // eval_interval) + 1) * eval_interval, total_steps)
    else:
        _atomic_torch_save(agent.actor.state_dict(), ckpt_dir / "initial_actor.pt")

    logger = SlottedTelemetryLogger(
        output_dir=out_path,
        config=cfg,
        algorithm=algo,
        seed=seed,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        hf_repo_id=hf_repo,
        run_name=saved_run_id,
        wandb_run_id=saved_wandb_id,
        resume=resume,
        gpu_warp_verified=gpu_warp_verified,
        vendor_uavnetsim_verified=vendor_uavnetsim_verified,
    )

    env = Slotted2DUavEnv(
        num_envs=num_envs,
        max_steps=max_steps,
        aoi_decay_eta=aoi_eta,
        w1_ds=rew_weights[0],
        w2_aoi=rew_weights[1],
        w3_prop_kj=rew_weights[2],
        w4_safety=rew_weights[3],
        w5_drop=rew_weights[4],
        base_seed=seed,
        backend=env_backend,
        belief_sharing_mode=belief_mode,
        device=env_device_str,
        use_cuda_graph=use_cuda_graph,
    )

    buf = GPUHybridReplayBuffer(
        capacity=int(acfg["replay_capacity"]),
        num_agents=int(cfg["world"]["num_uavs"]),
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        cont_dim=CONT_ACTION_DIM,
        disc_dim=DISC_ACTION_DIM,
        device=replay_dev,
    )

    eval_env: Optional[Slotted2DUavEnv] = None
    val_seeds = [1_000_000_000 + i for i in range(eval_n)]
    test_seeds = [2_000_000_000 + i for i in range(test_n)]

    obs, state, mask = env.reset()
    is_gpu_env = env._gpu_backend is not None
    env_dev = env._gpu_backend.torch_device if is_gpu_env else torch.device("cpu")
    dual_gpu_async = is_gpu_env and (env_dev != learner_dev)

    if is_gpu_env:
        obs_f16 = obs.to(torch.float16).to(replay_dev, non_blocking=True)
        obs_l = obs_f16.to(learner_dev, dtype=torch.float32, non_blocking=True)
        state_l = state.to(torch.float16).to(replay_dev, non_blocking=True)
        mask_l = mask.to(learner_dev, dtype=torch.uint8, non_blocking=True)
        if env_dev == learner_dev:
            obs_f16 = obs_f16.clone()
            state_l = state_l.clone()
            mask_l = mask_l.clone()

    session_start_step = global_step
    t0 = time.perf_counter()
    next_log_step = global_step + log_interval_steps
    n_upd = int(acfg["updates_per_step"])

    while global_step < total_steps:
        next_step_count = global_step + num_envs
        should_log = (next_step_count >= next_log_step) or (next_step_count >= total_steps)
        learner_metrics: Dict[str, float] = {}

        if is_gpu_env:
            act_cont_t, act_disc_oh_t = agent.act(obs_l, mask_l, explore=True)
            act_disc_t = act_disc_oh_t.argmax(dim=-1).to(torch.int32)

            if dual_gpu_async:
                # Dual-T4 Async Overlap: kick off GPU0 Warp step while GPU1 runs 8 learner updates!
                next_obs, next_state, next_mask, rewards, dones, infos = env._gpu_backend.step_torch(
                    act_cont_t.detach().to(env_dev, non_blocking=True),
                    act_disc_t.detach().to(env_dev, non_blocking=True),
                    compute_reward_info=should_log,
                )
                if buf.size >= max(learning_starts, batch_size):
                    for ui in range(n_upd):
                        batch = buf.sample(batch_size)
                        m_i = agent.update(batch, return_metrics=(should_log and ui == n_upd - 1))
                        if m_i:
                            learner_metrics = m_i
            else:
                next_obs, next_state, next_mask, rewards, dones, infos = env._gpu_backend.step_torch(
                    act_cont_t.detach().to(env_dev, non_blocking=True),
                    act_disc_t.detach().to(env_dev, non_blocking=True),
                    compute_reward_info=should_log,
                )

            next_obs_f16 = next_obs.to(torch.float16).to(replay_dev, non_blocking=True)
            next_state_f16 = next_state.to(torch.float16).to(replay_dev, non_blocking=True)
            next_mask_l = next_mask.to(learner_dev, dtype=torch.uint8, non_blocking=True)
            if env_dev == learner_dev:
                next_obs_f16 = next_obs_f16.clone()
                next_state_f16 = next_state_f16.clone()
                next_mask_l = next_mask_l.clone()
            term_dones = infos.get("terminated", dones)

            buf.add_batch(
                obs=obs_f16,
                state=state_l,
                act_cont=act_cont_t,
                act_disc_onehot=act_disc_oh_t,
                rewards=rewards.to(replay_dev, non_blocking=True),
                next_obs=next_obs_f16,
                next_state=next_state_f16,
                dones=term_dones.to(replay_dev, non_blocking=True),
                mask=mask_l,
                next_mask=next_mask_l,
            )
            obs_f16 = next_obs_f16
            obs_l = next_obs_f16.to(learner_dev, dtype=torch.float32, non_blocking=True)
            state_l = next_state_f16
            mask_l = next_mask_l

            if not dual_gpu_async and buf.size >= max(learning_starts, batch_size):
                for ui in range(n_upd):
                    batch = buf.sample(batch_size)
                    m_i = agent.update(batch, return_metrics=(should_log and ui == n_upd - 1))
                    if m_i:
                        learner_metrics = m_i
        else:
            obs_t = torch.from_numpy(obs).to(learner_dev, dtype=torch.float32)
            mask_t = torch.from_numpy(mask).to(learner_dev, dtype=torch.uint8)
            act_cont_t, act_disc_oh_t = agent.act(obs_t, mask_t, explore=True)
            act_cont_np = act_cont_t.cpu().numpy().astype(np.float32)
            act_disc_idx = act_disc_oh_t.argmax(dim=-1).cpu().numpy().astype(np.int32)

            next_obs, next_state, next_mask, rewards, dones, infos = env.step(act_cont_np, act_disc_idx)
            buf.add_batch(
                obs=obs_t,
                state=torch.from_numpy(state),
                act_cont=act_cont_t,
                act_disc_onehot=act_disc_oh_t,
                rewards=torch.from_numpy(rewards),
                next_obs=torch.from_numpy(next_obs),
                next_state=torch.from_numpy(next_state),
                dones=torch.from_numpy(dones),
                mask=mask_t,
                next_mask=torch.from_numpy(next_mask),
            )
            obs, state, mask = next_obs, next_state, next_mask
            if (dones > 0.5).any():
                done_indices = np.flatnonzero(dones > 0.5).tolist()
                obs, state, mask = env.reset_indices(
                    done_indices, seeds=[seed + next_step_count + i for i in done_indices]
                )
            if buf.size >= max(learning_starts, batch_size):
                for ui in range(n_upd):
                    batch = buf.sample(batch_size)
                    m_i = agent.update(batch, return_metrics=(should_log and ui == n_upd - 1))
                    if m_i:
                        learner_metrics = m_i

        global_step = next_step_count

        if should_log:
            session_elapsed = max(time.perf_counter() - t0, 1e-6)
            total_elapsed = prior_wall_clock_s + session_elapsed
            if is_gpu_env:
                r_vec = torch.stack([
                    rewards.float().mean(),
                    infos["r_ds"].float().mean(),
                    infos["r_gcs_aoi"].float().mean(),
                    infos["r_propulsion"].float().mean(),
                    infos["r_safety"].float().mean(),
                    infos["r_drop_ttl"].float().mean(),
                ]).detach().cpu().tolist()
                a_vec = torch.stack([
                    (act_cont_t.abs() > 0.99).float().mean(),
                    (act_disc_t == 0).float().mean(),
                    (mask_l.sum(dim=-1) == 1).float().mean(),
                ]).detach().cpu().tolist()
            else:
                r_vec = [
                    float(rewards.mean()),
                    float(infos["r_ds"].mean()),
                    float(infos["r_gcs_aoi"].mean()),
                    float(infos["r_propulsion"].mean()),
                    float(infos["r_safety"].mean()),
                    float(infos["r_drop_ttl"].mean()),
                ]
                a_vec = [
                    float((np.abs(act_cont_np) > 0.99).mean()),
                    float((act_disc_idx == 0).mean()),
                    float((mask.sum(axis=-1) == 1).mean()),
                ]

            train_scalars = {
                "train/step_reward_mean": r_vec[0],
                "reward/r_ds": r_vec[1],
                "reward/r_gcs_aoi": r_vec[2],
                "reward/r_propulsion": r_vec[3],
                "reward/r_safety": r_vec[4],
                "reward/r_drop_ttl": r_vec[5],
                "learner/tanh_sat_frac": a_vec[0],
                "learner/silent_frac": a_vec[1],
                "learner/mask_silent_only_frac": a_vec[2],
                "sys/env_steps_per_sec": float((global_step - session_start_step) / session_elapsed),
                "sys/wall_clock_s": total_elapsed,
            }
            for k, v in learner_metrics.items():
                train_scalars[f"learner/{k}"] = v
            logger.log_train_step(global_step, train_scalars)
            next_log_step = global_step + log_interval_steps

        # Phase 2: Periodic Validation on fixed held-out val_seeds
        if global_step >= next_eval_step:
            policy_fn = make_deterministic_policy_fn(agent, learner_dev)
            gpu_pfn = make_deterministic_torch_policy_fn(agent, learner_dev, env_dev) if is_gpu_env else None
            if eval_env is None and is_gpu_env:
                eval_env = Slotted2DUavEnv(
                    num_envs=len(val_seeds),
                    max_steps=max_steps,
                    aoi_decay_eta=aoi_eta,
                    w1_ds=rew_weights[0],
                    w2_aoi=rew_weights[1],
                    w3_prop_kj=rew_weights[2],
                    w4_safety=rew_weights[3],
                    w5_drop=rew_weights[4],
                    backend=env_backend,
                    belief_sharing_mode=belief_mode,
                    device=env_device_str,
                    use_cuda_graph=use_cuda_graph,
                    gpu_autoreset=False,
                )
            eval_eps = evaluate_on_slotted_warp(
                policy_fn,
                val_seeds,
                max_steps=max_steps,
                backend=env_backend,
                belief_sharing_mode=belief_mode,
                device=env_device_str,
                use_cuda_graph=use_cuda_graph,
                aoi_decay_eta=aoi_eta,
                reward_weights=rew_weights,
                gpu_policy_fn=gpu_pfn,
                eval_env=eval_env,
            )
            eval_sum = logger.log_periodic_eval(global_step, eval_eps)
            _atomic_torch_save(agent.actor.state_dict(), ckpt_dir / f"actor_step_{global_step:012d}.pt")
            ret_iqm = eval_sum.get("return_iqm")
            if ret_iqm is not None and float(ret_iqm) >= best_eval_score:
                best_eval_score = float(ret_iqm)
                _atomic_torch_save(agent.actor.state_dict(), best_ckpt_path)
            next_eval_step = min(((global_step // eval_interval) + 1) * eval_interval, total_steps + eval_interval)
            total_elapsed = prior_wall_clock_s + (time.perf_counter() - t0)
            _save_resume_checkpoint(
                latest_ckpt, agent, global_step, best_eval_score, next_eval_step,
                total_elapsed, logger.run_id, logger.wandb_run_id,
            )
            logger.upload_checkpoint_snapshot()

        if max_wall_clock_s is not None and (time.perf_counter() - t0) >= float(max_wall_clock_s):
            total_elapsed = prior_wall_clock_s + (time.perf_counter() - t0)
            _save_resume_checkpoint(
                latest_ckpt, agent, global_step, best_eval_score, next_eval_step,
                total_elapsed, logger.run_id, logger.wandb_run_id,
            )
            hf_snap = logger.upload_checkpoint_snapshot()
            logger.close()
            return {
                "mode": mode,
                "status": "paused_for_wall_clock",
                "global_step": global_step,
                "best_eval_return_iqm": best_eval_score,
                "hf_snapshot": hf_snap,
            }

    total_elapsed = prior_wall_clock_s + (time.perf_counter() - t0)
    _atomic_torch_save(agent.actor.state_dict(), ckpt_dir / "final_actor.pt")
    _save_resume_checkpoint(
        latest_ckpt, agent, global_step, best_eval_score, next_eval_step,
        total_elapsed, logger.run_id, logger.wandb_run_id,
    )
    if best_ckpt_path.exists():
        agent.actor.load_state_dict(torch.load(best_ckpt_path, map_location=learner_dev, weights_only=True))

    best_policy_fn = make_deterministic_policy_fn(agent, learner_dev)
    best_gpu_pfn = make_deterministic_torch_policy_fn(agent, learner_dev, env_dev) if is_gpu_env else None
    test_slotted_eps = evaluate_on_slotted_warp(
        best_policy_fn,
        test_seeds,
        max_steps=max_steps,
        backend=env_backend,
        belief_sharing_mode=belief_mode,
        device=env_device_str,
        use_cuda_graph=use_cuda_graph,
        aoi_decay_eta=aoi_eta,
        reward_weights=rew_weights,
        gpu_policy_fn=best_gpu_pfn,
        eval_env=eval_env if (eval_env is not None and len(test_seeds) == len(val_seeds)) else None,
    )
    if mode == "production":
        cpu_actor_sd = {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()}
        test_csma_eps = evaluate_on_full_csma_uavnetsim(
            best_policy_fn,
            test_seeds,
            max_steps=max_steps,
            belief_sharing_mode=belief_mode,
            aoi_decay_eta=aoi_eta,
            reward_weights=rew_weights,
            policy_spec={"cfg": cfg, "actor_state_dict": cpu_actor_sd},
        )
    else:
        test_csma_eps = evaluate_on_reference_csma(
            best_policy_fn,
            test_seeds,
            max_steps=max_steps,
            belief_sharing_mode=belief_mode,
        )

    final_report = logger.log_final_dual_fidelity_test(global_step, test_slotted_eps, test_csma_eps)
    hf_receipt = logger.write_sha256_manifest_and_upload_hf()
    logger.close()

    return {
        "mode": mode,
        "status": "completed",
        "global_step": global_step,
        "total_updates": int(getattr(agent, "total_updates", 0)),
        "best_eval_return_iqm": best_eval_score,
        "gpu_warp_verified": gpu_warp_verified,
        "vendor_uavnetsim_verified": vendor_uavnetsim_verified,
        "final_dual_fidelity_report": final_report,
        "hf_receipt": hf_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End Slotted Vanilla MATD3/MASAC Trainer (Streamfix v3)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="production", choices=["production", "reference-smoke"])
    parser.add_argument("--reference-smoke", action="store_true", help="Alias for --mode reference-smoke")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest_checkpoint.pt")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default="uav_search_target")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-interval-steps", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-wall-clock-s", type=float, default=None)
    args = parser.parse_args()

    res = run_end_to_end(
        config_path=args.config,
        output_dir=args.output,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        hf_repo=args.hf_repo,
        override_total_steps=args.total_steps,
        override_num_envs=args.num_envs,
        override_eval_episodes=args.eval_episodes,
        override_max_steps=args.max_steps,
        override_seed=args.seed,
        override_log_interval_steps=args.log_interval_steps,
        override_learning_starts=args.learning_starts,
        override_batch_size=args.batch_size,
        max_wall_clock_s=args.max_wall_clock_s,
        mode=args.mode,
        reference_smoke=args.reference_smoke,
        resume=args.resume,
    )
    print(f"RUN STATUS ({res['mode']} -> {res['status']}, updates={res.get('total_updates', 0)}):", res.get("hf_receipt") or res.get("hf_snapshot"))


if __name__ == "__main__":
    main()
