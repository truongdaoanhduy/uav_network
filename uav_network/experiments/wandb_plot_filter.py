"""W&B charts-only: bounded numeric learning/outcome/safety/hardware indicators.

All other detailed per-episode/learner/PHY/CBF evidence remains in local JSONL
and public Hugging Face .tar.gz, not in growing W&B run history or Files.
"""
from __future__ import annotations
from numbers import Real
import math

# Prefer actual outcome and evaluation over intermediate proxies.
PLOT_FIELDS=(
 "learner/actor_motion_pretanh_abs_mean",
 "learner/actor_motion_tanh_derivative_mean",
 "learner/gradprobe/critic_facing_dloss_da_x_abs_mean",
 "learner/gradprobe/pretanh_dloss_dz_x_abs_mean",
 "action_probe/recipient_silent_fraction",
 "action_probe/mask_eligible_fraction",
 "action_probe/motion_command_saturation_fraction",
 "action_probe/post_safety_blocked_fraction",
 "action_probe/raw_boundary_fraction",
 "global_step","episode/index","evaluation/checkpoint_episode",
 "train/episodes_completed","train/updates","train/replay_size",
 "train/step_reward_mean","train/step_reward_std",
 "train/step_reward_mean_2","train/step_reward_mean_4","train/step_reward_mean_8",
 "train/step_reward_std_2","train/step_reward_std_4","train/step_reward_std_8",
 "train/reward_window_transitions_2","train/reward_window_transitions_4",
 "train/reward_window_transitions_8",
 "train/reward_window_transitions","train/reward_window_collections",
 "train/reward_window_start_step",
 "train/episode_return_mean","train/episode_return_std","train/episode_window_count",
 "train/joint_transitions_per_s","run/elapsed_hours",
 "progress/search_rate_percent","progress/delivery_rate_percent",
 "progress/mean_battery_fraction",
 "cohort/return_mean","cohort/search_rate_percent_mean",
 "cohort/delivery_rate_percent_mean","cohort/success_mean",
 "rolling/return_mean","rolling/search_rate_percent","rolling/delivery_rate_percent",
 "rolling/success_rate",
 "episode/return","episode/length","outcome/search_rate_percent",
 "outcome/delivery_rate_percent","outcome/success",
 "eval/mean_return","eval/mean_delivery_rate_percent","eval/mean_search_rate_percent",
 "eval/success_rate","eval/checkpoint_episode",
 "evaluation/checkpoint_global_step","evaluation/episodes_completed",
 "evaluation/mean_delivery_rate_percent","evaluation/mean_search_rate_percent",
 "evaluation/success_rate",
 "evaluation/return","evaluation/mean_return","evaluation/search_rate_percent","evaluation/delivery_rate_percent",
 "evaluation/success",
 # Exact per-episode scientific outcomes (no fabricated values).
 "training/search_delivery_gap_percentage_points",
 "training/targets_found","training/targets_delivered",
 "training/mission_success_fraction","training/discovery_auc",
 "training/gcs_delivery_auc","training/delivery_auc",
 "training/discovery_rate_percent","training/gcs_delivery_rate_percent",
 "training/mission_success_rate_percent","training/discovery_time_capped_mean_s",
 "training/gcs_delivery_time_capped_mean_s",
 "training/mission_completion_time_capped_s",
 "training/total_uav_energy_j","training/energy_per_delivered_target_j",
 "training/delivery_given_discovery_fraction",
 "training/first_discovery_to_gcs_delay_delivered_mean_s",
 "training/false_confirmation_per_1000_measurements","training/false_confirmation_rate_percent",
 "training/map_coverage_percent","training/packet_pdr_percent",
 # Additional real held-out checkpoint metrics (not in-sample training losses).
 "evaluation/core/discovery_rate_percent","evaluation/core/gcs_delivery_rate_percent",
 "evaluation/core/discovery_time_capped_mean_s",
 "evaluation/core/gcs_delivery_time_capped_mean_s",
 "evaluation/core/energy_per_delivered_target_j",
 "evaluation/discovery_auc","evaluation/delivery_auc",
 "evaluation/mission_success","evaluation/mission_completion_time_capped_s",
 "evaluation/pooled_uav_j_per_gcs_delivery","evaluation/delivery_given_discovery_fraction",
 "evaluation/map_coverage_percent","evaluation/false_confirmation_per_1000_measurements",
 "evaluation/first_discovery_to_gcs_delay_delivered_mean_s",
 "evaluation/packet_pdr_fraction",
 "mission_probe/delivery_given_found_fraction",
 "mission_probe/found_target_fraction","mission_probe/delivered_target_fraction",
 "mission_probe/pending_without_fifo_or_active_packet_fraction",
 "mission_probe/pending_with_report_but_no_optimistic_path_fraction",
 "mission_probe/reward_accounting_abs_discrepancy_mean",
 "action_probe/cbf_projected_fraction","action_probe/cbf_correction_velocity_mean_mps",
 "action_probe/motion_saturation_fraction","action_probe/command_z_mean",
 "safety/cbf_intervention_rate","safety/raw_unsafe_action_rate",
 "reports/expired","reports/dropped","network/pdr_percent",
 "energy/mean_battery_fraction",
 "health/nonfinite_learner_metric","health/gpu0_low_physical_headroom",
 "health/gpu1_low_physical_headroom","health/actor_motion_tanh_saturated",
 "health/reward_component_inconsistent","health/found_but_delivery_stalled",
 "health/critic_large_td_error",
 "memory/cuda0_pytorch_peak_reserved_gib",
 "memory/cuda1_pytorch_peak_reserved_gib",
 "memory/cuda0_driver_peak_used_observed_gib",
 "memory/cuda1_driver_peak_used_observed_gib",
 "resources/host_process_rss_gib",
 "pql/critic_updates_per_s","pql/actor_updates_per_s",
 "pql/collector_wait_seconds","pql/queued_collections",
 "replay/evicted_unsampled_fraction",
)
EXTRA_ALLOWED_PREFIXES=("learner/critic_loss","learner/actor_loss",
 "learner/critic_grad","learner/actor_grad","learner/alpha",
 "learner/td_error","learner/q_", "timing/optimizer_update",
 "memory/cuda0_", "memory/cuda1_")
# Scientific mission outcomes are observability, not training controls.
# Include these even when compact plotting-only logging is enabled.
MISSION_OUTCOME_PREFIXES=("training/","gpu_eval_periodic/","train_cohort/","train_progress/","train_loss/","train_episode_step/")
MAX_LOG_FIELDS=256


def select_plot_metrics(values):
    selected={}
    for key in PLOT_FIELDS:
        if key in values:
            selected[key]=values[key]
    for key in sorted(values):
        if len(selected)>=MAX_LOG_FIELDS:break
        if key in selected:continue
        if (any(key.startswith(prefix) for prefix in EXTRA_ALLOWED_PREFIXES)
                or any(key.startswith(prefix) for prefix in MISSION_OUTCOME_PREFIXES)):
            selected[key]=values[key]
    return {k:float(v) if isinstance(v,Real) else v for k,v in selected.items()
            if isinstance(v,Real) and math.isfinite(float(v))}
