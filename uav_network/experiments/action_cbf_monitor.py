"""Non-invasive, bounded runtime audit of Actor commands and CBF execution.

The Warp motion kernel already produces nominal/CBF velocities and motion_flags.
Read them at an infrequent completed step: no change to environment physics,
CBF constraints, rewards, replay or Actor outputs. The stored two examples
are a diagnostic subset, not a frame-by-frame movie.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch


FLAG_NAMES = {
    1: "cbf_projected", 2: "post_safety_blocked",
    3: "obstacle_blocked", 4: "boundary_clipped",
    5: "raw_trajectory_unsafe", 6: "raw_boundary_violation",
    7: "raw_obstacle_intersection", 8: "raw_peer_proximity",
}


class ActionCBFMonitor:
    def __init__(self, config, output):
        # The No-CBF release still uses a common Training loop, but must not
        # execute expensive CBF-specific observation/transfers on every cadence.
        # The legacy monitor was SILENT in reject-mode despite the rest of
        # training treating its output as the gate for mission/health logs.
        # Observe the safety execution for both modes; never change motion.
        self.motion_safety = str(config.world.motion_safety)
        self.enabled = (config.runtime.environment_backend == "warp_packet"
                        and self.motion_safety in ("cbf", "reject"))
        self.interval = max(1, int(config.runtime.log_interval_steps))
        self.next_step = 0
        self.num_uavs = config.world.num_uavs
        self.map_size_m = config.world.map_size_m
        self.altitude_min_m = config.world.altitude_min_m
        self.altitude_max_m = config.world.altitude_max_m
        self.path = Path(output) / "action-cbf-samples.jsonl"
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
        self.last_values = {}

    def due(self, step):
        return self.enabled and step >= self.next_step

    @torch.no_grad()
    def sample(self, vector, before_obs, action, step, *, phase="policy"):
        """Summaries over ALL envs; at most two concrete CBF examples.

        Called AFTER step_wait, BEFORE reset. No random numbers or grads.
        The velocity difference is in m/s; this is not a normalized XYZ action
        change and not proof a specific constraint caused a CBF projection.
        """
        if not self.due(step) or not hasattr(vector, "s"):
            return {}
        import warp as wp
        self.next_step = step + self.interval
        state = vector.s
        commands = torch.as_tensor(action.continuous).detach()
        destination = torch.as_tensor(action.destination).detach()
        local = torch.as_tensor(before_obs).detach()
        flags = wp.to_torch(state.motion_flags).detach()
        work = wp.to_torch(state.xyz_work).detach()
        # Nominal/CBF fields change from acceleration to predicted velocity
        # inside the xyz_motion kernel and remain available after step_wait.
        nominal, chosen = work[:, :, 0], work[:, :, 1]
        actual = torch.cat((wp.to_torch(state.velocities),
                            wp.to_torch(state.vertical_velocity)[..., None]), dim=-1)
        active = (local[..., 9] > .5) & (flags[..., 0] == 0)
        den = active.to(dtype=torch.float64).sum().clamp(min=1)
        num_envs = int(local.shape[0])
        device = commands.device
        if flags.device != device:
            raise ValueError("Action and CBF state must live on the same device")
        valid = local[..., -(self.num_uavs+1):] > .5
        valid = valid.clone()
        valid[..., 0] = True
        options = valid.sum(-1)
        selected_valid = valid.gather(-1, destination.long().unsqueeze(-1)).squeeze(-1)
        correction_velocity = torch.linalg.vector_norm(chosen - nominal, dim=-1)
        fallback_velocity = torch.linalg.vector_norm(actual - chosen, dim=-1)
        nominal_actual_velocity = torch.linalg.vector_norm(actual - nominal, dim=-1)
        actual_speed = torch.linalg.vector_norm(actual, dim=-1)
        command_strength = torch.linalg.vector_norm(commands[..., :3], dim=-1)
        position_xy = wp.to_torch(state.positions).detach()
        altitudes = wp.to_torch(state.altitudes).detach()
        x, y = position_xy[..., 0], position_xy[..., 1]
        near_x0, near_y0 = x < 1.0, y < 1.0
        near_xmax = x > self.map_size_m - 1.0
        near_ymax = y > self.map_size_m - 1.0
        near_floor = altitudes < self.altitude_min_m + 1.0
        near_ceiling = altitudes > self.altitude_max_m - 1.0
        near_boundary = near_x0 | near_xmax | near_y0 | near_ymax | near_floor | near_ceiling
        command_outward = ((near_x0 & (commands[..., 0] < -0.1)) |
                           (near_xmax & (commands[..., 0] > 0.1)) |
                           (near_y0 & (commands[..., 1] < -0.1)) |
                           (near_ymax & (commands[..., 1] > 0.1)) |
                           (near_floor & (commands[..., 2] < -0.1)) |
                           (near_ceiling & (commands[..., 2] > 0.1)))
        projected = flags[..., 1] != 0
        raw_unsafe = flags[..., 5] != 0
        def rate(condition):
            return float(((condition & active).to(dtype=torch.float64).sum() / den).item())
        def active_mean(values):
            return float(((values.double() * active).sum() / den).item())
        def scalar(values):
            return float(values.detach().double().mean().item())
        out = {
            "action_probe/eligible_agent_count": int(active.sum().item()),
            "action_probe/worlds": num_envs,
            "action_probe/phase_random_warmup": int(phase == "random_warmup"),
            "action_probe/mask_silent_only_fraction": rate(options == 1),
            "action_probe/mask_eligible_fraction": rate(options > 1),
            "action_probe/selected_invalid_fraction": rate(~selected_valid),
            "action_probe/recipient_silent_fraction": rate(destination == 0),
            "action_probe/recipient_peer_fraction": rate((destination > 0) & (destination < self.num_uavs)),
            "action_probe/recipient_gcs_fraction": rate(destination == self.num_uavs),
            "action_probe/rf_power_command_abs_mean": active_mean(commands[..., 3].abs()),
            "action_probe/rf_power_nonzero_fraction": rate(commands[..., 3].abs() > 1e-7),
            "action_probe/motion_command_abs_mean": scalar(commands[..., :3].abs()),
            "action_probe/motion_command_saturation_fraction":
                scalar((commands[..., :3].abs() > .95).to(dtype=torch.float32)),
            "action_probe/cbf_projected_fraction": rate(projected),
            # Corrections grouped by magnitude; a projection can be tiny.
            "action_probe/cbf_correction_gt_0_1mps_fraction": rate(correction_velocity > 0.1),
            "action_probe/cbf_correction_gt_0_5mps_fraction": rate(correction_velocity > 0.5),
            "action_probe/cbf_correction_gt_1mps_fraction": rate(correction_velocity > 1.0),
            "action_probe/cbf_correction_gt_2mps_fraction": rate(correction_velocity > 2.0),
            "action_probe/nominal_actual_difference_gt_1mps_fraction": rate(nominal_actual_velocity > 1.0),
            "action_probe/near_boundary_1m_fraction": rate(near_boundary),
            "action_probe/near_ceiling_1m_fraction": rate(near_ceiling),
            "action_probe/near_floor_1m_fraction": rate(near_floor),
            "action_probe/command_outward_at_boundary_fraction": rate(command_outward),
            "action_probe/stalled_with_strong_command_fraction": rate((actual_speed < 0.1) & (command_strength > 0.5)),
            "action_probe/post_safety_blocked_at_boundary_fraction": rate((flags[..., 2] != 0) & near_boundary),
            "action_probe/raw_unsafe_fraction": rate(raw_unsafe),
            "action_probe/raw_boundary_fraction": rate(flags[..., 6] != 0),
            "action_probe/raw_obstacle_fraction": rate(flags[..., 7] != 0),
            "action_probe/raw_peer_fraction": rate(flags[..., 8] != 0),
            "action_probe/cbf_plus_raw_unsafe_fraction": rate(projected & raw_unsafe),
            "action_probe/cbf_without_raw_unsafe_fraction": rate(projected & ~raw_unsafe),
            "action_probe/post_safety_blocked_fraction": rate(flags[..., 2] != 0),
            "action_probe/obstacle_blocked_fraction": rate(flags[..., 3] != 0),
            "action_probe/boundary_clipped_fraction": rate(flags[..., 4] != 0),
            "action_probe/cbf_correction_velocity_mean_mps": active_mean(correction_velocity),
            "action_probe/cbf_correction_velocity_max_mps": float(
                correction_velocity.masked_fill(~active, 0).max().item()),
            "action_probe/post_cbf_fallback_velocity_mean_mps": active_mean(fallback_velocity),
            "action_probe/actual_speed_mean_mps": active_mean(actual_speed),
            "action_probe/step_shield_correction_mean":scalar(wp.to_torch(state.shield_correction_step)),
            "action_probe/step_shield_intervention_mean":scalar(wp.to_torch(state.shield_intervention_step)),
            "action_probe/step_raw_unsafe_mean":scalar(wp.to_torch(state.unsafe_action_step)),
        }
        if active.any():
            out["action_probe/cbf_correction_p95_mps"] = float(
                torch.quantile(correction_velocity[active], .95).item())
            out["action_probe/nominal_actual_difference_p95_mps"] = float(
                torch.quantile(nominal_actual_velocity[active], .95).item())
        else:
            out["action_probe/cbf_correction_p95_mps"] = 0.0
            out["action_probe/nominal_actual_difference_p95_mps"] = 0.0
        for axis, label in enumerate("xyz"):
            values=commands[..., axis]
            out[f"action_probe/command_{label}_mean"] = active_mean(values)
            out[f"action_probe/command_{label}_abs_mean"] = active_mean(values.abs())
            out[f"action_probe/command_{label}_positive_saturation_fraction"] = rate(values > .95)
            out[f"action_probe/command_{label}_negative_saturation_fraction"] = rate(values < -.95)
        for receiver in range(self.num_uavs + 1):
            out[f"action_probe/recipient_{receiver}_fraction"] = rate(destination == receiver)
        # Agent-specific drift/saturation diagnosis. Every percentage is over
        # currently active worlds for that UAV, never over total envs blindly.
        for agent in range(self.num_uavs):
            own = active[:, agent]
            own_den = own.double().sum().clamp(min=1)
            def own_mean(values):
                return float((values.double() * own).sum().div(own_den).item())
            def own_rate(values):
                return own_mean(values.float())
            prefix = f"action_probe/uav_{agent}/"
            for axis, label in enumerate("xyz"):
                out[prefix + f"command_{label}_mean"] = own_mean(commands[:,agent,axis])
            out[prefix + "rf_power_abs_mean"] = own_mean(commands[:,agent,3].abs())
            out[prefix + "motion_saturation_fraction"] = own_rate(
                (commands[:,agent,:3].abs() > .95).float().mean(-1))
            out[prefix + "silent_fraction"] = own_rate(destination[:,agent] == 0)
            out[prefix + "cbf_projected_fraction"] = own_rate(projected[:,agent])
            out[prefix + "raw_unsafe_fraction"] = own_rate(raw_unsafe[:,agent])
            out[prefix + "near_boundary_1m_fraction"] = own_rate(near_boundary[:,agent])
            out[prefix + "command_outward_at_boundary_fraction"] = own_rate(command_outward[:,agent])
            out[prefix + "stalled_with_strong_command_fraction"] = own_rate(
                (actual_speed[:,agent] < 0.1) & (command_strength[:,agent] > 0.5))
        # Extract concise reproducible examples ONLY when CBF actually intervened.
        events = projected & active
        where = torch.nonzero(events, as_tuple=False)
        examples = []
        if where.numel():
            scores = correction_velocity[events]
            top = torch.topk(scores, k=min(2, int(scores.numel()))).indices
            for selection in top.tolist():
                env_index, agent_index = (int(x) for x in where[selection].tolist())
                flag_values = flags[env_index,agent_index].tolist()
                trigger_flags = [name for idx, name in FLAG_NAMES.items()
                                 if flag_values[idx]]
                # Raw trajectory labels describe the command, not a proof of
                # which CBF halfspace became active.
                examples.append({
                    "env_index": env_index, "uav_index": agent_index,
                    "actor_command_xyz_rf": [round(float(x), 5) for x in commands[env_index,agent_index].tolist()],
                    "destination_index": int(destination[env_index,agent_index].item()),
                    "battery_fraction_before":round(float(local[env_index,agent_index,8]),5),
                    "position_after_m": [
                        round(float(x),5) for x in [
                            *wp.to_torch(state.positions)[env_index,agent_index].tolist(),
                            wp.to_torch(state.altitudes)[env_index,agent_index].item()
                        ]
                    ],
                    "nominal_velocity_mps": [round(float(x), 5) for x in nominal[env_index,agent_index].tolist()],
                    "cbf_selected_velocity_mps": [round(float(x), 5) for x in chosen[env_index,agent_index].tolist()],
                    "actual_velocity_mps": [round(float(x), 5) for x in actual[env_index,agent_index].tolist()],
                    "cbf_velocity_correction_mps": round(float(correction_velocity[env_index,agent_index]), 5),
                    "post_cbf_fallback_velocity_mps": round(float(fallback_velocity[env_index,agent_index]), 5),
                    "flags": trigger_flags,
                    "note": "Raw violation flags can co-occur and are not exclusive active CBF constraint IDs.",
                })
        # In reject mode there is no CBF intervention. Record the *actual*
        # boundary/obstacle/peer rejection examples instead; never label them
        # CBF projections and never imply physical actions were redirected.
        reject_examples = []
        if self.motion_safety == "reject":
            rejected = (raw_unsafe | (flags[...,2]!=0) | (flags[...,4]!=0)) & active
            indices = torch.nonzero(rejected, as_tuple=False)
            for env_id, agent_id in indices[:min(4, len(indices))].tolist():
                reject_examples.append({
                    "env_index":int(env_id),"uav_index":int(agent_id),
                    "actor_command_xyz_rf":commands[env_id,agent_id].cpu().tolist(),
                    "route_destination_index":int(destination[env_id,agent_id]),
                    "nominal_velocity_mps":nominal[env_id,agent_id].cpu().tolist(),
                    "actual_velocity_mps":actual[env_id,agent_id].cpu().tolist(),
                    "post_rejection_blocked":bool(flags[env_id,agent_id,2]),
                    "raw_boundary_violation":bool(flags[env_id,agent_id,6]),
                    "raw_obstacle_violation":bool(flags[env_id,agent_id,7]),
                    "raw_peer_violation":bool(flags[env_id,agent_id,8]),
                    "boundary_clipped":bool(flags[env_id,agent_id,4]),
                    "position_after_m": [*position_xy[env_id,agent_id].cpu().tolist(),
                                         float(altitudes[env_id,agent_id])],
                    "scope":"actual_nominal_and_post_rejection_motion",
                })
        out["action_probe/safety_reject_mode"] = int(self.motion_safety == "reject")
        out["action_probe/reject_example_count"] = len(reject_examples)
        record = {
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "motion_safety":self.motion_safety,
            "evidence_scope":"all_world_scalar_rates_plus_up_to_four_rejection_examples",
            "global_step":int(step), "training_phase":phase,
            "denominator":"active UAVs for rates unless otherwise stated",
            "metrics":out, "cbf_examples":examples,
            "reject_examples":reject_examples,
        }
        with self.path.open("a") as file:
            file.write(json.dumps(record,allow_nan=False)+"\n")
        self.last_values = out
        return out
