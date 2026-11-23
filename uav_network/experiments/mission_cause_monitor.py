"""Read-only cross-layer MATD3 mission evidence at a bounded logging cadence.

Separates sensing, confirmation, report custody, geometric path potential,
battery, reward, and map exploration. These are snapshots/conditional proxy
metrics, not proof of a causal mechanism or a replacement for episode records.
No Warp kernels are launched, no model forward/backward or random draws.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import torch


REWARD_NAMES = (
    "discovery", "delivery", "blocked", "expired", "dropped", "step",
    "communication", "propulsion", "communication_efficiency",
    "shield_correction", "information_gain", "investigation",
    "unsafe_action", "shield_intervention",
)
SENSOR_NAMES = (
    "measurements", "positive", "false_positive", "false_negative",
    "false_confirmation", "new_true_confirmation",
)
REPORT_NAMES = ("expired", "dropped", "regenerated")


class MissionCauseMonitor:
    """A sample of the *same* pre-reset Warp vector state as the CBF monitor."""

    def __init__(self, config, output, *, belief_worlds=16):
        self.enabled = config.runtime.environment_backend == "warp_packet"
        self.num_uavs = int(config.world.num_uavs)
        self.num_targets = int(config.world.num_targets)
        self.report_bytes = int(config.network.report_bytes)
        self.buffer_bytes = int(config.network.buffer_bytes)
        self.ttl_s = float(config.network.report_ttl_s)
        self.dt_s = float(config.world.dt_s)
        self.capacity_j = float(config.energy.battery_capacity_j)
        self.peer_range_m = float(config.network.peer_range_m)
        self.gcs_range_m = float(config.network.gcs_range_m)
        self.belief_worlds = int(belief_worlds)
        self.path = Path(output) / "mission-cause-snapshots.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._previous = None
        self.last_values = {}

    @staticmethod
    def _mean(values):
        return values.to(torch.float64).mean() if values.numel() else torch.tensor(0.0, device=values.device)

    @staticmethod
    def _ratio(num, den):
        return num.double() / den.double().clamp_min(1)

    def _deltas(self, steps, seeds, sensor, report, add):
        """Only worlds with same seed and strictly increasing episode steps.

        Sampling can skip resets; deliberately omit windows that crossed a reset.
        """
        previous = self._previous
        metrics = {}
        if previous is not None and previous["steps"].shape == steps.shape:
            eligible = (seeds == previous["seeds"]) & (steps > previous["steps"])
            # Negative counter deltas indicate either reset or invalid state.
            ds = sensor - previous["sensor"]
            dr = report - previous["report"]
            eligible &= (ds >= 0).all(-1) & (dr >= 0).all(-1)
            add("evidence/interval_comparable_world_fraction", eligible.double().mean())
            comparable = int(eligible.sum().item())
            if comparable:
                sensor_counts = ds[eligible].sum(0)
                report_counts = dr[eligible].sum(0)
                for i, name in enumerate(SENSOR_NAMES):
                    add(f"sensing/interval_{name}_count", sensor_counts[i].double())
                for i, name in enumerate(REPORT_NAMES):
                    add(f"report/interval_{name}_count", report_counts[i].double())
                denom = sensor_counts[0].double()
                if float(denom.item()) > 0:
                    add("sensing/interval_false_confirmation_per_1000_measurements",
                        1000.0 * sensor_counts[4] / denom)
                    add("sensing/interval_false_positive_fraction_of_measurements",
                        sensor_counts[2] / denom)
                    add("sensing/interval_false_negative_fraction_of_measurements",
                        sensor_counts[3] / denom)
                else:
                    metrics["sensing/interval_no_measurements"] = 1.0
            else:
                metrics["evidence/no_comparable_interval_worlds"] = 1.0
        self._previous = {
            "steps": steps.clone(), "seeds": seeds.clone(),
            "sensor": sensor.clone(), "report": report.clone(),
        }
        return metrics

    @torch.no_grad()
    def sample(self, vector, step, *, phase="policy", episode_returns=None):
        if not self.enabled or not hasattr(vector, "s"):
            return {}
        import warp as wp

        state = vector.s
        tensor = lambda name: wp.to_torch(getattr(state, name)).detach()
        steps = tensor("steps")
        seeds = tensor("seeds")
        known = tensor("known") > 0
        delivered = tensor("delivered") > 0
        present = tensor("report_present") > 0
        ages = tensor("report_created")
        hops = tensor("report_hops")
        pkt_status = tensor("pkt_status")
        pkt_target = tensor("pkt_target")
        pkt_attempt = tensor("pkt_attempt")
        session_active = tensor("session_active")
        battery = tensor("battery")
        positions = tensor("positions")
        altitudes = tensor("altitudes")
        gcs = tensor("gcs")
        sensor_counts = tensor("sensor_counts")
        report_counts = tensor("report_counts")
        reward = tensor("reward")
        reward_parts = tensor("reward_components")
        results = {}

        def add(name, value):
            results["mission_probe/" + name] = value

        envs, n, targets = known.shape
        if getattr(vector.p, 'cell_reports', 0):
            from .nooracle_telemetry import true_cell_progress
            stats = true_cell_progress(state, vector.p.grid_n, float(vector.p.cell_width))
            found = stats['found']
            received = stats['delivered']
            if found.shape != (envs,self.num_targets) or n != self.num_uavs:
                raise ValueError('Truth-rated cell report metrics have wrong shape')
            add('target_discovery_fraction', found.double().mean())
            add('target_gcs_delivery_fraction', received.double().mean())
            add('false_confirmed_cells_per_world',stats['false_confirmed_cells'].double().mean())
            add('false_gcs_reports_per_world',stats['false_gcs_reports'].double().mean())
            add('sensor_confirmed_hypotheses_per_world',stats['hypothesis_count'].double().mean())
            add('mean_battery_fraction',(battery.double()/max(1,self.capacity_j)).mean())
            add('mean_pending_reports_per_uav',present.double().sum(-1).mean())
            add('mean_active_uavs_per_world',(battery>0).double().sum(-1).mean())
            # Ground-truth is restricted to telemetry; none of these statistics
            # enters policy observation or the dynamic reporting pipeline.
            output={key:float(value.detach().cpu()) for key,value in results.items()}
            record={'global_step':int(step),'phase':str(phase),
                    'population_worlds':int(envs),'evidence_scope':'truth_scored_telemetry_only',
                    'metrics':output}
            with self.path.open('a') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            self.last_values=output
            return output
        if n != self.num_uavs or targets != self.num_targets:
            raise ValueError("Mission probe WorldState shape differs from config")
        found = known.any(1)
        pending = found & ~delivered
        report_any = present.any(1)
        report_held_pending = pending & report_any
        pending_count = pending.sum(1)
        stored_count = present.sum(-1)
        occupancy = stored_count.double() * self.report_bytes / max(1, self.buffer_bytes)
        alive = battery > 0
        battery_fraction = battery.double() / max(self.capacity_j, 1)
        add("worlds", torch.tensor(float(envs), device=battery.device))
        add("found_target_fraction", found.double().mean())
        add("delivered_target_fraction", delivered.double().mean())
        add("found_undelivered_target_fraction", pending.double().mean())
        add("found_target_observation_count", found.double().sum())
        add("delivered_target_observation_count", delivered.double().sum())
        add("delivery_given_found_fraction", self._ratio(delivered.sum(), found.sum()))
        add("pending_target_count_mean_per_world", pending_count.double().mean())
        add("pending_target_observation_count", pending.double().sum())
        add("pending_with_report_in_fifo_fraction", self._ratio(report_held_pending.sum(), pending.sum()))
        add("pending_not_in_fifo_fraction", self._ratio((pending & ~report_any).sum(), pending.sum()))
        # A confirmed target may have packet fragments currently pending even
        # when no whole report is in FIFO. Do NOT call this packet-lost yet.
        pkt_pending = (pkt_status == 1) | (pkt_status == 2)
        pkt_active = pkt_pending & (pkt_target >= 0) & (pkt_target < targets)
        target_packet_counts = torch.zeros((envs, targets), dtype=torch.int32,
                                           device=pkt_target.device)
        target_packet_counts.scatter_add_(
            1, pkt_target.long().clamp(0, targets - 1), pkt_active.to(torch.int32))
        pending_packet = target_packet_counts > 0
        add("active_packet_count_mean_per_world", pkt_pending.double().sum(-1).mean())
        add("queued_packet_count_mean_per_world", (pkt_status == 1).double().sum(-1).mean())
        add("transmitting_packet_count_mean_per_world", (pkt_status == 2).double().sum(-1).mean())
        add("active_session_count_mean_per_world", (session_active > 0).double().sum(-1).mean())
        add("active_packet_retry_mean", self._ratio(
            (pkt_attempt.double() * pkt_active).sum(), pkt_active.sum()))
        add("pending_with_active_packet_fraction", self._ratio(
            (pending & pending_packet).sum(), pending.sum()))
        add("pending_without_fifo_or_active_packet_fraction", self._ratio(
            (pending & ~report_any & ~pending_packet).sum(), pending.sum()))
        add("pending_with_fifo_and_packet_fraction", self._ratio(
            (pending & report_any & pending_packet).sum(), pending.sum()))
        add("report_count_mean_per_uav", stored_count.double().mean())
        add("buffer_occupancy_fraction_mean", occupancy.mean())
        add("buffer_full_fraction", (occupancy >= 1.0 - 1e-9).double().mean())
        add("battery_fraction_mean", battery_fraction.mean())
        add("battery_fraction_min", battery_fraction.min())
        add("battery_below_10pct_fraction", (battery_fraction < .1).double().mean())
        add("battery_below_20pct_fraction", (battery_fraction < .2).double().mean())
        add("inactive_uav_fraction", (~alive).double().mean())
        add("reports_on_inactive_uavs_fraction", self._ratio(
            (present & ~alive[:, :, None]).sum(), present.sum()))
        add("mission_success_world_fraction", delivered.all(-1).double().mean())

        valid_report = present & (~delivered[:, None, :])
        report_age = (steps[:, None, None].double() * self.dt_s - ages.double()).clamp_min(0)
        report_number = valid_report.sum()
        add("pending_report_observation_count", report_number.double())
        add("pending_report_age_mean_s", self._ratio((report_age * valid_report).sum(), report_number))
        add("pending_report_ttl_gt_75pct_fraction",
            self._ratio((valid_report & (report_age >= self.ttl_s * .75)).sum(), report_number))
        add("pending_report_hops_mean", self._ratio((hops.double() * valid_report).sum(), report_number))

        # Optimistic geometry only: distance bounds, no occlusion, MAC, SNR,
        # report admission/queue or power budget. It is an upper bound on
        # geometric routing feasibility, *not* proof of link delivery.
        pos = torch.cat((positions.double(), altitudes.double().unsqueeze(-1)), -1)
        gcs3 = torch.cat((gcs.double(), torch.zeros((envs, 1), dtype=torch.double,
                                                     device=gcs.device)), -1)
        dist_to_gcs = torch.linalg.vector_norm(pos - gcs3[:, None], dim=-1)
        direct = (dist_to_gcs <= self.gcs_range_m) & alive
        peer_d = torch.linalg.vector_norm(pos[:, :, None, :] - pos[:, None, :, :], dim=-1)
        edges = (peer_d <= self.peer_range_m) & alive[:, :, None] & alive[:, None, :]
        reachable = direct
        for _ in range(self.num_uavs):
            reachable = reachable | (edges & reachable[:, None, :]).any(-1)
        report_reachable = (present & reachable[:, :, None]).any(1)
        add("active_uav_direct_gcs_range_fraction", self._ratio(direct.sum(), alive.sum()))
        add("world_with_direct_gcs_candidate_fraction", direct.any(-1).double().mean())
        add("active_uav_optimistic_gcs_reachable_fraction", self._ratio(reachable.sum(), alive.sum()))
        add("pending_with_report_and_optimistic_gcs_path_fraction",
            self._ratio((pending & report_reachable).sum(), pending.sum()))
        add("pending_with_report_but_no_optimistic_path_fraction",
            self._ratio((pending & report_any & ~report_reachable).sum(), pending.sum()))
        add("pending_with_report_and_optimistic_path_count_mean_per_world",
            (pending & report_reachable).sum(-1).double().mean())

        reward_parts = reward_parts.double()
        part_sum = reward_parts.sum(-1)
        # GPU reward_components is accumulated over the WHOLE episode; the
        # s.reward array holds the CURRENT vector step. Comparing these
        # quantities produces a false alarm from the second step onward.
        # Trainer-provided episode_returns is updated AFTER this step and
        # BEFORE any world resets. Use the same episode horizon for both.
        if episode_returns is not None:
            cumulative = torch.as_tensor(episode_returns, dtype=torch.float64,
                                         device=reward.device)
            if cumulative.shape != part_sum.shape:
                raise ValueError("episode_returns shape differs from GPU worlds")
            comparable = torch.ones_like(steps, dtype=torch.bool)
        else:
            # Direct simulator callers may not have the collector return.
            # A first step has identical cumulative and step reward; later
            # steps have unknown cumulative return and MUST NOT be treated
            # as zero discrepancy.
            cumulative = reward.double()
            comparable = steps == 1
        add("reward_accounting_comparable_world_fraction", comparable.double().mean())
        if bool(comparable.any()):
            add("reward_accounting_abs_discrepancy_mean",
                (part_sum[comparable] - cumulative[comparable]).abs().mean())
            add("reward_accounting_abs_discrepancy_max",
                (part_sum[comparable] - cumulative[comparable]).abs().max())
        add("reward_mean", reward.double().mean())
        add("reward_episode_components_cumulative_mean", part_sum.mean())
        abs_total = reward_parts.abs().sum(-1).clamp_min(1e-12)
        for i, name in enumerate(REWARD_NAMES):
            component = reward_parts[..., i]
            add("reward_component_" + name + "_mean", component.mean())
            # Old metric retained for dashboard compatibility. The values
            # are *cumulative* since the beginning of the current episode.
        shaping = reward_parts[..., 10:12].abs().sum(-1)
        safety = reward_parts[..., [2, 9, 12, 13]].abs().sum(-1)
        add("reward_shaping_share_of_abs_components_mean", (shaping / abs_total).mean())
        add("reward_safety_share_of_abs_components_mean", (safety / abs_total).mean())
        # This is an observation at the end of a collection step; many rewards
        # (notably discovery/delivery) are expected to be sparse.

        # Bounded deterministic spatial sample. Never claim this small group
        # represents the complete training population at all time steps.
        sample_n = min(envs, self.belief_worlds)
        if sample_n:
            visited = tensor("visited")[:sample_n] > 0
            belief = tensor("belief")[:sample_n].double()
            visited_any = visited.any(dim=1)
            add("belief_sample_world_count", torch.tensor(float(sample_n), device=battery.device))
            add("visited_sample_union_fraction", visited_any.double().mean())
            add("visited_sample_per_uav_fraction", visited.double().mean())
            add("visited_sample_uav_overlap_fraction",
                (visited.sum(dim=1) >= 2).double().mean())
            add("belief_sample_extreme_fraction",
                ((belief < .01) | (belief > .99)).double().mean())
            add("belief_sample_mean_abs_departure_from_prior_0_5",
                (belief - .5).abs().mean())

        delta_metadata = self._deltas(
            steps, seeds, sensor_counts, report_counts,
            lambda name, value: add(name, value))
        for key, value in delta_metadata.items():
            add(key, value)
        # Bring all device scalars across the boundary in one batch to avoid
        # dozens of tiny GPU synchronization points.
        names = list(results)
        values = torch.stack([torch.as_tensor(results[k], dtype=torch.float64, device=battery.device)
                              for k in names]).cpu().tolist()
        output = dict(zip(names, values))
        document = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "global_step": int(step), "phase": str(phase),
                    "population_worlds": int(envs),
                    "belief_sample_worlds": int(sample_n),
                    "reward_accounting_scope": "episode_cumulative_vs_episode_returns" if episode_returns is not None else "first_step_only",
                    "interpretation": "Observational snapshot; geometric reachability is optimistic and cannot prove PHY connectivity.",
                    "metrics": output}
        with self.path.open("a") as stream:
            stream.write(json.dumps(document, allow_nan=False) + "\n")
        self.last_values = output
        return output
