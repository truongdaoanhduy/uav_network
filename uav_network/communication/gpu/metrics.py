"""Host-boundary scientific summaries; copy only the selected world row."""
import math
import numpy as np
import warp as wp
from ..upstream import REVISION
from ..diagnostics import network_diagnostics, report_diagnostics


def row(array, index):
    return wp.to_torch(array)[index].detach().cpu().numpy()


def hypothesis_truth_summary(s,p,index,known=None,delivered=None,present=None,created=None):
    """Read-only host scoring: truth is NEVER used to create sensor reports or policy inputs."""
    if known is None:known=row(s.known,index)
    if delivered is None:delivered=row(s.delivered,index)
    if present is None:present=row(s.report_present,index)
    if created is None:created=row(s.report_created,index)
    truth=row(s.targets,index)
    cell_indices=row(s.report_cell,index)
    g=int(p.grid_n);width=float(p.cell_width)
    target_cells=[min(int(x/width),g-1)+g*min(int(y/width),g-1) for x,y in truth]
    occupied=set(target_cells)
    true_found=0;true_delivered=0
    report_truth_present=np.zeros((int(p.n),int(p.targets_count)),dtype=bool)
    report_truth_created=np.zeros((int(p.n),int(p.targets_count)),dtype=np.float64)
    for t,cell in enumerate(target_cells):
        slots=np.flatnonzero(cell_indices==cell)
        if len(slots):
            true_found+=int(known[:,slots].any())
            true_delivered+=int(delivered[slots].any())
            report_truth_present[:,t]=present[:,slots].any(axis=1)
            for i in range(int(p.n)):
                active=slots[np.flatnonzero(present[i,slots])]
                if len(active):report_truth_created[i,t]=created[i,active[0]]
    invalid=np.array([c not in occupied and c>=0 for c in cell_indices],dtype=bool)
    false_confirmed=int(known[:,invalid].any(axis=0).sum())
    false_gcs=int(delivered[invalid].sum())
    return dict(targets_found=true_found,targets_delivered=true_delivered,
                false_confirmed_cells=false_confirmed,false_reports_delivered=false_gcs,
                total_hypotheses=int((cell_indices>=0).sum()),
                truth_present=report_truth_present,truth_created=report_truth_created)

def network_info(s, p, index=0):
    counts = row(s.packet_counts,index)
    energy = row(s.communication_energy,index)
    tx_energy = row(s.tx_energy,index)
    rx_energy = row(s.rx_energy,index)
    selection = row(s.selection_counts,index)
    delivered = int(row(s.report_delivery_count,index))
    true_delivery= (hypothesis_truth_summary(s,p,index)["targets_delivered"] if p.cell_reports else int(row(s.delivered,index).sum()))
    received = int(counts[1])
    phy = int(counts[2]+counts[3])
    elapsed = float(row(s.packet_time,index))/1e6
    return {
        "diagnostics": network_diagnostics(row(s.diagnostic_counts,index), row(s.diagnostic_stats,index),
                                            row(s.power_histogram,index), row(s.delivery_hop_histogram,index),
                                            tx_power_max_w=float(p.tx_power_max)),
        "enabled": bool(p.network_enabled), "backend": "warp_packet", "upstream_revision": REVISION,
        "selection_counts": dict(zip(("selected","accepted","receiver_rejected",
                                      "out_of_range_or_inactive","no_buffer"),map(int,selection))),
        # HELLO is a distinct RF frame class: do not inflate DATA PDR/packet counts.
        "hello_enabled": bool(p.hello_enabled),
        "hello_tx_frames": int(row(s.hello_tx_count,index)),
        "hello_rx_frames": int(row(s.hello_rx_count,index)),
        "hello_decode_failures": int(row(s.hello_rx_fail_count,index)),
        "hello_success_percent": (100*int(row(s.hello_rx_count,index))/
                                  max(int(row(s.hello_rx_count,index))+
                                      int(row(s.hello_rx_fail_count,index)),1)),
        "generated_packets": int(counts[0]), "received_packets": received,
        "packet_pdr_percent": received/max(int(counts[0]),1)*100,
        "packet_e2e_delay_ms": float(row(s.packet_delay_sum,index))/max(received,1)/1000,
        "phy_successes": int(counts[2]), "phy_failures": int(counts[3]),
        "phy_success_percent": int(counts[2])/max(phy,1)*100, "collisions": int(counts[4]),
        "communication_energy_j": float(energy.sum()),
        "uav_communication_energy_j": float(energy[:-1].sum()),
        "gcs_communication_energy_j": float(energy[-1]),
        "uav_tx_energy_j": float(tx_energy[:-1].sum()),
        "uav_rx_energy_j": float(rx_energy[:-1].sum()),
        "gcs_tx_energy_j": float(tx_energy[-1]),
        "gcs_rx_energy_j": float(rx_energy[-1]),
        "energy_blocked_transmissions": int(counts[5]),
        "obstacle_extra_loss_db": -10*math.log10(p.nlos_scale)-p.nlos_excess_loss_db,
        "payload_throughput_kbps": int(counts[6])*8/max(elapsed,1e-12)/1000,
        "targets_delivered": true_delivery,
        "raw_reports_delivered": delivered,
        "true_report_e2e_delay_s": (
            float(row(s.true_report_delay_sum,index))/true_delivery
            if p.cell_reports and true_delivery else None),
        "true_report_average_hops": (
            float(row(s.true_report_hop_sum,index))/true_delivery
            if p.cell_reports and true_delivery else None),
        # This raw radio metric includes false reports for network evaluation.
        "report_e2e_delay_s": float(row(s.report_delay_sum,index))/max(delivered,1),
        "report_average_hops": float(row(s.report_hop_sum,index))/max(delivered,1)}


def world_info(s, p, index=0):
    known = row(s.known,index)
    delivered = row(s.delivered,index)
    present = row(s.report_present,index)
    battery = row(s.battery,index)
    found = int(np.any(known,axis=0).sum())
    count = int(delivered.sum())
    hypothesis=None
    if p.cell_reports:
        hypothesis=hypothesis_truth_summary(s,p,index,known=known,delivered=delivered,present=present)
        found=hypothesis['targets_found']
        count=hypothesis['targets_delivered']
    steps = int(row(s.steps,index))
    active = battery > 0 if p.energy_enabled else np.ones(p.n,dtype=bool)
    report_counts = row(s.report_counts,index)
    motion = row(s.motion_counts,index)
    reward = row(s.reward_components,index)
    actions = row(s.action_counts,index)
    velocity = row(s.velocities,index)
    extra = {}
    if p.xyz:
        velocity = np.column_stack((velocity,row(s.vertical_velocity,index)))
        extra = {"sensing": dict(zip(("measurements","positive","false_positive","false_negative","false_confirmation","confirmations"),map(int,row(s.sensor_counts,index)))),
                 "coverage_percent": 100*float(row(s.visited,index).any(axis=0).mean()),
                 "coverage_auc": float(row(s.coverage_auc_weight_sum,index)) /
                    (float(p.max_steps) * float(p.grid_n * p.grid_n)),
                 "altitudes_m": row(s.altitudes,index).tolist()}
    if hypothesis is not None:
        extra.update(false_confirmed_cells=hypothesis['false_confirmed_cells'],
                     false_reports_delivered=hypothesis['false_reports_delivered'],
                     total_hypotheses=hypothesis['total_hypotheses'],
                     report_identity='observed_cell_id',
                     confirmation_source=('posterior_plus_likelihood_ratio_no_truth_filter'
                                          if p.confirmation_mode==1 else 'posterior_only_no_truth_filter'))
    return {
        **extra,
        "generation_fallback": bool(row(s.generation_fallback,index)),
        "generation_attempts": int(row(s.generation_attempts,index)),
        "report_diagnostics": report_diagnostics(row(s.first_discovery,index),row(s.first_delivery,index),
                                                  (hypothesis['truth_present'] if hypothesis is not None else present.astype(bool)),
                                                  (hypothesis['truth_created'] if hypothesis is not None else row(s.report_created,index)),
                                                  row(s.buffer_peak,index),steps*p.dt,
                                                  float(row(s.true_report_delay_sum,index) if p.cell_reports else row(s.report_delay_sum,index)),
                                                  float(row(s.true_report_hop_sum,index) if p.cell_reports else row(s.report_hop_sum,index))),
        "step": steps, "targets_found": found, "targets_delivered": count,
        "search_rate_percent": 100*found/p.targets_count,
        "delivery_rate_percent": 100*count/p.targets_count, "success": count == p.targets_count,
        "termination_reason": ("success" if count == p.targets_count else
                               "battery_exhausted" if not active.any() else
                               "time_limit" if steps >= p.max_steps else None),
        "buffer_reports": int(present.sum()), "buffer_bytes": int(present.sum())*p.report_bytes,
        "mean_speed_mps": float(np.linalg.norm(velocity,axis=1).mean()),
        "blocked_moves": int(row(s.blocked,index)), "boundary_clips": int(row(s.clipped,index)),
        "expired_reports": int(report_counts[0]), "dropped_reports": int(report_counts[1]),
        "regenerated_reports": int(report_counts[2]),
        "motion_counts": {**dict(zip(("obstacle_blocks","peer_blocks","commanded_stalls",
                                   "shield_interventions","shield_obstacle_interventions",
                                   "shield_peer_interventions","shield_boundary_interventions",
                                   "shield_infeasible","raw_unsafe_attempts","raw_boundary_attempts",
                                   "raw_obstacle_attempts","raw_peer_attempts","shield_rescues"),map(int,motion[:13]))),
                          "shield_correction": float(row(s.shield_correction,index))},
        "reward_components": dict(zip(("discovery","delivery","blocked_motion","expired_report",
                                       "dropped_report","step_cost","communication_energy",
                                       "propulsion_energy","communication_efficiency","shield_correction",
                                       "information_gain","investigation_shaping","unsafe_action",
                                       "shield_intervention"),
                                      map(float,reward))),
        "action_counts": dict(zip(("silent","peer","gcs","saturated"),map(int,actions))),
        "energy": {"per_uav_propulsion_energy_j": row(s.flight_energy,index).tolist(),
                   "per_uav_rf_energy_j": row(s.communication_energy,index)[:-1].tolist(),
                   "per_uav_tx_radio_energy_j": row(s.tx_energy,index)[:-1].tolist(),
                   "per_uav_rx_radio_energy_j": row(s.rx_energy,index)[:-1].tolist(),
                   "per_uav_total_energy_j": (row(s.flight_energy,index)+row(s.communication_energy,index)[:-1]).tolist(),
                   "battery_j": battery.tolist(), "battery_fraction": (battery/p.capacity).tolist(),
                   "active_uavs": int(active.sum()),
                   "propulsion_energy_j": float(row(s.flight_energy,index).sum())},
        "network": network_info(s,p,index) if p.network_enabled else {"enabled": False}}
