"""Host-boundary scientific summaries; copy only the selected world row."""
import math
import numpy as np
import warp as wp
from ..upstream import REVISION


def row(array, index):
    return wp.to_torch(array)[index].detach().cpu().numpy()


def network_info(s, p, index=0):
    counts = row(s.packet_counts,index)
    energy = row(s.communication_energy,index)
    selection = row(s.selection_counts,index)
    delivered = int(row(s.report_delivery_count,index))
    received = int(counts[1])
    phy = int(counts[2]+counts[3])
    elapsed = float(row(s.packet_time,index))/1e6
    return {
        "enabled": bool(p.network_enabled), "backend": "warp_packet", "upstream_revision": REVISION,
        "selection_counts": dict(zip(("selected","accepted","receiver_rejected",
                                      "out_of_range_or_inactive","no_buffer"),map(int,selection))),
        "generated_packets": int(counts[0]), "received_packets": received,
        "packet_pdr_percent": received/max(int(counts[0]),1)*100,
        "packet_e2e_delay_ms": float(row(s.packet_delay_sum,index))/max(received,1)/1000,
        "phy_successes": int(counts[2]), "phy_failures": int(counts[3]),
        "phy_success_percent": int(counts[2])/max(phy,1)*100, "collisions": int(counts[4]),
        "communication_energy_j": float(energy.sum()),
        "uav_communication_energy_j": float(energy[:-1].sum()),
        "gcs_communication_energy_j": float(energy[-1]),
        "energy_blocked_transmissions": int(counts[5]),
        "obstacle_extra_loss_db": -10*math.log10(p.nlos_scale)-p.nlos_excess_loss_db,
        "payload_throughput_kbps": int(counts[6])*8/max(elapsed,1e-12)/1000,
        "targets_delivered": int(row(s.delivered,index).sum()),
        "report_e2e_delay_s": float(row(s.report_delay_sum,index))/max(delivered,1),
        "report_average_hops": float(row(s.report_hop_sum,index))/max(delivered,1)}


def world_info(s, p, index=0):
    known = row(s.known,index)
    delivered = row(s.delivered,index)
    present = row(s.report_present,index)
    battery = row(s.battery,index)
    found = int(np.any(known,axis=0).sum())
    count = int(delivered.sum())
    steps = int(row(s.steps,index))
    active = battery > 0 if p.energy_enabled else np.ones(p.n,dtype=bool)
    report_counts = row(s.report_counts,index)
    motion = row(s.motion_counts,index)
    reward = row(s.reward_components,index)
    actions = row(s.action_counts,index)
    return {
        "step": steps, "targets_found": found, "targets_delivered": count,
        "search_rate_percent": 100*found/p.targets_count,
        "delivery_rate_percent": 100*count/p.targets_count, "success": count == p.targets_count,
        "termination_reason": ("success" if count == p.targets_count else
                               "battery_exhausted" if not active.any() else
                               "time_limit" if steps >= p.max_steps else None),
        "buffer_reports": int(present.sum()), "buffer_bytes": int(present.sum())*p.report_bytes,
        "mean_speed_mps": float(np.linalg.norm(row(s.velocities,index),axis=1).mean()),
        "blocked_moves": int(row(s.blocked,index)), "boundary_clips": int(row(s.clipped,index)),
        "expired_reports": int(report_counts[0]), "dropped_reports": int(report_counts[1]),
        "regenerated_reports": int(report_counts[2]),
        "motion_counts": dict(zip(("obstacle_blocks","peer_blocks","commanded_stalls",
                                   "shield_interventions","shield_obstacle_interventions",
                                   "shield_peer_interventions","shield_boundary_interventions",
                                   "shield_infeasible"),map(int,motion[:8]))),
        "reward_components": dict(zip(("discovery","delivery","blocked_motion","expired_report",
                                       "dropped_report","step_cost","communication_energy",
                                       "propulsion_energy"),map(float,reward))),
        "action_counts": dict(zip(("silent","peer","gcs","saturated"),map(int,actions))),
        "energy": {"battery_j": battery.tolist(), "battery_fraction": (battery/p.capacity).tolist(),
                   "active_uavs": int(active.sum()),
                   "propulsion_energy_j": float(row(s.flight_energy,index).sum())},
        "network": network_info(s,p,index) if p.network_enabled else {"enabled": False}}
