"""Bounded episode telemetry. Capture is passive: no random draws or extra PHY queries."""
import numpy as np

COUNTERS = (
    "data_attempts", "data_retries", "data_phy_successes", "data_phy_failures",
    "ack_attempts", "ack_phy_successes", "ack_phy_failures", "data_decode_count",
    "data_weak_signal_failures", "data_interference_failures",
    "data_overlap_decodes", "data_collision_failures",
    "data_los_decodes", "data_los_successes", "data_nlos_decodes", "data_nlos_successes",
    "data_peer_decodes", "data_peer_successes", "data_gcs_decodes", "data_gcs_successes",
    "queue_full", "packet_lifetime", "retry_limit", "energy_denials",
    "cca_checks", "cca_busy", "backoff_draws", "backoff_freezes",
    "custody_peer_transfers", "direct_gcs_reports", "relayed_gcs_reports",
    "action_slots", "action_active", "action_report_bearing", "action_receiver_eligible",
    "action_silent", "action_no_opportunity", "data_energy_denials", "ack_energy_denials",
    "ack_timeouts", "data_below_required_power_attempts", "relay_toward_gcs_transfers", "relay_away_from_gcs_transfers",
)
STATS = ("data_tx_power_w", "ack_tx_power_w", "data_peer_tx_power_w", "data_gcs_tx_power_w",
         "data_snr_db", "data_sinr_db", "data_rssi_dbm", "data_distance_m",
         "data_required_power_w", "data_power_margin_db", "relay_gcs_distance_improvement_m")
# Six bins: <0.1, [0.1,0.2), [0.2,0.3), [0.3,0.4), [0.4,0.5), >=0.5 W.
POWER_EDGES = (0.1, 0.2, 0.3, 0.4, 0.5)
HOP_BINS = 8  # 1..7, >=8.
COUNT = {name: i for i, name in enumerate(COUNTERS)}
STAT = {name: i for i, name in enumerate(STATS)}


def add_sample(stats, index, value):
    row = stats[index]
    if row[0] == 0:
        row[2] = row[3] = value
    else:
        row[2], row[3] = min(row[2], value), max(row[3], value)
    row[0] += 1
    row[1] += value


def network_diagnostics(counts, stats, power_histogram, hops):
    result = dict(zip(COUNTERS, map(int, counts)))
    for name, (count, total, low, high) in zip(STATS, stats):
        result[name] = dict(count=int(count), sum=float(total),
                            mean=float(total / count) if count else None,
                            min=float(low) if count else None, max=float(high) if count else None)
    result["data_tx_power_histogram"] = list(map(int, power_histogram[0]))
    result["ack_tx_power_histogram"] = list(map(int, power_histogram[1]))
    result["delivery_hop_histogram"] = list(map(int, hops))
    return result


def report_diagnostics(first_discovery, first_delivery, present, created, peaks, now,
                       report_delay_sum=0., report_hop_sum=0.):
    discovered, delivered = first_discovery >= 0, first_delivery >= 0
    pending = discovered & ~delivered
    custody = np.any(present, axis=0)
    delays = first_delivery[delivered] - first_discovery[delivered]
    oldest = [float(np.max(now - created[i, mask])) if mask.any() else None
              for i, mask in enumerate(present)]
    return {
        "first_discovery_s": [float(v) if v >= 0 else None for v in first_discovery],
        "first_delivery_s": [float(v) if v >= 0 else None for v in first_delivery],
        "first_discovery_to_delivery_s": [
            float(end-start) if end >= 0 and start >= 0 else None
            for start, end in zip(first_discovery, first_delivery)],
        "delivery_count": int(delivered.sum()),
        "delivery_hops_mean": float(report_hop_sum / delivered.sum()) if delivered.any() else None,
        "delivered_report_age_mean_s": float(report_delay_sum / delivered.sum()) if delivered.any() else None,
        "first_discovery_to_delivery_mean_s": float(delays.mean()) if len(delays) else None,
        "discovered_undelivered": int(pending.sum()),
        "discovered_undelivered_with_custody": int((pending & custody).sum()),
        "discovered_undelivered_without_custody": int((pending & ~custody).sum()),
        "buffer_peak_bytes": list(map(int, peaks)),
        "oldest_report_age_s": oldest,
    }
