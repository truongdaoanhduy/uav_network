"""Pool completed outcomes by raw sums/counts; never average packet or energy ratios."""
from copy import deepcopy
import math


def flatten(values, prefix):
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, (list, tuple)):
            result.update({f"{name}/bin_{i}": v for i, v in enumerate(value)})
        else:
            result[name] = value
    return result


def diagnostic_rates(raw):
    result = {}
    for name, numerator, denominator in (
        ("data_phy_success_fraction", "data_phy_successes", "data_decode_count"),
        ("ack_phy_success_fraction", "ack_phy_successes", "ack_attempts"),
        ("data_retry_fraction", "data_retries", "data_attempts"),
        ("cca_busy_fraction", "cca_busy", "cca_checks"),
        ("action_receiver_eligible_fraction", "action_receiver_eligible", "action_slots"),
        ("action_silent_fraction", "action_silent", "action_slots"),
        ("action_no_opportunity_fraction", "action_no_opportunity", "action_slots"),
        ("action_silent_given_eligible_fraction", "action_eligible_silent", "action_receiver_eligible"),
        ("action_silent_given_progress_peer_fraction", "action_progress_peer_silent", "action_progress_peer_opportunity"),
        ("data_los_success_fraction", "data_los_successes", "data_los_decodes"),
        ("data_nlos_success_fraction", "data_nlos_successes", "data_nlos_decodes"),
        ("data_peer_success_fraction", "data_peer_successes", "data_peer_decodes"),
        ("data_gcs_success_fraction", "data_gcs_successes", "data_gcs_decodes"),
        ("data_below_required_power_fraction", "data_below_required_power_attempts", "data_attempts"),
        ("relay_toward_gcs_fraction", "relay_toward_gcs_transfers", "custody_peer_transfers"),
    ):
        if numerator in raw and denominator in raw:
            n = raw[denominator]
            result[name] = raw[numerator] / n if n else None
            result[f"{name}_count"] = n
    return result


def episode_telemetry(record):
    """Detailed timestamps stay in JSON; W&B receives bounded scalar keys."""
    result = {}
    energy, network = record.get("energy", {}), record.get("network", {})
    for source, name in (
        ("battery_j", "battery_j"), ("battery_fraction", "battery_fraction"),
        ("per_uav_propulsion_energy_j", "propulsion_energy_j"),
        ("per_uav_rf_energy_j", "rf_energy_j"),
        ("per_uav_tx_radio_energy_j", "tx_radio_energy_j"),
        ("per_uav_rx_radio_energy_j", "rx_radio_energy_j"),
        ("per_uav_total_energy_j", "total_energy_j"),
    ):
        for i, value in enumerate(energy.get(source, [])):
            result[f"energy/uav_{i}/{name}"] = value
    raw = network.get("diagnostics")
    result["network/diagnostics/available"] = int(raw is not None)
    if raw is not None:
        result.update(flatten({**raw, **diagnostic_rates(raw)}, "network/diagnostics"))
    reports = record.get("report_diagnostics")
    result["reports/diagnostics_available"] = int(reports is not None)
    if reports is not None:
        for key, value in reports.items():
            if key in {"buffer_peak_bytes", "oldest_report_age_s"}:
                result.update({f"reports/uav_{i}/{key}": v for i, v in enumerate(value)})
            elif not isinstance(value, list):
                result[f"reports/{key}"] = value
        delivered = reports["delivery_count"]
        total = energy.get("propulsion_energy_j", 0.) + network.get("uav_communication_energy_j", 0.)
        result["energy/j_per_gcs_delivery"] = total / delivered if delivered else None
        result["energy/gcs_delivery_count"] = delivered
    return result


class OutcomePool:
    """Fixed-size accumulator, independent of the number or duration of episodes."""
    def __init__(self):
        self.n = 0
        self.sums = {}
        self.raw = {}
        self.network_n = self.diagnostics_n = self.reports_n = 0
        self.delivered = 0
        self.report_stats = {}
        self.uavs = {}

    def _sum(self, key, value):
        self.sums[key] = self.sums.get(key, 0.) + value

    def add(self, records):
        # Seed/worker identify real worlds; sorting removes simultaneous worker order.
        for r in sorted(records, key=lambda r: (r.get("seed", 0), r.get("worker", 0),
                                                r.get("episode", 0), r["return"])):
            self.n += 1
            for source, target in (("return", "return"), ("search_rate_percent", "search_rate_percent"),
                                   ("delivery_rate_percent", "delivery_rate_percent"), ("success", "success")):
                self._sum(target, r[source])
            self._sum("return_squared", r["return"] ** 2)
            energy, network = r.get("energy", {}), r.get("network", {})
            self._sum("energy/total_uav_energy_j", energy.get("propulsion_energy_j", 0.) +
                      network.get("uav_communication_energy_j", 0.))
            for source, target in (("propulsion_energy_j", "propulsion_energy_j"),):
                self._sum(f"energy/{target}", energy.get(source, 0.))
            self._sum("energy/uav_rf_energy_j", network.get("uav_communication_energy_j", 0.))
            self._sum("energy/uav_tx_radio_energy_j", network.get("uav_tx_energy_j", 0.))
            self._sum("energy/uav_rx_radio_energy_j", network.get("uav_rx_energy_j", 0.))
            if network.get("enabled") and "generated_packets" in network:
                self.network_n += 1
                for key in ("generated_packets", "received_packets"):
                    self._sum(f"network/{key}", network[key])
            raw = network.get("diagnostics")
            if raw is not None:
                self.diagnostics_n += 1
                for key, value in raw.items():
                    if isinstance(value, dict):
                        previous = self.raw.setdefault(key, {"count": 0, "sum": 0., "min": None, "max": None})
                        previous["count"] += value["count"]
                        previous["sum"] += value["sum"]
                        if value["count"]:
                            previous["min"] = value["min"] if previous["min"] is None else min(previous["min"], value["min"])
                            previous["max"] = value["max"] if previous["max"] is None else max(previous["max"], value["max"])
                    elif isinstance(value, list):
                        old = self.raw.setdefault(key, [0] * len(value))
                        self.raw[key] = [a + b for a, b in zip(old, value)]
                    elif value is not None:
                        self.raw[key] = self.raw.get(key, 0) + value
            reports = r.get("report_diagnostics")
            if reports is not None:
                self.reports_n += 1
                count = reports["delivery_count"]
                self.delivered += count
                for key in ("delivery_hops_mean", "delivered_report_age_mean_s", "first_discovery_to_delivery_mean_s"):
                    if key in reports and reports[key] is not None:
                        total, n = self.report_stats.get(key, (0., 0))
                        self.report_stats[key] = (total + reports[key] * count, n + count)
                for key in ("discovered_undelivered", "discovered_undelivered_with_custody",
                            "discovered_undelivered_without_custody"):
                    if key in reports:
                        self._sum(f"reports/{key}", reports[key])
            for key, value in episode_telemetry(r).items():
                if key.startswith(("energy/uav_", "reports/uav_")) and value is not None:
                    total, n = self.uavs.get(key, (0., 0))
                    self.uavs[key] = (total + value, n + 1)

    def metrics(self):
        values = {"episodes": self.n}
        for name in ("return", "search_rate_percent", "delivery_rate_percent", "success"):
            values[f"{name}_mean"] = self.sums.get(name, 0.) / self.n if self.n else None
        values["return_std"] = (math.sqrt(max(0., self.sums["return_squared"] / self.n -
                                                values["return_mean"] ** 2)) if self.n else None)
        values.update({key: value for key, value in self.sums.items() if "/" in key})
        values.update({"network/available_episodes": self.network_n,
                       "network/diagnostics/available_episodes": self.diagnostics_n,
                       "reports/available_episodes": self.reports_n,
                       "reports/delivery_count": self.delivered})
        generated = self.sums.get("network/generated_packets", 0)
        values["network/packet_pdr_fraction"] = (self.sums.get("network/received_packets", 0) / generated
                                                 if generated else None)
        values["network/generated_packets"] = generated
        values["network/received_packets"] = self.sums.get("network/received_packets", 0)
        values["energy/j_per_gcs_delivery"] = (self.sums.get("energy/total_uav_energy_j", 0.) / self.delivered
                                                if self.delivered and self.reports_n == self.n else None)
        values["energy/radio_j_per_gcs_delivery"] = (self.sums.get("energy/uav_rf_energy_j", 0.) / self.delivered
                                                      if self.delivered and self.reports_n == self.n else None)
        raw = deepcopy(self.raw)
        for stat in raw.values():
            if isinstance(stat, dict):
                stat["mean"] = stat["sum"] / stat["count"] if stat["count"] else None
        values.update(flatten({**raw, **diagnostic_rates(raw)}, "network/diagnostics"))
        for key in ("delivery_hops_mean", "delivered_report_age_mean_s", "first_discovery_to_delivery_mean_s"):
            total, n = self.report_stats.get(key, (0., 0))
            values[f"reports/{key}"] = total / n if n else None
            values[f"reports/{key}_count"] = n
        for key, (total, n) in self.uavs.items():
            values[f"{key}_mean"] = total / n
            values[f"{key}_count"] = n
        return values
