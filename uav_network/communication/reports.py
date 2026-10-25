"""Report custody, finite buffers, target knowledge and original TTL."""
from dataclasses import dataclass, replace
import numpy as np
from .diagnostics import HOP_BINS, report_diagnostics


@dataclass(frozen=True)
class Report:
    target_id: int
    source_id: int
    created_s: float
    size_bytes: int
    hops: int = 0

    @property
    def key(self):
        return self.target_id, self.source_id, self.created_s


class ReportStore:
    def __init__(self, config):
        self.config = config.network
        self.buffers = [dict() for _ in range(config.world.num_uavs)]
        self.known = np.zeros((config.world.num_uavs, config.world.num_targets), dtype=bool)
        self.recovery_pending = np.zeros_like(self.known)
        self.delivered = np.zeros(config.world.num_targets, dtype=bool)
        self.expired = 0
        self.dropped = 0
        self.regenerated = 0
        self.delivery_times_s = []
        self.delivery_hops = []
        self.first_discovery = np.full(config.world.num_targets, -1.)
        self.first_delivery = np.full(config.world.num_targets, -1.)
        self.buffer_peak = np.zeros(config.world.num_uavs, dtype=np.int64)
        self.delivery_hop_histogram = np.zeros(HOP_BINS, dtype=np.int64)
        self.custody_peer_transfers = self.direct_gcs_reports = self.relayed_gcs_reports = 0

    def bytes_used(self, uav_id):
        return sum(r.size_bytes for r in self.buffers[uav_id].values())

    def discover(self, uav_id, target_id, time_s):
        newly_known = not self.known[uav_id, target_id]
        recover = self.config.rediscovery_recovery and self.recovery_pending[uav_id, target_id]
        if not newly_known and not recover:
            return False
        self.known[uav_id, target_id] = True
        if self.first_discovery[target_id] < 0:
            self.first_discovery[target_id] = time_s
        # A remote GCS receipt is not a local acknowledgment of this sighting.
        if self.bytes_used(uav_id) + self.config.report_bytes <= self.config.buffer_bytes:
            # Called only after a fresh camera observation. Relay never refreshes TTL.
            self.buffers[uav_id][target_id] = Report(target_id, uav_id, time_s,
                                                  self.config.report_bytes)
            self.recovery_pending[uav_id, target_id] = False
            self.buffer_peak[uav_id] = max(self.buffer_peak[uav_id], self.bytes_used(uav_id))
            self.regenerated += int(not newly_known)
        else:
            self.recovery_pending[uav_id, target_id] = True
            # Continuing to observe a full buffer is not another dropped report.
            self.dropped += int(newly_known)
        return bool(newly_known)

    def contains(self, uav_id, report):
        return self.buffers[uav_id].get(report.target_id) == report

    def can_receive(self, destination, report):
        if destination == len(self.buffers):
            return True
        return (not self.known[destination, report.target_id]
                and self.bytes_used(destination) + report.size_bytes <= self.config.buffer_bytes)

    def can_forward(self, source, destination, report, time_s):
        return (self.contains(source, report)
                and time_s - report.created_s < self.config.report_ttl_s
                and self.can_receive(destination, report))

    def transfer(self, source, destination, report, time_s):
        if (not self.contains(source, report)
                or time_s - report.created_s >= self.config.report_ttl_s
                or not self.can_receive(destination, report)):
            return False
        del self.buffers[source][report.target_id]
        if destination == len(self.buffers):
            if not self.delivered[report.target_id]:
                self.delivery_times_s.append(time_s - report.created_s)
                self.delivery_hops.append(report.hops + 1)
                self.delivered[report.target_id] = True
                self.first_delivery[report.target_id] = time_s
                self.delivery_hop_histogram[min(report.hops, HOP_BINS - 1)] += 1
                self.direct_gcs_reports += int(report.hops == 0)
                self.relayed_gcs_reports += int(report.hops > 0)
        else:
            self.known[destination, report.target_id] = True
            self.buffers[destination][report.target_id] = replace(report, hops=report.hops + 1)
            self.custody_peer_transfers += 1
            self.buffer_peak[destination] = max(self.buffer_peak[destination], self.bytes_used(destination))
        return True

    def diagnostics(self, time_s):
        present = np.zeros_like(self.known)
        created = np.zeros(self.known.shape)
        for i, buffer in enumerate(self.buffers):
            for target, report in buffer.items():
                present[i, target], created[i, target] = True, report.created_s
        return report_diagnostics(self.first_discovery, self.first_delivery, present, created,
                                  self.buffer_peak, time_s, sum(self.delivery_times_s), sum(self.delivery_hops))

    def expire(self, time_s):
        for uav_id, buffer in enumerate(self.buffers):
            for target_id, report in list(buffer.items()):
                if time_s - report.created_s >= self.config.report_ttl_s:
                    del buffer[target_id]
                    self.recovery_pending[uav_id, target_id] = True
                    self.expired += 1
