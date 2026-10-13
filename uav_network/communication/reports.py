"""Report custody, finite buffers, target knowledge and original TTL."""
from dataclasses import dataclass, replace
import numpy as np


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
        self.delivered = np.zeros(config.world.num_targets, dtype=bool)
        self.expired = 0
        self.dropped = 0
        self.delivery_times_s = []
        self.delivery_hops = []

    def bytes_used(self, uav_id):
        return sum(r.size_bytes for r in self.buffers[uav_id].values())

    def discover(self, uav_id, target_id, time_s):
        if self.known[uav_id, target_id]:
            return False
        self.known[uav_id, target_id] = True
        if not self.delivered[target_id]:
            if self.bytes_used(uav_id) + self.config.report_bytes <= self.config.buffer_bytes:
                self.buffers[uav_id][target_id] = Report(target_id, uav_id, time_s,
                                                      self.config.report_bytes)
            else:
                self.dropped += 1
        return True

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
        else:
            self.known[destination, report.target_id] = True
            self.buffers[destination][report.target_id] = replace(report, hops=report.hops + 1)
        return True

    def expire(self, time_s):
        for buffer in self.buffers:
            for target_id, report in list(buffer.items()):
                if time_s - report.created_s >= self.config.report_ttl_s:
                    del buffer[target_id]
                    self.expired += 1
