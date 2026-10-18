"""Real packet transmission with fragment reassembly and report custody."""
from dataclasses import dataclass, field
import numpy as np

from .upstream import REVISION, start_simulator
from .energy import install_obstacle_loss, install_phy_budget
from .diagnostics import COUNTERS, STATS, COUNT, STAT, add_sample, network_diagnostics
from .telemetry import install_telemetry
from .link_budget import power_from_action, nominal_required_power_w

_ACTIVE = None  # Upstream configuration and packet IDs are process globals.


@dataclass
class Transfer:
    source: int
    destination: int
    report: object
    window_start: int = 0  # fragment index, not byte offset
    arrived_mask: int = 0
    pending: dict = field(default_factory=dict)  # byte offset -> (packet ID, expiry seconds)
    power_w: float = 0.0


class UavNetSimNetwork:
    def __init__(self, config, reports, scene, seed, battery_j=None):
        global _ACTIVE
        if _ACTIVE is not None:
            raise RuntimeError("UavNetSim supports only one live network instance per process")
        self.config, self.reports, self.scene = config, reports, scene
        self.sessions = {}
        self.communication_events = None  # Opt-in single-mission visualization only.
        self.energy_j = np.zeros(config.world.num_uavs + 1)
        self.tx_energy_j = np.zeros(config.world.num_uavs + 1)
        self.rx_energy_j = np.zeros(config.world.num_uavs + 1)
        self.battery_j = (battery_j if battery_j is not None else
                          np.full(config.world.num_uavs, config.energy.battery_capacity_j, dtype=float))
        self.selection_counts = {"selected": 0, "accepted": 0, "receiver_rejected": 0,
                                 "out_of_range_or_inactive": 0, "no_buffer": 0}
        self.energy_blocked_transmissions = 0
        self.diagnostic_counts = np.zeros(len(COUNTERS), dtype=np.int64)
        self.diagnostic_stats = np.zeros((len(STATS), 4))
        self.power_histogram = np.zeros((2, 6), dtype=np.int64)
        self.received_payload_bytes = 0
        self._closed = False
        self.step_end_us = 0.0
        self.coalesce_empty_step = False
        _ACTIVE = self
        try:
            (self.simulator, self.clock, self.native, self.DataPacket,
             self.directory, self.previous) = start_simulator(self, scene, seed)
            install_phy_budget(self)
            install_obstacle_loss(self)
            install_telemetry(self)
            self.sync_positions(scene.positions)
            self._event_sequence = 0
        except Exception:
            _ACTIVE = None
            self._closed = True
            raise

    def sync_positions(self, positions):
        for i, drone in enumerate(self.simulator.drones):
            xy = positions[i] if i < len(positions) else self.scene.gcs
            z = self.config.sensor.altitude_m if i < len(positions) else 0.0
            # Mission geometry already validated; the native coords setter tests
            # a previous random spawn-to-current path, not this mission movement.
            drone._coords = [float(xy[0]), float(xy[1]), z]
            drone.velocity = [0.0, 0.0, 0.0]
            drone.sleep = not self.node_active(i)
            if i < len(positions) and self.config.energy.enabled:
                drone.residual_energy = float(self.battery_j[i])

    def node_active(self, node):
        return (node == self.config.world.num_uavs or not self.config.energy.enabled
                or self.battery_j[node] > 0)

    def _destination(self, source, index):
        n = self.config.world.num_uavs
        if index == 0:
            return None
        if index == n:
            return n
        peers = [i for i in range(n) if i != source]
        return peers[index - 1]

    def _eligible(self, source, destination, positions, report):
        n = self.config.network
        peer = destination < len(positions)
        xy = positions[destination] if peer else self.scene.gcs
        reach = n.peer_range_m if peer else n.gcs_range_m
        feasible = True
        if n.power_limited_admission:
            feasible = nominal_required_power_w(positions[source], xy, self.scene.obstacles, n,
                                                self.config.sensor.altitude_m, not peer) <= n.tx_power_max_w
        return (self.node_active(source) and self.node_active(destination)
                and np.linalg.norm(positions[source] - xy) <= reach and feasible
                and self.reports.can_forward(source, destination, report, self.clock.now / 1e6))

    def _fragment_count(self, report):
        payload = self.config.network.payload_bytes
        return (report.size_bytes + payload - 1) // payload

    def _window_count(self, transfer):
        remaining = self._fragment_count(transfer.report) - transfer.window_start
        return min(self.config.network.max_queue_packets, remaining)

    def _window_mask(self, transfer):
        count = self._window_count(transfer)
        return (1 << count) - 1 if count else 0

    def _inject(self, source, destination, power, report):
        n, now = self.config.network, self.clock.now / 1e6
        key = (source, report.key, destination)
        transfer = self.sessions.setdefault(key, Transfer(source, destination, report))
        transfer.power_w = float(power)
        sender = self.simulator.drones[source]
        count = self._window_count(transfer)
        for relative in range(count):
            bit = 1 << relative
            fragment = transfer.window_start + relative
            offset = fragment * n.payload_bytes
            if transfer.arrived_mask & bit:
                continue
            pending = transfer.pending.get(offset)
            if pending and pending[1] > now:
                continue
            if sender.transmitting_queue.qsize() >= n.max_queue_packets:
                break
            remaining = report.created_s + n.report_ttl_s - now
            if remaining <= 0:
                break
            self.native.GL_ID_DATA_PACKET += 1
            payload = min(n.payload_bytes, report.size_bytes - offset)
            header = self.native.IP_HEADER_LENGTH + self.native.MAC_HEADER_LENGTH + self.native.PHY_HEADER_LENGTH
            packet = self.DataPacket(src_drone=sender,
                                     dst_drone=self.simulator.drones[destination],
                                     creation_time=self.clock.now,
                                     data_packet_id=self.native.GL_ID_DATA_PACKET,
                                     data_packet_length=payload * 8 + header,
                                     simulator=self.simulator, channel_id=1)
            lifetime = min(n.packet_lifetime_s, remaining)
            packet.deadline = lifetime * 1e6
            packet.transmission_mode = 0
            packet.waiting_start_time = self.clock.now
            packet.mission_power_w = transfer.power_w
            packet.mission_session = key
            packet.mission_offset = offset
            packet.mission_payload_bytes = payload
            sender.transmitting_queue.put(packet)
            self.simulator.metrics.record_generated(packet)
            transfer.pending[offset] = (packet.packet_id, now + lifetime)

    def packet_received(self, packet, receiver):
        key = getattr(packet, "mission_session", None)
        transfer = self.sessions.get(key)
        if transfer is None or receiver != transfer.destination or not self.node_active(receiver):
            return
        now = self.clock.now / 1e6
        if (not self.reports.contains(transfer.source, transfer.report)
                or now - transfer.report.created_s >= self.config.network.report_ttl_s):
            return
        payload = self.config.network.payload_bytes
        offset = packet.mission_offset
        fragment = offset // payload
        relative = fragment - transfer.window_start
        if relative < 0 or relative >= self._window_count(transfer):
            return
        bit = 1 << relative
        if not (transfer.arrived_mask & bit):
            self.received_payload_bytes += packet.mission_payload_bytes
            transfer.arrived_mask |= bit
        transfer.pending.pop(offset, None)
        if transfer.arrived_mask == self._window_mask(transfer):
            next_start = transfer.window_start + self._window_count(transfer)
            if next_start >= self._fragment_count(transfer.report):
                if self.reports.transfer(transfer.source, receiver, transfer.report, now):
                    if receiver < self.config.world.num_uavs:
                        source_xy = np.asarray(self.simulator.drones[transfer.source].coords[:2], dtype=float)
                        destination_xy = np.asarray(self.simulator.drones[receiver].coords[:2], dtype=float)
                        progress = float(np.linalg.norm(source_xy - self.scene.gcs) - np.linalg.norm(destination_xy - self.scene.gcs))
                        counter = "relay_toward_gcs_transfers" if progress > 0 else "relay_away_from_gcs_transfers"
                        self.diagnostic_counts[COUNT[counter]] += 1
                        add_sample(self.diagnostic_stats, STAT["relay_gcs_distance_improvement_m"], progress)
                    if self.communication_events is not None:
                        self.communication_events.append({
                            "event": "report_received", "time_s": float(now),
                            "source": int(transfer.source), "destination": int(receiver),
                            "target": int(transfer.report.target_id), "origin": int(transfer.report.source_id),
                            "created_s": float(transfer.report.created_s), "hops": int(transfer.report.hops + 1),
                            "power_w": float(packet.mission_power_w), "power_scope": "last_fragment",
                            "source_xy": [float(v) for v in self.simulator.drones[transfer.source].coords[:2]],
                            "destination_xy": [float(v) for v in self.simulator.drones[receiver].coords[:2]]})
                    self._discard_report_sessions(transfer.source, transfer.report)
                else:
                    # The receiver may have filled after admission. Drop this
                    # reassembly; source custody stays valid for later retransmission.
                    del self.sessions[key]
                return
            transfer.window_start = next_start
            transfer.arrived_mask = 0
        # Refill the bounded window immediately. Failed fragments remain in
        # pending until _collect_drops at the policy-interval boundary, so a
        # retry-limit failure still needs a later policy action (possibly with
        # a new TX power) rather than silently retrying forever.
        self._inject(transfer.source, transfer.destination, transfer.power_w, transfer.report)

    def _discard_report_sessions(self, source, report):
        for key in list(self.sessions):
            if key[:2] == (source, report.key):
                del self.sessions[key]

    def _collect_drops(self):
        events = self.simulator.event_bus.since(self._event_sequence)
        failed = {e["data"].get("packet_id") for e in events if e["event_type"] == "packet_dropped"}
        self._event_sequence = self.simulator.event_bus.sequence
        for key, transfer in list(self.sessions.items()):
            if not self.reports.contains(transfer.source, transfer.report):
                del self.sessions[key]
                continue
            mac = self.simulator.drones[transfer.source].mac_protocol
            for offset, (packet_id, expiry) in list(transfer.pending.items()):
                ack_key = f"wait_ack{transfer.source}_{packet_id}"
                # Upstream terminal ACK timeouts do not publish packet_dropped.
                if (packet_id in failed or expiry <= self.clock.now / 1e6
                        or mac.wait_ack_process_finish.get(ack_key, 0) == 1):
                    del transfer.pending[offset]

    def step(self, action, positions, time_s):
        action.validate(self.config.world.num_uavs)
        if abs(time_s - self.clock.now / 1e6 - self.config.world.dt_s) > 1e-8:
            raise ValueError("Mission and packet simulator clocks must advance by exactly dt_s")
        self.sync_positions(positions)
        before = self.energy_j[:-1].sum()
        self._collect_drops()
        # Post-motion/post-sensing at interval start; independent of the selected destination.
        for source in range(self.config.world.num_uavs):
            active = self.node_active(source)
            bearing = active and bool(self.reports.buffers[source])
            n = self.config.world.num_uavs
            receivers = {destination for destination in range(n + 1) if destination != source
                         and bearing and any(self._eligible(source, destination, positions, report)
                             for report in self.reports.buffers[source].values())}
            eligible = bool(receivers)
            # A closer eligible peer is a local opportunity, not an end-to-end route guarantee.
            progress_peer = n not in receivers and any(
                np.linalg.norm(positions[destination] - self.scene.gcs)
                < np.linalg.norm(positions[source] - self.scene.gcs) - 1e-9
                for destination in receivers if destination < n)
            silent = action.destination[source] == 0
            for name, value in (("action_slots", 1), ("action_active", active),
                                ("action_report_bearing", bearing), ("action_receiver_eligible", eligible),
                                ("action_silent", silent), ("action_no_opportunity", not eligible),
                                ("action_eligible_silent", eligible and silent),
                                ("action_progress_peer_opportunity", progress_peer),
                                ("action_progress_peer_silent", progress_peer and silent)):
                self.diagnostic_counts[COUNT[name]] += int(value)
        for source, index in enumerate(action.destination):
            destination = self._destination(source, int(index))
            buffer = self.reports.buffers[source]
            if destination is None:
                continue
            self.selection_counts["selected"] += 1
            if not buffer:
                self.selection_counts["no_buffer"] += 1
                continue
            report = next((r for r in buffer.values()
                           if self._eligible(source, destination, positions, r)), None)
            if report is not None:
                power = power_from_action(float(action.continuous[source, 2]), self.config.network)
                self.selection_counts["accepted"] += 1
                self._inject(source, destination, power, report)
            elif all(not self.reports.can_forward(source, destination, r, self.clock.now / 1e6)
                     for r in buffer.values()):
                self.selection_counts["receiver_rejected"] += 1
            else:
                self.selection_counts["out_of_range_or_inactive"] += 1
        self.step_end_us = time_s * 1e6
        # Only skip polls in a wholly idle radio interval. Alongside any live
        # frame/ACK/retry, preserve even empty-node polls and their event order.
        self.coalesce_empty_step = (self.config.runtime.coalesce_idle_polls
            and not any(t.end_time >= self.clock.now for t in self.simulator.channel.transmissions)
            and all(drone.transmitting_queue.empty() and not drone.inbox.items
                    and not any(p.is_alive for p in drone.mac_process_dict.values())
                    and not any(p.is_alive for p in drone.mac_protocol.wait_ack_process_dict.values())
                    for drone in self.simulator.drones))
        self.clock.run(until=self.step_end_us)
        self._collect_drops()
        return float(self.energy_j[:-1].sum() - before)

    def inflight_counts(self):
        counts = np.zeros(self.config.world.num_uavs)
        for transfer in self.sessions.values():
            counts[transfer.source] += len(transfer.pending)
        return counts

    def metrics(self):
        m = self.simulator.metrics.snapshot()
        elapsed = self.clock.now / 1e6
        counts = self.diagnostic_counts.copy()
        for name in ("custody_peer_transfers", "direct_gcs_reports", "relayed_gcs_reports"):
            counts[COUNT[name]] = getattr(self.reports, name)
        diagnostics = network_diagnostics(counts, self.diagnostic_stats, self.power_histogram,
                                          self.reports.delivery_hop_histogram)
        return {"diagnostics": diagnostics, "selection_counts": dict(self.selection_counts), "enabled": True, "backend": "uavnetsim", "upstream_revision": REVISION,
                "generated_packets": m["generated"], "received_packets": m["delivered"],
                "packet_pdr_percent": m["pdr_percent"], "packet_e2e_delay_ms": m["e2e_delay_ms"],
                "phy_successes": m["phy_success"], "phy_failures": m["phy_failure"],
                "phy_success_percent": m["phy_success_percent"], "collisions": m["collisions"],
                "communication_energy_j": float(self.energy_j.sum()),
                "uav_communication_energy_j": float(self.energy_j[:-1].sum()),
                "gcs_communication_energy_j": float(self.energy_j[-1]),
                "uav_tx_energy_j": float(self.tx_energy_j[:-1].sum()),
                "uav_rx_energy_j": float(self.rx_energy_j[:-1].sum()),
                "gcs_tx_energy_j": float(self.tx_energy_j[-1]),
                "gcs_rx_energy_j": float(self.rx_energy_j[-1]),
                "energy_blocked_transmissions": self.energy_blocked_transmissions,
                "obstacle_extra_loss_db": self.config.network.obstacle_extra_loss_db,
                "payload_throughput_kbps": self.received_payload_bytes * 8 / max(elapsed, 1e-12) / 1000,
                "targets_delivered": int(self.reports.delivered.sum()),
                "report_e2e_delay_s": float(np.mean(self.reports.delivery_times_s)) if self.reports.delivery_times_s else 0.0,
                "report_average_hops": float(np.mean(self.reports.delivery_hops)) if self.reports.delivery_hops else 0.0}

    def close(self):
        global _ACTIVE
        if self._closed:
            return
        try:
            self.simulator.close()
        finally:
            for name, value in self.previous.items():
                setattr(self.native, name, value)
            self.directory.cleanup()
            self.sessions.clear()
            self._closed = True
            if _ACTIVE is self:
                _ACTIVE = None
