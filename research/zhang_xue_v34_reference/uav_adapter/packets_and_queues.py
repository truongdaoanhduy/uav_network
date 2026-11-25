import heapq
import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

from uav_adapter.ds_belief import BeliefTable
from uav_adapter.phy_patch import sinr_db_patched

HEADER_FMT = ">HBBIHH"  # 12 B: episode_id, src, dst, seq_id, entry_count, flags
ENTRY_FMT = ">BHHHB"    # 8 B: origin_id, cell_id, n_pos, n_neg, reserved
APP_HEADER_BYTES = struct.calcsize(HEADER_FMT)  # 12
ENTRY_BYTES = struct.calcsize(ENTRY_FMT)        # 8

IP_HEADER_BYTES = 20
MAC_HEADER_BYTES = 14
PHY_HEADER_BYTES = 24
TOTAL_NET_HEADER_BYTES = IP_HEADER_BYTES + MAC_HEADER_BYTES + PHY_HEADER_BYTES  # 58 B
MAC_ACK_BYTES = 30
E2E_ACK_PAYLOAD_BYTES = 32
MAX_PAYLOAD_BYTES = 512
MAX_ENTRIES_PER_PACKET = (MAX_PAYLOAD_BYTES - APP_HEADER_BYTES) // ENTRY_BYTES  # 62


@dataclass(frozen=True)
class SearchDataPacket:
    """Immutable application packet with real binary payload sizing and strict MTU validation."""
    packet_id: int
    episode_id: int
    src_id: int
    dst_id: int
    seq_id: int
    creation_time: float
    deadline: float
    entries: Tuple[Tuple[int, int, int, int], ...]  # (origin_id, cell_id, n_pos, n_neg)
    flags: int = 0
    max_payload_bytes: int = MAX_PAYLOAD_BYTES

    def __post_init__(self):
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, tuple) or len(entry) != 4
            for entry in self.entries
        ):
            raise TypeError("entries must be an immutable tuple of 4-field tuples")
        if not (math.isfinite(self.creation_time) and math.isfinite(self.deadline)
                and self.creation_time >= 0 and self.deadline > 0):
            raise ValueError("creation_time must be finite/nonnegative; deadline finite/positive")
        if self.max_payload_bytes < APP_HEADER_BYTES:
            raise ValueError("max_payload_bytes too small for application header")
        if self.payload_bytes > self.max_payload_bytes:
            max_n = (self.max_payload_bytes - APP_HEADER_BYTES) // ENTRY_BYTES
            raise ValueError(
                f"Payload {self.payload_bytes} B exceeds MTU {self.max_payload_bytes} B "
                f"(max {max_n} entries, got {len(self.entries)})"
            )
        try:
            self.to_bytes()  # Validate every on-wire integer before queue admission.
        except (struct.error, OverflowError) as exc:
            raise ValueError("Packet field is outside its binary wire range") from exc

    @property
    def payload_bytes(self) -> int:
        return APP_HEADER_BYTES + ENTRY_BYTES * len(self.entries)

    @property
    def frame_bytes(self) -> int:
        return TOTAL_NET_HEADER_BYTES + self.payload_bytes

    def to_bytes(self) -> bytes:
        buf = bytearray(struct.pack(
            HEADER_FMT, self.episode_id, self.src_id, self.dst_id,
            self.seq_id, len(self.entries), self.flags
        ))
        for origin_id, cell_id, n_pos, n_neg in self.entries:
            buf.extend(struct.pack(ENTRY_FMT, origin_id, cell_id, n_pos, n_neg, 0))
        return bytes(buf)


@dataclass(frozen=True)
class E2EAckPacket:
    """Routable end-to-end ACK (32 B payload -> 90 B frame), EXCLUDED from Data PDR."""
    packet_id: int
    episode_id: int
    src_id: int
    dst_id: int
    acked_origin_id: int
    acked_seq_id: int
    creation_time: float
    deadline: float
    payload_bytes: int = E2E_ACK_PAYLOAD_BYTES

    @property
    def frame_bytes(self) -> int:
        return TOTAL_NET_HEADER_BYTES + self.payload_bytes  # 90 B


class NetworkMetricsFilter:
    def __init__(self):
        self.data_generated = 0
        self.data_delivered_ids = set()
        self.e2e_ack_delivered = 0
        self.total_airtime_s = 0.0

    def on_generate(self, pkt):
        if isinstance(pkt, SearchDataPacket):
            self.data_generated += 1

    def on_deliver(self, pkt):
        if isinstance(pkt, E2EAckPacket):
            self.e2e_ack_delivered += 1
        elif isinstance(pkt, SearchDataPacket):
            self.data_delivered_ids.add(pkt.packet_id)

    def record_hop_airtime(self, pkt, bit_rate_bps: float = 2e6, include_mac_overhead: bool = True) -> float:
        tx_s = (pkt.frame_bytes * 8.0) / bit_rate_bps
        overhead_s = (
            ((50.0 + 310.0 + 10.0) * 1e-6 + (MAC_ACK_BYTES * 8.0) / bit_rate_bps)
            if include_mac_overhead
            else 0.0
        )
        hop_s = tx_s + overhead_s
        self.total_airtime_s += hop_s
        return hop_s

    @property
    def data_pdr(self) -> float:
        return 0.0 if self.data_generated == 0 else len(self.data_delivered_ids) / self.data_generated



class PacketQueueManager:
    """Bounded node-local packet queues.

    The total of scheduled and waiting packets never exceeds max_size.
    Rejections are counted and never scheduled as TX events.
    """
    def __init__(self, max_size: int = 50):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.transmitting_queue: List = []
        self.waiting_list: List = []
        self.expired_count = 0
        self.dropped_full_count = 0

    def enqueue(self, pkt) -> bool:
        if pkt in self.transmitting_queue or pkt in self.waiting_list:
            return False
        if len(self.transmitting_queue) + len(self.waiting_list) >= self.max_size:
            self.dropped_full_count += 1
            return False
        self.transmitting_queue.append(pkt)
        return True

    def mark_transmitted(self, pkt):
        if pkt in self.transmitting_queue:
            self.transmitting_queue.remove(pkt)

    def park_for_retry(self, pkt) -> bool:
        self.mark_transmitted(pkt)
        if pkt in self.waiting_list:
            return True
        if len(self.transmitting_queue) + len(self.waiting_list) >= self.max_size:
            self.dropped_full_count += 1
            return False
        self.waiting_list.append(pkt)
        return True

    def release_waiting(self, now: float) -> List:
        self.purge_expired_packets(now)
        capacity = self.max_size - len(self.transmitting_queue)
        ready = self.waiting_list[:capacity]
        del self.waiting_list[:capacity]
        self.transmitting_queue.extend(ready)
        return ready

    def purge_expired_packets(self, now: float) -> int:
        before = len(self.transmitting_queue) + len(self.waiting_list)
        self.transmitting_queue = [
            p for p in self.transmitting_queue if now < p.creation_time + p.deadline
        ]
        self.waiting_list = [
            p for p in self.waiting_list if now < p.creation_time + p.deadline
        ]
        removed = before - len(self.transmitting_queue) - len(self.waiting_list)
        self.expired_count += removed
        return removed


class MiniDiscreteEventNet:
    """Small deterministic protocol reference, not the production UavNetSim PHY/MAC.

    Queue-full admission is enforced before TX. If a route or sender is disabled,
    the packet stays in a bounded waiting list until the next simulation advance;
    RX blackout is modelled as a failed link with retransmission at the prior hop.
    Deadlines remain relative to original creation_time on every hop and retry.
    """
    def __init__(self, positions: Dict[int, Tuple[float, float]], belief: BeliefTable,
                 tx_power_w: float = 0.5, sinr_threshold_db: float = 6.0):
        self.positions = dict(positions)
        self.radio_enabled = {u: True for u in positions}
        self.queues = {u: PacketQueueManager(max_size=50) for u in positions}
        self.belief = belief
        self.tx_power_w = tx_power_w
        self.sinr_threshold_db = sinr_threshold_db
        self.metrics = NetworkMetricsFilter()
        self.watermarks: Dict[Tuple[int, int], int] = {}
        self._seq_counter = 1000
        self._event_serial = 0
        self.now = 0.0
        self.event_q: List = []

    def _push_event(self, when: float, event_type: str, node: int, pkt, prev=None):
        self._event_serial += 1
        heapq.heappush(
            self.event_q, (when, self._event_serial, event_type, node, pkt, prev)
        )

    def _dist(self, a: int, b: int) -> float:
        xa, ya = self.positions[a]
        xb, yb = self.positions[b]
        return math.hypot(xa - xb, ya - yb)

    def _greedy_next_hop(self, cur: int, dst: int):
        if cur == dst:
            return dst
        best_hop, best_dist = None, self._dist(cur, dst)
        for nb in self.positions:
            if nb == cur or not self.radio_enabled[nb]:
                continue
            sinr = sinr_db_patched(
                self.tx_power_w, self._dist(cur, nb), [],
                tx_radio_enabled=self.radio_enabled[cur],
                rx_radio_enabled=self.radio_enabled[nb],
            )
            if sinr >= self.sinr_threshold_db and self._dist(nb, dst) < best_dist:
                best_dist = self._dist(nb, dst)
                best_hop = nb
        return best_hop

    def send_data(self, now: float, pkt: SearchDataPacket) -> bool:
        if now < self.now:
            raise ValueError("Cannot send data in the simulation past")
        if pkt.src_id not in self.queues:
            raise ValueError("Unknown packet source")
        if pkt.dst_id not in self.queues or pkt.dst_id == pkt.src_id:
            raise ValueError("Data destination must be a distinct valid UAV")
        self.metrics.on_generate(pkt)  # includes rejected attempts, so PDR cannot exceed 1
        if not self.queues[pkt.src_id].enqueue(pkt):
            return False
        self._push_event(now, "TX", pkt.src_id, pkt)
        return True

    def _release_waiting_at(self, now: float):
        for u, queue in self.queues.items():
            if self.radio_enabled[u]:
                for pkt in queue.release_waiting(now):
                    self._push_event(now, "TX", u, pkt)
            else:
                queue.purge_expired_packets(now)

    def run_until(self, t_end: float) -> float:
        if t_end < self.now:
            raise ValueError("Cannot run the simulator backwards")
        shared_gain = 0.0
        self._release_waiting_at(self.now)
        while self.event_q and self.event_q[0][0] <= t_end:
            now, _, ev_type, node, pkt, prev = heapq.heappop(self.event_q)
            queue = self.queues[node]
            queue.purge_expired_packets(now)
            if now >= pkt.creation_time + pkt.deadline:
                queue.mark_transmitted(pkt)
                continue
            if ev_type == "TX":
                if not self.radio_enabled[node]:
                    queue.park_for_retry(pkt)
                    continue
                nxt = self._greedy_next_hop(node, pkt.dst_id)
                if nxt is None:
                    queue.park_for_retry(pkt)
                    continue
                queue.mark_transmitted(pkt)
                hop_s = self.metrics.record_hop_airtime(pkt)
                self._push_event(now + hop_s, "RX", nxt, pkt, prev=node)
            elif ev_type == "RX":
                if not self.radio_enabled[node]:
                    if prev is not None:
                        self.queues[prev].park_for_retry(pkt)
                    continue
                if node == pkt.dst_id:
                    self.metrics.on_deliver(pkt)
                    if isinstance(pkt, SearchDataPacket):
                        shared_gain += self.belief.merge_packet(node, pkt.entries)
                        self._seq_counter += 1
                        ack = E2EAckPacket(
                            packet_id=self._seq_counter,
                            episode_id=pkt.episode_id,
                            src_id=node,
                            dst_id=pkt.src_id,
                            acked_origin_id=pkt.src_id,
                            acked_seq_id=pkt.seq_id,
                            creation_time=now,
                            deadline=pkt.deadline,
                        )
                        if self.queues[node].enqueue(ack):
                            self._push_event(now, "TX", node, ack)
                    elif isinstance(pkt, E2EAckPacket):
                        key = (node, pkt.src_id)
                        self.watermarks[key] = max(
                            self.watermarks.get(key, 0), pkt.acked_seq_id
                        )
                elif queue.enqueue(pkt):
                    self._push_event(now, "TX", node, pkt)
        for queue in self.queues.values():
            queue.purge_expired_packets(t_end)
        self.now = t_end
        return shared_gain
