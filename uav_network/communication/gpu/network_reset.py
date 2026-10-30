"""Cold-start network reset isolated from the packet scheduler module."""
import warp as wp

from ...environment.gpu.state import WorldParams, WorldState
from .rng import seed_mt
from .reports import reset_reward_state
from .events import push
from . import diagnostics as diag

wp.set_module_options({"enable_fp_fma": False, "enable_backward": False})


@wp.kernel
def reset_network(s: WorldState, p: WorldParams, mask: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] != 0:
        s.steps[e] = 0
        s.terminated[e] = 0
        s.blocked[e] = 0
        s.clipped[e] = 0
        s.errors[e] = 0
        s.frame_count[e] = 0
        s.event_count[e] = 0
        s.event_sequence[e] = wp.int64(0)
        s.packet_sequence[e] = wp.int64(0)
        s.report_sequence[e] = wp.int64(0)
        s.packet_time[e] = wp.float64(0)
        s.mission_time[e] = wp.float64(0)
        s.packet_delay_sum[e] = wp.float64(0)
        s.report_delay_sum[e] = wp.float64(0)
        s.report_hop_sum[e] = wp.float64(0)
        s.report_delivery_count[e] = 0
        s.trace_count[e] = 0
        reset_reward_state(s, e)
        for j in range(diag.COUNTER_COUNT):
            s.diagnostic_counts[e, j] = wp.int64(0)
        for j in range(8):
            s.delivery_hop_histogram[e, j] = wp.int64(0)
        for j in range(diag.STAT_COUNT):
            for k in range(4):
                s.diagnostic_stats[e, j, k] = wp.float64(0)
        for j in range(2):
            for k in range(6):
                s.power_histogram[e, j, k] = wp.int64(0)
        for i in range(p.n):
            s.buffer_peak[e, i] = wp.int64(0)
        for t in range(p.targets_count):
            s.first_discovery[e, t] = wp.float64(-1)
            s.first_delivery[e, t] = wp.float64(-1)
        for i in range(p.n + 1):
            seed_mt(s, e, i)
            s.node_queue_count[e, i] = 0
            s.node_current[e, i] = -1
            s.node_feed_alive[e, i] = 1
            s.node_wait_finish[e, i] = 1
            s.node_wait_packet[e, i] = -1
            s.node_receiving[e, i] = 0
            s.communication_energy[e, i] = wp.float64(0)
            s.tx_energy[e, i] = wp.float64(0)
            s.rx_energy[e, i] = wp.float64(0)
            if p.network_enabled != 0:
                push(s, p, e, p.queue_poll, 1, i, 0, 1)
            if i < p.n:
                s.battery[e, i] = p.capacity
                s.flight_energy[e, i] = wp.float64(0)
                s.inflight[e, i] = wp.float64(0)
                for t in range(p.targets_count):
                    s.known[e, i, t] = 0
                    s.recovery[e, i, t] = 0
                    s.report_present[e, i, t] = 0
                    s.report_origin[e, i, t] = 0
                    s.report_hops[e, i, t] = 0
                    s.report_created[e, i, t] = wp.float64(0)
                    s.report_order[e, i, t] = wp.int64(0)
        for t in range(p.targets_count):
            s.delivered[e, t] = 0
        for j in range(9):
            s.motion_counts[e, j] = 0
        for j in range(8):
            s.packet_counts[e, j] = 0
        for j in range(5):
            s.selection_counts[e, j] = 0
        for j in range(4):
            s.action_counts[e, j] = wp.int64(0)
        for j in range(3):
            s.report_counts[e, j] = 0
        for j in range(p.packet_capacity):
            s.pkt_status[e, j] = 0
            s.pkt_generation[e, j] = 0
            s.pkt_backoff_version[e, j] = 0
        for j in range(p.frame_capacity):
            s.tx_active[e, j] = 0
        for j in range((p.n + 1) * (p.n + 1)):
            s.pair_time[e, j] = wp.float64(-1)
        for j in range(p.session_capacity):
            s.session_active[e, j] = 0
            s.session_id[e, j] = wp.int64(0)
            s.session_window_start[e, j] = 0
            s.session_arrived_mask[e, j] = wp.uint64(0)
            s.session_pending_mask[e, j] = wp.uint64(0)
            s.session_created[e, j] = wp.float64(0)
            s.session_power[e, j] = wp.float64(0)
