"""Deterministic SimPy-order packet scheduler; one Warp thread per world."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState
from .rng import seed_mt, randbelow
from .reports import (active, bytes_used, can_receive, transfer, begin_step, sense_reports,
                      masked_sense_reports, end_step, reset_reward_state)
from .phy import xy, busy, emit, evaluate, arrival_end, latest_arrival_end
from .events import push, pop
from .telemetry import record_action
from .power import power_from_action, link_power_feasible
from .streaming import (clear_pending, fill_session, fragment_bit, inject, window_count, window_mask)
from . import diagnostics as diag

# Preserve CPython scalar FP64 operation boundaries on CUDA as well as CPU.
wp.set_module_options({"enable_fp_fma": False, "enable_backward": False})
# SimPy urgent (0) versus normal (1); insertion order breaks equal-time ties.
POLL, START, MAC, IDLE, IDLE_DONE, LISTEN, INTERRUPT = range(1,8)
BACKOFF_DONE, TX_START, DATA_END, DELIVER, ACK_TIMEOUT, RELEASE = range(8,14)
RECEIVE, ACK_START, RECEIVE_DONE = range(14,17)

@wp.func
def next_receive(s: WorldState, p: WorldParams, e: int, node: int, now: wp.float64):
    if s.node_receiving[e,node] == 0:
        frame = int(-1)
        for j in range(p.frame_capacity):
            if s.tx_active[e,j] != 0 and s.tx_destination[e,j] == node and s.tx_receive[e,j] == 1:
                if frame < 0 or s.tx_receive_order[e,j] < s.tx_receive_order[e,frame]:
                    frame = j
        if frame >= 0:
            s.node_receiving[e,node] = 1
            s.tx_receive[e,frame] = 2
            push(s,p,e,now,14,frame,0,0)

@wp.func
def receive_data(s: WorldState, p: WorldParams, e: int, frame: int, now: wp.float64):
    packet = s.tx_packet[e,frame]
    if s.pkt_generation[e,packet] == s.tx_generation[e,frame] and s.pkt_delivered[e,packet] == 0:
        s.pkt_delivered[e,packet] = 1
        s.packet_counts[e,1] += 1
        s.packet_delay_sum[e] += now-s.tx_created[e,frame]
        source = s.tx_source[e,frame]
        dest = s.tx_destination[e,frame]
        target = s.tx_target[e,frame]
        session = (source*p.targets_count+target)*(p.n+1)+dest
        if (s.session_active[e,session] != 0 and s.session_id[e,session] == s.tx_session[e,frame]
            and s.report_present[e,source,target] != 0
            and s.report_created[e,source,target] == s.tx_report_created[e,frame]
            and s.report_origin[e,source,target] == s.tx_origin[e,frame]
            and s.report_hops[e,source,target] == s.tx_hops[e,frame]
            and now/wp.float64(1e6)-s.tx_report_created[e,frame] < p.report_ttl):
            fragment = s.tx_fragment[e,frame]
            bit = fragment_bit(s,e,session,fragment)
            if bit != wp.uint64(0):
                if (s.session_arrived_mask[e,session] & bit) == wp.uint64(0):
                    s.packet_counts[e,6] += s.tx_payload[e,frame]
                    s.session_arrived_mask[e,session] = s.session_arrived_mask[e,session] | bit
                s.session_pending_mask[e,session] = s.session_pending_mask[e,session] & ~bit
                start = s.session_window_start[e,session]
                current_mask = window_mask(p,start)
                if s.session_arrived_mask[e,session] == current_mask:
                    next_start = start+window_count(p,start)
                    if next_start >= p.fragments:
                        if not transfer(s,p,e,source,dest,target,s.tx_origin[e,frame],
                                        s.tx_report_created[e,frame],s.tx_hops[e,frame],now/wp.float64(1e6),
                                        s.tx_power[e,frame]):
                            s.session_active[e,session] = 0
                    else:
                        s.session_window_start[e,session] = next_start
                        s.session_arrived_mask[e,session] = wp.uint64(0)
                        s.session_pending_mask[e,session] = wp.uint64(0)
                        fill_session(s,p,e,session,source,dest,target,now)
                else:
                    fill_session(s,p,e,session,source,dest,target,now)

@wp.func
def poll_delay(s: WorldState, p: WorldParams, e: int, now: wp.float64) -> wp.float64:
    delay = p.queue_poll
    if (s.coalesce_step[e] != 0 and wp.floor(p.queue_poll) == p.queue_poll
        and wp.floor(now) == now):
        end = (s.mission_time[e]+p.dt)*wp.float64(1e6)
        delay *= wp.max(wp.float64(1),wp.ceil((end-now)/p.queue_poll)-wp.float64(1))
    return delay


@wp.func
def event(s: WorldState, p: WorldParams, e: int, kind: int, arg: int,
          generation: int, now: wp.float64):
    if kind == 1:
        node = arg
        if s.node_feed_alive[e,node] != 0:
            packet = int(-1)
            if (s.node_queue_count[e,node] > 0 and s.node_current[e,node] < 0
                and s.node_wait_finish[e,node] != 0):
                for j in range(p.packet_capacity):
                    if s.pkt_status[e,j] == 1 and s.pkt_source[e,j] == node:
                        if packet < 0 or s.pkt_order[e,j] < s.pkt_order[e,packet]:
                            packet = j
            if packet >= 0:
                s.node_queue_count[e,node] -= 1
                if now >= s.pkt_deadline[e,packet] or s.pkt_attempt[e,packet] >= p.max_retries:
                    reason = diag.RETRY_LIMIT
                    if now >= s.pkt_deadline[e,packet]:
                        reason = diag.PACKET_LIFETIME
                    s.diagnostic_counts[e,reason] += wp.int64(1)
                    s.pkt_status[e,packet] = 4
                    s.packet_counts[e,7] += 1
                    push(s,p,e,now+poll_delay(s,p,e,now),1,node,0,1)
                else:
                    s.pkt_status[e,packet] = 2
                    s.pkt_feed[e,packet] = 1
                    push(s,p,e,now,2,packet,s.pkt_generation[e,packet],0)
            else:
                push(s,p,e,now+poll_delay(s,p,e,now),1,node,0,1)
    elif kind == 2:
        packet = arg
        node = s.pkt_source[e,packet]
        if s.pkt_generation[e,packet] == generation and active(s,p,e,node):
            if s.node_current[e,node] < 0:
                s.node_current[e,node] = packet
                s.pkt_attempt[e,packet] += 1
                push(s,p,e,now,3,packet,generation,0)
            else:
                push(s,p,e,now+p.slot,2,packet,generation,1)
    elif kind == 3:
        packet = arg
        node = s.pkt_source[e,packet]
        attempt = s.pkt_attempt[e,packet]
        window = (p.cw_min+1)*(1 << (attempt-1))-1
        s.diagnostic_counts[e,diag.BACKOFF_DRAWS] += wp.int64(1)
        s.pkt_backoff[e,packet] = wp.float64(randbelow(s,e,node,window))*p.slot
        s.pkt_backoff_version[e,packet] += 1
        push(s,p,e,now,4,packet,s.pkt_backoff_version[e,packet],0)
    elif kind == 4:
        packet = arg
        if s.pkt_backoff_version[e,packet] == generation:
            if busy(s,p,e,s.pkt_source[e,packet],now):
                push(s,p,e,now+p.slot,4,packet,generation,1)
            else:
                push(s,p,e,now,5,packet,generation,1)
    elif kind == 5:
        packet = arg
        if s.pkt_backoff_version[e,packet] == generation:
            if s.pkt_attempt[e,packet] == 1:
                s.pkt_first_attempt[e,packet] = now
            s.pkt_wait_start[e,packet] = now
            push(s,p,e,now,6,packet,generation,0)
            push(s,p,e,now+p.difs+s.pkt_backoff[e,packet],8,packet,generation,1)
    elif kind == 6:
        packet = arg
        if s.pkt_backoff_version[e,packet] == generation:
            if busy(s,p,e,s.pkt_source[e,packet],now):
                push(s,p,e,now,7,packet,generation,0)
            else:
                push(s,p,e,now+wp.float64(1),6,packet,generation,1)
    elif kind == 7:
        packet = arg
        if s.pkt_backoff_version[e,packet] == generation:
            s.diagnostic_counts[e,diag.BACKOFF_FREEZES] += wp.int64(1)
            remaining = p.difs+s.pkt_backoff[e,packet]-(now-s.pkt_wait_start[e,packet])
            if remaining <= s.pkt_backoff[e,packet]:
                s.pkt_backoff[e,packet] = remaining
            s.pkt_backoff_version[e,packet] += 1
            push(s,p,e,now,4,packet,s.pkt_backoff_version[e,packet],0)
    elif kind == 8:
        packet = arg
        if s.pkt_backoff_version[e,packet] == generation:
            s.pkt_backoff_version[e,packet] += 1
            push(s,p,e,now,9,packet,s.pkt_generation[e,packet],1)
    elif kind == 9:
        packet = arg
        source = s.pkt_source[e,packet]
        dest = s.pkt_destination[e,packet]
        s.pkt_ttl[e,packet] += 1
        frame = emit(s,p,e,packet,source,dest,0,now)
        duration = (wp.float64(wp.min(p.payload_bytes,p.report_bytes-s.pkt_fragment[e,packet]*p.payload_bytes)*8)+p.header_bits)/p.bit_rate*wp.float64(1e6)
        push(s,p,e,now+duration,10,packet,generation,1)
        if frame >= 0:
            push(s,p,e,arrival_end(s,p,e,frame,dest),11,frame,0,1)
    elif kind == 10:
        packet = arg
        source = s.pkt_source[e,packet]
        s.node_wait_finish[e,source] = 0
        s.node_wait_packet[e,source] = packet
        push(s,p,e,now+p.sifs+p.ack_bits/p.bit_rate*wp.float64(1e6),13,packet,generation,1)
        push(s,p,e,now+p.ack_timeout,12,packet,generation,1)
    elif kind == 11:
        frame = arg
        if evaluate(s,p,e,frame,now):
            s.tx_receive[e,frame] = 1
            s.tx_receive_order[e,frame] = s.event_sequence[e]
            next_receive(s,p,e,s.tx_destination[e,frame],now)
        else:
            s.tx_receive[e,frame] = 3
    elif kind == 12:
        packet = arg
        source = s.pkt_source[e,packet]
        if (s.pkt_generation[e,packet] == generation and s.node_wait_packet[e,source] == packet
            and s.node_wait_finish[e,source] == 0):
            s.diagnostic_counts[e,diag.ACK_TIMEOUTS] += wp.int64(1)
            if s.pkt_attempt[e,packet] < p.max_retries:
                s.pkt_feed[e,packet] = 0
                push(s,p,e,now,2,packet,generation,0)
            else:
                s.diagnostic_counts[e,diag.RETRY_LIMIT] += wp.int64(1)
                s.node_wait_finish[e,source] = 1
                s.pkt_status[e,packet] = 4
    elif kind == 13:
        packet = arg
        source = s.pkt_source[e,packet]
        if s.node_current[e,source] == packet:
            s.node_current[e,source] = -1
        if s.pkt_feed[e,packet] != 0:
            if active(s,p,e,source):
                push(s,p,e,now+poll_delay(s,p,e,now),1,source,0,1)
            else:
                s.node_feed_alive[e,source] = 0
    elif kind == 14:
        frame = arg
        node = s.tx_destination[e,frame]
        if not active(s,p,e,node) or s.tx_ttl[e,frame] >= p.n+2:
            s.tx_receive[e,frame] = 3
            s.node_receiving[e,node] = 0
            next_receive(s,p,e,node,now)
        elif s.tx_kind[e,frame] == 0:
            receive_data(s,p,e,frame,now)
            push(s,p,e,now+p.sifs,15,frame,0,1)
        else:
            packet = s.tx_packet[e,frame]
            if s.pkt_generation[e,packet] == s.tx_generation[e,frame]:
                if s.node_wait_packet[e,node] == packet:
                    s.node_wait_finish[e,node] = 1
                s.pkt_status[e,packet] = 4
            s.tx_receive[e,frame] = 3
            s.node_receiving[e,node] = 0
            next_receive(s,p,e,node,now)
    elif kind == 15:
        frame = arg
        packet = s.tx_packet[e,frame]
        ack = emit(s,p,e,packet,s.tx_destination[e,frame],s.tx_source[e,frame],1,now)
        duration = p.ack_bits/p.bit_rate*wp.float64(1e6)
        push(s,p,e,now+duration,16,frame,0,1)
        if ack >= 0:
            push(s,p,e,arrival_end(s,p,e,ack,s.tx_source[e,frame]),11,ack,0,1)
    elif kind == 16:
        frame = arg
        node = s.tx_destination[e,frame]
        s.tx_receive[e,frame] = 3
        s.node_receiving[e,node] = 0
        next_receive(s,p,e,node,now)


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
        reset_reward_state(s,e)
        for j in range(diag.COUNTER_COUNT):
            s.diagnostic_counts[e,j] = wp.int64(0)
        for j in range(8):
            s.delivery_hop_histogram[e,j] = wp.int64(0)
        for j in range(diag.STAT_COUNT):
            for k in range(4):
                s.diagnostic_stats[e,j,k] = wp.float64(0)
        for j in range(2):
            for k in range(6):
                s.power_histogram[e,j,k] = wp.int64(0)
        for i in range(p.n):
            s.buffer_peak[e,i] = wp.int64(0)
        for t in range(p.targets_count):
            s.first_discovery[e,t] = wp.float64(-1)
            s.first_delivery[e,t] = wp.float64(-1)
        for i in range(p.n+1):
            seed_mt(s,e,i)
            s.node_queue_count[e,i] = 0
            s.node_current[e,i] = -1
            s.node_feed_alive[e,i] = 1
            s.node_wait_finish[e,i] = 1
            s.node_wait_packet[e,i] = -1
            s.node_receiving[e,i] = 0
            s.communication_energy[e,i] = wp.float64(0)
            s.tx_energy[e,i] = wp.float64(0)
            s.rx_energy[e,i] = wp.float64(0)
            if p.network_enabled != 0:
                push(s,p,e,p.queue_poll,1,i,0,1)
            if i < p.n:
                s.battery[e,i] = p.capacity
                s.flight_energy[e,i] = wp.float64(0)
                s.inflight[e,i] = wp.float64(0)
                for t in range(p.targets_count):
                    s.known[e,i,t] = 0
                    s.recovery[e,i,t] = 0
                    s.report_present[e,i,t] = 0
                    s.report_origin[e,i,t] = 0
                    s.report_hops[e,i,t] = 0
                    s.report_created[e,i,t] = wp.float64(0)
                    s.report_order[e,i,t] = wp.int64(0)
        for t in range(p.targets_count):
            s.delivered[e,t] = 0
        for j in range(9):
            s.motion_counts[e,j] = 0
        for j in range(8):
            s.packet_counts[e,j] = 0
        for j in range(5):
            s.selection_counts[e,j] = 0
        for j in range(4):
            s.action_counts[e,j] = wp.int64(0)
        for j in range(3):
            s.report_counts[e,j] = 0
        for j in range(p.packet_capacity):
            s.pkt_status[e,j] = 0
            s.pkt_generation[e,j] = 0
            s.pkt_backoff_version[e,j] = 0
        for j in range(p.frame_capacity):
            s.tx_active[e,j] = 0
        for j in range((p.n+1)*(p.n+1)):
            s.pair_time[e,j] = wp.float64(-1)
        for j in range(p.session_capacity):
            s.session_active[e,j] = 0
            s.session_id[e,j] = wp.int64(0)
            s.session_window_start[e,j] = 0
            s.session_arrived_mask[e,j] = wp.uint64(0)
            s.session_pending_mask[e,j] = wp.uint64(0)
            s.session_created[e,j] = wp.float64(0)
            s.session_power[e,j] = wp.float64(0)


@wp.kernel
def packet_step(s: WorldState, p: WorldParams, continuous: wp.array3d(dtype=wp.float32),
                destination: wp.array2d(dtype=wp.int64)):
    e = wp.tid()
    now = s.packet_time[e]
    end = (s.mission_time[e]+p.dt)*wp.float64(1e6)
    clear_pending(s,p,e,now/wp.float64(1e6))
    for source in range(p.n):
        index = int(destination[e,source])
        record_action(s,p,e,source,index,continuous,now)
        if p.network_enabled == 0 or index == 0:
            continue
        dest = p.n
        if index < p.n:
            dest = index-1
            if dest >= source:
                dest += 1
        s.selection_counts[e,0] += 1
        if bytes_used(s,p,e,source) == 0:
            s.selection_counts[e,4] += 1
            continue
        selected = int(-1)
        can_forward = bool(False)
        for t in range(p.targets_count):
            forward = (s.report_present[e,source,t] != 0
                       and now/wp.float64(1e6)-s.report_created[e,source,t] < p.report_ttl
                       and can_receive(s,p,e,dest,t))
            can_forward = can_forward or forward
            reach = p.peer_range
            if dest == p.n:
                reach = p.gcs_range
            if (forward and active(s,p,e,source) and active(s,p,e,dest)
                and wp.length(xy(s,p,e,source)-xy(s,p,e,dest)) <= reach and link_power_feasible(s,p,e,source,dest)):
                if selected < 0 or s.report_order[e,source,t] < s.report_order[e,source,selected]:
                    selected = t
        if selected >= 0:
            power = power_from_action(p,continuous[e,source,2])
            s.selection_counts[e,1] += 1
            inject(s,p,e,source,dest,selected,power,now)
        elif can_forward:
            s.selection_counts[e,3] += 1
        else:
            s.selection_counts[e,2] += 1
    idle = p.coalesce_idle_polls != 0
    for frame in range(p.frame_capacity):
        if s.tx_active[e,frame] != 0 and latest_arrival_end(s,p,e,frame) >= now:
            idle = False
    for node in range(p.n+1):
        if (s.node_queue_count[e,node] > 0 or s.node_current[e,node] >= 0
            or s.node_wait_finish[e,node] == 0 or s.node_receiving[e,node] != 0):
            idle = False
    s.coalesce_step[e] = int(idle)
    while s.event_count[e] > 0 and s.errors[e] == 0:
        time = s.event_time[e,0]
        if time >= end:
            break
        kind = s.event_kind[e,0]
        arg = s.event_arg[e,0]
        generation = s.event_generation[e,0]
        pop(s,e)
        s.packet_time[e] = time
        event(s,p,e,kind,arg,generation,time)
    s.packet_time[e] = end
    clear_pending(s,p,e,end/wp.float64(1e6))


def transmission_times(s, index=0):
    count = int(s.trace_count.numpy()[index])
    attempts = s.trace_kind.numpy()[index,:count] < 2
    return s.trace_time.numpy()[index,:count][attempts].tolist()


def metrics_at(s, p, index=0):
    from .metrics import network_info
    return network_info(s,p,index)
