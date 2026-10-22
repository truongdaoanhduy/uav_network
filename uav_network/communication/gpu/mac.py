"""Deterministic SimPy-order packet scheduler; one Warp thread per world."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState
from .rng import randbelow
from .reports import (active, bytes_used, can_receive, transfer, begin_step, sense_reports,
                      masked_sense_reports, end_step)
from .phy import xy, busy, emit, evaluate, arrival_start, arrival_end, latest_arrival_end
from .events import push, pop, replace_root, next_time_after_root
from .streaming import (fill_session, fragment_bit, window_count, window_mask)
from . import diagnostics as diag
from .packet_step_support import packet_step_preamble, packet_step_finish
from .receiver_index import receiver_push, next_receive_indexed

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


@wp.func
def event_rxindex(s: WorldState, p: WorldParams, e: int, kind: int, arg: int,
                  generation: int, now: wp.float64,
                  frames: wp.array3d(dtype=wp.int32), sizes: wp.array2d(dtype=wp.int32)):
    if kind == DELIVER:
        frame = arg
        if evaluate(s,p,e,frame,now):
            s.tx_receive[e,frame] = 1
            s.tx_receive_order[e,frame] = s.event_sequence[e]
            receiver = s.tx_destination[e,frame]
            receiver_push(s,p,e,receiver,frames,sizes,frame)
            next_receive_indexed(s,p,e,receiver,now,RECEIVE,frames,sizes)
        else:
            s.tx_receive[e,frame] = 3
    elif kind == RECEIVE:
        frame = arg
        node = s.tx_destination[e,frame]
        if not active(s,p,e,node) or s.tx_ttl[e,frame] >= p.n+2:
            s.tx_receive[e,frame] = 3
            s.node_receiving[e,node] = 0
            next_receive_indexed(s,p,e,node,now,RECEIVE,frames,sizes)
        elif s.tx_kind[e,frame] == 0:
            receive_data(s,p,e,frame,now)
            push(s,p,e,now+p.sifs,ACK_START,frame,0,1)
        else:
            packet = s.tx_packet[e,frame]
            if s.pkt_generation[e,packet] == s.tx_generation[e,frame]:
                if s.node_wait_packet[e,node] == packet:
                    s.node_wait_finish[e,node] = 1
                s.pkt_status[e,packet] = 4
            s.tx_receive[e,frame] = 3
            s.node_receiving[e,node] = 0
            next_receive_indexed(s,p,e,node,now,RECEIVE,frames,sizes)
    elif kind == RECEIVE_DONE:
        frame = arg
        node = s.tx_destination[e,frame]
        s.tx_receive[e,frame] = 3
        s.node_receiving[e,node] = 0
        next_receive_indexed(s,p,e,node,now,RECEIVE,frames,sizes)
    else:
        event(s,p,e,kind,arg,generation,now)


@wp.func
def handle_fast_root(s: WorldState, p: WorldParams, e: int, end: wp.float64) -> bool:
    time=s.event_time[e,0]; kind=s.event_kind[e,0]; arg=s.event_arg[e,0]
    generation=s.event_generation[e,0]
    if kind == LISTEN and s.pkt_backoff_version[e,arg] == generation:
        node=s.pkt_source[e,arg]
        if not busy(s,p,e,node,time):
            boundary=next_time_after_root(s,e,end)
            for k in range(s.frame_count[e]):
                frame=s.frame_order[e,k]
                if s.tx_source[e,frame] != node:
                    start=arrival_start(s,p,e,frame,node)
                    if time < start: boundary=wp.min(boundary,start)
            jumps=int(wp.ceil(boundary-time)); jumps=wp.max(jumps,1)
            if jumps > 1:
                s.event_sequence[e] += wp.int64(jumps-1)
                s.diagnostic_counts[e,diag.CCA_CHECKS] += wp.int64(jumps-1)
            replace_root(s,e,time+wp.float64(jumps),LISTEN,arg,generation,1)
            return True
    if kind == POLL and s.node_feed_alive[e,arg] != 0:
        if not (s.node_queue_count[e,arg] > 0 and s.node_current[e,arg] < 0
                and s.node_wait_finish[e,arg] != 0):
            delay=poll_delay(s,p,e,time)
            if (s.coalesce_step[e] == 0 and wp.floor(p.queue_poll) == p.queue_poll
                and wp.floor(time) == time):
                boundary=end
                for k in range(1,s.event_count[e]):
                    if s.event_kind[e,k] != POLL: boundary=wp.min(boundary,s.event_time[e,k])
                jumps=int(wp.ceil((boundary-time)/p.queue_poll)); jumps=wp.max(jumps,1)
                if jumps > 1: s.event_sequence[e] += wp.int64(jumps-1)
                delay=wp.float64(jumps)*p.queue_poll
            replace_root(s,e,time+delay,POLL,arg,0,1)
            return True
    return False


@wp.func
def try_macro_event(s: WorldState, p: WorldParams, e: int, kind: int, arg: int,
                    generation: int, now: wp.float64,
                    counters: wp.array2d(dtype=wp.int64)) -> bool:
    # Performance-qualified Gen2 candidate: only transmission-side fusion is
    # retained. All other event families execute through the exact Gen1 path.
    if kind == BACKOFF_DONE:
        valid = s.pkt_backoff_version[e,arg] == generation
        event(s,p,e,kind,arg,generation,now)
        if s.errors[e] != 0:
            return True
        end = (s.mission_time[e]+p.dt)*wp.float64(1e6)
        txgen = s.pkt_generation[e,arg]
        if (valid and s.event_count[e] > 0 and s.event_time[e,0] < end
            and s.event_kind[e,0] == TX_START and s.event_arg[e,0] == arg
            and s.event_generation[e,0] == txgen):
            time = s.event_time[e,0]
            pop(s,e)
            event(s,p,e,TX_START,arg,txgen,time)
            if (s.event_count[e] > 0 and s.event_time[e,0] < end
                and s.event_kind[e,0] == DATA_END and s.event_arg[e,0] == arg
                and s.event_generation[e,0] == txgen):
                time = s.event_time[e,0]
                pop(s,e)
                event(s,p,e,DATA_END,arg,txgen,time)
        return True

    if kind != TX_START:
        return False

    event(s,p,e,kind,arg,generation,now)
    if s.errors[e] != 0:
        return True
    end = (s.mission_time[e]+p.dt)*wp.float64(1e6)
    if (s.event_count[e] > 0 and s.event_time[e,0] < end
        and s.event_kind[e,0] == DATA_END and s.event_arg[e,0] == arg
        and s.event_generation[e,0] == generation):
        time = s.event_time[e,0]
        pop(s,e)
        event(s,p,e,DATA_END,arg,generation,time)
    return True


from .reset import reset_network


@wp.func
def run_event_loop_gen1(s: WorldState, p: WorldParams, e: int, end: wp.float64):
    while s.event_count[e] > 0 and s.errors[e] == 0:
        time=s.event_time[e,0]
        if time >= end: break
        kind=s.event_kind[e,0]; arg=s.event_arg[e,0]; generation=s.event_generation[e,0]
        s.packet_time[e]=time
        if handle_fast_root(s,p,e,end): continue
        pop(s,e); event(s,p,e,kind,arg,generation,time)


@wp.func
def run_event_loop_gen2(s: WorldState, p: WorldParams, e: int, end: wp.float64,
                        counters: wp.array2d(dtype=wp.int64)):
    while s.event_count[e] > 0 and s.errors[e] == 0:
        time=s.event_time[e,0]
        if time >= end: break
        kind=s.event_kind[e,0]; arg=s.event_arg[e,0]; generation=s.event_generation[e,0]
        s.packet_time[e]=time
        if handle_fast_root(s,p,e,end): continue
        pop(s,e)
        if not try_macro_event(s,p,e,kind,arg,generation,time,counters):
            event(s,p,e,kind,arg,generation,time)


@wp.func
def run_event_loop_rxindex(s: WorldState, p: WorldParams, e: int, end: wp.float64,
                           frames: wp.array3d(dtype=wp.int32),
                           sizes: wp.array2d(dtype=wp.int32)):
    while s.event_count[e] > 0 and s.errors[e] == 0:
        time=s.event_time[e,0]
        if time >= end: break
        kind=s.event_kind[e,0]; arg=s.event_arg[e,0]; generation=s.event_generation[e,0]
        s.packet_time[e]=time
        if handle_fast_root(s,p,e,end): continue
        pop(s,e)
        event_rxindex(s,p,e,kind,arg,generation,time,frames,sizes)


@wp.func
def packet_step_world(s: WorldState, p: WorldParams, continuous: wp.array3d(dtype=wp.float32),
                      destination: wp.array2d(dtype=wp.int64), e: int):
    end = packet_step_preamble(s,p,continuous,destination,e)
    run_event_loop_gen1(s,p,e,end)
    packet_step_finish(s,p,e,end)


@wp.func
def packet_step_world_gen2(s: WorldState, p: WorldParams, continuous: wp.array3d(dtype=wp.float32),
                           destination: wp.array2d(dtype=wp.int64), e: int,
                           counters: wp.array2d(dtype=wp.int64)):
    end = packet_step_preamble(s,p,continuous,destination,e)
    run_event_loop_gen2(s,p,e,end,counters)
    packet_step_finish(s,p,e,end)


@wp.func
def packet_step_world_rxindex(s: WorldState, p: WorldParams, continuous: wp.array3d(dtype=wp.float32),
                              destination: wp.array2d(dtype=wp.int64), e: int,
                              frames: wp.array3d(dtype=wp.int32),
                              sizes: wp.array2d(dtype=wp.int32)):
    end = packet_step_preamble(s,p,continuous,destination,e)
    run_event_loop_rxindex(s,p,e,end,frames,sizes)
    packet_step_finish(s,p,e,end)



@wp.kernel
def packet_step(s: WorldState, p: WorldParams, continuous: wp.array3d(dtype=wp.float32),
                destination: wp.array2d(dtype=wp.int64)):
    e = wp.tid()
    packet_step_world(s,p,continuous,destination,e)


@wp.kernel
def packet_step_gen2(s: WorldState, p: WorldParams, continuous: wp.array3d(dtype=wp.float32),
                     destination: wp.array2d(dtype=wp.int64),
                     counters: wp.array2d(dtype=wp.int64)):
    e = wp.tid()
    packet_step_world_gen2(s,p,continuous,destination,e,counters)


@wp.kernel
def packet_step_rxindex(s: WorldState, p: WorldParams, continuous: wp.array3d(dtype=wp.float32),
                        destination: wp.array2d(dtype=wp.int64),
                        frames: wp.array3d(dtype=wp.int32), sizes: wp.array2d(dtype=wp.int32)):
    e = wp.tid()
    packet_step_world_rxindex(s,p,continuous,destination,e,frames,sizes)


def transmission_times(s, index=0):
    count = int(s.trace_count.numpy()[index])
    attempts = s.trace_kind.numpy()[index,:count] < 2
    return s.trace_time.numpy()[index,:count][attempts].tolist()


def metrics_at(s, p, index=0):
    from .metrics import network_info
    return network_info(s,p,index)
