"""Bounded streaming report windows for the Warp packet backend."""
import warp as wp

from ...environment.gpu.state import WorldParams, WorldState
from .reports import discard_sessions
from .phy import latest_arrival_end

wp.set_module_options({"enable_fp_fma": False, "enable_backward": False})

@wp.func
def window_count(p: WorldParams, start: int) -> int:
    return wp.min(p.max_queue_packets, p.fragments-start)


@wp.func
def window_mask(p: WorldParams, start: int) -> wp.uint64:
    count = window_count(p,start)
    mask = wp.uint64(0)
    for bit_index in range(64):
        if bit_index < count:
            mask = mask | (wp.uint64(1) << wp.uint64(bit_index))
    return mask


@wp.func
def fragment_bit(s: WorldState, e: int, session: int, fragment: int) -> wp.uint64:
    offset = fragment-s.session_window_start[e,session]
    if offset < 0 or offset >= 64:
        return wp.uint64(0)
    return wp.uint64(1) << wp.uint64(offset)


@wp.func
def clear_pending(s: WorldState, p: WorldParams, e: int, now: wp.float64):
    # Session validity is report-custody based. Terminal packet slots keep their
    # pending bit until the next policy interval so retry exhaustion does not
    # silently create another attempt under the same action.
    for session in range(p.session_capacity):
        if s.session_active[e,session] != 0:
            source = session//((p.n+1)*p.targets_count)
            target = (session//(p.n+1))%p.targets_count
            if (s.report_present[e,source,target] == 0
                or s.report_created[e,source,target] != s.session_created[e,session]
                or s.report_origin[e,source,target] != s.session_origin[e,session]
                or s.report_hops[e,source,target] != s.session_hops[e,session]):
                discard_sessions(s,p,e,source,target)
    for packet in range(p.packet_capacity):
        if s.pkt_status[e,packet] == 4:
            session = int(s.pkt_session[e,packet])
            if (session >= 0 and session < p.session_capacity
                and s.session_active[e,session] != 0
                and s.pkt_session_id[e,packet] == s.session_id[e,session]):
                bit = fragment_bit(s,e,session,s.pkt_fragment[e,packet])
                if bit != wp.uint64(0):
                    s.session_pending_mask[e,session] = s.session_pending_mask[e,session] & ~bit


@wp.func
def fill_session(s: WorldState, p: WorldParams, e: int, session: int,
                 source: int, dest: int, target: int, now: wp.float64):
    start = s.session_window_start[e,session]
    count = window_count(p,start)
    for offset in range(64):
        if offset >= count:
            continue
        bit = wp.uint64(1) << wp.uint64(offset)
        if ((s.session_arrived_mask[e,session] & bit) != wp.uint64(0)
            or (s.session_pending_mask[e,session] & bit) != wp.uint64(0)):
            continue
        queued = int(0)
        for j in range(p.packet_capacity):
            if s.pkt_status[e,j] == 1 and s.pkt_source[e,j] == source:
                queued += 1
        if queued >= p.max_queue_packets:
            break
        remaining = s.report_created[e,source,target]+p.report_ttl-now/wp.float64(1e6)
        if remaining <= wp.float64(0):
            break
        slot = int(-1)
        for j in range(p.packet_capacity):
            if slot < 0 and (s.pkt_status[e,j] == 0 or s.pkt_status[e,j] == 4):
                retained = bool(False)
                for frame in range(p.frame_capacity):
                    if (s.tx_active[e,frame] != 0 and s.tx_packet[e,frame] == j
                        and s.tx_generation[e,frame] == s.pkt_generation[e,j]
                        and (s.tx_receive[e,frame] == 1 or s.tx_receive[e,frame] == 2
                             or (s.tx_receive[e,frame] < 3 and latest_arrival_end(s,p,e,frame) >= now))):
                        retained = True
                if not retained:
                    slot = j
        if slot < 0:
            s.errors[e] = 2
            break
        fragment = start+offset
        s.packet_sequence[e] += wp.int64(1)
        s.pkt_id[e,slot] = s.packet_sequence[e]
        s.pkt_order[e,slot] = s.packet_sequence[e]
        s.pkt_status[e,slot] = 1
        s.pkt_generation[e,slot] += 1
        s.pkt_backoff_version[e,slot] += 1
        s.pkt_source[e,slot] = source
        s.pkt_destination[e,slot] = dest
        s.pkt_target[e,slot] = target
        s.pkt_fragment[e,slot] = fragment
        s.pkt_attempt[e,slot] = 0
        s.pkt_ttl[e,slot] = 0
        s.pkt_delivered[e,slot] = 0
        s.pkt_session[e,slot] = wp.int64(session)
        s.pkt_session_id[e,slot] = s.session_id[e,session]
        s.pkt_origin[e,slot] = s.report_origin[e,source,target]
        s.pkt_hops[e,slot] = s.report_hops[e,source,target]
        s.pkt_report_created[e,slot] = s.report_created[e,source,target]
        s.pkt_created[e,slot] = now
        s.pkt_deadline[e,slot] = now+wp.min(p.packet_lifetime,remaining)*wp.float64(1e6)
        s.pkt_power[e,slot] = s.session_power[e,session]
        s.pkt_first_attempt[e,slot] = wp.float64(-1)
        s.session_pending_mask[e,session] = s.session_pending_mask[e,session] | bit
        s.packet_counts[e,0] += 1
        s.node_queue_count[e,source] += 1


@wp.func
def inject(s: WorldState, p: WorldParams, e: int, source: int, dest: int,
           target: int, power: wp.float64, now: wp.float64):
    session = (source*p.targets_count+target)*(p.n+1)+dest
    if s.session_active[e,session] == 0:
        s.session_active[e,session] = 1
        s.session_id[e,session] += wp.int64(1)
        s.session_origin[e,session] = s.report_origin[e,source,target]
        s.session_created[e,session] = s.report_created[e,source,target]
        s.session_hops[e,session] = s.report_hops[e,source,target]
        s.session_window_start[e,session] = 0
        s.session_arrived_mask[e,session] = wp.uint64(0)
        s.session_pending_mask[e,session] = wp.uint64(0)
    s.session_power[e,session] = power
    fill_session(s,p,e,session,source,dest,target,now)
