"""Shared packet-step setup/finalization outside the MAC event loop."""
import warp as wp

from ...environment.gpu.state import WorldParams, WorldState
from .reports import active, bytes_used, can_receive
from .phy import xy, latest_arrival_end
from .telemetry import record_action
from .power import power_from_action, link_power_feasible
from .streaming import clear_pending, inject


@wp.func
def packet_step_preamble(s: WorldState, p: WorldParams, continuous: wp.array3d(dtype=wp.float32),
                         destination: wp.array2d(dtype=wp.int64), e: int) -> wp.float64:
    now = s.packet_time[e]
    end = (s.mission_time[e]+p.dt)*wp.float64(1e6)
    # Motion is fixed during packet_step; invalidate once, then reuse exact link gains.
    for pair in range((p.n+1)*(p.n+1)):
        s.pair_time[e,pair] = wp.float64(-1)
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
    return end


@wp.func
def packet_step_finish(s: WorldState, p: WorldParams, e: int, end: wp.float64):
    s.packet_time[e] = end
    clear_pending(s,p,e,end/wp.float64(1e6))

