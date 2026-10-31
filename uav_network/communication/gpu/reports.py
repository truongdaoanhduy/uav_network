"""Native FIFO report custody and source TTL, serial within each world."""
import warp as wp
from ...environment.gpu.xyz_geometry import range_distance
from ...environment.gpu.state import WorldParams, WorldState
from ...environment.gpu.bayesian import investigation_potential
from . import diagnostics as diag

wp.set_module_options({"enable_backward": False})


@wp.func
def active(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, node: int) -> bool:
    return node == p.n or p.energy_enabled == 0 or s.battery[e,node] > wp.float64(0)


@wp.func
def bytes_used(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, node: int) -> int:
    count = int(0)
    for t in range(p.targets_count):
        count += s.report_present[e,node,t]
    return count*p.report_bytes


@wp.func
def can_receive(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, node: int, target: int) -> bool:
    if node == p.n:
        return True
    return s.known[e,node,target] == 0 and bytes_used(s,p,e,node)+p.report_bytes <= p.buffer_bytes


@wp.func
def discard_sessions(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, source: int, target: int):
    for dest in range(p.n+1):
        session = (source*p.targets_count+target)*(p.n+1)+dest
        s.session_active[e,session] = 0
        s.session_window_start[e,session] = 0
        s.session_arrived_mask[e,session] = wp.uint64(0)
        s.session_pending_mask[e,session] = wp.uint64(0)
        s.session_power[e,session] = wp.float64(0)


@wp.func
def expire(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, now: wp.float64):
    for i in range(p.n):
        for t in range(p.targets_count):
            if s.report_present[e,i,t] != 0 and now-s.report_created[e,i,t] >= p.report_ttl:
                s.report_present[e,i,t] = 0
                s.recovery[e,i,t] = 1
                s.report_counts[e,0] += 1


@wp.func
def insert(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, node: int, target: int,
           origin: int, created: wp.float64, hops: int):
    s.report_sequence[e] += wp.int64(1)
    s.report_present[e,node,target] = 1
    s.report_origin[e,node,target] = origin
    s.report_created[e,node,target] = created
    s.report_hops[e,node,target] = hops
    s.report_order[e,node,target] = s.report_sequence[e]
    s.buffer_peak[e,node] = wp.max(s.buffer_peak[e,node],wp.int64(bytes_used(s,p,e,node)))


@wp.func
def transfer(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, source: int, destination: int,
             target: int, origin: int, created: wp.float64, hops: int, now: wp.float64,
             power: wp.float64) -> bool:
    if (s.report_present[e,source,target] == 0 or s.report_origin[e,source,target] != origin
        or s.report_created[e,source,target] != created or s.report_hops[e,source,target] != hops
        or now-created >= p.report_ttl or not can_receive(s,p,e,destination,target)):
        return False
    s.report_present[e,source,target] = 0
    if destination == p.n:
        if s.delivered[e,target] == 0:
            s.delivered[e,target] = 1
            s.first_delivery[e,target] = now
            s.delivery_hop_histogram[e,wp.min(hops,7)] += wp.int64(1)
            delivery = diag.RELAYED_GCS_REPORTS
            if hops == 0:
                delivery = diag.DIRECT_GCS_REPORTS
            s.diagnostic_counts[e,delivery] += wp.int64(1)
            s.report_delivery_count[e] += 1
            s.report_delay_sum[e] += now-created
            s.report_hop_sum[e] += wp.float64(hops+1)
    else:
        s.diagnostic_counts[e,diag.CUSTODY_PEER_TRANSFERS] += wp.int64(1)
        source_distance = range_distance(s,p,e,source,p.n)
        destination_distance = range_distance(s,p,e,destination,p.n)
        progress = source_distance-destination_distance
        if progress > wp.float64(0):
            s.diagnostic_counts[e,diag.RELAY_TOWARD_GCS_TRANSFERS] += wp.int64(1)
        else:
            s.diagnostic_counts[e,diag.RELAY_AWAY_FROM_GCS_TRANSFERS] += wp.int64(1)
        diag.sample(s,e,diag.STAT_RELAY_GCS_DISTANCE_IMPROVEMENT_M,progress)
        s.known[e,destination,target] = 1
        insert(s,p,e,destination,target,origin,created,hops+1)
    discard_sessions(s,p,e,source,target)
    if p.trace_enabled != 0:
        trace = s.trace_count[e]
        if trace >= p.trace_capacity:
            s.errors[e] = 4
        else:
            # Kind 2 is complete report custody, distinct from DATA/ACK attempts.
            s.trace_time[e,trace] = now*wp.float64(1e6)
            s.trace_source[e,trace] = source
            s.trace_destination[e,trace] = destination
            s.trace_kind[e,trace] = 2
            s.trace_report[e,trace,0] = wp.float64(target)
            s.trace_report[e,trace,1] = wp.float64(origin)
            s.trace_report[e,trace,2] = created
            s.trace_report[e,trace,3] = wp.float64(hops+1)
            s.trace_report[e,trace,4] = power
            start = s.positions[e,source]
            end = s.gcs[e]
            if destination < p.n:
                end = s.positions[e,destination]
            s.trace_report[e,trace,5] = start[0]
            s.trace_report[e,trace,6] = start[1]
            s.trace_report[e,trace,7] = end[0]
            s.trace_report[e,trace,8] = end[1]
            s.trace_count[e] += 1
    return True



@wp.func
def reset_reward_state(s: wp.ref[WorldState], e: int):
    s.reward[e] = wp.float32(0)
    s.shield_correction_step[e] = wp.float64(0)
    s.shield_correction[e] = wp.float64(0)
    for j in range(8):
        s.before[e,j] = wp.float64(0)
    for j in range(12):
        s.reward_components[e,j] = wp.float64(0)

@wp.kernel
def begin_step(s: WorldState, p: WorldParams):
    e = wp.tid()
    found = int(0)
    delivered = int(0)
    flight = wp.float64(0)
    comm = wp.float64(0)
    for t in range(p.targets_count):
        known = int(0)
        for i in range(p.n):
            known = wp.max(known,s.known[e,i,t])
        found += known
        delivered += s.delivered[e,t]
    for i in range(p.n):
        flight += s.flight_energy[e,i]
        comm += s.communication_energy[e,i]
    s.before[e,0] = wp.float64(found)
    s.before[e,1] = wp.float64(delivered)
    s.before[e,2] = wp.float64(s.report_counts[e,0])
    s.before[e,3] = wp.float64(s.report_counts[e,1])
    s.before[e,4] = wp.float64(s.blocked[e])
    s.before[e,5] = flight
    s.before[e,6] = comm
    s.before[e,7] = s.belief_potential[e] if p.xyz!=0 else wp.float64(0)
    s.reward[e] = wp.float32(investigation_potential(s,p,e) if p.xyz!=0 else wp.float64(0))
    s.trace_count[e] = 0
    # Materialize native mission time in a prior kernel so CUDA cannot contract
    # steps*dt + dt into an FMA with different IEEE-754 rounding than Python.
    s.mission_time[e] = wp.float64(s.steps[e])*p.dt
    expire(s,p,e,s.mission_time[e])


@wp.func
def sense_reports_at(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int):
    now = wp.float64(s.steps[e])*p.dt
    for i in range(p.n):
        for t in range(p.targets_count):
            fresh = s.known[e,i,t] == 0
            recover = p.recovery_enabled != 0 and s.recovery[e,i,t] != 0
            if active(s,p,e,i) and s.visible[e,i,t] != 0 and (fresh or recover):
                s.known[e,i,t] = 1
                if s.first_discovery[e,t] < wp.float64(0):
                    s.first_discovery[e,t] = now
                # Remote GCS delivery does not acknowledge this local sighting.
                if bytes_used(s,p,e,i)+p.report_bytes <= p.buffer_bytes:
                    insert(s,p,e,i,t,i,now,0)
                    s.recovery[e,i,t] = 0
                    if not fresh:
                        s.report_counts[e,2] += 1
                else:
                    s.recovery[e,i,t] = 1
                    if fresh:
                        s.report_counts[e,1] += 1


@wp.kernel
def sense_reports(s: WorldState, p: WorldParams):
    sense_reports_at(s,p,wp.tid())


@wp.kernel
def masked_sense_reports(s: WorldState, p: WorldParams, mask: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] != 0:
        sense_reports_at(s,p,e)


@wp.kernel
def end_step(s: WorldState, p: WorldParams):
    e = wp.tid()
    s.steps[e] += 1
    expire(s,p,e,wp.float64(s.steps[e])*p.dt)
    found = int(0)
    delivered = int(0)
    live = int(0)
    flight = wp.float64(0)
    comm = wp.float64(0)
    for t in range(p.targets_count):
        known = int(0)
        for i in range(p.n):
            known = wp.max(known,s.known[e,i,t])
        found += known
        delivered += s.delivered[e,t]
    for i in range(p.n):
        flight += s.flight_energy[e,i]
        comm += s.communication_energy[e,i]
        if active(s,p,e,i):
            live += 1
        else:
            s.velocities[e,i] = wp.vec2d(wp.float64(0),wp.float64(0))
            if p.xyz!=0:s.vertical_velocity[e,i]=wp.float64(0)
        pending = int(0)
        for packet in range(p.packet_capacity):
            if (s.pkt_source[e,packet] == i
                and (s.pkt_status[e,packet] == 1 or s.pkt_status[e,packet] == 2)):
                pending += 1
        s.inflight[e,i] = wp.float64(pending)
    c0 = p.reward_discovery*(wp.float64(found)-s.before[e,0])
    c1 = p.reward_delivery*(wp.float64(delivered)-s.before[e,1])
    c2 = -p.reward_blocked*(wp.float64(s.blocked[e])-s.before[e,4])
    c3 = -p.reward_expired*(wp.float64(s.report_counts[e,0])-s.before[e,2])
    c4 = -p.reward_dropped*(wp.float64(s.report_counts[e,1])-s.before[e,3])
    c5 = -p.reward_step
    c6 = -p.reward_communication*(comm-s.before[e,6])
    c7 = -p.reward_propulsion*(flight-s.before[e,5])/wp.float64(1000)
    c8 = -p.reward_communication_efficiency*(comm-s.before[e,6])/p.report_reference_tx_energy
    c9 = -p.reward_shield_correction*s.shield_correction_step[e]
    terminal = delivered == p.targets_count or live == 0 or s.steps[e] >= p.max_steps
    old_investigation=wp.float64(s.reward[e]);next_belief=s.belief_potential[e];next_investigation=investigation_potential(s,p,e)
    if terminal:
        next_belief=wp.float64(0);next_investigation=wp.float64(0)
    c10=wp.float64(0);c11=wp.float64(0)
    if p.xyz!=0:
        c10=p.reward_information_gain*(p.reward_shaping_gamma*next_belief-s.before[e,7])
        c11=p.reward_investigation*(p.reward_shaping_gamma*next_investigation-old_investigation)
    s.reward[e] = wp.float32(c0+c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11)
    s.reward_components[e,0] += c0
    s.reward_components[e,1] += c1
    s.reward_components[e,2] += c2
    s.reward_components[e,3] += c3
    s.reward_components[e,4] += c4
    s.reward_components[e,5] += c5
    s.reward_components[e,6] += c6
    s.reward_components[e,7] += c7
    s.reward_components[e,8] += c8
    s.reward_components[e,9] += c9
    s.reward_components[e,10] += c10
    s.reward_components[e,11] += c11
    s.terminated[e] = int(terminal)
