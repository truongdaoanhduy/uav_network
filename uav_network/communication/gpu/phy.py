"""Native polygon LOS, cached A2A gain, airtime interference and battery budget."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState
from .reports import active
from .frame_free import (apply_reclaims_and_take_smallest, mark_reclaim_if_ready, set_free_bit)
from . import diagnostics as diag

PI = wp.constant(wp.float64(3.141592653589793))
C = wp.constant(wp.float64(299792458.0))


@wp.func
def xy(s: WorldState, p: WorldParams, e: int, node: int) -> wp.vec2d:
    point = wp.vec2d(s.gcs[e])
    if node < p.n:
        point = wp.vec2d(s.positions[e,node])
    return point


@wp.func
def link_distance(s: WorldState, p: WorldParams, e: int, first: int, second: int) -> wp.float64:
    a = xy(s,p,e,first)
    b = xy(s,p,e,second)
    dz = wp.float64(0)
    if (first == p.n) != (second == p.n):
        dz = p.altitude
    return wp.sqrt(wp.dot(a-b,a-b)+dz*dz)


@wp.func
def cross(a: wp.vec2d, b: wp.vec2d) -> wp.float64:
    return a[0]*b[1]-a[1]*b[0]


@wp.func
def los(s: WorldState, p: WorldParams, e: int, a: wp.vec2d, b: wp.vec2d) -> bool:
    d = b-a
    for o in range(p.obstacles_count):
        obstacle = s.obstacles[e,o]
        center = wp.vec2d(obstacle[0],obstacle[1])
        radius = obstacle[2]
        low = wp.float64(0)
        high = wp.float64(1)
        intersects = bool(True)
        for j in range(p.polygon_sides):
            angle = wp.float64(j)*wp.float64(2)*PI/wp.float64(p.polygon_sides)
            next_angle = wp.float64(j+1)*wp.float64(2)*PI/wp.float64(p.polygon_sides)
            start = center+radius*wp.vec2d(wp.cos(angle),wp.sin(angle))
            end = center+radius*wp.vec2d(wp.cos(next_angle),wp.sin(next_angle))
            f = cross(end-start,a-start)
            slope = cross(end-start,d)
            if slope == wp.float64(0):
                if f < wp.float64(0):
                    intersects = False
            elif slope > wp.float64(0):
                low = wp.max(low,-f/slope)
            else:
                high = wp.min(high,-f/slope)
        if intersects and low <= high:
            return False
    return True


@wp.func
def gain(s: WorldState, p: WorldParams, e: int, transmitter: int, receiver: int,
         now: wp.float64) -> wp.float64:
    if transmitter == receiver:
        return wp.float64(1)
    a = xy(s,p,e,transmitter)
    b = xy(s,p,e,receiver)
    index = transmitter*(p.n+1)+receiver
    if (s.pair_time[e,index] < wp.float64(0)
        or wp.length(a-s.pair_transmitter[e,index]) >= wp.float64(1)
        or wp.length(b-s.pair_receiver[e,index]) >= wp.float64(1)):
        distance = wp.max(wp.float64(1),link_distance(s,p,e,transmitter,receiver))
        reference = (C/p.frequency/(wp.float64(4)*PI))**wp.float64(2)
        result = reference*distance**wp.float64(-2)
        if not los(s,p,e,a,b):
            result = reference*distance**(-p.nlos_exponent)*p.nlos_scale
        s.pair_gain[e,index] = result
        s.pair_time[e,index] = now
        s.pair_transmitter[e,index] = a
        s.pair_receiver[e,index] = b
    return s.pair_gain[e,index]



@wp.func
def nominal_required_power(s: WorldState, p: WorldParams, e: int, transmitter: int, receiver: int) -> wp.float64:
    a = xy(s,p,e,transmitter)
    b = xy(s,p,e,receiver)
    distance = wp.max(wp.float64(1), link_distance(s,p,e,transmitter,receiver))
    reference = (C/p.frequency/(wp.float64(4)*PI))**wp.float64(2)
    nominal = reference*distance**wp.float64(-2)
    if not los(s,p,e,a,b):
        nominal = reference*distance**(-p.nlos_exponent)*p.nlos_scale
    threshold = wp.pow(wp.float64(10), p.sinr_threshold/wp.float64(10))
    return threshold*p.noise_power/nominal

@wp.func
def propagation_delay_us(s: WorldState, p: WorldParams, e: int, frame: int, node: int) -> wp.float64:
    if p.propagation_delay_enabled != 0:
        return s.tx_propagation_delay[e,frame,node]
    return wp.float64(0)


@wp.func
def arrival_start(s: WorldState, p: WorldParams, e: int, frame: int, node: int) -> wp.float64:
    return s.tx_start[e,frame]+propagation_delay_us(s,p,e,frame,node)


@wp.func
def arrival_end(s: WorldState, p: WorldParams, e: int, frame: int, node: int) -> wp.float64:
    return s.tx_end[e,frame]+propagation_delay_us(s,p,e,frame,node)


@wp.func
def latest_arrival_end(s: WorldState, p: WorldParams, e: int, frame: int) -> wp.float64:
    result = s.tx_end[e,frame]
    if p.propagation_delay_enabled != 0:
        for node in range(p.n+1):
            result = wp.max(result,s.tx_end[e,frame]+s.tx_propagation_delay[e,frame,node])
    return result


@wp.func
def busy(s: WorldState, p: WorldParams, e: int, node: int, now: wp.float64) -> bool:
    s.diagnostic_counts[e,diag.CCA_CHECKS] += wp.int64(1)
    power = wp.float64(0)
    for k in range(s.frame_count[e]):
        j = s.frame_order[e,k]
        if (s.tx_source[e,j] != node
            and arrival_start(s,p,e,j,node) <= now and now < arrival_end(s,p,e,j,node)):
            power += s.tx_power[e,j]*gain(s,p,e,s.tx_source[e,j],node,now)
    if power <= wp.float64(0):
        return False
    result = wp.float64(10)*wp.log10(power*wp.float64(1000)) >= p.cca_threshold
    if result:
        s.diagnostic_counts[e,diag.CCA_BUSY] += wp.int64(1)
    return result


@wp.func
def emit(s: WorldState, p: WorldParams, e: int, packet: int, source: int, destination: int,
         kind: int, now: wp.float64) -> int:
    power = s.pkt_power[e,packet]
    bits = wp.float64(wp.min(p.payload_bytes,p.report_bytes-s.pkt_fragment[e,packet]*p.payload_bytes)*8)+p.header_bits
    if kind == 1:
        bits = p.ack_bits
        if p.ack_uses_data_power == 0:
            power = p.tx_power_min
    airtime = bits/p.bit_rate
    cost = airtime*(p.tx_circuit_power + power/p.pa_efficiency)
    if not active(s,p,e,source) or (source < p.n and p.energy_enabled != 0 and cost > s.battery[e,source]):
        s.packet_counts[e,5] += 1
        s.diagnostic_counts[e,diag.ENERGY_DENIALS] += wp.int64(1)
        count = diag.DATA_ENERGY_DENIALS
        if kind == 1:
            count = diag.ACK_ENERGY_DENIALS
        s.diagnostic_counts[e,count] += wp.int64(1)
        return -1
    s.communication_energy[e,source] += cost
    s.tx_energy[e,source] += cost
    if source < p.n and p.energy_enabled != 0:
        s.battery[e,source] = wp.max(wp.float64(0),s.battery[e,source]-cost)
    retained = int(0)
    for k in range(s.frame_count[e]):
        j = s.frame_order[e,k]
        if latest_arrival_end(s,p,e,j) >= now-p.retention:
            s.frame_order[e,retained] = j
            retained += 1
        else:
            # PHY retention and the serialized receiver inbox have different lifetimes.
            s.tx_active[e,j] = 0
            if s.tx_receive[e,j] == 1 or s.tx_receive[e,j] == 2:
                s.tx_active[e,j] = 2
    s.frame_count[e] = retained
    frame = int(-1)
    for j in range(p.frame_capacity):
        if s.tx_active[e,j] == 2 and s.tx_receive[e,j] == 3:
            s.tx_active[e,j] = 0
        if frame < 0 and s.tx_active[e,j] == 0:
            frame = j
    if frame < 0:
        s.errors[e] = 3
        return -1
    diag.emission(s,p,e,packet,destination,kind,power)
    if kind == 0:
        required = nominal_required_power(s,p,e,source,destination)
        diag.sample(s,e,diag.STAT_DATA_REQUIRED_POWER_W,required)
        diag.sample(s,e,diag.STAT_DATA_POWER_MARGIN_DB,wp.float64(10)*wp.log10(power/required))
        if power < required:
            s.diagnostic_counts[e,diag.DATA_BELOW_REQUIRED_POWER_ATTEMPTS] += wp.int64(1)
    s.frame_order[e,s.frame_count[e]] = frame
    s.frame_count[e] += 1
    s.tx_active[e,frame] = 1
    s.tx_kind[e,frame] = kind
    s.tx_packet[e,frame] = packet
    s.tx_generation[e,frame] = s.pkt_generation[e,packet]
    s.tx_source[e,frame] = source
    s.tx_destination[e,frame] = destination
    s.tx_target[e,frame] = s.pkt_target[e,packet]
    s.tx_fragment[e,frame] = s.pkt_fragment[e,packet]
    s.tx_payload[e,frame] = wp.min(p.payload_bytes,p.report_bytes-s.pkt_fragment[e,packet]*p.payload_bytes)
    s.tx_ttl[e,frame] = s.pkt_ttl[e,packet]
    if kind == 1:
        s.tx_ttl[e,frame] = 1
    s.tx_origin[e,frame] = s.pkt_origin[e,packet]
    s.tx_hops[e,frame] = s.pkt_hops[e,packet]
    s.tx_session[e,frame] = s.pkt_session_id[e,packet]
    s.tx_id[e,frame] = s.pkt_id[e,packet]
    s.tx_power[e,frame] = power
    s.tx_start[e,frame] = now
    s.tx_end[e,frame] = now+bits/p.bit_rate*wp.float64(1e6)
    if p.propagation_delay_enabled != 0:
        for node in range(p.n+1):
            s.tx_propagation_delay[e,frame,node] = link_distance(s,p,e,source,node)/C*wp.float64(1e6)
    s.tx_created[e,frame] = s.pkt_created[e,packet]
    s.tx_report_created[e,frame] = s.pkt_report_created[e,packet]
    s.tx_receive[e,frame] = 0
    if p.trace_enabled != 0:
        trace = s.trace_count[e]
        if trace >= p.trace_capacity:
            s.errors[e] = 4
        else:
            s.trace_time[e,trace] = now
            s.trace_source[e,trace] = source
            s.trace_destination[e,trace] = destination
            s.trace_kind[e,trace] = kind
            s.trace_report[e,trace,4] = power
            s.trace_count[e] += 1
    return frame


@wp.func
def emit_framefree(s: WorldState, p: WorldParams, e: int, packet: int, source: int, destination: int,
                   kind: int, now: wp.float64,
                   free_words: wp.array2d(dtype=wp.uint64),
                   reclaim_words: wp.array2d(dtype=wp.uint64)) -> int:
    """Gen1 emission semantics with derived bitmap allocation only."""
    power = s.pkt_power[e,packet]
    bits = wp.float64(wp.min(p.payload_bytes,p.report_bytes-s.pkt_fragment[e,packet]*p.payload_bytes)*8)+p.header_bits
    if kind == 1:
        bits = p.ack_bits
        if p.ack_uses_data_power == 0:
            power = p.tx_power_min
    airtime = bits/p.bit_rate
    cost = airtime*(p.tx_circuit_power + power/p.pa_efficiency)
    if not active(s,p,e,source) or (source < p.n and p.energy_enabled != 0 and cost > s.battery[e,source]):
        s.packet_counts[e,5] += 1
        s.diagnostic_counts[e,diag.ENERGY_DENIALS] += wp.int64(1)
        count = diag.DATA_ENERGY_DENIALS
        if kind == 1:
            count = diag.ACK_ENERGY_DENIALS
        s.diagnostic_counts[e,count] += wp.int64(1)
        return -1
    s.communication_energy[e,source] += cost
    s.tx_energy[e,source] += cost
    if source < p.n and p.energy_enabled != 0:
        s.battery[e,source] = wp.max(wp.float64(0),s.battery[e,source]-cost)
    retained = int(0)
    for k in range(s.frame_count[e]):
        j = s.frame_order[e,k]
        if latest_arrival_end(s,p,e,j) >= now-p.retention:
            s.frame_order[e,retained] = j
            retained += 1
        else:
            s.tx_active[e,j] = 0
            if s.tx_receive[e,j] == 1 or s.tx_receive[e,j] == 2:
                s.tx_active[e,j] = 2
                mark_reclaim_if_ready(s,e,j,reclaim_words)
            else:
                set_free_bit(free_words,e,j)
    s.frame_count[e] = retained
    frame = apply_reclaims_and_take_smallest(s,p,e,free_words,reclaim_words)
    if frame < 0:
        s.errors[e] = 3
        return -1
    diag.emission(s,p,e,packet,destination,kind,power)
    if kind == 0:
        required = nominal_required_power(s,p,e,source,destination)
        diag.sample(s,e,diag.STAT_DATA_REQUIRED_POWER_W,required)
        diag.sample(s,e,diag.STAT_DATA_POWER_MARGIN_DB,wp.float64(10)*wp.log10(power/required))
        if power < required:
            s.diagnostic_counts[e,diag.DATA_BELOW_REQUIRED_POWER_ATTEMPTS] += wp.int64(1)
    s.frame_order[e,s.frame_count[e]] = frame
    s.frame_count[e] += 1
    s.tx_active[e,frame] = 1
    s.tx_kind[e,frame] = kind
    s.tx_packet[e,frame] = packet
    s.tx_generation[e,frame] = s.pkt_generation[e,packet]
    s.tx_source[e,frame] = source
    s.tx_destination[e,frame] = destination
    s.tx_target[e,frame] = s.pkt_target[e,packet]
    s.tx_fragment[e,frame] = s.pkt_fragment[e,packet]
    s.tx_payload[e,frame] = wp.min(p.payload_bytes,p.report_bytes-s.pkt_fragment[e,packet]*p.payload_bytes)
    s.tx_ttl[e,frame] = s.pkt_ttl[e,packet]
    if kind == 1:
        s.tx_ttl[e,frame] = 1
    s.tx_origin[e,frame] = s.pkt_origin[e,packet]
    s.tx_hops[e,frame] = s.pkt_hops[e,packet]
    s.tx_session[e,frame] = s.pkt_session_id[e,packet]
    s.tx_id[e,frame] = s.pkt_id[e,packet]
    s.tx_power[e,frame] = power
    s.tx_start[e,frame] = now
    s.tx_end[e,frame] = now+bits/p.bit_rate*wp.float64(1e6)
    if p.propagation_delay_enabled != 0:
        for node in range(p.n+1):
            s.tx_propagation_delay[e,frame,node] = link_distance(s,p,e,source,node)/C*wp.float64(1e6)
    s.tx_created[e,frame] = s.pkt_created[e,packet]
    s.tx_report_created[e,frame] = s.pkt_report_created[e,packet]
    s.tx_receive[e,frame] = 0
    if p.trace_enabled != 0:
        trace = s.trace_count[e]
        if trace >= p.trace_capacity:
            s.errors[e] = 4
        else:
            s.trace_time[e,trace] = now
            s.trace_source[e,trace] = source
            s.trace_destination[e,trace] = destination
            s.trace_kind[e,trace] = kind
            s.trace_report[e,trace,4] = power
            s.trace_count[e] += 1
    return frame


@wp.func
def evaluate(s: WorldState, p: WorldParams, e: int, frame: int, now: wp.float64) -> bool:
    receiver = s.tx_destination[e,frame]
    start = arrival_start(s,p,e,frame,receiver)
    end = arrival_end(s,p,e,frame,receiver)
    duration = end-start
    signal = s.tx_power[e,frame]*gain(s,p,e,s.tx_source[e,frame],receiver,now)
    interference = wp.float64(0)
    collided = bool(False)
    for k in range(s.frame_count[e]):
        j = s.frame_order[e,k]
        if j != frame:
            overlap = wp.max(wp.float64(0),wp.min(end,arrival_end(s,p,e,j,receiver))
                             -wp.max(start,arrival_start(s,p,e,j,receiver)))
            if overlap > wp.float64(0):
                collided = True
                interference += s.tx_power[e,j]*gain(s,p,e,s.tx_source[e,j],receiver,now)*(overlap/duration)
    sinr = wp.float64(-200)
    if signal > wp.float64(0):
        sinr = wp.float64(10)*wp.log10(signal/(p.noise_power+interference))
    success = sinr >= p.sinr_threshold
    rx_cost = duration/wp.float64(1e6)*p.rx_circuit_power
    if rx_cost > wp.float64(0) and active(s,p,e,receiver):
        consumed = rx_cost
        affordable = bool(True)
        if receiver < p.n and p.energy_enabled != 0:
            if rx_cost > s.battery[e,receiver]:
                consumed = s.battery[e,receiver]
                affordable = False
            s.battery[e,receiver] = wp.max(wp.float64(0),s.battery[e,receiver]-consumed)
        s.communication_energy[e,receiver] += consumed
        s.rx_energy[e,receiver] += consumed
        if not affordable:
            success = False
    snr = wp.float64(-200)
    rssi = wp.float64(-200)
    if signal > wp.float64(0):
        snr = wp.float64(10)*wp.log10(signal/p.noise_power)
        rssi = wp.float64(10)*wp.log10(signal*wp.float64(1000))
    pair = s.tx_source[e,frame]*(p.n+1)+receiver
    # Classify the cached gain's geometry, matching native A2A snapshot LOS.
    line_of_sight = los(s,p,e,s.pair_transmitter[e,pair],s.pair_receiver[e,pair])
    diag.decode(s,p,e,s.tx_kind[e,frame],receiver,snr,sinr,rssi,
                link_distance(s,p,e,s.tx_source[e,frame],receiver),line_of_sight,collided,success)
    if success:
        s.packet_counts[e,2] += 1
    else:
        s.packet_counts[e,3] += 1
        if collided:
            s.packet_counts[e,4] += 1
    return success
