"""Native polygon LOS, cached A2A gain, airtime interference and battery budget."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState
from .reports import active

PI = wp.constant(wp.float64(3.141592653589793))


@wp.func
def xy(s: WorldState, p: WorldParams, e: int, node: int) -> wp.vec2d:
    point = wp.vec2d(s.gcs[e])
    if node < p.n:
        point = wp.vec2d(s.positions[e,node])
    return point


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
        or now-s.pair_time[e,index] >= wp.float64(100000)
        or wp.length(a-s.pair_transmitter[e,index]) >= wp.float64(1)
        or wp.length(b-s.pair_receiver[e,index]) >= wp.float64(1)):
        dz = wp.float64(0)
        if (transmitter == p.n) != (receiver == p.n):
            dz = p.altitude
        distance = wp.max(wp.float64(1),wp.sqrt(wp.dot(a-b,a-b)+dz*dz))
        reference = (wp.float64(299792458)/p.frequency/(wp.float64(4)*PI))**wp.float64(2)
        result = reference*distance**wp.float64(-2)
        if not los(s,p,e,a,b):
            result = reference*distance**wp.float64(-3)*p.nlos_scale
        s.pair_gain[e,index] = result
        s.pair_time[e,index] = now
        s.pair_transmitter[e,index] = a
        s.pair_receiver[e,index] = b
    return s.pair_gain[e,index]


@wp.func
def busy(s: WorldState, p: WorldParams, e: int, node: int, now: wp.float64) -> bool:
    power = wp.float64(0)
    for k in range(s.frame_count[e]):
        j = s.frame_order[e,k]
        if (s.tx_source[e,j] != node
            and s.tx_start[e,j] <= now and now < s.tx_end[e,j]):
            power += s.tx_power[e,j]*gain(s,p,e,s.tx_source[e,j],node,now)
    if power <= wp.float64(0):
        return wp.float64(-200) >= p.cca_threshold
    return wp.float64(10)*wp.log10(power*wp.float64(1000)) >= p.cca_threshold


@wp.func
def emit(s: WorldState, p: WorldParams, e: int, packet: int, source: int, destination: int,
         kind: int, now: wp.float64) -> int:
    power = s.pkt_power[e,packet]
    bits = wp.float64(wp.min(p.payload_bytes,p.report_bytes-s.pkt_fragment[e,packet]*p.payload_bytes)*8)+p.header_bits
    if kind == 1:
        bits = p.ack_bits
        if p.ack_uses_data_power == 0:
            power = p.tx_power_min
    cost = bits/p.bit_rate*power
    if not active(s,p,e,source) or (source < p.n and p.energy_enabled != 0 and cost > s.battery[e,source]):
        s.packet_counts[e,5] += 1
        return -1
    s.communication_energy[e,source] += cost
    if source < p.n and p.energy_enabled != 0:
        s.battery[e,source] = wp.max(wp.float64(0),s.battery[e,source]-cost)
    retained = int(0)
    for k in range(s.frame_count[e]):
        j = s.frame_order[e,k]
        if s.tx_end[e,j] >= now-p.retention:
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
    duration = s.tx_end[e,frame]-s.tx_start[e,frame]
    signal = s.tx_power[e,frame]*gain(s,p,e,s.tx_source[e,frame],receiver,now)
    interference = wp.float64(0)
    collided = bool(False)
    for k in range(s.frame_count[e]):
        j = s.frame_order[e,k]
        if j != frame:
            overlap = wp.max(wp.float64(0),wp.min(s.tx_end[e,frame],s.tx_end[e,j])
                             -wp.max(s.tx_start[e,frame],s.tx_start[e,j]))
            if overlap > wp.float64(0):
                collided = True
                interference += s.tx_power[e,j]*gain(s,p,e,s.tx_source[e,j],receiver,now)*(overlap/duration)
    sinr = wp.float64(-200)
    if signal > wp.float64(0):
        sinr = wp.float64(10)*wp.log10(signal/(p.noise_power+interference))
    success = sinr >= p.sinr_threshold
    if success:
        s.packet_counts[e,2] += 1
    else:
        s.packet_counts[e,3] += 1
        if collided:
            s.packet_counts[e,4] += 1
    return success
