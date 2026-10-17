"""Native actor observation and privileged critic layout written on device."""
import warp as wp

from .state import WorldParams, WorldState
from ...communication.gpu.phy import xy, nominal_required_power
from ...communication.gpu.power import power_fraction

# Keep native NumPy operation rounding; these simulator kernels need no gradients.
wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.func
def _active(s: WorldState, p: WorldParams, e: int, i: int):
    return p.energy_enabled == 0 or s.battery[e, i] > wp.float64(0.0)


@wp.func
def _report_count(s: WorldState, p: WorldParams, e: int, i: int):
    count = int(0)
    for t in range(p.targets_count):
        count = count + s.report_present[e, i, t]
    return count


@wp.func
def report_age(s: WorldState, p: WorldParams, e: int, i: int, t: int):
    age = wp.float64(0.0)
    if p.recovery_enabled != 0 and s.recovery[e, i, t] != 0:
        age = wp.float64(-1.0)
    if s.report_present[e, i, t] != 0:
        age = wp.max(wp.float64(0.0), wp.float64(1.0)
                     - (wp.float64(s.steps[e]) * p.dt - s.report_created[e, i, t]) / p.report_ttl)
    return age


@wp.func
def _admissible(s: WorldState, p: WorldParams, e: int, i: int, destination: int):
    available = bool(False)
    receiver_space = True
    if destination < p.n:
        receiver_space = (_report_count(s, p, e, destination) + 1) * p.report_bytes <= p.buffer_bytes
    for t in range(p.targets_count):
        if (s.report_present[e, i, t] != 0
                and wp.float64(s.steps[e]) * p.dt - s.report_created[e, i, t] < p.report_ttl):
            if destination == p.n:
                available = True
            elif receiver_space and s.known[e, destination, t] == 0:
                available = True
    return available


@wp.func
def destination_allowed(s: WorldState, p: WorldParams, e: int, i: int, destination: int):
    allowed = (p.network_enabled != 0 and _active(s, p, e, i)
               and _report_count(s, p, e, i) > 0)
    if destination < p.n:
        allowed = (allowed and _active(s, p, e, destination)
                   and wp.length(s.positions[e, destination] - s.positions[e, i]) <= p.peer_range)
    else:
        allowed = allowed and wp.length(s.gcs[e] - s.positions[e, i]) <= p.gcs_range
    if p.power_limited_admission != 0:
        allowed = allowed and nominal_required_power(s,p,e,i,destination) <= p.tx_power_max
    if p.admission_mask != 0:
        allowed = allowed and _admissible(s, p, e, i, destination)
    return allowed


@wp.kernel
def observe(s: WorldState, p: WorldParams):
    e, i = wp.tid()
    for column in range(s.observation.shape[2]):
        s.observation[e, i, column] = wp.float32(0.0)
    position, velocity = s.positions[e, i], s.velocities[e, i]
    active = _active(s, p, e, i)
    for axis in range(2):
        s.observation[e, i, axis] = wp.float32(position[axis] / p.map_size)
        s.observation[e, i, 2 + axis] = wp.float32(velocity[axis] / p.max_speed)
        s.observation[e, i, 4 + axis] = wp.float32((s.gcs[e][axis] - position[axis]) / p.map_size)
    s.observation[e, i, 6] = wp.float32(wp.float64(s.steps[e]) / wp.float64(p.max_steps))
    s.observation[e, i, 7] = wp.float32(wp.float64(_report_count(s, p, e, i) * p.report_bytes) / wp.float64(p.buffer_bytes))
    s.observation[e, i, 8] = wp.float32(s.battery[e, i] / p.capacity)
    s.observation[e, i, 9] = wp.float32(int(active))
    for t in range(p.targets_count):
        s.observation[e, i, 10 + t] = wp.float32(s.known[e, i, t])
        s.observation[e, i, 10 + p.targets_count + t] = wp.float32(report_age(s, p, e, i, t))
    offset = 10 + 2 * p.targets_count
    # Rank visible items by distance and ID; stable ties match Python's sort.
    for t in range(p.targets_count):
        if active and s.visible[e, i, t] != 0:
            distance = wp.length(s.targets[e, t] - position)
            rank = int(0)
            for other in range(p.targets_count):
                if s.visible[e, i, other] != 0:
                    other_distance = wp.length(s.targets[e, other] - position)
                    if other_distance < distance or (other_distance == distance and other < t):
                        rank = rank + 1
            if rank < p.target_slots:
                relative = (s.targets[e, t] - position) / p.map_size
                s.observation[e, i, offset + 4 * rank] = wp.float32(relative[0])
                s.observation[e, i, offset + 4 * rank + 1] = wp.float32(relative[1])
                s.observation[e, i, offset + 4 * rank + 2] = wp.float32(s.known[e, i, t])
                s.observation[e, i, offset + 4 * rank + 3] = wp.float32(1.0)
    offset = offset + 4 * p.target_slots
    for o in range(p.obstacles_count):
        circle = s.obstacles[e, o]
        distance = wp.length(wp.vec2d(circle[0], circle[1]) - position)
        if active and distance <= p.footprint + circle[2]:
            rank = int(0)
            for other in range(p.obstacles_count):
                other_circle = s.obstacles[e, other]
                other_distance = wp.length(wp.vec2d(other_circle[0], other_circle[1]) - position)
                if other_distance <= p.footprint + other_circle[2]:
                    if other_distance < distance or (other_distance == distance and other < o):
                        rank = rank + 1
            if rank < p.obstacle_slots:
                relative = (wp.vec2d(circle[0], circle[1]) - position) / p.map_size
                s.observation[e, i, offset + 4 * rank] = wp.float32(relative[0])
                s.observation[e, i, offset + 4 * rank + 1] = wp.float32(relative[1])
                s.observation[e, i, offset + 4 * rank + 2] = wp.float32(circle[2] / p.map_size)
                s.observation[e, i, offset + 4 * rank + 3] = wp.float32(1.0)
    offset = offset + 4 * p.obstacle_slots
    slot = int(0)
    for j in range(p.n):
        if j != i:
            if active and wp.length(s.positions[e, j] - position) <= p.peer_range:
                relative = (s.positions[e, j] - position) / p.map_size
                for axis in range(2):
                    s.observation[e, i, offset + 6 * slot + axis] = wp.float32(relative[axis])
                    s.observation[e, i, offset + 6 * slot + 2 + axis] = wp.float32(s.velocities[e, j][axis] / p.max_speed)
                s.observation[e, i, offset + 6 * slot + 4] = wp.float32(s.battery[e, j] / p.capacity)
                s.observation[e, i, offset + 6 * slot + 5] = wp.float32(int(_active(s, p, e, j)))
            slot = slot + 1
    offset = offset + 6 * (p.n - 1)
    if p.link_state_observation != 0:
        source_gcs = wp.length(s.positions[e,i]-s.gcs[e])
        diagonal = wp.sqrt(wp.float64(2))*p.map_size
        slot = int(0)
        for j in range(p.n):
            if j != i:
                required = nominal_required_power(s,p,e,i,j)
                progress = wp.clamp((source_gcs-wp.length(s.positions[e,j]-s.gcs[e]))/diagonal,
                                    wp.float64(-1),wp.float64(1))
                free = wp.clamp((wp.float64(p.buffer_bytes)-wp.float64(_report_count(s,p,e,j)*p.report_bytes))
                                /wp.float64(p.buffer_bytes),wp.float64(0),wp.float64(1))
                s.observation[e,i,offset+3*slot] = wp.float32(power_fraction(p,required))
                s.observation[e,i,offset+3*slot+1] = wp.float32(progress)
                s.observation[e,i,offset+3*slot+2] = wp.float32(free)
                slot += 1
        required = nominal_required_power(s,p,e,i,p.n)
        s.observation[e,i,offset+3*slot] = wp.float32(power_fraction(p,required))
        s.observation[e,i,offset+3*slot+1] = wp.float32(wp.clamp(source_gcs/diagonal,wp.float64(-1),wp.float64(1)))
        s.observation[e,i,offset+3*slot+2] = wp.float32(1.0)
        offset = offset + 3*p.n
    s.observation[e, i, offset] = wp.float32(1.0)
    slot = int(0)
    for j in range(p.n):
        if j != i:
            s.observation[e, i, offset + slot + 1] = wp.float32(int(destination_allowed(s, p, e, i, j)))
            slot = slot + 1
    s.observation[e, i, offset + p.n] = wp.float32(int(destination_allowed(s, p, e, i, p.n)))


@wp.kernel
def centralized(s: WorldState, p: WorldParams):
    e = wp.tid()
    for i in range(p.n):
        for axis in range(2):
            s.state[e, 2 * i + axis] = wp.float32(s.positions[e, i][axis] / p.map_size)
            s.state[e, 2 * p.n + 2 * i + axis] = wp.float32(s.velocities[e, i][axis] / p.max_speed)
    offset = 4 * p.n
    for t in range(p.targets_count):
        for axis in range(2):
            s.state[e, offset + 2 * t + axis] = wp.float32(s.targets[e, t][axis] / p.map_size)
    offset = offset + 2 * p.targets_count
    for o in range(p.obstacles_count):
        for axis in range(3):
            s.state[e, offset + 3 * o + axis] = wp.float32(s.obstacles[e, o][axis] / p.map_size)
    offset = offset + 3 * p.obstacles_count
    for i in range(p.n):
        for t in range(p.targets_count):
            s.state[e, offset + i * p.targets_count + t] = wp.float32(s.known[e, i, t])
            s.state[e, offset + p.n * p.targets_count + i * p.targets_count + t] = wp.float32(report_age(s, p, e, i, t))
    offset = offset + 2 * p.n * p.targets_count
    for t in range(p.targets_count):
        s.state[e, offset + t] = wp.float32(s.delivered[e, t])
    offset = offset + p.targets_count
    for i in range(p.n):
        s.state[e, offset + i] = wp.float32(s.inflight[e, i] / wp.float64(p.max_queue_packets))
        s.state[e, offset + p.n + i] = wp.float32(s.battery[e, i] / p.capacity)
        s.state[e, offset + 2 * p.n + i] = wp.float32(int(_active(s, p, e, i)))
    s.state[e, offset + 3 * p.n] = wp.float32(wp.float64(s.steps[e]) / wp.float64(p.max_steps))
