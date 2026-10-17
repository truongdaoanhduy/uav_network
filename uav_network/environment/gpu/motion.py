"""Ordered dynamic-window reservations and funded FP64 motion on device."""
import warp as wp

from .geometry import (brake, capped, outside_map, peer_path, segment_distance,
                       static_path, static_point)
from .state import WorldParams, WorldState

# Keep native NumPy operation rounding; these simulator kernels need no gradients.
wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.func
def _clip_box(value: wp.vec2d, lower: wp.vec2d, upper: wp.vec2d):
    return wp.vec2d(wp.clamp(value[0], lower[0], upper[0]),
                    wp.clamp(value[1], lower[1], upper[1]))


@wp.func
def reachable(desired: wp.vec2d, previous: wp.vec2d, p: WorldParams):
    step = p.max_accel * p.dt
    lower, upper = previous - wp.vec2d(step, step), previous + wp.vec2d(step, step)
    result = _clip_box(desired, lower, upper)
    if wp.length(result) <= p.max_speed:
        return result
    if wp.length(_clip_box(wp.vec2d(0.0, 0.0), lower, upper)) > p.max_speed:
        return capped(result, p.max_speed)
    low, high = wp.float64(0.0), wp.float64(1.0)
    while wp.length(_clip_box(desired / (wp.float64(1.0) + high), lower, upper)) > p.max_speed:
        high = high * wp.float64(2.0)
    for k in range(40):
        middle = (low + high) / wp.float64(2.0)
        candidate = _clip_box(desired / (wp.float64(1.0) + middle), lower, upper)
        if wp.length(candidate) > p.max_speed:
            low = middle
        else:
            high = middle
    return _clip_box(desired / (wp.float64(1.0) + high), lower, upper)


@wp.func
def _reachable(candidate: wp.vec2d, previous: wp.vec2d, p: WorldParams):
    step = p.max_accel * p.dt + wp.float64(1e-9)
    return wp.abs(candidate[0] - previous[0]) <= step and wp.abs(candidate[1] - previous[1]) <= step


@wp.func
def _others_safe(s: WorldState, p: WorldParams, e: int, i: int, velocity: wp.vec2d):
    valid = bool(True)
    for j in range(p.n):
        if j != i and not peer_path(p, s.positions[e, i], velocity,
                                    s.positions[e, j], s.motion_work[e, j, 1]):
            valid = bool(False)
    return valid


@wp.func
def _tangent(normal: wp.vec2d, previous: wp.vec2d, nominal: wp.vec2d):
    unit = normal / wp.max(wp.length(normal), wp.float64(1e-12))
    tangent = wp.vec2d(-unit[1], unit[0])
    component = wp.dot(previous, tangent)
    if wp.abs(component) < wp.float64(1e-8):
        component = wp.dot(nominal, tangent)
    direction = wp.float64(1.0)
    if component < wp.float64(-1e-8):
        direction = wp.float64(-1.0)
    return tangent * direction * wp.length(nominal)


@wp.func
def _guidance(s: WorldState, p: WorldParams, e: int, i: int):
    position, previous = wp.vec2d(s.positions[e, i]), wp.vec2d(s.velocities[e, i])
    nominal = wp.vec2d(s.motion_work[e, i, 0])
    step = p.max_accel * p.dt
    horizon = int(wp.ceil(p.max_speed / step)) + 1
    nearest, nearest_distance = int(-1), wp.float64(1e300)
    for o in range(p.obstacles_count):
        circle = s.obstacles[e, o]
        center = wp.vec2d(circle[0], circle[1])
        point, speed = wp.vec2d(position), wp.vec2d(nominal)
        distance = wp.float64(1e300)
        for k in range(horizon):
            end = point + speed * p.dt
            distance = wp.min(distance, segment_distance(point, end, center))
            point, speed = end, brake(speed, step)
        gap = distance - circle[2] - p.obstacle_clearance
        current_distance = wp.length(position - center)
        if gap < wp.float64(1e-9) and current_distance < nearest_distance:
            nearest, nearest_distance = o, current_distance
    if nearest >= 0:
        circle = s.obstacles[e, nearest]
        center = wp.vec2d(circle[0], circle[1])
        guide = _tangent(position - center, previous, nominal)
        axis = int(0)
        if wp.abs(guide[1]) > wp.abs(guide[0]):
            axis = 1
        direction = wp.float64(1.0)
        if guide[axis] < wp.float64(0.0):
            direction = wp.float64(-1.0)
        radius = circle[2] + p.obstacle_clearance + wp.float64(1e-6)
        first, second = wp.vec2d(center), wp.vec2d(center)
        first[axis] = first[axis] + direction * radius
        second[axis] = second[axis] - direction * radius
        if not static_point(s, p, e, first) and static_point(s, p, e, second):
            guide = -guide
        return guide, int(0)
    point, speed = wp.vec2d(position), wp.vec2d(nominal)
    outside_x, outside_y = bool(False), bool(False)
    below_x, below_y = bool(False), bool(False)
    for k in range(horizon + 1):
        below_x = below_x or point[0] < wp.float64(0.0)
        below_y = below_y or point[1] < wp.float64(0.0)
        outside_x = outside_x or below_x or point[0] > p.map_size
        outside_y = outside_y or below_y or point[1] > p.map_size
        point, speed = point + speed * p.dt, brake(speed, step)
    if outside_x or outside_y:
        guide = wp.vec2d(nominal)
        if outside_x:
            guide[0] = wp.float64(0.0)
        if outside_y:
            guide[1] = wp.float64(0.0)
        aimed = reachable(previous + _clip_box(guide - previous,
            wp.vec2d(-step, -step), wp.vec2d(step, step)), previous, p)
        guide_safe = (static_path(s, p, e, position, aimed)
                      and _others_safe(s, p, e, i, aimed))
        if (wp.length(guide) <= wp.float64(1e-8)
                or wp.dot(guide, previous) < wp.float64(-1e-8) or not guide_safe):
            # Match the native certified tangent fallback and established side.
            stop_guide, has_stop = wp.vec2d(0.0, 0.0), bool(False)
            for axis in range(2):
                affected = (axis == 0 and outside_x) or (axis == 1 and outside_y)
                if affected:
                    normal = wp.vec2d(0.0, 0.0)
                    normal[axis] = wp.float64(1.0)
                    if (axis == 0 and below_x) or (axis == 1 and below_y):
                        normal[axis] = wp.float64(-1.0)
                    tangent = _tangent(normal, previous, nominal)
                    for reverse in range(2):
                        alternative = wp.vec2d(tangent)
                        if reverse == 1:
                            alternative = -tangent
                        aimed = reachable(previous + _clip_box(alternative - previous,
                            wp.vec2d(-step, -step), wp.vec2d(step, step)), previous, p)
                        if (static_path(s, p, e, position, aimed)
                                and _others_safe(s, p, e, i, aimed)):
                            if wp.length(aimed) > wp.float64(1e-8):
                                return alternative, int(2)
                            if not has_stop:
                                stop_guide, has_stop = alternative, bool(True)
            if has_stop:
                return stop_guide, int(2)
        return guide, int(2)
    for j in range(p.n):
        if j != i and not peer_path(p, position, nominal, s.positions[e, j], s.motion_work[e, j, 1]):
            return _tangent(position - s.positions[e, j], previous, nominal), int(1)
    return nominal, int(-1)


@wp.func
def _move(s: WorldState, p: WorldParams, e: int, actions: wp.array3d(dtype=wp.float32)):
    first_counts, shield_counts = wp.vec4i(0, 0, 0, 0), wp.vec4i(0, 0, 0, 0)
    step = p.max_accel * p.dt
    all_safe = bool(True)
    for i in range(p.n):
        previous = wp.vec2d(s.velocities[e, i])
        # NumPy scales FP32 actions before adding them to the FP64 velocity.
        action_delta = wp.vec2f(actions[e, i, 0], actions[e, i, 1]) * wp.float32(p.max_accel) * wp.float32(p.dt)
        desired = previous + wp.vec2d(action_delta)
        nominal = capped(desired, p.max_speed)
        if p.motion_safety == 1:
            nominal = reachable(desired, previous, p)
        if s.motion_flags[e, i, 0] != 0:
            nominal = wp.vec2d(0.0, 0.0)
        s.motion_work[e, i, 0] = nominal
        s.motion_work[e, i, 1] = nominal
        if p.motion_safety == 1:
            static = static_path(s, p, e, s.positions[e, i], nominal) and _reachable(nominal, previous, p)
            s.motion_flags[e, i, 1] = int(static)
            all_safe = all_safe and static
    if p.motion_safety == 1:
        for i in range(p.n):
            for j in range(i):
                if not peer_path(p, s.positions[e, i], s.motion_work[e, i, 0],
                                 s.positions[e, j], s.motion_work[e, j, 0]):
                    all_safe = bool(False)
        if not all_safe:
            for i in range(p.n):
                chosen = brake(s.velocities[e, i], step)
                if s.motion_flags[e, i, 0] != 0:
                    chosen = wp.vec2d(0.0, 0.0)
                s.motion_work[e, i, 1] = chosen
            for i in range(p.n):
                if s.motion_flags[e, i, 0] != 0:
                    continue
                nominal, previous = wp.vec2d(s.motion_work[e, i, 0]), wp.vec2d(s.velocities[e, i])
                if s.motion_flags[e, i, 1] != 0 and _others_safe(s, p, e, i, nominal):
                    s.motion_work[e, i, 1] = nominal
                    continue
                guide, kind = _guidance(s, p, e, i)
                aimed = previous + _clip_box(guide - previous, wp.vec2d(-step, -step), wp.vec2d(step, step))
                found, best_score = bool(False), wp.float64(1e300)
                chosen = wp.vec2d(s.motion_work[e, i, 1])
                # Native candidate order is aimed, nominal, brake, then X/Y grid.
                for k in range(28):
                    desired = wp.vec2d(aimed)
                    if k == 1:
                        desired = wp.vec2d(nominal)
                    elif k == 2:
                        desired = wp.vec2d(s.motion_work[e, i, 1])
                    elif k >= 3:
                        x = wp.float64((k - 3) // 5) * wp.float64(0.5) - wp.float64(1.0)
                        y = wp.float64((k - 3) % 5) * wp.float64(0.5) - wp.float64(1.0)
                        desired = previous + wp.vec2d(x, y) * step
                    candidate = reachable(desired, previous, p)
                    if (_reachable(candidate, previous, p)
                            and static_path(s, p, e, s.positions[e, i], candidate)
                            and _others_safe(s, p, e, i, candidate)):
                        delta, nominal_delta = candidate - guide, candidate - nominal
                        score = wp.dot(delta, delta) + wp.float64(0.01) * wp.dot(nominal_delta, nominal_delta)
                        if score < best_score:
                            found, best_score, chosen = bool(True), score, candidate
                if not found:
                    shield_counts[3] = shield_counts[3] + 1
                s.motion_work[e, i, 1] = chosen
                if wp.abs(chosen[0] - nominal[0]) > wp.float64(1e-9) or wp.abs(chosen[1] - nominal[1]) > wp.float64(1e-9):
                    first_counts[3] = first_counts[3] + 1
                    if kind >= 0:
                        shield_counts[kind] = shield_counts[kind] + 1
    for i in range(p.n):
        position = wp.vec2d(s.positions[e, i])
        unbounded = position + s.motion_work[e, i, 1] * p.dt
        proposed = _clip_box(unbounded, wp.vec2d(0.0, 0.0), wp.vec2d(p.map_size, p.map_size))
        if s.motion_flags[e, i, 0] != 0:
            proposed = position
        s.motion_flags[e, i, 4] = int(proposed[0] != unbounded[0] or proposed[1] != unbounded[1])
        blocked = bool(False)
        for o in range(p.obstacles_count):
            circle = s.obstacles[e, o]
            if segment_distance(position, proposed, wp.vec2d(circle[0], circle[1])) < circle[2] + p.obstacle_clearance + wp.float64(1e-9):
                blocked = bool(True)
        s.motion_flags[e, i, 2] = int(blocked)
        s.motion_flags[e, i, 3] = int(blocked)
        s.motion_work[e, i, 2] = proposed
    # Freeze resolved endpoints for each pass before setting more blocked bits.
    for iteration in range(p.n + 1):
        for i in range(p.n):
            if s.motion_flags[e, i, 2] != 0:
                s.motion_work[e, i, 2] = s.positions[e, i]
        changed = bool(False)
        for i in range(p.n):
            for j in range(i):
                if segment_distance(s.positions[e, i] - s.positions[e, j],
                                    s.motion_work[e, i, 2] - s.motion_work[e, j, 2],
                                    wp.vec2d(0.0, 0.0)) < p.safety_distance - wp.float64(1e-9):
                    if s.motion_flags[e, i, 2] == 0 or s.motion_flags[e, j, 2] == 0:
                        changed = bool(True)
                    s.motion_flags[e, i, 2] = 1
                    s.motion_flags[e, j, 2] = 1
        if not changed:
            break
    blocked_count, clipped_count = int(0), int(0)
    for i in range(p.n):
        if s.motion_flags[e, i, 2] != 0:
            s.motion_work[e, i, 2] = s.positions[e, i]
        if s.motion_flags[e, i, 0] == 0:
            blocked_count = blocked_count + s.motion_flags[e, i, 2]
            clipped_count = clipped_count + s.motion_flags[e, i, 4]
            first_counts[0] = first_counts[0] + s.motion_flags[e, i, 3]
            if s.motion_flags[e, i, 2] != 0 and s.motion_flags[e, i, 3] == 0:
                first_counts[1] = first_counts[1] + 1
            command = wp.vec2d(wp.float64(actions[e, i, 0]), wp.float64(actions[e, i, 1]))
            actual = (s.motion_work[e, i, 2] - s.positions[e, i]) / p.dt
            if wp.length(command) > wp.float64(0.1) and wp.length(actual) < wp.float64(1e-8):
                first_counts[2] = first_counts[2] + 1
    return first_counts, shield_counts, blocked_count, clipped_count


@wp.func
def _cost(previous: wp.vec2d, realized: wp.vec2d, p: WorldParams):
    acceleration = (realized - previous) / p.dt
    power = (p.idle_power + p.hover_power + p.speed_coefficient * wp.dot(realized, realized)
             + p.accel_coefficient * wp.dot(acceleration, acceleration))
    return power * p.dt


@wp.kernel
def motion_step(s: WorldState, p: WorldParams, continuous: wp.array3d(dtype=wp.float32)):
    e = wp.tid()
    if s.errors[e] != 0 or s.terminated[e] != 0:
        return
    for i in range(p.n):
        s.motion_flags[e, i, 0] = int(p.energy_enabled != 0 and s.battery[e, i] <= wp.float64(0.0))
    counts, shield = wp.vec4i(0, 0, 0, 0), wp.vec4i(0, 0, 0, 0)
    blocked_count, clipped_count = int(0), int(0)
    for iteration in range(p.n + 1):
        counts, shield, blocked_count, clipped_count = _move(s, p, e, continuous)
        unaffordable = bool(False)
        if p.energy_enabled != 0:
            for i in range(p.n):
                realized = (s.motion_work[e, i, 2] - s.positions[e, i]) / p.dt
                cost = _cost(s.velocities[e, i], realized, p)
                if s.battery[e, i] > wp.float64(0.0) and s.motion_flags[e, i, 0] == 0 and cost > s.battery[e, i]:
                    s.motion_flags[e, i, 0] = 1
                    unaffordable = bool(True)
        if not unaffordable:
            break
    correction_sum = wp.float64(0.0)
    active_count = int(0)
    for i in range(p.n):
        if s.motion_flags[e, i, 0] == 0:
            correction_sum += wp.length(s.motion_work[e, i, 1] - s.motion_work[e, i, 0]) / p.max_speed
            active_count += 1
    correction = wp.float64(0.0)
    if active_count > 0:
        correction = correction_sum / wp.float64(active_count)
    s.shield_correction_step[e] = correction
    s.shield_correction[e] += correction
    for i in range(p.n):
        realized = (s.motion_work[e, i, 2] - s.positions[e, i]) / p.dt
        cost = _cost(s.velocities[e, i], realized, p)
        if p.energy_enabled != 0:
            if s.battery[e, i] <= wp.float64(0.0):
                cost = wp.float64(0.0)
            elif s.motion_flags[e, i, 0] != 0:
                cost = s.battery[e, i]
            s.battery[e, i] = wp.max(wp.float64(0.0), s.battery[e, i] - cost)
        s.flight_energy[e, i] = s.flight_energy[e, i] + cost
        s.positions[e, i] = s.motion_work[e, i, 2]
        s.velocities[e, i] = realized
    s.blocked[e] = s.blocked[e] + blocked_count
    s.clipped[e] = s.clipped[e] + clipped_count
    for k in range(4):
        s.motion_counts[e, k] = s.motion_counts[e, k] + counts[k]
        s.motion_counts[e, k + 4] = s.motion_counts[e, k + 4] + shield[k]
