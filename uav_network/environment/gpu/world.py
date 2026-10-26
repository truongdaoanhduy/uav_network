"""Device reset with the original NumPy PCG64 uniform/rejection sequence."""
import warp as wp

from .state import WorldParams, WorldState

# Keep native NumPy operation rounding; these simulator kernels need no gradients.
wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.func
def _multiply_high(a: wp.uint64, b: wp.uint64):
    mask = wp.uint64(0xffffffff)
    a0, a1 = a & mask, a >> wp.uint64(32)
    b0, b1 = b & mask, b >> wp.uint64(32)
    first = a0 * b0
    middle = a1 * b0 + (first >> wp.uint64(32))
    middle_low = middle & mask
    middle_high = middle >> wp.uint64(32)
    middle2 = a0 * b1 + middle_low
    return a1 * b1 + middle_high + (middle2 >> wp.uint64(32))


@wp.func
def pcg_uniform(rows: wp.array2d(dtype=wp.uint64), e: int):
    # PCG XSL RR 128/64: advance the 128-bit LCG, then rotate high XOR low.
    old_low, old_high = rows[e, 0], rows[e, 1]
    multiplier_low = wp.uint64(0x4385df649fccf645)
    multiplier_high = wp.uint64(0x2360ed051fc65da4)
    product_low = old_low * multiplier_low
    low = product_low + rows[e, 2]
    carry = wp.uint64(0)
    if low < product_low:
        carry = wp.uint64(1)
    high = (_multiply_high(old_low, multiplier_low) + old_high * multiplier_low
            + old_low * multiplier_high + rows[e, 3] + carry)
    rows[e, 0] = low
    rows[e, 1] = high
    mixed = high ^ low
    rotation = high >> wp.uint64(58)
    # Mask the opposite shift so a zero rotation never shifts by 64.
    raw = ((mixed >> rotation) |
           (mixed << ((wp.uint64(64) - rotation) & wp.uint64(63))))
    return wp.float64(raw >> wp.uint64(11)) * wp.float64(1.1102230246251565e-16)


@wp.func
def _uniform(s: WorldState, e: int, low: wp.float64, high: wp.float64):
    return low + (high - low) * pcg_uniform(s.pcg_state, e)


@wp.func
def _outside(s: WorldState, e: int, point: wp.vec2d, count: int, clearance: wp.float64):
    valid = bool(True)
    for o in range(count):
        circle = s.obstacles[e, o]
        if wp.length(point - wp.vec2d(circle[0], circle[1])) < circle[2] + clearance + wp.float64(1e-9):
            valid = bool(False)
    return valid


@wp.kernel
def reset_geometry(s: WorldState, p: WorldParams, mask: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] == 0:
        return
    s.errors[e] = 0
    s.gcs[e] = wp.vec2d(p.gcs_x, p.gcs_y)
    s.steps[e] = 0
    s.terminated[e] = 0
    s.blocked[e] = 0
    s.clipped[e] = 0
    for k in range(9):
        s.motion_counts[e, k] = 0
    for i in range(p.n):
        s.velocities[e, i] = wp.vec2d(0.0, 0.0)
        s.battery[e, i] = p.capacity
        s.flight_energy[e, i] = wp.float64(0.0)
    if p.obstacles_count > 0 and wp.float64(2.0) * (p.obstacle_max_radius + p.obstacle_clearance + wp.float64(1e-6)) >= p.map_size:
        s.errors[e] = 101
        return
    # Disjoint clearance disks inside the map keep the free flight region connected.
    for o in range(p.obstacles_count):
        radius = _uniform(s, e, p.obstacle_min_radius, p.obstacle_max_radius)
        placed = bool(False)
        for attempt in range(p.placement_attempts):
            padding = radius + p.obstacle_clearance + wp.float64(1e-6)
            x = _uniform(s, e, padding, p.map_size - padding)
            y = _uniform(s, e, padding, p.map_size - padding)
            center = wp.vec2d(x, y)
            if (wp.length(center - s.gcs[e]) > p.launch_radius + padding
                    and _outside(s, e, center, o, radius + wp.float64(2.0) * p.obstacle_clearance + wp.float64(1e-6))):
                s.obstacles[e, o] = wp.vec3d(x, y, radius)
                placed = bool(True)
                break
        if not placed:
            s.errors[e] = 102
            return
    for i in range(p.n):
        placed = bool(False)
        for attempt in range(p.placement_attempts):
            x = _uniform(s, e, -p.launch_radius, p.launch_radius)
            y = _uniform(s, e, -p.launch_radius, p.launch_radius)
            point = s.gcs[e] + wp.vec2d(x, y)
            valid = (point[0] >= wp.float64(0.0) and point[1] >= wp.float64(0.0)
                     and point[0] <= p.map_size and point[1] <= p.map_size
                     and wp.length(point - s.gcs[e]) <= p.launch_radius
                     and _outside(s, e, point, p.obstacles_count, p.obstacle_clearance))
            for j in range(i):
                if wp.length(point - s.positions[e, j]) < p.launch_spacing:
                    valid = bool(False)
            if valid:
                s.positions[e, i] = point
                placed = bool(True)
                break
        if not placed:
            s.errors[e] = 103
            return
    target_clearance = wp.max(wp.float64(0.0), p.obstacle_clearance - p.footprint) + wp.float64(1e-6)
    for t in range(p.targets_count):
        placed = bool(False)
        for attempt in range(p.placement_attempts):
            x = _uniform(s, e, wp.float64(0.0), p.map_size)
            y = _uniform(s, e, wp.float64(0.0), p.map_size)
            point = wp.vec2d(x, y)
            if wp.length(point - s.gcs[e]) >= p.target_exclusion and _outside(s, e, point, p.obstacles_count, target_clearance):
                s.targets[e, t] = point
                placed = bool(True)
                break
        if not placed:
            s.errors[e] = 104
            return
