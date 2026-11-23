"""FP64 continuous geometry shared by motion and camera visibility."""
import warp as wp

from .state import WorldParams, WorldState
from .bayesian import profile
from .xyz_geometry import position3,camera_clear

# Keep native NumPy operation rounding; these simulator kernels need no gradients.
wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.func
def segment_distance(start: wp.vec2d, end: wp.vec2d, point: wp.vec2d):
    delta = end - start
    squared = wp.dot(delta, delta)
    fraction = wp.float64(0.0)
    if squared > wp.float64(0.0):
        fraction = wp.clamp(wp.dot(point - start, delta) / squared,
                            wp.float64(0.0), wp.float64(1.0))
    return wp.length(start + fraction * delta - point)


@wp.func
def capped(velocity: wp.vec2d, maximum: wp.float64):
    return velocity * wp.min(wp.float64(1.0),
                             maximum / wp.max(wp.length(velocity), wp.float64(1e-12)))


@wp.func
def brake(velocity: wp.vec2d, step: wp.float64):
    return velocity * wp.max(wp.float64(0.0),
                             wp.float64(1.0) - step / wp.max(wp.length(velocity), wp.float64(1e-12)))


@wp.func
def outside_map(point: wp.vec2d, p: wp.ref[WorldParams]):
    return point[0] < wp.float64(0.0) or point[1] < wp.float64(0.0) or point[0] > p.map_size or point[1] > p.map_size


@wp.func
def static_point(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, point: wp.vec2d):
    valid = not outside_map(point, p)
    for o in range(p.obstacles_count):
        circle = s.obstacles[e, o]
        if wp.length(point - wp.vec2d(circle[0], circle[1])) < circle[2] + p.obstacle_clearance + wp.float64(1e-9):
            valid = bool(False)
    return valid


@wp.func
def static_path(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, position: wp.vec2d, velocity: wp.vec2d):
    valid = not outside_map(position, p)
    point = position
    speed = velocity
    step = p.max_accel * p.dt
    horizon = int(wp.ceil(p.max_speed / step)) + 1
    for k in range(horizon):
        end = point + speed * p.dt
        if outside_map(end, p):
            valid = bool(False)
        for o in range(p.obstacles_count):
            circle = s.obstacles[e, o]
            center = wp.vec2d(circle[0], circle[1])
            if segment_distance(point, end, center) < circle[2] + p.obstacle_clearance + wp.float64(1e-9):
                valid = bool(False)
        point = end
        speed = brake(speed, step)
    return valid


@wp.func
def peer_path(p: wp.ref[WorldParams], position: wp.vec2d, velocity: wp.vec2d,
              other_position: wp.vec2d, other_velocity: wp.vec2d):
    valid = bool(True)
    point, other = position, other_position
    speed, other_speed = velocity, other_velocity
    step = p.max_accel * p.dt
    horizon = int(wp.ceil(p.max_speed / step)) + 1
    for k in range(horizon):
        end, other_end = point + speed * p.dt, other + other_speed * p.dt
        if segment_distance(point - other, end - other_end, wp.vec2d(0.0, 0.0)) < p.safety_distance - wp.float64(1e-9):
            valid = bool(False)
        point, other = end, other_end
        speed, other_speed = brake(speed, step), brake(other_speed, step)
    return valid


@wp.func
def target_visible(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, i: int, t: int):
    visible = p.energy_enabled == 0 or s.battery[e, i] > wp.float64(0.0)
    position, target = s.positions[e, i], s.targets[e, t]
    if wp.length(target - position) > p.footprint + wp.float64(1e-9):
        visible = bool(False)
    if p.occlusion != 0:
        for o in range(p.obstacles_count):
            circle = s.obstacles[e, o]
            if segment_distance(position, target, wp.vec2d(circle[0], circle[1])) < circle[2] + wp.float64(1e-9):
                visible = bool(False)
    return visible


@wp.kernel
def sense(s: WorldState, p: WorldParams):
    e, i, t = wp.tid()
    s.visible[e, i, t] = int(target_visible(s, p, e, i, t))


@wp.kernel
def scan_coverage(s: WorldState, p: WorldParams):
    e, i, cell = wp.tid()
    if p.energy_enabled != 0 and s.battery[e,i] <= wp.float64(0):
        return
    width = p.map_size / wp.float64(p.coverage_grid_cells)
    point = wp.vec2d((wp.float64(cell % p.coverage_grid_cells)+wp.float64(.5))*width,
                     (wp.float64(cell // p.coverage_grid_cells)+wp.float64(.5))*width)
    position = s.positions[e,i]
    footprint=p.footprint
    if p.xyz!=0:
        if s.altitudes[e,i]<=wp.float64(0):return
        footprint=profile(p,s.altitudes[e,i])[2]
    if wp.length(point-position) > footprint + wp.float64(1e-9):
        return
    visible = bool(True)
    if p.occlusion != 0 and p.xyz!=0:
        visible=camera_clear(s,p,e,position3(s,e,i),point)
    elif p.occlusion != 0:
        for o in range(p.obstacles_count):
            circle = s.obstacles[e,o]
            if segment_distance(position,point,wp.vec2d(circle[0],circle[1])) < circle[2]+wp.float64(1e-9):
                visible = bool(False)
    if visible:
        s.coverage[e,i,cell] = 1
