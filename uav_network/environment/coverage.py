"""Small per-UAV binary scan memory; no target truth or shared map."""
import numpy as np

from .geometry import line_is_clear


def scan_coverage(env):
    cells = env.config.sensor.coverage_grid_cells
    if not cells:
        return
    width = env.config.world.map_size_m / cells
    points = np.array([[(x+.5)*width, (y+.5)*width]
                       for y in range(cells) for x in range(cells)])
    sensor = env.config.sensor
    for i, position in enumerate(env.positions):
        if not env.active[i]:
            continue
        xyz = env.config.motion_dim == 3
        from .bayesian import sensor_profile
        radius = sensor_profile(sensor,position[2])[2] if xyz else sensor.footprint_radius_m
        if xyz and position[2]<=0:continue
        candidates = np.flatnonzero(np.linalg.norm(points-position[:2], axis=1)
                                    <= radius + 1e-9)
        for cell in candidates:
            from .xyz_geometry import camera_clear
            clear = camera_clear(position,points[cell],env.obstacles,env.obstacle_heights) if xyz else line_is_clear(position,points[cell],env.obstacles)
            if not sensor.obstacle_occlusion or clear:
                env.coverage[i,cell] = 1
