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
        candidates = np.flatnonzero(np.linalg.norm(points-position, axis=1)
                                    <= sensor.footprint_radius_m + 1e-9)
        for cell in candidates:
            if not sensor.obstacle_occlusion or line_is_clear(position, points[cell], env.obstacles):
                env.coverage[i,cell] = 1
