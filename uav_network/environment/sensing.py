"""Exact target detections inside a fixed-altitude nadir-camera footprint."""
import numpy as np

from .geometry import line_is_clear


def visible_target_ids(position, targets, obstacles, sensor):
    distances = np.linalg.norm(targets - position, axis=1)
    candidates = np.flatnonzero(distances <= sensor.footprint_radius_m + 1e-9)
    visible = [i for i in candidates if not sensor.obstacle_occlusion
               or line_is_clear(position, targets[i], obstacles)]
    return sorted(visible, key=lambda i: distances[i])
