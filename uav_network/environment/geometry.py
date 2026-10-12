"""Continuous segment geometry used by motion and occlusion."""
import numpy as np


def segment_distance(start, end, point):
    start, end, point = (np.asarray(x, dtype=float) for x in (start, end, point))
    delta = end - start
    norm_sq = float(delta @ delta)
    fraction = np.clip((point - start) @ delta / norm_sq, 0, 1) if norm_sq else 0.0
    return float(np.linalg.norm(start + fraction * delta - point))


def segment_hits_circle(start, end, center, radius):
    return segment_distance(start, end, center) < radius + 1e-9


def line_is_clear(start, end, obstacles):
    return not any(segment_hits_circle(start, end, o[:2], o[2]) for o in obstacles)
