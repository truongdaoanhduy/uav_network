"""CPU-only fixed-tilt controllable-azimuth optical cone (V123 experimental).

The camera ray axis points down at tilt alpha from nadir and rotates through yaw
psi. The cone is geometric; detection Pd/Pf are inherited unchanged from the
V122 scenario, which is *not* an empirical oblique-image detector calibration.
Do not report this as an engineering-ready UAV camera or as Warp parity.

Cone condition: (d dot optical_axis) / ||d|| >= cos(beta), d=(dx,dy,-h).
Camera projection is grounded in ray-plane/pinhole geometry; choice of pilot
parameters (alpha=beta=25 deg, slew=45 deg/s) is PROJECT HYPOTHESIS.
"""
from __future__ import annotations
import math
import numpy as np


def max_ground_radius_m(alt_m: float, tilt_deg: float, half_fov_deg: float) -> float:
    if not all(math.isfinite(x) for x in (alt_m, tilt_deg, half_fov_deg)):
        raise ValueError("nonfinite geometry")
    if alt_m <= 0 or not (0 <= tilt_deg < 85 and 0 < half_fov_deg < 85 and tilt_deg + half_fov_deg < 85):
        raise ValueError("ground-facing cone requires positive altitude and tilt + beta < 85 degrees")
    return alt_m * math.tan(math.radians(tilt_deg + half_fov_deg))


def is_cell_in_gimbal_cone(dx: float, dy: float, alt_m: float,
                           yaw_rad: float, tilt_deg: float, half_fov_deg: float) -> bool:
    """Ground-cell visibility in an ideal camera cone; occlusion must be tested separately."""
    rmax = max_ground_radius_m(alt_m, tilt_deg, half_fov_deg)
    if not all(math.isfinite(x) for x in (dx, dy, yaw_rad)):
        raise ValueError("nonfinite relative cell or yaw")
    if dx*dx+dy*dy > (rmax+1e-6)**2:
        return False
    alpha = math.radians(tilt_deg)
    beta = math.radians(half_fov_deg)
    numerator = math.sin(alpha) * (math.cos(yaw_rad)*dx + math.sin(yaw_rad)*dy) + math.cos(alpha) * alt_m
    return numerator >= math.cos(beta) * math.sqrt(dx*dx + dy*dy + alt_m*alt_m) - 1e-9


def slew_azimuth(current_rad: np.ndarray, target_rad: np.ndarray, max_step_rad: float) -> np.ndarray:
    """Shortest-arc saturated pan update with observed orientation state."""
    current = np.asarray(current_rad, dtype=np.float64)
    target = np.asarray(target_rad, dtype=np.float64)
    if current.shape != target.shape or not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("pan target must match current vector and be finite")
    if not math.isfinite(max_step_rad) or max_step_rad <= 0:
        raise ValueError("positive slew budget required")
    delta = (target - current + math.pi) % (2*math.pi) - math.pi
    return (current + np.clip(delta, -max_step_rad, max_step_rad) + math.pi) % (2*math.pi) - math.pi
