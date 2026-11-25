"""Steady, level rotary-wing propulsion for the CPU six-radio pilot.

Zeng, Xu and Zhang, IEEE TWC 2019, Eq. (12), DOI 10.1109/TWC.2019.2902559.
This speed-only model does not model acceleration, climb, wind, or RF energy.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np


@dataclass(frozen=True)
class ZengPropulsionParameters:
    p0_blade_hover_w: float = 79.86
    pi_induced_hover_w: float = 88.63
    u_tip_mps: float = 120.0
    v0_hover_mps: float = 4.03
    fuselage_drag_d0: float = 0.6
    air_density_rho: float = 1.225
    rotor_solidity_s: float = 0.05
    rotor_disc_area_m2: float = 0.503

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, (bool, str)) or
                    not isinstance(value, (int, float, np.number)) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f"{field.name} must be positive and finite")
        if self.rotor_solidity_s > 1:
            raise ValueError("rotor_solidity_s must be within (0, 1]")


def steady_level_propulsion_power_w(
    speed_mps, parameters: ZengPropulsionParameters | None = None,
):
    """Return Eq. (12) propulsion watts for nonnegative, finite speed(s).

    Avionics watts are added by the environment. The induced term uses the
    algebraically equivalent reciprocal form to avoid subtractive cancellation.
    """
    p = parameters or ZengPropulsionParameters()
    speed = np.asarray(speed_mps, dtype=np.float64)
    if not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError("speed_mps must be nonnegative and finite")
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = (speed / p.v0_hover_mps) ** 2 / 2.0
        power = (
            p.p0_blade_hover_w * (1.0 + 3.0 * (speed / p.u_tip_mps) ** 2)
            + p.pi_induced_hover_w / np.sqrt(np.hypot(1.0, ratio) + ratio)
            + 0.5 * p.fuselage_drag_d0 * p.air_density_rho
            * p.rotor_solidity_s * p.rotor_disc_area_m2 * speed ** 3
        )
    if not np.isfinite(power).all():
        raise ValueError("speed_mps or propulsion parameters overflow the power model")
    return float(power) if power.ndim == 0 else power
