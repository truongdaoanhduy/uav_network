"""Warp-side normalized action to TX-power mapping."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState
from .phy import nominal_required_power


@wp.func
def power_from_action(p: wp.ref[WorldParams], action_value: wp.float32) -> wp.float64:
    u = wp.clamp((wp.float64(action_value) + wp.float64(1)) / wp.float64(2),
                 wp.float64(0), wp.float64(1))
    power = p.tx_power_min + u * (p.tx_power_max - p.tx_power_min)
    if p.tx_power_mapping_log != 0:
        power = p.tx_power_min * wp.pow(p.tx_power_max / p.tx_power_min, u)
    return power


@wp.func
def power_fraction(p: wp.ref[WorldParams], power: wp.float64) -> wp.float64:
    result = wp.float64(0)
    if p.tx_power_max > p.tx_power_min:
        result = (power - p.tx_power_min) / (p.tx_power_max - p.tx_power_min)
        if p.tx_power_mapping_log != 0 and power > wp.float64(0):
            result = wp.log(power / p.tx_power_min) / wp.log(p.tx_power_max / p.tx_power_min)
    return wp.clamp(result, wp.float64(0), wp.float64(1))


@wp.func
def link_power_feasible(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, source: int, destination: int) -> bool:
    return p.power_limited_admission == 0 or nominal_required_power(s,p,e,source,destination) <= p.tx_power_max


@wp.func
def selected_power(s: wp.ref[WorldState], p: wp.ref[WorldParams], e: int, source: int,
                   destination: int, action_value: wp.float32) -> wp.float64:
    power = power_from_action(p, action_value)
    if p.guided_power != 0:
        power = wp.clamp(nominal_required_power(s,p,e,source,destination) * p.power_margin_factor,
                         p.tx_power_min, p.tx_power_max)
    return power
