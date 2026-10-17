"""Warp-side normalized action to TX-power mapping."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState
from .phy import nominal_required_power


@wp.func
def power_from_action(p: WorldParams, action_value: wp.float32) -> wp.float64:
    u = wp.clamp((wp.float64(action_value) + wp.float64(1)) / wp.float64(2),
                 wp.float64(0), wp.float64(1))
    power = p.tx_power_min + u * (p.tx_power_max - p.tx_power_min)
    if p.tx_power_mapping_log != 0:
        power = p.tx_power_min * wp.pow(p.tx_power_max / p.tx_power_min, u)
    return power


@wp.func
def power_fraction(p: WorldParams, power: wp.float64) -> wp.float64:
    result = wp.float64(0)
    if p.tx_power_max > p.tx_power_min:
        result = (power - p.tx_power_min) / (p.tx_power_max - p.tx_power_min)
        if p.tx_power_mapping_log != 0 and power > wp.float64(0):
            result = wp.log(power / p.tx_power_min) / wp.log(p.tx_power_max / p.tx_power_min)
    return wp.clamp(result, wp.float64(0), wp.float64(1))


@wp.func
def link_power_feasible(s: WorldState, p: WorldParams, e: int, source: int, destination: int) -> bool:
    return p.power_limited_admission == 0 or nominal_required_power(s,p,e,source,destination) <= p.tx_power_max
