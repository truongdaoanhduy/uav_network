"""Warp-side normalized action to TX-power mapping."""
import warp as wp
from ...environment.gpu.state import WorldParams


@wp.func
def power_from_action(p: WorldParams, action_value: wp.float32) -> wp.float64:
    u = wp.clamp((wp.float64(action_value) + wp.float64(1)) / wp.float64(2),
                 wp.float64(0), wp.float64(1))
    power = p.tx_power_min + u * (p.tx_power_max - p.tx_power_min)
    if p.tx_power_mapping_log != 0:
        power = p.tx_power_min * wp.pow(p.tx_power_max / p.tx_power_min, u)
    return power
