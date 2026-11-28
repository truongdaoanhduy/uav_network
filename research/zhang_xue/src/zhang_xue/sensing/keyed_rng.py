"""Counter-based common-random-numbers for observational CPU ablations only.

Not cryptographic, not calibrated physical correlation, not current Warp RNG.
No source randomness or ground-truth target positions are exposed to an RL actor.
The canonical scenario uses its unchanged sequential PCG generator.
"""
from __future__ import annotations

_MASK = (1 << 64) - 1

def sensor_uniform(seed: int, episode: int, tick: int, uav: int, cell: int) -> float:
    """Stable uniform value keyed by independent observation identity, independent of FoV loops."""
    if not (0 <= episode <= (1<<32) and 0 <= tick <= (1<<32)
            and 0 <= uav < (1<<16) and 0 <= cell < (1<<32)):
        raise ValueError('Invalid keyed observation identity')
    x = (int(seed) & _MASK) ^ 0x7f4a7c159e3779b9
    for val, salt in ((episode, 0x9e3779b97f4a7c15),
                      (tick, 0xbf58476d1ce4e5b9),
                      (uav, 0x94d049bb133111eb),
                      (cell, 0x2545f4914f6cdd1d)):
        x = (x ^ ((int(val) + 1) * salt)) & _MASK
        x ^= x >> 30
        x = (x * 0xbf58476d1ce4e5b9) & _MASK
        x ^= x >> 27
        x = (x * 0x94d049bb133111eb) & _MASK
        x ^= x >> 31
    return (x >> 11) * (1.0 / (1 << 53))
