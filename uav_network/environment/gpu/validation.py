"""Checked device action boundary before mutating any world state."""
import warp as wp
from .state import WorldState, WorldParams


@wp.kernel
def validate_actions(s: WorldState,p: WorldParams,continuous: wp.array3d(dtype=wp.float32),
                     destination: wp.array2d(dtype=wp.int64)):
    e = wp.tid()
    if s.terminated[e] != 0:
        s.errors[e] = 92
    for i in range(p.n):
        if destination[e,i] < 0 or destination[e,i] > wp.int64(p.n):
            s.errors[e] = 91
        for j in range(3):
            value = continuous[e,i,j]
            if not wp.isfinite(value) or wp.abs(value) > wp.float32(1.0):
                s.errors[e] = 90
