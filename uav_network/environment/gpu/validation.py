"""Checked device action boundary before mutating any world state."""
import warp as wp
from .state import WorldState, WorldParams


@wp.kernel
def validate_actions(s: WorldState,p: WorldParams,continuous: wp.array3d(dtype=wp.float32),
                     destination: wp.array2d(dtype=wp.int64)):
    e = wp.tid()
    if s.terminated[e] != 0:
        # GPU evaluator runs many independent seeds simultaneously. Some worlds
        # finish early, but must remain in the batch while other worlds finish.
        # A terminal world is legal ONLY with an all-zero no-op in the new
        # no-oracle cell-report MDP; historical training keeps its strict ABI.
        noop = p.cell_reports != 0
        for i in range(p.n):
            if destination[e,i] != wp.int64(0):
                noop=False
            for j in range(p.continuous_dim):
                if continuous[e,i,j] != wp.float32(0.0):
                    noop=False
        if not noop:
            s.errors[e] = 92
        return
    for i in range(p.n):
        if destination[e,i] < 0 or destination[e,i] > wp.int64(p.n):
            s.errors[e] = 91
        for j in range(p.continuous_dim):
            value = continuous[e,i,j]
            if not wp.isfinite(value) or wp.abs(value) > wp.float32(1.0):
                s.errors[e] = 90
