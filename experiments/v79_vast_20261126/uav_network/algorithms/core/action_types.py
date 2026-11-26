"""Standalone learner action types; no old UAV environment dependency."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Action:
    continuous: object
    destination: object

    def validate(self, num_uavs: int, continuous_dim: int = 3):
        a, d = np.asarray(self.continuous), np.asarray(self.destination)
        if (a.shape != (num_uavs, continuous_dim) or
                not np.issubdtype(a.dtype, np.number) or
                not np.isfinite(a).all() or np.any(np.abs(a)>1)):
            raise ValueError("Continuous actions must be finite [-1,1] of shape (N,D)")
        if (d.shape != (num_uavs,) or
                not np.issubdtype(d.dtype, np.integer) or
                np.any(d<0) or np.any(d>num_uavs)):
            raise ValueError("Destination IDs must be integer 0..N")
