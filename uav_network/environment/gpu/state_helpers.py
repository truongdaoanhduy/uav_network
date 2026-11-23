"""NumPy PCG seed representation and Warp allocation accounting helpers."""
import numpy as np
import warp as wp


def seed_rows(seeds):
    """NumPy PCG64's 128-bit state for the geometry kernel."""
    rows = []
    for seed in seeds:
        d = np.random.PCG64(int(seed)).state["state"]
        rows.append([d["state"] & ((1<<64)-1), d["state"] >> 64,
                     d["inc"] & ((1<<64)-1), d["inc"] >> 64])
    return np.asarray(rows, dtype=np.uint64)


def allocated_bytes(s):
    return sum(value.capacity for value in vars(s).values() if isinstance(value, wp.array))
