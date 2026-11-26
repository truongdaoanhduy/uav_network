"""Explicit comparability metrics for variable-length communication episodes.

Not policy shaping. A terminal episode is padded counterfactually with *no new*
GCS deliveries, as required to compare unequal episode lengths at a common H.
"""
from __future__ import annotations
import math


def fixed_horizon_aoi_no_new_deliveries(*, observed_sum_cell_seconds: float,
    current_mean_cell_age_s: float, t: int, horizon: int, num_cells: int) -> float:
    """Mean censored AoI over H steps assuming no updates after step t.

    observed_sum_cell_seconds=sum_{s=1..t} sum_c age_c(s).
    For the remaining m=H-t steps, each age increases by 1 per step.
    The padding sum is m*(num_cells*current_mean_age) +
    num_cells * m*(m+1)/2. This is a *conditional* terminal convention.
    """
    if not (isinstance(t,int) and isinstance(horizon,int) and isinstance(num_cells,int)
            and 1 <= t <= horizon and num_cells > 0):
        raise ValueError('require 1 <= t <= horizon and num_cells > 0')
    if (not math.isfinite(observed_sum_cell_seconds)
            or not math.isfinite(current_mean_cell_age_s)
            or min(observed_sum_cell_seconds,current_mean_cell_age_s) < 0.0):
        raise ValueError('AoI sums and current mean must be finite and non-negative')
    remaining = horizon - t
    total = (observed_sum_cell_seconds + remaining*num_cells*current_mean_cell_age_s
             + num_cells*remaining*(remaining+1)/2.0)
    return total / (horizon * num_cells)
