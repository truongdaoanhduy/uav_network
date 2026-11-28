"""v5.4 Complete 7-Node Packet-Level RL Environment (6 Actors + 1 Passive GCS).

Integrates:
  - Real SevenRadioCsmaCaNetwork (CSMA/CA, half-duplex, SINR, 3D NLoS, 2 blackouts, Q=64, E2E ACK)
  - Numerically stable canonical DS (FP32/FP64 int-diff log-space)
  - Per-UAV Zeng 2019 physical battery + acceleration clamping (a_max = 5 m/s^2)
  - Regime (c) termination: battery depletion or 15/15 GCS confirm -> terminated=True; H=1500 -> truncated=True
  - AoI metrics: confirm latency (microsecond timestamps), delivered-cell AoI and censored map-wide age
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from zhang_xue.compat.network import (
    ACTOR_N,
    CANONICAL_BLACKOUT_CENTERS_2,
    CANONICAL_BUILDINGS_5, scale_canonical_geometry,
    CELL_SIZE_M,
    GCS_ID,
    GCS_POS_M,
    GRID_W,
    NUM_CELLS,
    RADIO_N,
    SevenRadioCsmaCaNetwork,
    CsmaProtocolConfig,
    is_in_blackout,
    segment_hits_cylinder_3d_fp32,
)
from zhang_xue.compat.belief_mobility import (
    ZengBatteryConfig,
    clamp_velocity_with_accel,
    counts_to_canonical_ds_stable,
    fuse_provenance_counts,
    kinematic_safety_shield,
    trajectory_safety_violations,
    sensor_likelihood_to_bba,
    uav_step_energy_j,
)

DISC_DIM: int = 7  # 0=NO_SEND, 1..5=5 peer UAVs (bijective), 6=GCS_ID


def action_to_dest_radio_id(actor_id: int, action_idx: int) -> int:
    if not 0 <= action_idx < DISC_DIM or not 0 <= actor_id < ACTOR_N:
        raise ValueError(f"Invalid comm action {action_idx} for actor {actor_id}")
    if action_idx == 0:
        return -1
    if action_idx == 6:
        return GCS_ID
    peer = action_idx - 1
    return peer + 1 if peer >= actor_id else peer
