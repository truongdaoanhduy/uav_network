"""v6.5 **buffer allocation only** compressed Warp schema (379,848 B/env before data-mule).

Allocates `make_shapes_v63(num_uavs, grid_w)` with complete dtype coverage (`int8`, `int16`,
`float16`, `uint8`, `uint16`, `int32`, `uint32`, `float32`) and selected UavNetSim PHY constants
(`tx_power_w=0.1`, `cca_threshold_w=6.3095734448e-12` [-82 dBm], `noise_w=4.3895770929e-13`).
No packet-step kernel is implemented and data-mule provenance ACK tensors are missing.
This is NOT a functional GPU simulator and has no verified event parity.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .layout_v62 import backing_bytes_per_environment_v63, make_shapes_v63
from .state_v65 import (
    HAS_WARP,
    K_V65,
    Q_V65,
    UAVNETSIM_CCA_W,
    UAVNETSIM_NOISE_W,
    UAVNETSIM_SNR_TH_LIN,
    UAVNETSIM_TX_POWER_W,
    wp,
)


def v65_gpu_memory_budget(
    num_envs_per_gpu: int = 2048,
    num_uavs: int = 6,
    grid_w: int = 20,
    replay_capacity: int = 300_000,
    obs_dim: int = 514,
) -> Dict[str, float]:
    """Schema-only VRAM projection; separately report 4D data-mule ACK memory.

    Not a GPU benchmark, not packet-event parity, and not a complete
    training-memory estimate. The missing 4D arrays are NOT allocated.
    """
    from zhang_xue.learners.state_derived_replay import estimate_replay_bytes

    soa_bytes = backing_bytes_per_environment_v63(num_uavs=num_uavs, grid_w=grid_w)
    network_gib = (soa_bytes * num_envs_per_gpu) / (1024 ** 3)
    # v6.5 CPU data-mule owns three 4D ACK watermark arrays:
    # uint16 positive + uint16 negative + int32 generation timestamp.
    # These arrays do NOT exist in make_shapes_v63 and are not executable
    # on Warp: budget their missing footprint explicitly rather than
    # advertising the compressed schema as a complete data-mule network.
    missing_ack_bytes = num_uavs * (num_uavs + 1) * num_uavs * grid_w**2 * (2 + 2 + 4)
    missing_ack_gib = (missing_ack_bytes * num_envs_per_gpu) / (1024 ** 3)
    replay_gib = estimate_replay_bytes(
        replay_capacity, num_uavs, obs_dim, 2, 3, store_state=False
    ) / (1024 ** 3)
    pytorch_cuda_reserve_gib = 3.0
    total_gib = network_gib + replay_gib + pytorch_cuda_reserve_gib
    return {
        "layout_id": "v65_compressed_param",
        "num_envs_per_gpu": float(num_envs_per_gpu),
        "num_uavs": float(num_uavs),
        "grid_w": float(grid_w),
        "soa_bytes_per_env": float(soa_bytes),
        "network_soa_gib": round(network_gib, 3),
        "datamule_ack_bytes_per_env": missing_ack_bytes,
        "datamule_ack_gib": round(missing_ack_gib, 3),
        "network_plus_missing_datamule_ack_gib": round(network_gib + missing_ack_gib, 3),
        "datamule_packet_fields_implemented": False,
        "estimate_only_not_a_gpu_benchmark": True,
        "replay_buffer_gib": round(replay_gib, 3),
        "pytorch_cuda_reserve_gib": pytorch_cuda_reserve_gib,
        "estimated_total_gib": round(total_gib, 3),
        "t4_vram_gib": 16.0,
        "fits_single_t4_16gib": float(total_gib < 14.5),
    }


def resolve_compact3_actions_np(
    actions: np.ndarray,
    positions_xy: np.ndarray,
    alive_mask: np.ndarray,
    enable_peer_dissemination: bool = True,
    *,
    dirty_dest: np.ndarray | None = None,
    can_tx: np.ndarray | None = None,
) -> np.ndarray:
    """Resolve actions, using the CPU policy's nearest *dirty* alive peer.

    `dirty_dest[e, src, dst]` records sender-local, receiver-specific pending
    evidence for dst=0..N (last index is GCS). A caller claiming CPU parity
    MUST provide it: the fallback without dirty_dest is only a topology-only
    diagnostic for old tests, not a full actor-to-radio executor.
    `can_tx[e, src]` also gates blackout/dead-radio sources.
    """
    act = np.asarray(actions)
    pos = np.asarray(positions_xy, dtype=np.float32)
    alive = np.asarray(alive_mask, dtype=bool)
    if act.ndim != 2 or not np.issubdtype(act.dtype, np.integer):
        raise ValueError("actions must be a 2D integer array")
    e_n, u_n = act.shape
    if pos.shape != (e_n, u_n, 2) or alive.shape != (e_n, u_n):
        raise ValueError("positions_xy/alive_mask shape must match actions")
    if np.any((act < 0) | (act > 2)):
        raise ValueError("compact_3 action must be in [0,2]")
    if dirty_dest is not None:
        dirty = np.asarray(dirty_dest, dtype=bool)
        if dirty.shape != (e_n, u_n, u_n + 1):
            raise ValueError("dirty_dest must have shape (envs, uavs, uavs+1)")
    else:
        dirty = None
    if can_tx is not None:
        tx = np.asarray(can_tx, dtype=bool)
        if tx.shape != (e_n, u_n):
            raise ValueError("can_tx must have shape (envs, uavs)")
        enabled = alive & tx
    else:
        # Compatibility diagnostic: legacy callers only supplied peer-aliveness,
        # not transmitter availability. This does not assert packet execution.
        enabled = np.ones_like(alive, dtype=bool)
    out = np.full((e_n, u_n), -1, dtype=np.int32)
    out[(act == 1) & enabled] = u_n
    peer_mask = (act == 2) & enabled
    if not enable_peer_dissemination or not np.any(peer_mask):
        return out
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    invalid = np.eye(u_n, dtype=bool)[None, :, :] | (~alive[:, None, :])
    if dirty is not None:
        invalid |= ~dirty[:, :, :u_n]
    dist = np.where(invalid, np.inf, dist)
    nearest = np.argmin(dist, axis=-1)
    min_d = np.take_along_axis(dist, nearest[..., None], axis=-1)[..., 0]
    valid_peer = peer_mask & np.isfinite(min_d)
    out[valid_peer] = nearest[valid_peer]
    return out


class WarpCompressedRadioNetwork:
    """Parameterized compressed Warp SoA network (dual-path alongside legacy `WarpSevenRadioNetwork`)."""

    @staticmethod
    def active_layout_id() -> str:
        return "v65_compressed_param"

    @staticmethod
    def active_soa_bytes_per_environment(num_uavs: int = 6, grid_w: int = 20) -> int:
        return backing_bytes_per_environment_v63(num_uavs=num_uavs, grid_w=grid_w)

    @staticmethod
    def default_phy_params() -> Dict[str, float]:
        return {
            "tx_power_w": UAVNETSIM_TX_POWER_W,
            "cca_threshold_w": UAVNETSIM_CCA_W,
            "noise_w": UAVNETSIM_NOISE_W,
            "snr_threshold": UAVNETSIM_SNR_TH_LIN,
        }

    def __init__(
        self,
        num_envs: int,
        num_uavs: int = 6,
        grid_w: int = 20,
        comm_action_mode: str = "compact_3",
        device: str = "cpu",
        seed: int = 20261125,
    ):
        if not HAS_WARP:
            raise RuntimeError("warp-lang is required for WarpCompressedRadioNetwork allocation")
        if num_envs < 1 or not (1 <= int(num_uavs) <= 30):
            raise ValueError("num_envs must be >= 1 and num_uavs in [1, 30]")
        if comm_action_mode not in ("compact_3", "per_peer"):
            raise ValueError("comm_action_mode must be compact_3 or per_peer")
        from .state_v65 import NetworkBuffersV65, SimDimsV65

        self.envs = int(num_envs)
        self.num_uavs = int(num_uavs)
        self.radio_n = self.num_uavs + 1
        self.gcs_id = self.num_uavs
        self.grid_w = int(grid_w)
        self.num_cells = self.grid_w * self.grid_w
        self.comm_action_mode = comm_action_mode
        self.device = device
        self.seed = int(seed)
        self.dims = SimDimsV65()
        self.dims.n_uav = self.num_uavs
        self.dims.n_radio = self.radio_n
        self.dims.gcs_id = self.gcs_id
        self.dims.grid_w = self.grid_w
        self.dims.num_cells = self.num_cells
        self.dims.q_cap = Q_V65
        self.dims.p_cap = max(1, self.num_uavs)
        self.dims.k_entries = K_V65
        self.dims.scalable_wire = 1 if (self.num_uavs > 8 or self.num_cells > 512) else 0
        self.dims.node_mask_all = (1 << self.radio_n) - 1

        self.state = NetworkBuffersV65()
        self.data = {}
        dt_map = {
            np.dtype(np.int8): wp.int8,
            np.dtype(np.uint8): wp.uint8,
            np.dtype(np.int16): wp.int16,
            np.dtype(np.uint16): wp.uint16,
            np.dtype(np.int32): wp.int32,
            np.dtype(np.uint32): wp.uint32,
            np.dtype(np.float16): wp.float16,
            np.dtype(np.float32): wp.float32,
        }
        shapes = make_shapes_v63(num_uavs=self.num_uavs, grid_w=self.grid_w, q_cap=Q_V65, k_entries=K_V65)
        for name, (shape, dtype) in shapes.items():
            dt = dt_map[np.dtype(dtype)]
            arr = wp.zeros((self.envs, *shape) if shape != (1,) else (self.envs,), dtype=dt, device=device)
            self.data[name] = arr
            setattr(self.state, name, arr)
