"""v6.7/v7.1 Executable Parameterized Compressed Warp Radio Runtime."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from ..communication.common import CsmaProtocolConfig
from .layout_v67 import (
    ENV_SCALAR_FIELDS_V67,
    K_V67,
    Q_V67,
    Q_V67_MAX,
    assert_network_allocation_fits,
    backing_bytes_per_environment_v67,
    make_shapes_v67,
    v67_gpu_memory_budget,
)
from .state_v67 import (
    FAULT_GENERATION,
    HAS_WARP,
    UAVNETSIM_CCA_W,
    UAVNETSIM_NOISE_W,
    UAVNETSIM_SNR_TH_LIN,
    UAVNETSIM_TX_POWER_W,
    wp,
)


def resolve_compact3_actions_v67_np(
    actions: np.ndarray,
    positions_xy: np.ndarray,
    alive_mask: np.ndarray,
    enable_peer_dissemination: bool = True,
    dirty_dest: np.ndarray | None = None,
    can_tx: np.ndarray | None = None,
) -> np.ndarray:
    """Legacy global-position resolver, separate from Warp's default HELLO-local policy."""
    act = np.asarray(actions, dtype=np.int32)
    pos = np.asarray(positions_xy, dtype=np.float32)
    alive = np.asarray(alive_mask, dtype=bool)
    e_n, u_n = act.shape
    gcs_id = u_n
    tx_ok = alive if can_tx is None else (alive & np.asarray(can_tx, dtype=bool))
    dirty = None if dirty_dest is None else np.asarray(dirty_dest, dtype=bool)
    out = np.full((e_n, u_n), -1, dtype=np.int32)
    gcs_mask = (act == 1) & tx_ok
    if dirty is not None:
        gcs_mask = gcs_mask & dirty[:, :, gcs_id]
    out[gcs_mask] = gcs_id
    if not enable_peer_dissemination:
        return out
    peer_mask = (act == 2) & tx_ok
    if not np.any(peer_mask):
        return out
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    eye = np.eye(u_n, dtype=bool)[None, :, :]
    invalid = eye | (~alive[:, None, :])
    if dirty is not None:
        invalid = invalid | (~dirty[:, :, :u_n])
    dist = np.where(invalid, np.inf, dist)
    nearest = np.argmin(dist, axis=-1)
    min_d = np.take_along_axis(dist, nearest[..., None], axis=-1)[..., 0]
    valid_peer = peer_mask & np.isfinite(min_d)
    out[valid_peer] = nearest[valid_peer]
    return out


def normalize_sensing_inputs(cell_id, detected, t_s, *, envs: int, uavs: int, cells: int):
    """Validate signed cell sentinel -1, binary/bool outcome and int16 generation."""
    c = np.asarray(cell_id)
    d = np.asarray(detected)
    if c.shape != (envs, uavs) or d.shape != (envs, uavs):
        raise ValueError("sense_batch requires (envs,uavs) cell and detection arrays")
    if c.dtype.kind not in "iu" or d.dtype.kind not in "iub":
        raise ValueError("cell must be integer and detection must be integer or bool")
    d_int = d.astype(np.int32)
    if np.any((c < -1) | (c >= cells)) or np.any((d_int < 0) | (d_int > 1)):
        raise ValueError("cell_id out of bounds or nonbinary detected flag")
    if isinstance(t_s, bool) or not isinstance(t_s, (int, np.integer)):
        raise ValueError("t_s must be an integer in int16-generation range [0,32767]")
    t_val = int(t_s)
    if not (0 <= t_val <= 32767):
        raise ValueError("t_s must fit int16-generation timestamp range [0,32767]")
    return c.astype(np.int32), d_int, t_val


def normalize_compact_actions(actions, pos_xy, *, envs: int, uavs: int):
    """Validate compact_3 policy-to-network interface without silent casts."""
    a, p = np.asarray(actions), np.asarray(pos_xy, dtype=np.float32)
    if a.shape != (envs, uavs) or a.dtype.kind not in "iu" or np.any((a < 0) | (a > 2)):
        raise ValueError("compact_3 actions must be integer (envs,uavs) in {0,1,2}")
    if p.shape != (envs, uavs, 2) or not np.isfinite(p).all():
        raise ValueError("pos_xy must be finite (envs,uavs,2)")
    return a.astype(np.int32), p


def pending_slot_available(valid: int, now_us: int, expiry_us: int) -> bool:
    """A pending E2E ACK owns its destination slot until expiry or clear."""
    return not bool(valid) or int(now_us) >= int(expiry_us)


def normalize_phy_inputs(positions, buildings, blackout_centers, *, envs: int, radios: int):
    """Fail closed on invalid shapes/NaN *before* allocating or launching GPU kernels."""
    p = np.asarray(positions, dtype=np.float32)
    if p.shape != (envs, radios, 3) or not np.isfinite(p).all():
        raise ValueError(f"positions must be finite ({envs}, {radios}, 3)")
    try:
        b = np.asarray(buildings, dtype=np.float32).reshape((-1, 4))
        bo = np.asarray(blackout_centers, dtype=np.float64).reshape((-1, 2))
    except (TypeError, ValueError) as exc:
        raise ValueError("buildings must be (N,4), blackout centers (M,2)") from exc
    if not np.isfinite(b).all() or not np.isfinite(bo).all():
        raise ValueError("buildings and blackout centers must be finite")
    if (b.shape[0] > 0 and np.any(b[:, 2:] <= 0)) or (bo.shape[0] > 0 and not np.all(bo == np.floor(bo))):
        raise ValueError("building radius/height must be positive; blackout centers integers")
    return p, b, bo.astype(np.int32)


class WarpCompressedRadioNetworkV67:
    """Experimental EWMA packet radio; CUDA parity and RL integration unverified.

    Generation-only ACK watermarks enforce strictly increasing per-cell sensing
    generations. Raw-count mode is unsupported by this compressed layout.
    """

    @staticmethod
    def active_layout_id() -> str:
        return "v67_datamule_compressed"

    @staticmethod
    def active_soa_bytes_per_environment(
        num_uavs: int = 6, grid_w: int = 20, q_cap: int = Q_V67
    ) -> int:
        return backing_bytes_per_environment_v67(num_uavs=num_uavs, grid_w=grid_w, q_cap=q_cap)

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
        enable_data_mule: bool = True,
        enable_peer: bool = True,
        aoi_ewma: bool = True,
        aoi_eta: float = 0.02,
        rx_lock_on_cca: bool = False,
        cw_inclusive_upper: int = 1,
        device: str = "cpu",
        seed: int = 20261125,
        decentralized_peer_select: bool = True,
        enable_peer_telemetry: bool = False,
        q_cap: int = Q_V67,
        protocol: CsmaProtocolConfig | None = None,
    ):
        if not HAS_WARP:
            raise RuntimeError("warp-lang is required for WarpCompressedRadioNetworkV67")
        for name, value in (("num_envs", num_envs), ("num_uavs", num_uavs), ("grid_w", grid_w)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if not aoi_ewma:
            raise NotImplementedError("generation-only ACK watermarks support only EWMA mode")
        if not np.isfinite(aoi_eta) or aoi_eta < 0:
            raise ValueError("aoi_eta must be finite and non-negative")
        if isinstance(q_cap, (bool, np.bool_)) or not isinstance(q_cap, (int, np.integer)):
            raise ValueError("q_cap must be an integer")
        if not (1 <= int(q_cap) <= Q_V67_MAX):
            raise ValueError(f"q_cap must lie in [1, {Q_V67_MAX}] (int8 active_slot limit)")
        proto = CsmaProtocolConfig() if protocol is None else protocol
        if not isinstance(proto, CsmaProtocolConfig):
            raise ValueError("protocol must be a CsmaProtocolConfig instance")
        if (
            int(num_envs) < 1
            or not (1 <= int(num_uavs) <= 30)
            or int(grid_w) < 1
            or (int(grid_w) * int(grid_w)) > 4096
        ):
            raise ValueError("Invalid num_envs / num_uavs (1..30) / grid_w (1 <= grid_w^2 <= 4096)")
        warp_device = wp.get_device(device)
        if warp_device.is_cuda:
            assert_network_allocation_fits(
                int(num_envs), int(num_uavs), int(grid_w),
                free_bytes=int(warp_device.free_memory), q_cap=int(q_cap),
            )
        from .state_v67 import NetworkBuffersV67, SimDimsV67

        self.envs = int(num_envs)
        self.num_uavs = int(num_uavs)
        self.radio_n = self.num_uavs + 1
        self.gcs_id = self.num_uavs
        self.grid_w = int(grid_w)
        self.num_cells = self.grid_w * self.grid_w
        self.device = device
        self.seed = int(seed)

        self.dims = SimDimsV67()
        self.dims.n_uav = self.num_uavs
        self.dims.n_radio = self.radio_n
        self.dims.gcs_id = self.gcs_id
        self.dims.grid_w = self.grid_w
        self.dims.num_cells = self.num_cells
        self.q_cap = int(q_cap)
        self.protocol = proto
        self.dims.q_cap = self.q_cap
        self.dims.p_cap = max(1, self.num_uavs)
        self.dims.k_entries = K_V67
        self.dims.max_hops = max(6, self.num_uavs)
        self.dims.scalable_wire = 1 if (self.num_uavs > 8 or self.num_cells > 512) else 0
        self.dims.enable_data_mule = 1 if enable_data_mule else 0
        self.dims.enable_peer = 1 if enable_peer else 0
        self.dims.enable_peer_telemetry = 1 if enable_peer_telemetry else 0
        self.dims.decentralized_peer_select = 1 if decentralized_peer_select else 0
        self.dims.aoi_ewma = 1 if aoi_ewma else 0
        self.dims.aoi_eta = float(aoi_eta)
        self.dims.rx_lock_on_cca = 1 if rx_lock_on_cca else 0
        self.dims.node_mask_all = (1 << self.radio_n) - 1
        self.dims.difs_us = int(proto.difs_us)
        self.dims.sifs_us = int(proto.sifs_us)
        self.dims.slot_us = int(proto.slot_us)
        self.dims.cw_min = int(proto.cw_min)
        if isinstance(cw_inclusive_upper, (bool, np.bool_)) or int(cw_inclusive_upper) not in (0, 1):
            raise ValueError('cw_inclusive_upper must be integer 0 ([0, CW-1]) or 1 ([0, CW])')
        self.dims.cw_inclusive_upper = int(cw_inclusive_upper)
        self.dims.max_retx = int(proto.max_retries)
        self.dims.ack_timeout_us = int(proto.ack_timeout_us)

        self.state = NetworkBuffersV67()
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
        shapes = make_shapes_v67(num_uavs=self.num_uavs, grid_w=self.grid_w, q_cap=self.q_cap)
        for name, (shape, dtype) in shapes.items():
            dt = dt_map[np.dtype(dtype)]
            full_shape = (self.envs,) if name in ENV_SCALAR_FIELDS_V67 else (self.envs, *shape)
            arr = wp.zeros(full_shape, dtype=dt, device=device)
            self.data[name] = arr
            setattr(self.state, name, arr)
        self._dest_buf = wp.zeros((self.envs, self.num_uavs), dtype=wp.int32, device=device)
        self.reset()

    def reset(self, done: np.ndarray | None = None) -> None:
        from .kernels.reset_v67 import reset_all_v67
        mask = np.ones(self.envs, dtype=np.uint8) if done is None else np.asarray(done, dtype=np.uint8)
        if mask.shape != (self.envs,):
            raise ValueError(f"done mask must have shape ({self.envs},), got {mask.shape}")
        reset_all_v67(self.state, self.dims, mask, self.envs, self.seed, self.device)

    def set_channel_gain(self, gain: np.ndarray) -> None:
        g = np.asarray(gain, dtype=np.float32)
        if g.shape != (self.envs, self.radio_n, self.radio_n) or not np.isfinite(g).all() or np.any(g < 0.0):
            raise ValueError("gain must be finite non-negative (envs, radio_n, radio_n)")
        wp.copy(self.data["gain"], wp.array(g, dtype=wp.float32, device=self.device))

    def set_radio_alive(self, alive: np.ndarray) -> None:
        a = np.asarray(alive, dtype=np.uint8)
        if a.shape != (self.envs, self.radio_n):
            raise ValueError("radio_alive shape mismatch")
        wp.copy(self.data["radio_alive"], wp.array(a, dtype=wp.uint8, device=self.device))

    def update_phy_from_positions(
        self,
        positions: np.ndarray,
        buildings: Sequence[Tuple[float, float, float, float]],
        blackout_centers: Sequence[Tuple[int, int]],
        blackout_r: int,
        cell_size_m: float,
        *,
        obstacle_model: str = "binary_nlos",
        per_building_loss_db: float = 12.0,
    ) -> None:
        if obstacle_model not in ("binary_nlos", "multi_obstacle", "rf_opaque"):
            raise ValueError("obstacle_model must be binary_nlos, multi_obstacle, or rf_opaque")
        if not np.isfinite(per_building_loss_db) or per_building_loss_db < 0:
            raise ValueError("per_building_loss_db must be finite and >= 0")
        model_id = ("binary_nlos", "multi_obstacle", "rf_opaque").index(obstacle_model)
        from .kernels.phy_v67 import compute_channel_gain_v67
        p, b, bo = normalize_phy_inputs(positions, buildings, blackout_centers,
                                        envs=self.envs, radios=self.radio_n)
        if not (np.isfinite(cell_size_m) and cell_size_m > 0) or blackout_r < 0:
            raise ValueError("cell_size_m > 0 and blackout_r >= 0 required")
        n_bld, n_bo = b.shape[0], bo.shape[0]
        b_safe = b if n_bld > 0 else np.zeros((1, 4), dtype=np.float32)
        bo_safe = bo if n_bo > 0 else np.zeros((1, 2), dtype=np.int32)
        p_gpu = wp.array(p, dtype=wp.float32, device=self.device)
        b_gpu = wp.array(b_safe, dtype=wp.float32, device=self.device)
        bo_gpu = wp.array(bo_safe, dtype=wp.int32, device=self.device)
        wp.launch(
            compute_channel_gain_v67,
            dim=(self.envs, self.radio_n),
            inputs=[
                p_gpu, self.data["radio_alive"], b_gpu, bo_gpu, self.data["gain"],
                self.radio_n, n_bld, n_bo, int(blackout_r),
                self.grid_w, float(cell_size_m), model_id, float(per_building_loss_db),
            ],
            device=self.device,
        )

    def sense_batch(
        self, cell_id: np.ndarray, detected: np.ndarray, t_s: int,
        check_host_guards: bool = True,
    ) -> None:
        """Validate each environment before sensing; timestamp faults require reset.

        Disabling host guards avoids readback but preserves the sticky device
        fault and blocks later sensing/event steps for invalid environments.
        """
        from .kernels.inject_v67 import sense_batch_v67, validate_sensing_generations_v67
        cell, hits, t_val = normalize_sensing_inputs(cell_id, detected, t_s,
                    envs=self.envs, uavs=self.num_uavs, cells=self.num_cells)
        c_gpu = wp.array(cell, dtype=wp.int32, device=self.device)
        d_gpu = wp.array(hits, dtype=wp.int32, device=self.device)
        wp.launch(validate_sensing_generations_v67, dim=self.envs,
                  inputs=[self.state, self.dims, c_gpu, int(t_val)], device=self.device)
        wp.launch(sense_batch_v67, dim=(self.envs, self.num_uavs), inputs=[self.state, self.dims, c_gpu, d_gpu, int(t_val)], device=self.device)
        if check_host_guards:
            faults = self.data["overflow"].numpy()
            if np.any(faults == FAULT_GENERATION):
                raise ValueError("EWMA sensing generations must be strictly increasing per source/cell; reset invalid environments")
            if np.any(faults):
                raise RuntimeError("Warp network state is invalid; reset before sensing")

    def inject_compact3(self, actions: np.ndarray, pos_xy: np.ndarray) -> None:
        from .kernels.inject_v67 import compute_dirty_dest_v67, inject_v67, resolve_compact3_v67
        actions, pos_xy = normalize_compact_actions(actions, pos_xy,
                                                envs=self.envs, uavs=self.num_uavs)
        a_gpu = wp.array(actions, dtype=wp.int32, device=self.device)
        p_gpu = wp.array(pos_xy, dtype=wp.float32, device=self.device)
        wp.launch(compute_dirty_dest_v67, dim=(self.envs, self.num_uavs, self.radio_n), inputs=[self.state, self.dims], device=self.device)
        wp.launch(resolve_compact3_v67, dim=(self.envs, self.num_uavs), inputs=[self.state, self.dims, a_gpu, p_gpu, self._dest_buf], device=self.device)
        wp.launch(inject_v67, dim=(self.envs, self.num_uavs), inputs=[self.state, self.dims, self._dest_buf], device=self.device)

    def step(
        self,
        duration_us: int = 1_000_000,
        max_events: int = 4096,
        tx_power_w: float = UAVNETSIM_TX_POWER_W,
        cca_threshold_w: float = UAVNETSIM_CCA_W,
        noise_w: float = UAVNETSIM_NOISE_W,
        snr_threshold: float = UAVNETSIM_SNR_TH_LIN,
        check_host_guards: bool = True,
    ) -> None:
        from .kernels.events_v67 import step_v67
        if type(duration_us) is not int or duration_us <= 0 or duration_us >= 2_000_000_000:
            raise ValueError("duration_us must be positive int32 microseconds")
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        for name, value in (("tx_power_w", tx_power_w), ("cca_threshold_w", cca_threshold_w),
                            ("noise_w", noise_w), ("snr_threshold", snr_threshold)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if check_host_guards and np.any(self.data["now"].numpy().astype(np.int64) + duration_us >= 2_000_000_000):
            raise OverflowError("Network event clock approaching NEVER_US int32 sentinel")
        self.data["tx_airtime"].zero_()
        wp.launch(
            step_v67,
            dim=self.envs,
            inputs=[self.state, self.dims, int(duration_us), int(max_events),
                    float(tx_power_w), float(cca_threshold_w), float(noise_w), float(snr_threshold)],
            device=self.device,
        )
        if check_host_guards:
            faults = self.data["overflow"].numpy()
            if np.any(faults == FAULT_GENERATION):
                raise ValueError("EWMA sensing generations must be strictly increasing per source/cell; reset invalid environments")
            if np.any(faults):
                raise RuntimeError("Warp packet-event budget exceeded; invalid transition blocked")

    def snapshot(self, names=("now", "overflow", "stats_gcs", "stats_peer", "stats_mac_ack", "stats_e2e", "q_count")):
        return {n: self.data[n].numpy().copy() for n in names}
