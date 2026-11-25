"""Warp per-environment allocation and opt-in event execution runtime."""
import numpy as np
from .state import *
from .layout import SHAPES

class WarpSevenRadioNetwork:
    """Batched all-on-device 7-radio MAC; no per-world Python packet scheduler."""

    def __init__(self, num_envs: int, device: str = "cpu", seed: int = 20261125):
        if not HAS_WARP:
            raise RuntimeError("warp-lang is needed for MAC kernel execution")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.envs = num_envs
        self.device = device
        self.seed = seed
        self.state = NetworkBuffers()
        self.data = {}
        dt_map = {
            np.dtype(np.int32): wp.int32,
            np.dtype(np.uint32): wp.uint32,
            np.dtype(np.uint16): wp.uint16,
            np.dtype(np.uint8): wp.uint8,
            np.dtype(np.float32): wp.float32,
        }
        for name, (shape, dtype) in SHAPES.items():
            dt = dt_map[np.dtype(dtype)]
            arr = wp.zeros((num_envs, *shape) if shape != (1,) else (num_envs,), dtype=dt, device=device)
            self.data[name] = arr
            setattr(self.state, name, arr)
        self.reset()

    def reset(self, done: np.ndarray | None = None):
        if done is None:
            done = np.ones(self.envs, dtype=np.uint8)
        d = np.asarray(done, dtype=np.uint8)
        if d.shape != (self.envs,):
            raise ValueError("reset mask shape mismatch")
        from v56_warp_reset_kernels import reset_all

        reset_all(self.state, d, self.envs, self.seed, self.device)

    def _device_int2d(self, x, shape):
        if HAS_WARP and isinstance(x, wp.array):
            if x.shape != shape or x.dtype != wp.int32:
                raise ValueError("Warp int32 shape or dtype mismatch")
            return x
        x = np.asarray(x, dtype=np.int32)
        if x.shape != shape:
            raise ValueError("input shape mismatch")
        return wp.array(x, dtype=wp.int32, device=self.device)

    def set_channel_gain(self, gain):
        if HAS_WARP and isinstance(gain, wp.array):
            if gain.shape != (self.envs, 7, 7) or gain.dtype != wp.float32:
                raise ValueError("gain shape/dtype mismatch")
            wp.copy(self.data["gain"], gain)
        else:
            g = np.asarray(gain, dtype=np.float32)
            if g.shape != (self.envs, 7, 7):
                raise ValueError("gain shape mismatch")
            wp.copy(self.data["gain"], wp.array(g, dtype=wp.float32, device=self.device))

    def set_radio_alive(self, alive):
        if HAS_WARP and isinstance(alive, wp.array):
            wp.copy(self.data["radio_alive"], alive)
        else:
            a = np.asarray(alive, dtype=np.uint8)
            if a.shape != (self.envs, 7):
                raise ValueError("radio_alive shape mismatch")
            wp.copy(self.data["radio_alive"], wp.array(a, dtype=wp.uint8, device=self.device))

    def sense(self, cell: np.ndarray, positive: np.ndarray, t_s: int):
        c = self._device_int2d(cell, (self.envs, 6))
        p = self._device_int2d(positive, (self.envs, 6))
        from v56_warp_injection_kernels import _add_observation

        wp.launch(_add_observation, dim=(self.envs, 6), inputs=[self.state, c, p, int(t_s)], device=self.device)

    def inject(self, actions: np.ndarray, t_s: int = 0):
        if not (HAS_WARP and isinstance(actions, wp.array)):
            a = np.asarray(actions, dtype=np.int32)
            if np.any((a < 0) | (a > 6)):
                raise ValueError("actions must contain indices 0..6")
        else:
            a = actions
        a = self._device_int2d(a, (self.envs, 6))
        from v56_warp_injection_kernels import _inject

        wp.launch(
            _inject,
            dim=(self.envs, 6),
            inputs=[self.state, a, wp.zeros((self.envs,), dtype=wp.int32, device=self.device)],
            device=self.device,
        )

    def run(
        self,
        duration_us: int = 1_000_000,
        max_events: int = 4096,
        tx_power_w: float = 0.5,
        cca_threshold_w: float = 2e-11,
        noise_w: float = 4e-11,
        snr_threshold: float = 3.9810717,
        experimental: bool = False,
    ):
        if duration_us < 1 or duration_us > 1_000_000 or max_events < 1:
            raise ValueError("duration_us 1..1_000_000; max_events >=1")
        if not experimental:
            raise RuntimeError("Gate 3 blocked: 7-radio event MAC kernel lacks T4 CUDA parity benchmark")
        from v56_warp_event_kernels import _step

        wp.launch(
            _step,
            dim=self.envs,
            inputs=[
                self.state,
                int(duration_us),
                int(max_events),
                float(tx_power_w),
                float(cca_threshold_w),
                float(noise_w),
                float(snr_threshold),
            ],
            device=self.device,
        )

    def snapshot(self, names=("now", "overflow", "stats_gcs", "stats_mac_ack", "stats_e2e", "q_count", "route_hop")):
        return {n: self.data[n].numpy().copy() for n in names}
