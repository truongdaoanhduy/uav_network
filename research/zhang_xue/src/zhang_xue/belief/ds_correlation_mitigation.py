"""Warp-ready O(1) temporal-correlation mitigation for Dempster-Shafer belief fusion.

Addresses C04/C23/C41/C66 (V121/V122): standard Dempster's rule assumes independent
evidence sources; repeated intra-UAV observations of the same cell under Markov
temporal correlation (rho > 0) cause rapid false-positive confirmation bursts.

State footprint per world matches GPU Warp constraints:
  - eff_pos, eff_neg: float32[num_uavs, num_cells] (or uint16 counts in cooldown mode)
  - last_obs_step: int16[num_uavs, num_cells] (-1 = unseen; 4.8 KB/world for 6x400)
  - uav_pos_mask: uint8[num_cells] (400 B/world bitmask of UAVs with positive hits)
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .dempster_shafer import counts_to_canonical_ds_effective


def ar1_effective_sample_weight(dt_steps: int | None, rho_hat: float) -> float:
    """AR(1) effective independent sample weight for a revisit after `dt_steps`.

    Returns 1.0 on first visit (`dt_steps is None`) or when `rho_hat == 0.0`.
    For `dt >= 1` and `0 < rho_hat < 1`, returns `(1 - rho_hat**dt) / (1 + rho_hat**dt)`,
    which is the classical Kish/AR(1) effective-sample-size discount factor.
    """
    rho = float(rho_hat)
    if not math.isfinite(rho) or not (0.0 <= rho < 1.0):
        raise ValueError(f"rho_hat must lie in [0, 1), got {rho_hat}")
    if dt_steps is None or rho == 0.0:
        return 1.0
    dt = int(dt_steps)
    if dt <= 0:
        return 0.0
    r_pow = math.pow(rho, dt)
    return float((1.0 - r_pow) / (1.0 + r_pow))


def revisit_cooldown_horizon_s(rho_hat: float, corr_eps: float = 0.05) -> int:
    """Minimum integer revisit interval tau such that rho_hat**tau <= corr_eps."""
    rho = float(rho_hat)
    eps = float(corr_eps)
    if not math.isfinite(rho) or not (0.0 <= rho < 1.0):
        raise ValueError(f"rho_hat must lie in [0, 1), got {rho_hat}")
    if not math.isfinite(eps) or not (0.0 < eps < 1.0):
        raise ValueError(f"corr_eps must lie in (0, 1), got {corr_eps}")
    if rho == 0.0:
        return 0
    return int(math.ceil(math.log(eps) / math.log(rho)))


class DSCorrelationMitigator:
    """Lightweight O(1) per-cell temporal-correlation mitigator for DS evidence."""

    VALID_MODES = ("raw_iid", "ar1_weighted", "revisit_cooldown", "hybrid_multi_uav_gate")

    def __init__(
        self,
        *,
        num_uavs: int = 6,
        num_cells: int = 400,
        mode: str = "ar1_weighted",
        rho_hat: float = 0.7,
        cooldown_s: int = 9,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {self.VALID_MODES}")
        self.num_uavs = int(num_uavs)
        self.num_cells = int(num_cells)
        self.mode = mode
        self.rho_hat = float(rho_hat)
        self.cooldown_s = int(cooldown_s)
        self.eff_pos = np.zeros((self.num_uavs, self.num_cells), dtype=np.float32)
        self.eff_neg = np.zeros((self.num_uavs, self.num_cells), dtype=np.float32)
        self.last_obs_step = np.full((self.num_uavs, self.num_cells), -1, dtype=np.int16)
        self.indep_pos_hits = np.zeros((self.num_cells,), dtype=np.uint16)
        self.uav_pos_mask = np.zeros((self.num_cells,), dtype=np.uint8)

    def record(self, uav_id: int, cell_id: int, det: bool, step_s: int) -> float:
        """Record one local observation and return the applied effective sample weight."""
        u = int(uav_id)
        c = int(cell_id)
        t = int(step_s)
        prev_t = int(self.last_obs_step[u, c])
        dt = None if prev_t < 0 else max(0, t - prev_t)

        if self.mode == "raw_iid":
            w = 1.0
        elif self.mode == "ar1_weighted":
            w = ar1_effective_sample_weight(dt, self.rho_hat)
        elif self.mode in ("revisit_cooldown", "hybrid_multi_uav_gate"):
            if dt is not None and dt < self.cooldown_s:
                return 0.0
            w = 1.0
        else:
            w = 1.0

        self.last_obs_step[u, c] = np.int16(min(32767, t))
        if det:
            self.eff_pos[u, c] = np.float32(float(self.eff_pos[u, c]) + w)
            self.uav_pos_mask[c] = np.uint8(int(self.uav_pos_mask[c]) | (1 << u))
            if dt is None or dt >= self.cooldown_s:
                self.indep_pos_hits[c] = np.uint16(min(65535, int(self.indep_pos_hits[c]) + 1))
        else:
            self.eff_neg[u, c] = np.float32(float(self.eff_neg[u, c]) + w)
        return float(w)

    def fused_cell(self, cell_id: int, *, mf: float = 0.80, me: float = 0.80) -> Tuple[float, float, float]:
        """Fuse effective counts across all UAV origins into canonical DS (mT, mF, mTheta)."""
        c = int(cell_id)
        ep = float(self.eff_pos[:, c].sum(dtype=np.float64))
        en = float(self.eff_neg[:, c].sum(dtype=np.float64))
        return counts_to_canonical_ds_effective(ep, en, mf=mf, me=me)

    def is_confirmed(
        self,
        cell_id: int,
        *,
        mf: float = 0.80,
        me: float = 0.80,
        confirm_th: float = 0.90,
        uncertainty_ceiling: float = 0.20,
    ) -> bool:
        """Evaluate GCS confirmation with optional hybrid multi-UAV / temporal gate."""
        m_t, _, m_u = self.fused_cell(cell_id, mf=mf, me=me)
        if not (m_t >= confirm_th and m_u <= uncertainty_ceiling):
            return False
        if self.mode == "hybrid_multi_uav_gate":
            distinct_uavs = int(bin(int(self.uav_pos_mask[int(cell_id)])).count("1"))
            return distinct_uavs >= 2 or int(self.indep_pos_hits[int(cell_id)]) >= 2
        return True
