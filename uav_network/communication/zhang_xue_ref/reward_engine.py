import numpy as np


class RewardEngine:
    """Disjoint TP/FP Team-Level Monotonic Latches + calibrated airtime cost (lambda_B = 30.0)."""

    def __init__(
        self,
        n_uavs: int,
        n_cells: int,
        true_target_mask: np.ndarray,
        tau: float = 0.95,
        omega_s: float = 30.0,
        omega_c: float = 15.0,
        omega_target: float = 10.0,
        omega_fp: float = 5.0,
        omega_fp_hold: float = 0.0,
        lambda_b: float = 30.0,
    ):
        self.n_uavs = n_uavs
        self.n_cells = n_cells
        self.true_target_mask = np.asarray(true_target_mask, dtype=bool)
        if self.true_target_mask.shape != (n_cells,):
            raise ValueError("true_target_mask shape mismatch")
        self.tau = tau
        self.omega_s = omega_s
        self.omega_c = omega_c
        self.omega_target = omega_target
        self.omega_fp = omega_fp
        self.omega_fp_hold = omega_fp_hold
        self.lambda_b = lambda_b

        self.team_tp_latch = np.zeros(n_cells, dtype=bool)
        self.team_fp_latch = np.zeros(n_cells, dtype=bool)
        self.prev_team_mf = np.zeros(n_cells, dtype=np.float32)

    def step_target_and_fp_reward(self, team_mf: np.ndarray, mode: str = "LATCHED"):
        """Returns (r_target, r_fp_penalty) with strict separation between true targets and empty cells."""
        above = team_mf >= self.tau
        if mode == "EXACT":
            crossed_tp = (self.prev_team_mf < self.tau) & above & self.true_target_mask
            r_target = float(self.omega_target * np.sum(crossed_tp))
        elif mode == "LATCHED":
            newly_tp = (~self.team_tp_latch) & above & self.true_target_mask
            self.team_tp_latch |= newly_tp
            r_target = float(self.omega_target * np.sum(newly_tp))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        empty_mask = ~self.true_target_mask
        newly_fp = (~self.team_fp_latch) & above & empty_mask
        self.team_fp_latch |= newly_fp
        hold_fp = above & empty_mask
        r_fp_penalty = float(
            self.omega_fp * np.sum(newly_fp) + self.omega_fp_hold * np.sum(hold_fp)
        )
        self.prev_team_mf = team_mf.astype(np.float32, copy=True)
        return r_target, r_fp_penalty

    def _target_only_information_gain(self, delta_by_cell, name: str) -> float:
        """Reward-only truth projection; never include this mask in actor observations.

        A non-zero aggregate scalar is unsafe: it cannot distinguish useful
        evidence from false-positive evidence, so it is rejected by design.
        """
        delta = np.asarray(delta_by_cell, dtype=np.float64)
        if delta.ndim == 0:
            if float(delta) == 0.0:
                return 0.0  # backwards-compatible no-information path
            raise ValueError(
                f"{name} must be a per-cell vector; a scalar hides false positives"
            )
        if delta.shape != (self.n_cells,):
            raise ValueError(f"{name} must have shape ({self.n_cells},)")
        if not np.all(np.isfinite(delta)):
            raise ValueError(f"{name} contains non-finite values")
        return float(np.sum(delta[self.true_target_mask]))

    def compute_unified_step_reward(
        self,
        delta_i_search,
        raw_delta_i_shared,
        team_mf: np.ndarray,
        airtime_fraction: float,
        safety_penalty: float = 0.0,
        boundary_penalty: float = 0.0,
    ) -> float:
        """R1 network-aware reward.

        Per-cell gains are credited only for true target cells to prevent a
        false-positive posterior from earning positive information reward.
        The simulator truth mask is restricted to this reward calculation.
        Penalties use positive cost magnitudes and are always subtracted.
        """
        mf = np.asarray(team_mf, dtype=np.float32)
        if mf.shape != (self.n_cells,) or not np.all(np.isfinite(mf)):
            raise ValueError(f"team_mf must be finite with shape ({self.n_cells},)")
        for name, cost in (
            ("airtime_fraction", airtime_fraction),
            ("safety_penalty", safety_penalty),
            ("boundary_penalty", boundary_penalty),
        ):
            if not np.isfinite(cost) or cost < 0:
                raise ValueError(f"{name} must be a finite nonnegative cost")
        info_search = self._target_only_information_gain(delta_i_search, "delta_i_search")
        info_shared = self._target_only_information_gain(raw_delta_i_shared, "raw_delta_i_shared")
        norm_shared = info_shared / float(max(self.n_uavs - 1, 1))
        r_target, r_fp_penalty = self.step_target_and_fp_reward(mf, mode="LATCHED")
        return float(
            self.omega_s * info_search
            + self.omega_c * norm_shared
            + r_target
            - r_fp_penalty
            - self.lambda_b * airtime_fraction
            - safety_penalty
            - boundary_penalty
        )
