import numpy as np


def ds_log_space_safe(n_pos, n_neg, mf: float = 0.8, me: float = 0.8):
    """Numerically stable DS belief computation for FP32 (supports scalar or arrays)."""
    if not (0.0 < mf < 1.0 and 0.0 < me < 1.0):
        raise ValueError(f"mf and me must be in (0, 1), got mf={mf}, me={me}")

    is_scalar = np.isscalar(n_pos) and np.isscalar(n_neg)
    p = np.asarray(n_pos, dtype=np.int32)
    n = np.asarray(n_neg, dtype=np.int32)
    if np.any(p < 0) or np.any(n < 0):
        raise ValueError("Counts must be non-negative")

    if mf == me:
        log_1m = np.log1p(np.float32(-mf))
        dn = p - n
        a_minus_M = np.maximum(dn, 0).astype(np.float32) * log_1m
        b_minus_M = np.maximum(-dn, 0).astype(np.float32) * log_1m
        q = np.maximum(p, n).astype(np.float32) * log_1m
        a = p.astype(np.float32) * log_1m
        b = n.astype(np.float32) * log_1m
        ea, eb, eq = np.exp(a_minus_M), np.exp(b_minus_M), np.exp(q)
        ds = ea + eb - eq
        m_f = (eb * (-np.expm1(a)) / ds).astype(np.float32)
        m_e = (ea * (-np.expm1(b)) / ds).astype(np.float32)
        m_u = (eq / ds).astype(np.float32)
    else:
        a64 = p.astype(np.float64) * np.log1p(-np.float64(mf))
        b64 = n.astype(np.float64) * np.log1p(-np.float64(me))
        M64 = np.maximum(a64, b64)
        q64 = np.minimum(a64, b64)
        ea, eb, eq = np.exp(a64 - M64), np.exp(b64 - M64), np.exp(q64)
        ds = ea + eb - eq
        m_f = (eb * (-np.expm1(a64)) / ds).astype(np.float32)
        m_e = (ea * (-np.expm1(b64)) / ds).astype(np.float32)
        m_u = (eq / ds).astype(np.float32)

    if is_scalar:
        return np.float32(m_f), np.float32(m_e), np.float32(m_u)
    return m_f, m_e, m_u


class BeliefTable:
    """Per-UAV uint16 per-origin count table supporting MAX (Zhang T-IV) and Max-plus-Sum (Zhang Drones 2024)."""

    def __init__(self, n_uavs: int, n_cells: int, fusion_mode: str = "MAX_PLUS_SUM", mf: float = 0.8, me: float = 0.8):
        if fusion_mode not in ("MAX_PLUS_SUM", "MAX"):
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.n_uavs = n_uavs
        self.n_cells = n_cells
        self.fusion_mode = fusion_mode
        self.mf = mf
        self.me = me
        self.pos = np.zeros((n_uavs, n_uavs, n_cells), dtype=np.uint16)
        self.neg = np.zeros((n_uavs, n_uavs, n_cells), dtype=np.uint16)

    def record_own_observation(self, uav_id: int, cell_id: int, detected_positive: bool):
        if detected_positive:
            if self.pos[uav_id, uav_id, cell_id] < 65535:
                self.pos[uav_id, uav_id, cell_id] += np.uint16(1)
        else:
            if self.neg[uav_id, uav_id, cell_id] < 65535:
                self.neg[uav_id, uav_id, cell_id] += np.uint16(1)

    def merge_packet(self, receiver_id: int, entries):
        affected_cells = sorted({int(c) for _, c, _, _ in entries})
        if not affected_cells:
            return 0.0
        _, _, mu_before = self.get_belief(receiver_id, affected_cells)
        for origin_id, cell_id, n_pos, n_neg in entries:
            o, c = int(origin_id), int(cell_id)
            p_in, n_in = np.uint16(n_pos), np.uint16(n_neg)
            if p_in > self.pos[receiver_id, o, c]:
                self.pos[receiver_id, o, c] = p_in
            if n_in > self.neg[receiver_id, o, c]:
                self.neg[receiver_id, o, c] = n_in
        _, _, mu_after = self.get_belief(receiver_id, affected_cells)
        return float(np.sum(mu_before - mu_after))

    def get_fused_counts(self, uav_id: int, cells=None):
        p_slice = self.pos[uav_id] if cells is None else self.pos[uav_id][:, cells]
        n_slice = self.neg[uav_id] if cells is None else self.neg[uav_id][:, cells]
        if self.fusion_mode == "MAX_PLUS_SUM":
            return np.sum(p_slice, axis=0, dtype=np.int32), np.sum(n_slice, axis=0, dtype=np.int32)
        return np.max(p_slice, axis=0).astype(np.int32), np.max(n_slice, axis=0).astype(np.int32)

    def get_belief(self, uav_id: int, cells=None):
        p_tot, n_tot = self.get_fused_counts(uav_id, cells)
        return ds_log_space_safe(p_tot, n_tot, self.mf, self.me)

    def get_team_own_belief(self):
        idx = np.arange(self.n_uavs)
        own_p = self.pos[idx, idx, :]
        own_n = self.neg[idx, idx, :]
        if self.fusion_mode == "MAX_PLUS_SUM":
            p_team = np.sum(own_p, axis=0, dtype=np.int32)
            n_team = np.sum(own_n, axis=0, dtype=np.int32)
        else:
            p_team = np.max(own_p, axis=0).astype(np.int32)
            n_team = np.max(own_n, axis=0).astype(np.int32)
        return ds_log_space_safe(p_team, n_team, self.mf, self.me)
