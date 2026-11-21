"""Dempster–Shafer target-presence map from Zhang et al., IEEE T-IV 2024.

This module is intentionally independent of simulator target coordinates/IDs.
It consumes *observed sensor events* only; crossing an evidence threshold is
not ground-truth verification. The paper's global fusion uses max counts, not
sum counts, and assumes fixed evidence reliability and instant connectivity.

For the user's 3D camera, event reliabilities may vary by altitude/occlusion.
Such events support local Dempster combination, but NOT the paper's count-only
fusion, which discards reliability information and would double count repeats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _valid_confidence(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("evidence confidence must be finite and in [0,1)")
    return value


def combine_evidence(masses, detected: bool, confidence: float = 0.8):
    """Combine (target, empty, unknown) with one *observed* binary reading.

    Eq. (6)-(10) of Zhang et al.; positive BPA is (c,0,1-c), negative
    BPA is (0,c,1-c). No true target label is accepted by this interface.
    """
    confidence = _valid_confidence(confidence)
    m_f, m_e, m_u = (float(x) for x in masses)
    if (not all(np.isfinite((m_f, m_e, m_u))) or
            min(m_f, m_e, m_u) < -1e-8 or
            abs(m_f + m_e + m_u - 1.0) > 1e-6):
        raise ValueError("input evidence masses must sum to one and be nonnegative")
    b_f, b_e = (confidence, 0.0) if bool(detected) else (0.0, confidence)
    b_u = 1.0 - confidence
    denominator = 1.0 - m_e * b_f - m_f * b_e
    if denominator <= 0.0:
        raise ValueError("total evidence conflict cannot be normalized")
    result = ((m_f * b_f + m_f * b_u + m_u * b_f) / denominator,
              (m_e * b_e + m_e * b_u + m_u * b_e) / denominator,
              (m_u * b_u) / denominator)
    return tuple(max(0.0, min(1.0, float(x))) for x in result)


def mass_from_counts(n_positive, n_negative, mf: float = 0.8, me: float = 0.8):
    """Eq. (17) with log-domain normalization for long/mixed histories.

    Handles arrays of non-negative integer counts. The reconstruction is exact
    only for constant mf and me (not altitude-dependent confidence).
    """
    mf, me = _valid_confidence(mf), _valid_confidence(me)
    pos, neg = np.broadcast_arrays(np.asarray(n_positive), np.asarray(n_negative))
    if not (np.issubdtype(pos.dtype, np.integer) and
            np.issubdtype(neg.dtype, np.integer)):
        raise ValueError("count inputs must be integers")
    if np.any(pos < 0) or np.any(neg < 0):
        raise ValueError("count inputs must be non-negative")
    loga = pos.astype(np.float64) * np.log1p(-mf)
    logb = neg.astype(np.float64) * np.log1p(-me)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_f = logb + np.log(-np.expm1(loga))
        log_e = loga + np.log(-np.expm1(logb))
    log_u = loga + logb
    norm = np.logaddexp(np.logaddexp(log_f, log_e), log_u)
    masses = np.stack([np.exp(log_f - norm),
                       np.exp(log_e - norm),
                       np.exp(log_u - norm)], axis=-1)
    return tuple(float(x) for x in masses) if masses.ndim == 1 else masses


@dataclass
class DSEvidenceMap:
    """Local 2D ground-plane occupancy evidences for UAVs flying in 3D.

    Map dimensions are (n_uav, row, column, channel [F,E,U]).
    `observe` deliberately has no true-target/target-ID parameter.
    No reports are emitted by this mapping class.
    """
    num_uavs: int
    grid_cells: int
    detection_confidence: float = 0.8
    empty_confidence: float = 0.8
    detection_threshold: float = 0.95
    _masses: np.ndarray = field(init=False, repr=False)
    _pos: np.ndarray = field(init=False, repr=False)
    _neg: np.ndarray = field(init=False, repr=False)
    _heterogeneous: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.num_uavs, int) or self.num_uavs < 1:
            raise ValueError("num_uavs must be a positive integer")
        if not isinstance(self.grid_cells, int) or self.grid_cells < 1:
            raise ValueError("grid_cells must be a positive integer")
        self.detection_confidence = _valid_confidence(self.detection_confidence)
        self.empty_confidence = _valid_confidence(self.empty_confidence)
        if not 0.0 < self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must lie in (0,1]")
        self.reset()

    def reset(self):
        shape = (self.num_uavs, self.grid_cells, self.grid_cells)
        self._masses = np.zeros(shape + (3,), dtype=np.float64)
        self._masses[..., 2] = 1.0
        self._pos = np.zeros(shape, dtype=np.int64)
        self._neg = np.zeros(shape, dtype=np.int64)
        self._heterogeneous = False

    def observe(self, uav_id: int, row: int, col: int,
                detected: bool, *, confidence: float | None = None):
        if not (0 <= uav_id < self.num_uavs and
                0 <= row < self.grid_cells and 0 <= col < self.grid_cells):
            raise IndexError("agent or cell index outside evidence map")
        baseline = (self.detection_confidence if detected else self.empty_confidence)
        c = baseline if confidence is None else _valid_confidence(confidence)
        old = self._masses[uav_id, row, col]
        new_mass = combine_evidence(old, detected, c)
        self._masses[uav_id, row, col] = new_mass
        if detected:
            self._pos[uav_id, row, col] += 1
        else:
            self._neg[uav_id, row, col] += 1
        if abs(c - baseline) > 1e-12:
            self._heterogeneous = True
        return new_mass

    def local_masses(self) -> np.ndarray:
        return self._masses.copy()

    def target_candidates(self, uav_id: int) -> np.ndarray:
        if not 0 <= uav_id < self.num_uavs:
            raise IndexError("invalid uav_id")
        return self._masses[uav_id, ..., 0] >= self.detection_threshold

    def paper_global_masses(self) -> np.ndarray:
        """Zhang Eq.(16)-(17), OFFLINE ideal fully-connected baseline.

        No message transport takes place. Do NOT call this local/realistic
        network fusion. It is prohibited with heterogeneous confidence.
        """
        if self._heterogeneous:
            raise ValueError("paper count fusion requires fixed evidence confidences")
        return mass_from_counts(self._pos.max(axis=0), self._neg.max(axis=0),
                                self.detection_confidence, self.empty_confidence)
