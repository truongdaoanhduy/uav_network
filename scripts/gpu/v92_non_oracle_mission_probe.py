#!/usr/bin/env python3
"""V92 Zero-Oracle Decentralized Frontier Heuristic Mission Probe (SCENARIO-FIRST, NO RL).

Measures non-oracle mission progress, empirical FP/FN, GCS delivery,
coverage, and safety on `scenario_v103_ds_only_1km.yaml` or `scenario_v89_static_1km.yaml`
using STRICTLY each UAV's own local observation `obs` (zero target truth, zero oracle).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]


def non_oracle_frontier_actions_np(obs: np.ndarray, mask: np.ndarray, *, v_max: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute decentralized 2D velocity commands and event-driven comm actions from local obs only.

    Uses ONLY:
    - Own normalized position: `obs[..., 0:2]` (in [0, 1] * 1000 m)
    - Own radio-alive flag: `obs[..., 5]`
    - Own dirty-GCS fraction: `obs[..., 11]`
    - Own 10x10 pooled uncertainty map: `obs[..., 214:314]` (each block = 100m x 100m)
    - Own local comm mask: `mask[..., 0:3]` (0=SILENT, 1=GCS, 2=PEER)
    Zero ground-truth target positions or global oracle state are ever read.
    """
    envs, n_uav, _ = obs.shape
    cont = np.zeros((envs, n_uav, 2), dtype=np.float32)
    dest = np.zeros((envs, n_uav), dtype=np.int32)

    # Precompute 10x10 block centers in meters (50, 150, ..., 950)
    grid_x = (np.arange(10, dtype=np.float32) + 0.5) * 100.0
    grid_y = (np.arange(10, dtype=np.float32) + 0.5) * 100.0
    bx, by = np.meshgrid(grid_x, grid_y)  # (10, 10)
    bx_flat = bx.reshape(-1)  # (100,)
    by_flat = by.reshape(-1)  # (100,)

    # Disjoint longitudinal sector preference per UAV ID to reduce overlap without oracle.
    sector_x = (np.arange(n_uav, dtype=np.float32) + 0.5) * (1000.0 / float(n_uav))

    for e in range(envs):
        for u in range(n_uav):
            x = float(obs[e, u, 0]) * 1000.0
            y = float(obs[e, u, 1]) * 1000.0
            radio_ok = float(obs[e, u, 5]) > 0.5
            dirty_gcs = float(obs[e, u, 11])
            u_map = obs[e, u, 214:314]  # (100,)

            # Event-driven zero-oracle comm trigger: GCS (1) > gated PEER (2 iff valid_gcs==0) > SILENT (0).
            # Gating PEER on obs[10] (valid_gcs < 0.5) prevents pending-ACK lockout (mask[1]==0)
            # from triggering unconditional PEER gossip flooding when a valid GCS route exists.
            valid_gcs = float(obs[e, u, 10]) > 0.5
            if mask[e, u, 1] > 0.5:
                dest[e, u] = 1
            elif (not valid_gcs) and mask[e, u, 2] > 0.5:
                dest[e, u] = 2
            else:
                dest[e, u] = 0

            # If inside blackout / radio disconnected with significant dirty evidence, head toward GCS (50,50)
            if (not radio_ok) and dirty_gcs >= 0.015:
                tx, ty = 50.0, 50.0
            else:
                dist = np.hypot(bx_flat - x, by_flat - y)
                sec_pen = np.abs(bx_flat - sector_x[u]) / 1000.0
                # Score: maximize local uncertainty, prefer nearby cells and own sector
                score = u_map - 0.35 * (dist / 1414.2) - 0.15 * sec_pen
                best_bid = int(np.argmax(score))
                tx, ty = float(bx_flat[best_bid]), float(by_flat[best_bid])

            dx, dy = tx - x, ty - y
            norm = float(np.hypot(dx, dy))
            if norm > 1e-6:
                cont[e, u, 0] = np.clip(dx / norm, -1.0, 1.0)
                cont[e, u, 1] = np.clip(dy / norm, -1.0, 1.0)
    return cont, dest


def main() -> int:
    # Shared evaluator captures each world at its first terminal transition.
    from v93_mission_audit import main as run_audit
    return run_audit()


if __name__ == "__main__":
    raise SystemExit(main())
