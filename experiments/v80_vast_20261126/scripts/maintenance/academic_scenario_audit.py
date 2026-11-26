#!/usr/bin/env python3
"""Reproducible, *direct-link-only* geometry/PHY gate for research maps.

Counts are not throughput, end-to-end reachability, or evidence of a useful
store-carry-forward policy. They exclude contention, MAC ACK and routing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow 'python scripts/maintenance/academic_scenario_audit.py' in a clean ZIP.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'research/zhang_xue/src'))

from zhang_xue.communication.common import (
    GCS_POS_M, is_in_blackout, link_path_gain,
    scale_canonical_geometry, segment_hits_cylinder_3d_fp32,
)

NOISE_W = 4.38957709293155e-13
SNR_THRESHOLD = 10.0 ** (6.0 / 10.0)
TX_POWER_W = 0.1


def audit_direct_gcs_link(map_size_m: float, grid_w: int, altitude_m: float = 50.0) -> dict:
    """Scan grid-cell centers using the same 3D cylinder channel as the CPU sim."""
    if not map_size_m > 0.0 or grid_w < 10 or not altitude_m > 0.0:
        raise ValueError('invalid map size/grid/altitude')
    buildings, zones, radius = scale_canonical_geometry(
        map_size_m, grid_w, scale_obstacles=True
    )
    scale = map_size_m / 1000.0
    gcs = [GCS_POS_M[0] * scale, GCS_POS_M[1] * scale, GCS_POS_M[2]]
    cell_size = map_size_m / grid_w
    blackout = nlos = overlap = not_decodable = not_cca_sensed = 0
    cca_w = 10.0 ** ((-82.0 - 30.0) / 10.0)
    for row in range(grid_w):
        for col in range(grid_w):
            p = [(col + 0.5) * cell_size, (row + 0.5) * cell_size, altitude_m]
            in_blackout = is_in_blackout(
                p, zones, manhattan_r=radius, grid_w=grid_w, cell_size_m=cell_size
            )
            in_nlos = any(segment_hits_cylinder_3d_fp32(p, gcs, *b) for b in buildings)
            blackout += int(in_blackout)
            nlos += int(in_nlos)
            overlap += int(in_blackout and in_nlos)
            power = TX_POWER_W * link_path_gain(
                p, gcs, buildings, zones,
                grid_w=grid_w, cell_size_m=cell_size, blackout_r=radius,
            )
            not_decodable += int(power / NOISE_W < SNR_THRESHOLD)
            # CCA is MAC carrier sensing, NOT a PHY decode requirement.
            # Count links with carrier sense below threshold separately.
            not_cca_sensed += int(power < cca_w)
    cells = grid_w**2
    return {
        'map_size_m': map_size_m,
        'grid_w': grid_w,
        'cells': cells,
        'blackout_cells': blackout,
        'nlos_cells': nlos,
        'blackout_and_nlos_cells': overlap,
        'blackout_or_nlos_cells': blackout + nlos - overlap,
        'no_direct_decodable_link_cells': not_decodable,
        'no_direct_decodable_link_pct': round(100 * not_decodable / cells, 3),
        'no_direct_receivable_link_cells': not_decodable,
        'no_direct_receivable_link_pct': round(100 * not_decodable / cells, 3),
        'no_direct_cca_sensed_link_cells': not_cca_sensed,
        'no_direct_cca_sensed_link_pct': round(100 * not_cca_sensed / cells, 3),
        'cca_threshold_dbm': -82.0,
        'tx_power_w': TX_POWER_W,
        'noise_power_w': NOISE_W,
        'snr_threshold_db': 6.0,
        'note': ('Direct PHY only; not packet delivery, congestion, routing, '
                 'GCS reachability through relays, or trained-policy ablation'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes-m', type=int, nargs='+', default=[1000, 2000, 3000])
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args()
    results = [audit_direct_gcs_link(float(size), size // 50) for size in args.sizes_m]
    result = json.dumps(results, ensure_ascii=False, indent=2)
    if args.json_out:
        root = Path(__file__).resolve().parents[2]
        output = args.json_out.resolve()
        if not output.is_relative_to((root / 'outputs').resolve()):
            parser.error('only gitignored outputs/ is an allowed target')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result + '\n', encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
