#!/usr/bin/env python3
"""Detect scientific confounders in UAV ablations before training/evaluation.

Compares *effective* settings (including schema defaults), not only YAML keys;
reports scenario map density, UAV density and controls that are not equivalent.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/zhang_xue/src"))
from zhang_xue.configuration.loader import load_scenario
CFG = ROOT / 'research/zhang_xue/configs'
ABLATED = frozenset({'enable_peer_dissemination'})


def _config(path: Path) -> dict:
    return asdict(load_scenario(path))


def audit_pair(a: Path, b: Path) -> dict:
    ca, cb = _config(Path(a)), _config(Path(b))
    diffs = {key: {'a': ca[key], 'b': cb[key]}
             for key in sorted(ca) if ca[key] != cb[key]}
    confounders = {key: value for key, value in diffs.items() if key not in ABLATED}
    return {
        'profile_a': Path(a).name, 'profile_b': Path(b).name,
        'status': 'CONFOUNDED' if confounders else 'MATCHED',
        'deliberately_changed_fields': sorted(set(diffs) & ABLATED),
        'confounders': confounders,
        'effective_battery_termination_mode': ca['battery_termination_mode']
            if ca['battery_termination_mode'] == cb['battery_termination_mode'] else 'MISMATCH',
    }


def inventory(directory: Path = CFG) -> dict:
    out = {}
    for p in sorted(Path(directory).glob('*.yaml')):
        try:
            c = _config(p)
            area_km2 = (float(c['map_size_m']) / 1000) ** 2
            out[p.name] = {
                'map_size_m': c['map_size_m'],
                'num_uavs': c['num_uavs'], 'num_targets': c['num_targets'],
                'grid_w': c['grid_w'],
                'target_density_per_km2': round(c['num_targets'] / area_km2, 8),
                'uavs_per_km2': round(c['num_uavs'] / area_km2, 8),
                'battery_termination_mode': c['battery_termination_mode'],
                'enable_peer_dissemination': c['enable_peer_dissemination'],
                'enable_data_mule_forwarding': c['enable_data_mule_forwarding'],
                'w_p2p': c['w_p2p'], 'confirm_th': c['confirm_th'],
                'aoi_reward_mode': c['aoi_reward_mode'],
                'mac_queue_capacity': c['mac_queue_capacity'],
            }
        except (OSError, ValueError, TypeError) as exc:
            out[p.name] = {'status': 'INVALID', 'error': str(exc)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, default=CFG)
    parser.add_argument('--compare', nargs=2, metavar=('A.yaml', 'B.yaml'))
    args = parser.parse_args()
    pairs = []
    for version, size in (('v67', '1km'), ('v67', '2km'), ('v70', '3km')):
        a = args.config_dir / f'scenario_{version}_datamule_{size}.yaml'
        b = args.config_dir / f'scenario_{version}_no_peer_{size}.yaml'
        if a.is_file() and b.is_file():
            pairs.append(audit_pair(a, b))
    if args.compare:
        pairs.append(audit_pair(*map(Path, args.compare)))
    report = {'matched_ablation_pairs': pairs, 'scenario_inventory': inventory(args.config_dir),
              'note': 'Matching YAML != matching policy, seeds, or packet-event distribution; report measured results separately.'}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if any(x['status'] != 'MATCHED' for x in pairs) else 0


if __name__ == '__main__':
    raise SystemExit(main())
