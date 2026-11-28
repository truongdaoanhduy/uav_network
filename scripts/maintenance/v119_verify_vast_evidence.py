#!/usr/bin/env python3
"""V119 verify complete V118/V119 raw GPU receipts without manufacturing PASS claims."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

CASES = ('base_random', 'rf_random', 'rf_boundary', 'rf_churn')
MODES = ('random_exploration', 'random_exploration', 'boundary_push', 'command_churn')
SOURCES = ('04_', '12_v119_')


def verify(log_dir: Path) -> dict:
    rounds = {}
    for label, prefix in zip(('V118', 'V119'), SOURCES):
        tested = {}
        for case, mode in zip(CASES, MODES):
            path = log_dir / f'{prefix}{case}.json'
            if not path.is_file():
                raise FileNotFoundError(f'Missing real GPU case {label}/{case}: {path}')
            raw = path.read_bytes()
            record = json.loads(raw)
            cond = (record.get('status') == 'PASS_STRESS_RESET_MODE'
                    and record.get('device') == 'cuda:0'
                    and record.get('envs') == 4096 and record.get('steps') == 900
                    and record.get('mode') == mode
                    and record.get('scenario_frozen') is False
                    and record.get('ever_safety_failure_worlds') == 0
                    and record.get('ever_overflow_worlds') == 0
                    and record.get('ever_nonfinite_worlds') == 0
                    and abs(float(record.get('max_reward_component_sum_error', float('inf')))) <= 1e-5)
            if not cond:
                raise AssertionError(f'No proven GPU PASS on {label}/{case}: {record}')
            tested[case] = {
                'sha256_json': hashlib.sha256(raw).hexdigest(),
                'status': record['status'], 'mode': mode,
                'peak_vram_mib': record['peak_vram_mib'],
                'world_steps_per_sec': record['world_steps_per_sec'],
                'ever_safety_failure_worlds': 0, 'ever_overflow_worlds': 0,
                'ever_nonfinite_worlds': 0,
                'reward_component_max_abs_error': record['max_reward_component_sum_error'],
                'terminal_events': record['terminal_events'],
                'scenario_sha256': record['scenario_sha256'],
            }
        rounds[label] = tested
    for case in CASES:
        if rounds['V118'][case]['scenario_sha256'] != rounds['V119'][case]['scenario_sha256']:
            raise AssertionError(f'{case} config changed, not only the kernel')
    return {'status': 'EIGHT_GPU_STRESS_RESULTS_VERIFIED_NOT_SCENARIO_FROZEN',
            'scenario_frozen': False, 'stress_reset_mode': True,
            'source': 'Actual JSONs on Vast, not independently rerun by validator',
            'cases': rounds}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
