#!/usr/bin/env python3
"""Four *explicit* V118 CUDA stress arms; captures raw stdout and checks JSON gates.

This runs scenario stress, not an RL policy or upstream SimPy event-level parity.
Use --dry-run for reproducibility without GPU. No scenario or algorithm source edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STRESS = ROOT / 'scripts/gpu/v112_adverse_action_stress_gate.py'
SCENARIOS = ROOT / 'research/zhang_xue/configs'
CASES = (
    ('base_random', 'scenario_v103_ds_only_1km.yaml', 'random_exploration'),
    ('rf_random', 'scenario_v111_ds_rf_stress_no_peer_1km.yaml', 'random_exploration'),
    ('rf_boundary', 'scenario_v111_ds_rf_stress_no_peer_1km.yaml', 'boundary_push'),
    ('rf_churn', 'scenario_v111_ds_rf_stress_no_peer_1km.yaml', 'command_churn'),
)


def build_commands(output_dir: Path, *, python: str = sys.executable,
                   envs: int = 4096, steps: int = 900, device: str = 'cuda:0',
                   seed: int = 44) -> dict[str, list[str]]:
    return {name: [
        python, str(STRESS), '--scenario', str(SCENARIOS / scenario),
        '--device', device, '--envs', str(envs), '--steps', str(steps),
        '--mode', mode, '--seed', str(seed), '--output', str(output_dir / f'{name}.json'),
    ] for name, scenario, mode in CASES}


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--envs', type=int, default=4096)
    ap.add_argument('--steps', type=int, default=900)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=44)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    if args.envs <= 0 or args.steps <= 0:
        ap.error('envs/steps must be positive')
    arms = build_commands(args.output_dir, envs=args.envs, steps=args.steps,
                          device=args.device, seed=args.seed)
    if args.dry_run:
        for name, command in arms.items():
            print(name, ' '.join(command))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {'gate': 'V119_EXPLICIT_FOUR_WAY_STRESS', 'scenario_frozen': False,
               'world_steps_per_arm': args.envs * args.steps,
               'environment': {'envs': args.envs, 'steps': args.steps, 'device': args.device},
               'cases': {}, 'status': 'RUNNING'}
    for name, command in arms.items():
        report = args.output_dir / f'{name}.json'
        log = args.output_dir / f'{name}.log'
        if report.exists() or log.exists():
            raise FileExistsError(f'{name}: output already exists; prevent accidental stale PASS reuse')
        with log.open('w', encoding='utf-8') as out:
            proc = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=False)
        record = {'command': command, 'exit_code': proc.returncode,
                  'log': str(log), 'sha256_log': hashlib.sha256(log.read_bytes()).hexdigest(),
                  'result': 'NO_JSON'}
        if report.exists():
            try:
                data = json.loads(report.read_text())
                valid = (proc.returncode == 0 and data.get('status') == 'PASS_STRESS_RESET_MODE'
                         and data.get('envs') == args.envs and data.get('steps') == args.steps
                         and data.get('mode') == command[command.index('--mode') + 1])
                record.update(result='PASS' if valid else 'FAIL', sha256_json=hashlib.sha256(report.read_bytes()).hexdigest(),
                              claimed_status=data.get('status'), scenario_sha256=data.get('scenario_sha256'),
                              ever_safety_failure_worlds=data.get('ever_safety_failure_worlds'),
                              ever_overflow_worlds=data.get('ever_overflow_worlds'))
            except (OSError, ValueError) as error:
                record['parse_error'] = str(error)
        receipt['cases'][name] = record
        (args.output_dir / 'gate_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(f'{name}: {record["result"]}', flush=True)
    receipt['status'] = ('PASS_STRESS_RESET_MODE_NOT_FROZEN' if all(x['result'] == 'PASS' for x in receipt['cases'].values())
                         and args.envs == 4096 and args.steps == 900 else 'FAIL_OR_INCOMPLETE')
    (args.output_dir / 'gate_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(receipt['status'])
    return 0 if receipt['status'].startswith('PASS') else 2


if __name__ == '__main__':
    raise SystemExit(run())
