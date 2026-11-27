#!/usr/bin/env python3
"""Bounded preflight for an unpacked V72 source on a Kaggle T4x2 runtime.

No account login, upload, competition submission or training is performed here.
Install this package and warp-lang==1.18.0 in the Kaggle notebook first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--envs', type=int, default=4, choices=range(1, 33))
    parser.add_argument('--expected-gpus', type=int, default=2, choices=(1, 2))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/kaggle_v72'))
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'created_utc':datetime.now(timezone.utc).isoformat(),
              'status':'BLOCKED', 'full_rl_training':False, 'gates':[]}
    try:
        import torch
        import warp as wp
        wp.init()
        gpu_count = torch.cuda.device_count()
        if not torch.cuda.is_available() or gpu_count < args.expected_gpus:
            raise RuntimeError(f'Requires {args.expected_gpus} CUDA GPUs, found {gpu_count}')
        report.update(torch=torch.__version__, warp=wp.__version__,
                      gpu_names=[torch.cuda.get_device_name(i) for i in range(gpu_count)])
    except (ImportError, RuntimeError) as exc:
        report['reason'] = str(exc)
        (output / 'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 3
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(ROOT / 'research/zhang_xue/src'),
                                       env.get('PYTHONPATH', '')))
    commands = [
        ('source_manifest', ['scripts/maintenance/verify_source_manifests.py', '--check']),
        ('source_policy', ['scripts/maintenance/check_project_policy.py', '--scope', 'full']),
        ('warp_cpu_regressions', ['-m', 'pytest', '-q', 'tests/test_v72_warp_execution.py']),
    ]
    for idx in range(args.expected_gpus):
        device = f'cuda:{idx}'
        commands.extend([
            (f'cuda_{idx}_kernel_regressions', ['-m', 'pytest', '-q', 'tests/test_v72_warp_execution.py']),
            (f'cuda_{idx}_packet', ['scripts/gpu/warp_packet_cuda_gate.py', '--envs', str(args.envs),
                                   '--device', device, '--uavs', '6', '--grid', '20']),
            (f'cuda_{idx}_endpoint_contract', ['scripts/gpu/packet_contract_gate.py', '--device', device]),
        ])
    for name, command in commands:
        try:
            command_env = env.copy()
            if name.startswith('cuda_') and name.endswith('_kernel_regressions'):
                command_env['UAV_WARP_TEST_DEVICE'] = 'cuda:' + name.split('_')[1]
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=command_env,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            text, code = result.stdout, result.returncode
        except subprocess.TimeoutExpired as exc:
            text = (exc.stdout or b'')
            text = text.decode(errors='replace') if isinstance(text, bytes) else text
            text, code = text + '\nTIMEOUT 300s', 124
        (output / f'{name}.log').write_text(text)
        report['gates'].append({'name':name, 'argv':[sys.executable, *command], 'exit_code':code})
        print(f'{name}: exit {code}', flush=True)
        if code:
            report['status'] = 'BLOCKED' if code == 3 else 'FAIL'
            break
    else:
        report['status'] = 'CUDA_PACKET_GATES_PASS_NOT_FULL_RL_OR_EVENT_PARITY'
    (output / 'results.json').write_text(json.dumps(report, indent=2))
    return 0 if report['status'].startswith('CUDA_PACKET_GATES_PASS') else 3 if report['status']=='BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
