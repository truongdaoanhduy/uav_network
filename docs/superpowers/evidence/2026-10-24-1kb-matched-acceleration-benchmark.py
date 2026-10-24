"""Matched 1 KiB acceleration benchmark for MATD3 BASE and MASAC/MATD3 PQL.

Diagnostic only. It never writes production checkpoints and tracking is disabled.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path('/kaggle/working')
PYTHON = sys.executable
CONFIG = 'configs/comparison-20k-hybrid75-1kb.yaml'
SUMMARY = {'scope': '1KiB matched acceleration benchmark; no production checkpoints', 'rows': []}


def run(cmd, log_path):
    with log_path.open('w') as log:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f'benchmark failed: {cmd}; inspect {log_path}')


def base_matd3(label, accelerated, rep):
    path = OUT / f'{label}-matd3-base-{rep}.json'
    cmd = [PYTHON, '-m', 'uav_network', 'benchmark', '--config', CONFIG,
           '--env-counts', '4096', '--steps', '32', '--learner-updates', '20', '--output', str(path)]
    overrides = ['runtime.algorithm=matd3', 'runtime.device=cuda', 'tracking.enabled=false']
    if accelerated:
        overrides += ['runtime.amp_mode=fp16', 'runtime.compile_models=true', 'runtime.fused_optimizers=true']
    else:
        overrides += ['runtime.amp_mode="off"', 'runtime.compile_models=false', 'runtime.fused_optimizers=false']
    for value in overrides:
        cmd += ['--set', value]
    run(cmd, OUT / f'{label}-matd3-base-{rep}.log')
    row = json.loads(path.read_text())['results'][0]
    return {'family': 'base', 'algorithm': 'matd3', 'label': label, 'rep': rep,
            'learner_update_ms': 1000 * row['learner_s'] / row['learner_updates'],
            'learner_updates_per_s': row['learner_updates_per_s'],
            'joint_transitions_per_s': row['joint_transitions_per_s'],
            'dual_gpu_verified': row.get('dual_gpu_verified'), 'devices': row.get('devices'),
            'peak_memory': row.get('peak_memory'), 'replay_bytes': row.get('replay_bytes')}


def pql(label, accelerated, rep):
    path = OUT / f'{label}-pql-{rep}.json'
    cmd = [PYTHON, 'scripts/benchmark_pql.py', '--config', CONFIG,
           '--algorithms', 'masac', 'matd3', '--batches', '8192', '--envs', '4096',
           '--steps', '40', '--output', str(path)]
    overrides = ['pql.enabled=true', 'runtime.device=cuda', 'tracking.enabled=false']
    if accelerated:
        overrides += ['runtime.amp_mode=fp16', 'runtime.compile_models=true', 'runtime.fused_optimizers=true']
    else:
        overrides += ['runtime.amp_mode="off"', 'runtime.compile_models=false', 'runtime.fused_optimizers=false']
    for value in overrides:
        cmd += ['--set', value]
    run(cmd, OUT / f'{label}-pql-{rep}.log')
    report = json.loads(path.read_text())
    rows = []
    for item in report['results']:
        if item['status'] != 'passed' or not item['cuda_pipeline_verified']:
            raise RuntimeError(f'PQL case did not pass: {item}')
        rows.append({'family': 'pql', 'algorithm': item['algorithm'], 'label': label, 'rep': rep,
                     'joint_transitions_per_s': item['joint_transitions_per_s'],
                     'critic_updates_per_s': item['critic_updates_per_s'],
                     'actor_updates_per_s': item['actor_updates_per_s'],
                     'critic_active_seconds': item['critic_active_seconds'],
                     'policy_seconds': item['policy_seconds'], 'elapsed_s': item['elapsed_s'],
                     'cuda_pipeline_verified': item['cuda_pipeline_verified'],
                     'devices': item['devices'], 'peak_memory': item['peak_memory'],
                     'replay_bytes': item['replay_bytes']})
    return rows


for label, accelerated, rep in [('baseline', False, 1), ('accelerated', True, 1),
                                ('accelerated', True, 2), ('baseline', False, 2)]:
    row = base_matd3(label, accelerated, rep)
    SUMMARY['rows'].append(row); print(json.dumps(row), flush=True)
for label, accelerated, rep in [('baseline', False, 1), ('accelerated', True, 1),
                                ('accelerated', True, 2), ('baseline', False, 2)]:
    for row in pql(label, accelerated, rep):
        SUMMARY['rows'].append(row); print(json.dumps(row), flush=True)

aggregate = {}
for family, algorithm, metric in [('base', 'matd3', 'learner_update_ms'),
                                  ('pql', 'masac', 'joint_transitions_per_s'),
                                  ('pql', 'matd3', 'joint_transitions_per_s')]:
    key = f'{algorithm}_{family}'
    baseline = [r[metric] for r in SUMMARY['rows'] if r['family']==family and r['algorithm']==algorithm and r['label']=='baseline']
    accelerated = [r[metric] for r in SUMMARY['rows'] if r['family']==family and r['algorithm']==algorithm and r['label']=='accelerated']
    b, a = statistics.median(baseline), statistics.median(accelerated)
    if metric.endswith('_ms'):
        speedup = b / a
    else:
        speedup = a / b
    aggregate[key] = {'metric': metric, 'baseline_median': b, 'accelerated_median': a, 'speedup_x': speedup}
SUMMARY['aggregate'] = aggregate
(OUT / 'matched-acceleration-summary.json').write_text(json.dumps(SUMMARY, indent=2, allow_nan=False))
print(json.dumps(aggregate, indent=2), flush=True)
