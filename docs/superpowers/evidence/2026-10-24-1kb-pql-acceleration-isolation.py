"""One-variable PQL acceleration isolation on the primary 1 KiB production shape.

Diagnostic only. Tracking is disabled and no training checkpoint is written.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path('/kaggle/working')
CONFIG = 'configs/comparison-20k-hybrid75-1kb.yaml'
PYTHON = sys.executable
CASES = {
    'baseline': ('"off"', False, False),
    'amp_only': ('fp16', False, False),
    'compile_only': ('"off"', True, False),
    'fused_only': ('"off"', False, True),
}
ORDER = [
    ('baseline', 1), ('amp_only', 1), ('compile_only', 1), ('fused_only', 1),
    ('fused_only', 2), ('compile_only', 2), ('amp_only', 2), ('baseline', 2),
]
summary = {'scope': 'PQL one-variable acceleration isolation; 1KiB/env4096/batch8192/replay1M; no checkpoints',
           'rows': []}


def run_case(label, rep):
    amp, compiled, fused = CASES[label]
    path = OUT / f'pql-isolation-{label}-{rep}.json'
    cmd = [PYTHON, 'scripts/benchmark_pql.py', '--config', CONFIG,
           '--algorithms', 'masac', 'matd3', '--batches', '8192', '--envs', '4096',
           '--steps', '40', '--output', str(path)]
    overrides = [
        'pql.enabled=true', 'runtime.device=cuda', 'tracking.enabled=false',
        f'runtime.amp_mode={amp}',
        f'runtime.compile_models={str(compiled).lower()}',
        f'runtime.fused_optimizers={str(fused).lower()}',
    ]
    for value in overrides:
        cmd += ['--set', value]
    with (OUT / f'pql-isolation-{label}-{rep}.log').open('w') as log:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{label} rep{rep} exited {completed.returncode}')
    report = json.loads(path.read_text())
    rows = []
    for item in report['results']:
        if item['status'] != 'passed' or not item['cuda_pipeline_verified']:
            raise RuntimeError(f'{label} rep{rep} failed: {item}')
        row = {
            'label': label, 'rep': rep, 'algorithm': item['algorithm'],
            'amp_mode': amp.strip('"'), 'compile_models': compiled, 'fused_optimizers': fused,
            'joint_transitions_per_s': item['joint_transitions_per_s'],
            'critic_updates_per_s': item['critic_updates_per_s'],
            'actor_updates_per_s': item['actor_updates_per_s'],
            'critic_active_seconds': item['critic_active_seconds'],
            'policy_seconds': item['policy_seconds'], 'elapsed_s': item['elapsed_s'],
            'cuda_pipeline_verified': item['cuda_pipeline_verified'],
            'devices': item['devices'], 'peak_memory': item['peak_memory'],
            'replay_bytes': item['replay_bytes'], 'learning_metrics': item.get('learning_metrics', {}),
        }
        rows.append(row)
    return rows


for label, rep in ORDER:
    rows = run_case(label, rep)
    summary['rows'].extend(rows)
    (OUT / 'pql-isolation-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    for row in rows:
        print(json.dumps({k: row[k] for k in ('label','rep','algorithm','joint_transitions_per_s','critic_updates_per_s','actor_updates_per_s','critic_active_seconds','policy_seconds')}, allow_nan=False), flush=True)

aggregate = {}
for algorithm in ('masac', 'matd3'):
    baseline = statistics.median(r['joint_transitions_per_s'] for r in summary['rows']
                                 if r['algorithm'] == algorithm and r['label'] == 'baseline')
    aggregate[algorithm] = {}
    for label in CASES:
        rows = [r for r in summary['rows'] if r['algorithm'] == algorithm and r['label'] == label]
        throughput = statistics.median(r['joint_transitions_per_s'] for r in rows)
        critic = statistics.median(r['critic_active_seconds'] for r in rows)
        policy = statistics.median(r['policy_seconds'] for r in rows)
        aggregate[algorithm][label] = {
            'median_joint_transitions_per_s': throughput,
            'speedup_vs_baseline_x': throughput / baseline,
            'median_critic_active_seconds': critic,
            'median_policy_seconds': policy,
            'repetitions': len(rows),
        }
summary['aggregate'] = aggregate
(OUT / 'pql-isolation-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
print(json.dumps(aggregate, indent=2), flush=True)
