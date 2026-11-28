"""V121 single-episode all-step CUDA audit. No episode reset is performed.

This is a bounded non-RL validation workload, not a proof over every policy,
nor whole-world CPU/GPU or upstream SimPy event equivalence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'research/zhang_xue/src')]


def evaluate_run(record):
    """Return unmet full4096x900 evidence conditions (missing evidence fails)."""
    reasons = []
    expected = dict(device_is_cuda=True, envs=4096, requested_steps=900,
                    observed_steps=900, reset_calls=0, min_episode_steps=900,
                    first_terminal_min_step=900, terminal_worlds=4096,
                    ever_safety_failure_worlds=0, ever_overflow_worlds=0,
                    nonfinite_worlds=0, energy_increase_worlds=0,
                    physical_violation_worlds=0)
    for key, value in expected.items():
        if record.get(key) != value:
            reasons.append(f'{key}: expected {value!r}, observed {record.get(key)!r}')
    for key, maximum in [('max_reward_sum_error', 2e-3), ('max_speed_excess', 2e-4)]:
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value > maximum:
            reasons.append(f'{key}: missing/nonfinite or exceeds {maximum}')
    for key in ['source_sha256', 'scenario_sha256']:
        value = record.get(key, '')
        if not isinstance(value, str) or len(value) !=64 or any(c not in '0123456789abcdef' for c in value):
            reasons.append(f'{key}: missing or invalid')
    return reasons


def source_identity():
    files = {}
    for folder in ['research/zhang_xue/src', 'uav_network/algorithms', 'scripts/gpu']:
        for path in sorted((ROOT / folder).rglob('*.py')):
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return digest, files


def vram_mib():
    try:
        return float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                     '--format=csv,noheader,nounits', '--id=0'], text=True, timeout=8).splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def run(args):
    import numpy as np
    import warp as wp
    from zhang_xue.configuration.loader import load_scenario
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
    from v121_horizon_monitor import monitor

    if not wp.get_device(args.device).is_cuda:
        raise ValueError('This audit requires a real CUDA device')
    cfg = load_scenario(args.scenario)
    if args.noise_power_w is not None:
        cfg = replace(cfg, mac_noise_power_w=args.noise_power_w)
    if cfg.enable_peer_dissemination or cfg.enable_data_mule_forwarding:
        raise ValueError('The single-episode audit requires isolated pure-GCS settings')
    if not 1 <= args.steps <= cfg.safety_horizon_s:
        raise ValueError('steps must be positive and no greater than scenario horizon')
    params = GPUWorldV80Params.from_scenario(cfg, envs=args.envs, device=args.device, seed=args.seed)
    world = WarpParallelMissionV80(params)
    buffers = world.buffers
    sticky = wp.zeros((args.envs, 6), dtype=wp.int32, device=args.device)
    errors = wp.zeros((args.envs, 2), dtype=wp.float32, device=args.device)
    previous_pos = wp.clone(buffers['positions'])
    previous_bat = wp.clone(buffers['battery'])
    action = wp.zeros((args.envs, 6, 2), dtype=wp.float32, device=args.device)
    comm = wp.array(np.ones((args.envs, 6), dtype=np.int32), dtype=wp.int32, device=args.device)
    rng = np.random.default_rng(args.seed + 121000)
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    memory = [vram_mib()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix('.jsonl')
    wp.synchronize_device(args.device)
    start = time.monotonic()
    with trace_path.open('w', encoding='utf-8') as trace:
        for step in range(1, args.steps +1):
            if args.motion == 'random':
                values = rng.uniform(-1, 1, (args.envs, 6, 2)).astype(np.float32)
            else:
                values = np.zeros((args.envs, 6, 2), dtype=np.float32)
                if args.motion == 'gentle_patrol':
                    values[:, :, 0] = .25*np.cos(angles) * (-1 if (step-1)//60 %2 else 1)
                    values[:, :, 1] = .25*np.sin(angles) * (-1 if (step-1)//60 %2 else 1)
            wp.copy(action, wp.array(values, dtype=wp.float32, device=args.device))
            world.step(action, comm)
            wp.launch(monitor, dim=args.envs, inputs=[buffers['positions'], buffers['velocities'],
                      buffers['battery'], buffers['reward'], buffers['reward_components'],
                      buffers['failed'], world.network.data['overflow'], buffers['done'],
                      previous_pos, previous_bat, world.build_wp, len(world.buildings),
                      sticky, errors, step, float(params.v_max_mps),
                      float(params.uav_safe_distance_m)], device=args.device)
            # Reading telemetry every150 steps synchronizes explicitly; no resets.
            if step %150 ==0 or step == args.steps:
                wp.synchronize_device(args.device)
                states = sticky.numpy()
                row = dict(step=step, safety_worlds=int(np.count_nonzero(states[:,0])),
                           overflow_worlds=int(np.count_nonzero(states[:,1])),
                           early_terminal_worlds=int(np.count_nonzero((states[:,4]>0)&(states[:,4]<cfg.safety_horizon_s))),
                           vram_mib=vram_mib(), elapsed_sec=time.monotonic()-start)
                memory.append(row['vram_mib'])
                trace.write(json.dumps(row)+'\n'); trace.flush()
                print(json.dumps(row), flush=True)
    wp.synchronize_device(args.device)
    states, err = sticky.numpy(), errors.numpy()
    steps = buffers['episode_step'].numpy()
    first = states[:,4]
    source_hash, files = source_identity()
    out = dict(gate='V121_UNINTERRUPTED_CUDA_4096X900', device_is_cuda=True,
               device=str(wp.get_device(args.device)), gpu_name=wp.get_device(args.device).name,
               warp_version=wp.__version__, seed=args.seed, motion=args.motion,
               envs=args.envs, requested_steps=args.steps, observed_steps=step,
               reset_calls=0, episode_handling='single episode, no auto-reset',
               min_episode_steps=int(steps.min()), max_episode_steps=int(steps.max()),
               first_terminal_min_step=int(first[first>0].min()) if np.any(first>0) else 0,
               terminal_worlds=int(np.count_nonzero(first)),
               first_terminal_histogram={str(k):int(v) for k,v in zip(*np.unique(first,return_counts=True))},
               ever_safety_failure_worlds=int(np.count_nonzero(states[:,0])),
               ever_overflow_worlds=int(np.count_nonzero(states[:,1])),
               nonfinite_worlds=int(np.count_nonzero(states[:,2])),
               energy_increase_worlds=int(np.count_nonzero(states[:,3])),
               physical_violation_worlds=int(np.count_nonzero(states[:,5])),
               max_reward_sum_error=float(err[:,0].max()), max_speed_excess=float(err[:,1].max()),
               final_min_battery_j=float(buffers['battery'].numpy().min()),
               source_sha256=source_hash, source_files=files,
               scenario_sha256=hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
               scenario_path=str(args.scenario), resolved_params=asdict(params),
               memory_samples_mib=memory, peak_vram_mib=max((v for v in memory if v is not None),default=None),
               elapsed_sec=time.monotonic()-start,
               claim_limit='Bounded workload only; not full CPU/GPU, SimPy equivalence or a universal safety proof.')
    out['unmet_conditions'] = evaluate_run(out)
    out['status'] = 'PASS' if not out['unmet_conditions'] else 'NOT_CERTIFIED'
    args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('source_files','resolved_params')},indent=2),flush=True)
    return 0 if out['status']=='PASS' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, default=ROOT/'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml')
    parser.add_argument('--envs', '--worlds', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--motion', choices=('stationary','gentle_patrol','random'), default='random')
    parser.add_argument('--noise-power-w', type=float)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
