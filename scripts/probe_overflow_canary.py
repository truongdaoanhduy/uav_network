"""Bounded fresh-data CUDA diagnostic; no production continuation or full replay export."""
import argparse
import json
import os
from pathlib import Path
import time
import torch

from uav_network.algorithms.checkpoints import make_learner
from uav_network.configuration import load_config
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.environment.vector import make_vector
from uav_network.experiments.evidence import json_safe
from uav_network.experiments.overflow_capture import attach_overflow_capture, replay_overflow
from uav_network.experiments.pql_replay import make_replay
from uav_network.experiments.pql_system import PQLSystem
from uav_network.experiments.pql_workers import inference_copy
from uav_network.experiments.training import seed_runtime, refresh_base_collector


def require_cuda():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Diagnostic requires two CUDA devices; no CPU fallback')


def run_case(algorithm, pql, output, seconds=120., collections=128):
    require_cuda()
    if not 0 < seconds <= 300 or not 1 <= collections <= 512:
        raise ValueError('Diagnostic bounds exceeded')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config('configs/comparison-20k-hybrid75-1kb.yaml')
    config.runtime.algorithm, config.pql.enabled = algorithm, pql
    config.runtime.amp_mode, config.runtime.compile_models = 'fp16', False
    config.runtime.vectorize_critics = True
    config.tracking.enabled = False
    config.validate()
    seed_runtime(config)
    started = time.monotonic()
    engine = vector = None
    report = {'purpose': 'fresh-data-overflow-diagnostic-not-production',
        'source_commit': os.environ.get('UAV_GIT_COMMIT', 'local'),
        'algorithm': algorithm, 'pql': pql, 'compiled_forward': False,
        'historical_failed_batch_replayed': False, 'resolved_config': config.to_dict(),
        'hardware': [torch.cuda.get_device_name(i) for i in range(2)],
        'collections': 0, 'global_step': 0, 'comparisons': [], 'status': 'initializing'}
    try:
        if pql:
            engine = PQLSystem(config)
            critic, collector = engine.critic, engine.collector
            observers = [attach_overflow_capture(critic, output, 'critic')]
            if engine.policy is not critic:
                observers.append(attach_overflow_capture(engine.policy, output, 'policy'))
        else:
            critic = make_learner(config, observation_dim(config), state_dim(config), 'cuda:1')
            collector = inference_copy(critic, 'cuda:0')
            replay = make_replay(config, (config.world.num_uavs, observation_dim(config)), state_dim(config), critic.device)
            observers = [attach_overflow_capture(critic, output, 'base')]
        vector = make_vector(config, device='cuda:0')
        count = config.runtime.num_envs
        next_seed = config.runtime.seed + count
        obs, _ = vector.reset(seed=list(range(config.runtime.seed, next_seed)))
        for index in range(collections):
            if time.monotonic() - started >= seconds:
                break
            warmup = index < config.pql.warmup_collection_steps if pql else report['global_step'] < config.algorithm_config.learning_starts
            action = vector.random_actions(obs['observation']) if warmup else collector.act_batch(
                obs['observation'], step=report['global_step'], return_tensors=True)
            following, reward, terminated, truncated, infos = vector.step({
                'continuous': action.continuous, 'destination': action.destination})
            transition = (obs['observation'], obs['state'], action, reward,
                          following['observation'], following['state'], terminated)
            if pql:
                engine.submit(transition)
            else:
                replay.add_batch(*transition)
                if len(replay) >= config.algorithm_config.batch_size and report['global_step'] + count >= config.algorithm_config.learning_starts:
                    for _ in range(config.algorithm_config.updates_per_step):
                        critic.update(replay.sample(config.algorithm_config.batch_size))
                    refresh_base_collector(collector, critic)
            report['collections'], report['global_step'] = index + 1, (index + 1) * count
            obs = following
            if any(observer.events for observer in observers):
                if engine is not None:
                    engine.flush()
                break
            done = terminated | truncated
            indices = list(infos['device_episode_infos'])
            if indices:
                seeds = [None] * count
                for i in indices:
                    seeds[i] = next_seed
                    next_seed += 1
                obs, _ = vector.reset(seed=seeds, options={'reset_mask': done.cpu().numpy()})
        if engine is not None:
            engine.flush()
        paths = sorted(output.glob('overflow-*.pt'))
        report['updates'] = critic.updates
        report['amp'] = critic.precision.metrics()
        report['status'] = 'overflow_captured' if paths else 'no_overflow_observed_within_bound'
        # Release large replay/worlds before isolated comparison constructs another learner.
        if vector is not None:
            vector.close()
            vector = None
        if engine is not None:
            engine.close()
            engine = None
        for path in paths:
            comparison = replay_overflow(path, device='cuda:1')
            comparison['snapshot_file'] = path.name
            report['comparisons'].append(comparison)
        return json_safe(report)
    finally:
        if vector is not None:
            vector.close()
        if engine is not None:
            engine.close()
        report['elapsed_seconds'] = time.monotonic() - started
        (output / 'result.json').write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=['masac', 'matd3'], required=True)
    parser.add_argument('--pql', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--collections', type=int, default=128)
    args = parser.parse_args()
    result = run_case(args.algorithm, args.pql, args.output, args.seconds, args.collections)
    print(json.dumps({k: result[k] for k in ['status', 'algorithm', 'pql', 'global_step']}, allow_nan=False))


if __name__ == '__main__':
    main()
