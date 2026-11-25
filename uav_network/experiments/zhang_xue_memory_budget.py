"""Explicit storage arithmetic; never a measured CUDA peak or complete VRAM budget.

Replay layout follows GPUHybridReplayBuffer in vanilla_baselines.py. Network
arrays describe the proposed seven-radio layout, not allocated production state.
"""
from __future__ import annotations

import argparse
import json
import numbers


def _shapes(*values):
    if any(isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < 1
           for x in values):
        raise ValueError('Array dimensions must be positive integers')


def replay_storage_bytes(capacity, agents, obs_dim, state_dim, cont_dim, disc_dim):
    """Persistent tensor bytes, including FP16 obs/state and both uint8 masks.

    capacity counts JOINT transitions (one world-step with all agents). It does
    not count vector steps or individual agent rows. Sampled FP32 batches,
    networks, gradients, Adam and allocator/context memory are additional.
    """
    _shapes(capacity, agents, obs_dim, state_dim, cont_dim, disc_dim)
    per_row = (2 * agents * obs_dim * 2 + 2 * state_dim * 2
               + agents * cont_dim * 4 + agents * disc_dim * 4
               + 2 * 4 + 2 * agents * disc_dim)
    return int(capacity * per_row)


def network_storage_lower_bound(envs, actors, radios, cells, queue_cap, entries):
    """Count specified candidate arrays only; routing/MAC/metadata are excluded.

    A receiver-by-origin ledger is distinct from sender ACK watermarks; both
    must be retained. Latest sensor-generation timestamps are also persistent.
    """
    _shapes(envs, actors, radios, cells, queue_cap, entries)
    if radios < actors:
        raise ValueError('Radio count must include every actor')
    arrays = {
        'packed_queue_payload_bytes': envs * radios * queue_cap * entries * 8,
        'ack_watermark_bytes': envs * actors * radios * cells * 2 * 2,
        'receiver_origin_counts_bytes': envs * radios * actors * cells * 2 * 2,
        'receiver_origin_timestamp_bytes': envs * radios * actors * cells * 4,
    }
    return {**arrays, 'total_bytes': int(sum(arrays.values())),
            'complete_runtime_budget': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envs-per-gpu', type=int, required=True)
    p.add_argument('--capacity', type=int, required=True)
    p.add_argument('--obs-dim', type=int, required=True)
    p.add_argument('--state-dim', type=int, required=True)
    p.add_argument('--disc-dim', type=int, required=True)
    a = p.parse_args()
    network = network_storage_lower_bound(a.envs_per_gpu, 6, 7, 400, 64, 62)
    replay = replay_storage_bytes(a.capacity, 6, a.obs_dim, a.state_dim, 2, a.disc_dim)
    print(json.dumps({'network_candidate': network, 'replay_bytes': replay,
                      'partial_total_gib': (network['total_bytes'] + replay) / 2**30,
                      'excluded': ['queue metadata', 'MAC/PHY events', 'routing',
                                   'belief mass caches', 'scene', 'actor/critic/targets',
                                   'Adam', 'FP32 minibatches and activations',
                                   'evaluation', 'CUDA contexts/allocator reserve'],
                      'cuda_peak_measured': False}, indent=2))


if __name__ == '__main__':
    main()
