"""One real, bounded packet workload shared by the CPU and CUDA gates.

This is network subsystem execution, not environment rollout or RL training.
"""
from __future__ import annotations

import numpy as np


def run_packet_workload(
    net, *, step_kwargs=None, buildings=(), blackout_centers=(), blackout_r=0,
    cell_size_m=50.0, positions=None, obstacle_model="binary_nlos",
    per_building_loss_db=12.0,
):
    """Observe once, transfer to a peer, then carry its original evidence to GCS."""
    envs, uavs = net.envs, net.num_uavs
    step_kwargs = {} if step_kwargs is None else dict(step_kwargs)
    use_peer_path = bool(net.dims.enable_peer and net.dims.enable_data_mule and uavs > 1)
    if positions is None:
        pos = np.zeros((envs, uavs + 1, 3), dtype=np.float32)
        pos[:, :, 0] = (100.0 + 25.0 * np.arange(uavs + 1))[None, :]
        pos[:, :, 1] = 100.0
        pos[:, :uavs, 2] = 50.0
    else:
        pos = np.asarray(positions, dtype=np.float32)
        if pos.shape != (envs, uavs + 1, 3) or not np.isfinite(pos).all():
            raise ValueError("positions must be finite (envs,num_uavs+1,3)")
    net.update_phy_from_positions(
        pos, buildings, blackout_centers, blackout_r, cell_size_m,
        obstacle_model=obstacle_model, per_building_loss_db=per_building_loss_db,
    )
    # Establish routes and align the generation timestamp with the event clock.
    net.step(duration_us=1_000_000, max_events=4096, **step_kwargs)
    cell = np.full((envs, uavs), -1, dtype=np.int32)
    cell[:, 0] = min(42, net.num_cells - 1)
    observed = np.zeros_like(cell)
    observed[:, 0] = 1
    net.sense_batch(cell, observed, t_s=1)
    actions = np.zeros((envs, uavs), dtype=np.int32)
    actions[:, 0] = 2 if use_peer_path else 1
    net.inject_compact3(actions, pos[:, :uavs, :2])
    net.step(duration_us=500_000, max_events=4096, **step_kwargs)
    if use_peer_path:
        actions[:] = 0
        actions[:, 1] = 1
        net.inject_compact3(actions, pos[:, :uavs, :2])
    net.step(duration_us=500_000, max_events=4096, **step_kwargs)
    snapshot = net.snapshot(names=('stats_gcs', 'stats_peer', 'stats_mac_ack',
        'stats_e2e', 'stats_drop', 'overflow', 'now', 'event_count'))
    if np.any(snapshot['overflow']):
        raise AssertionError('Warp event loop truncated')
    expected = 2 if use_peer_path else 1
    for name, minimum in (('stats_gcs', 1), ('stats_mac_ack', expected),
                          ('stats_e2e', expected), ('stats_peer', int(use_peer_path))):
        if np.any(snapshot[name] < minimum):
            raise AssertionError(f'Expected {name} >= {minimum} in every env')
    cid = min(42, net.num_cells - 1)
    # Check evidence itself: counters alone can hide payload corruption.
    if not np.all(net.data['ledger_pos'].numpy()[:, uavs, 0, cid] == 256):
        raise AssertionError('Original UQ8.8 evidence missing or counted twice at GCS')
    if not np.all(net.data['gcs_latest'].numpy()[:, cid] == 1):
        raise AssertionError('Original evidence generation timestamp was not preserved')
    return snapshot


def summarize_snapshot(snapshot):
    return {k: {'min': int(v.min()), 'max': int(v.max()), 'mean': float(v.mean())}
            for k, v in snapshot.items()}


def compare_contract_fixture(num_uavs=6, device='cpu'):
    """Compare two real implementations' endpoint contract on an easy link fixture.

    Different RNGs/event orders are allowed. This fixture cannot establish
    MAC collision timing, statistical simulator equivalence, or event parity.
    """
    from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    from zhang_xue.gpu.state_v67 import UAVNETSIM_NOISE_W

    warp_net = WarpCompressedRadioNetworkV67(1, num_uavs=num_uavs,
                                           device=device, seed=44)
    warp_stats = run_packet_workload(warp_net)
    cpu = SevenRadioCsmaCaNetwork(num_uavs=num_uavs, grid_w=20,
        tx_power_w=0.1, noise_power_w=UAVNETSIM_NOISE_W, queue_cap=32,
        buildings=[], blackout_centers=[], aoi_ewma_end_to_end=True,
        enable_data_mule_forwarding=True, seed=44)
    pos = np.zeros((num_uavs + 1, 3), dtype=np.float64)
    pos[:, 0] = 100.0 + 25.0 * np.arange(num_uavs + 1)
    pos[:, 1] = 100.0
    pos[:num_uavs, 2] = 50.0
    cpu.reset(pos)
    cpu.run_until_us(1_000_000)
    cpu.record_local_observation(0, 42, True, 1)
    first_dst = 1 if num_uavs > 1 else cpu.gcs_id
    cpu.enqueue_app_data(0, first_dst, cpu.get_dirty_entries_non_oracle(0, first_dst))
    cpu.run_until_us(1_500_000)
    if num_uavs > 1:
        cpu.enqueue_app_data(1, cpu.gcs_id,
            cpu.get_dirty_entries_non_oracle(1, cpu.gcs_id))
    cpu.run_until_us(2_000_000)
    gcs = num_uavs
    cpu_counts = [int(cpu.ledger_pos[gcs, 0, 42]), int(cpu.ledger_neg[gcs, 0, 42])]
    warp_counts = [int(warp_net.data[k].numpy()[0, gcs, 0, 42])
                   for k in ('ledger_pos', 'ledger_neg')]
    cpu_gen = int(cpu.gcs_latest_u_s[42])
    warp_gen = int(warp_net.data['gcs_latest'].numpy()[0, 42])
    if cpu_counts != warp_counts or cpu_counts != [256, 0] or cpu_gen != warp_gen or cpu_gen != 1:
        raise AssertionError(f'Endpoint evidence mismatch: CPU {cpu_counts}/{cpu_gen}, '
                             f'Warp {warp_counts}/{warp_gen}')
    minimum = 2 if num_uavs > 1 else 1
    if cpu.stats['e2e_ack_rx'] < minimum or warp_stats['stats_e2e'][0] < minimum:
        raise AssertionError('Missing real end-to-end acknowledgment')
    return {'status':'APPLICATION_CONTRACT_PASS_NOT_EVENT_PARITY',
            'num_uavs':num_uavs, 'warp_device':device,
            'cpu_gcs_counts':cpu_counts, 'warp_gcs_counts':warp_counts,
            'cpu_latest_generation_s':cpu_gen, 'warp_latest_generation_s':warp_gen,
            'cpu_e2e_ack_rx':cpu.stats['e2e_ack_rx'],
            'warp_e2e_ack_rx':int(warp_stats['stats_e2e'][0]),
            'event_parity_verified':False,
            'limitations':['Easy fully connected fixture only',
                           'No hidden-terminal or timing-distribution comparison',
                           'No upstream UavNetSim run']}
