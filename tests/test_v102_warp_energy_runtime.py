"""Runtime Zeng energy ledger: actual Warp step on CPU and CUDA, not a mirrored formula.

The 180 kJ battery is f32; subtraction near 180000 J has 0.015625 J ULP,
so requiring 1e-3 J total-battery parity would be physically numerically invalid.
"""
from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip('warp')
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
from zhang_xue.mobility.flight_safety_energy import uav_step_energy_j


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_warp_battery_matches_realized_velocity_and_mac_airtime(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('No CUDA GPU')
    params = GPUWorldV80Params.from_yaml(
        'research/zhang_xue/configs/scenario_v91_bayes_1km.yaml',
        envs=1, device=device, seed=44,
    )
    world = WarpParallelMissionV80(params)
    initial = world.buffers['battery'].numpy().copy()
    motion = wp.zeros((1, 6, 2), dtype=wp.float32, device=device)
    communication = wp.zeros((1, 6), dtype=wp.int32, device=device)
    world.step(motion, communication)
    actual = (initial - world.buffers['battery'].numpy())[0]
    velocities = world.buffers['velocities'].numpy()[0]
    airtime_us = world.network.data['tx_airtime'].numpy()[0]
    modeled = np.array([
        uav_step_energy_j(
            np.linalg.norm(velocities[u]), float(airtime_us[u]) / 1e6,
            dt_s=1.0, tx_power_w=params.tx_power_w,
        )
        for u in range(6)
    ])
    assert int(world.buffers['failed'].numpy()[0]) == 0
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0.0)
    assert np.max(np.abs(actual - modeled)) < 0.08, (
        f'Velocity/RF airtime/battery mismatch {device}: {actual} vs {modeled}'
    )


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_warp_peer_mask_allows_local_routable_peer_even_with_gcs_route(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('No CUDA GPU')
    from zhang_xue.gpu.kernels.mission_v79_reset_obs import _observe_core_v79
    params = GPUWorldV80Params.from_yaml(
        'research/zhang_xue/configs/scenario_v91_bayes_1km.yaml',
        envs=1, device=device, seed=44,
    )
    world = WarpParallelMissionV80(params)
    radio, buffs = world.network, world.buffers
    # Route/ACK data are in the actor's local radio table. Truth is unused.
    for dst in (1, 6):
        wp.to_torch(radio.data['route_hop'])[0, 0, dst] = dst
        wp.to_torch(radio.data['route_dist'])[0, 0, dst] = 1
        wp.to_torch(radio.data['route_expiry'])[0, 0, dst] = 1_000_000
        wp.to_torch(radio.data['dirty_dest'])[0, 0, dst] = 1
    wp.launch(
        _observe_core_v79, dim=(1, 6),
        inputs=[radio.state, radio.dims, buffs['positions'], buffs['velocities'],
                buffs['battery'], world.blackout_wp, len(world.blackouts),
                world.blackout_r, buffs['obs'], buffs['mask'],
                float(params.e_max_j), float(params.queue_capacity),
                float(params.v_max_mps)],
        device=device,
    )
    assert buffs['mask'].numpy()[0, 0, :].tolist() == [1, 1, 1]


@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_cpu_vs_warp_pooled_bayesian_belief_same_received_ledger(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('No CUDA GPU')
    from zhang_xue.configuration.loader import load_scenario
    from zhang_xue.environment.core import ZhangXue7NodePacketEnv
    cfg_path = 'research/zhang_xue/configs/scenario_v91_bayes_1km.yaml'
    cpu_env = ZhangXue7NodePacketEnv(load_scenario(cfg_path), seed=44)
    cpu_env.reset(seed=44)
    gpu_p = GPUWorldV80Params.from_yaml(cfg_path, envs=1, device=device, seed=44)
    gpu_env = WarpParallelMissionV80(gpu_p)
    # Controlled receiver-private ledger with two positive measurements.
    cid = 120
    cpu_env.net.ledger_pos[0, 0, cid] = 2
    cpu_env.net.ledger_gen_ts_s[0, 0, cid] = 1
    wp.to_torch(gpu_env.network.data['ledger_pos'])[0, 0, 0, cid] = 2
    wp.to_torch(gpu_env.network.data['ledger_gen'])[0, 0, 0, cid] = 1
    gpu_env._observe()
    cpu_map = cpu_env._local_belief_features(0).reshape(5, 100)
    gpu_map = gpu_env.buffers['obs'].numpy()[0, 0, 14:514].reshape(5, 100)
    np.testing.assert_allclose(cpu_map[:3], gpu_map[:3], atol=1e-5, rtol=0)
    assert float(cpu_map[0, 30]) > 0.9
