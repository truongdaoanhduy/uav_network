"""v6.2 Comprehensive Audit & Upgrade Verification Suite."""
from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhang_xue.belief.dempster_shafer import (
    DENG_MAX_BITS,
    decode_uq88,
    deng_entropy,
    encode_uq88,
    fuse_ewma_provenance_uq88,
)
from zhang_xue.communication.common import CANONICAL_BUILDINGS_5, GCS_ID
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
from zhang_xue.environment.configuration import EnvConfigV54, make_v62_config
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.layout import backing_bytes_per_environment
from zhang_xue.gpu.layout_v62 import backing_bytes_per_environment_v62, v62_gpu_memory_budget
from zhang_xue.learners.actor_adapter import decode_tx_power_w
from zhang_xue.mobility.flight_safety_energy import kinematic_safety_shield, rf_pa_dc_power_w


class TestV62Upgrades(unittest.TestCase):
    def test_v62_uq88_ewma_end_to_end_no_artificial_rejuvenation(self):
        env = ZhangXue7NodePacketEnv(make_v62_config(ds_aoi_eta=0.05), seed=101)
        env.reset(seed=101)
        # 5 old observations at t=0, then 1 fresh observation at t=100
        for _ in range(5):
            env.net.record_local_observation(0, 12, True, 0)
        env.net.record_local_observation(0, 12, True, 100)
        ep = decode_uq88(env.net.ledger_pos[0, 0, 12])
        expected = 5.0 * math.exp(-0.05 * 100.0) + 1.0
        self.assertAlmostEqual(ep, expected, delta=0.02)
        # Dirty selection uses monotonic gen_ts even though effective count shrank from 5.0 to 1.03!
        entries = env.net.get_dirty_entries_non_oracle(0, GCS_ID)
        self.assertEqual(len(entries), 1)

    def test_v62_deng_entropy_peer_reward_telescoping_and_conflict_safe(self):
        self.assertAlmostEqual(deng_entropy(0.0, 0.0, 1.0), DENG_MAX_BITS, places=9)
        self.assertAlmostEqual(deng_entropy(1.0, 0.0, 0.0), 0.0, places=9)
        net = SevenRadioCsmaCaNetwork(seed=77, aoi_ewma_end_to_end=True, aoi_eta=0.02)
        pos = np.array([
            [100.0, 100.0, 50.0],
            [180.0, 100.0, 50.0],
            [900.0, 900.0, 50.0],
            [900.0, 800.0, 50.0],
            [900.0, 700.0, 50.0],
            [900.0, 600.0, 50.0],
            [50.0, 50.0, 15.0],
        ], dtype=np.float64)
        net.reset(pos, episode_id=1)
        self.assertTrue(net.run_until_us(600_000))
        total_gain = 0.0
        for step_t in range(1, 10):
            net.step_peer_info_gain = 0.0
            net.record_local_observation(0, 5, True, step_t)
            entries = net.get_dirty_entries_non_oracle(0, 1)
            if entries:
                net.enqueue_app_data(0, 1, entries)
            self.assertTrue(net.run_until_us(net.sim_time_us + 500_000))
            total_gain += net.step_peer_info_gain
        # Telescoping guarantee: total peer Deng info gain for a single (peer, cell) can NEVER exceed 1.0!
        self.assertGreater(total_gain, 0.0)
        self.assertLessEqual(total_gain, 1.0 + 1e-9)

    def test_v62_continuous_tx_power_control_and_pa_energy(self):
        controls = np.zeros((6, 3), dtype=np.float64)
        controls[0, 2] = -1.0  # P_min = 0.05 W
        controls[1, 2] = 1.0   # P_max = 0.50 W
        powers = decode_tx_power_w(controls, p_min_w=0.05, p_max_w=0.50)
        self.assertAlmostEqual(float(powers[0]), 0.05, places=6)
        self.assertAlmostEqual(float(powers[1]), 0.50, places=6)
        self.assertAlmostEqual(rf_pa_dc_power_w(0.50), 3.0, places=6)
        self.assertAlmostEqual(rf_pa_dc_power_w(0.05), 0.75, places=6)
        env = ZhangXue7NodePacketEnv(make_v62_config(), seed=55)
        env.reset(seed=55)
        env.step(np.zeros((6, 2)), np.zeros(6, dtype=np.int32), tx_power_w=powers)
        self.assertAlmostEqual(float(env.net.tx_power_per_radio[0]), 0.05, places=6)
        self.assertAlmostEqual(float(env.net.tx_power_per_radio[1]), 0.50, places=6)

    def test_v62_braking_shield_prevents_headon_from_200m(self):
        # Two UAVs 200m apart approaching head-on at 25 m/s (braking dist = 125m -> must brake early!)
        pos = np.array([
            [200.0, 500.0],
            [400.0, 500.0],
            [100.0, 100.0],
            [100.0, 200.0],
            [100.0, 300.0],
            [100.0, 400.0],
        ], dtype=np.float64)
        vel = np.array([
            [25.0, 0.0],
            [-25.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float64)
        alive = np.ones(6, dtype=bool)
        for _ in range(8):
            v_safe, _ = kinematic_safety_shield(
                pos, vel, vel.copy(), alive, CANONICAL_BUILDINGS_5,
                map_size_m=1000.0, dt_s=1.0, v_max_mps=25.0, a_max_mps2=5.0, uav_safe_distance_m=50.0,
            )
            for u in range(6):
                self.assertLessEqual(float(np.linalg.norm(v_safe[u] - vel[u])), 5.0 + 1e-6)
            pos = pos + v_safe
            vel = v_safe
            self.assertGreaterEqual(float(np.linalg.norm(pos[0] - pos[1])), 50.0 - 1e-5)

    def test_v62_compressed_soa_layout_and_bool_config_rejection(self):
        b61 = backing_bytes_per_environment()
        b62 = backing_bytes_per_environment_v62()
        self.assertLess(b62, 420_000)
        self.assertLess(b62, b61 * 0.65)
        bud = v62_gpu_memory_budget(2048, 300_000, 522)
        self.assertEqual(bud["fits_single_t4_16gib"], 1.0)
        with self.assertRaises(ValueError):
            EnvConfigV54(num_targets=True)
        with self.assertRaises(ValueError):
            EnvConfigV54(reward_peer_cap_per_step=False)


if __name__ == "__main__":
    unittest.main()
