"""v75 audit regression tests: PBRS watermark, radio contracts, dynamic q_cap/MAC, loader."""
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from uav_network.experiments.radio_config import (
    network_kwargs,
    radio_config_manifest,
    scenario_protocol_config,
)
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.layout_v67 import (
    Q_V67,
    Q_V67_MAX,
    assert_network_allocation_fits,
    backing_bytes_per_environment_v67,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "research/zhang_xue/configs/scenario_v67_datamule_1km.yaml"


def _scenario(**kwargs):
    return replace(load_scenario(SCENARIO), **kwargs)


class TestV75AuditFixes(unittest.TestCase):
    def test_v75_1_no_oscillatory_reward_farming_under_ewma_aging(self):
        """Monotone minimum-uncertainty watermark prevents infinite revisit reward."""
        cfg = _scenario(
            num_uavs=1,
            w_loc_team=0.50,
            ds_aoi_eta=0.05,
            aoi_ewma_end_to_end=True,
            ds_fusion_mode="sum",
        )
        env = ZhangXue7NodePacketEnv(cfg, seed=7)
        env.reset(seed=7)
        # Step 1: hover to observe initial cells
        vel = np.zeros((1, 2), dtype=np.float64)
        comm = np.zeros((1,), dtype=np.int64)
        _, _, _, _, _, _, info1 = env.step(vel, comm)
        self.assertGreater(info1["r_loc_team"], 0.0)
        observed_cells = np.flatnonzero(env.team_u_prev < 1.0)
        self.assertGreater(len(observed_cells), 0)
        # Save exact minimum watermarks after initial pass
        wm_after_first = env.team_u_prev.copy()
        # Simulate 40 steps away (no sensing on those cells) by moving away and back,
        # or advancing time while verifying team_u_prev never increases on any cell.
        for _ in range(15):
            _, _, _, _, _, _, _ = env.step(np.array([[1.0, 0.0]]), comm)
            self.assertTrue(
                np.all(env.team_u_prev <= wm_after_first + 1e-12),
                "team_u_prev must be monotonically non-increasing (no aging reset)",
            )
            wm_after_first = np.minimum(wm_after_first, env.team_u_prev)

    def test_v75_2_zhang_max_disables_aoi_eta_in_network_kwargs(self):
        s_max = _scenario(ds_fusion_mode="zhang_max", ds_aoi_eta=0.02)
        kw_max = network_kwargs(s_max)
        self.assertEqual(kw_max["aoi_eta"], 0.0)
        self.assertIs(kw_max["rx_lock_on_cca"], False)
        s_sum = _scenario(ds_fusion_mode="sum", ds_aoi_eta=0.02)
        self.assertAlmostEqual(network_kwargs(s_sum)["aoi_eta"], 0.02)

    def test_v75_2_radio_config_manifest_fails_closed_on_unsupported_modes(self):
        for bad_kwargs in (
            {"comm_action_mode": "per_peer"},
            {"enable_tx_power_control": True},
            {"aoi_ewma_end_to_end": False},
            {"network_substeps": 2},
            {"coalesce_duplicate_queue": True},
        ):
            with self.subTest(bad_kwargs=bad_kwargs):
                with self.assertRaises(ValueError):
                    radio_config_manifest(_scenario(**bad_kwargs))

    def test_v75_3_dynamic_q_cap_and_int8_ceiling_guard(self):
        b32 = backing_bytes_per_environment_v67(6, 20, q_cap=32)
        b64 = backing_bytes_per_environment_v67(6, 20, q_cap=64)
        self.assertEqual(b32, 481_141)
        self.assertEqual(b64 - b32, 116_256)
        self.assertEqual(b64, 597_397)
        # Preflight accounts for q_cap=64
        fit = assert_network_allocation_fits(4096, 6, 20, free_bytes=16 * 1024**3, q_cap=64)
        self.assertEqual(fit["estimated_network_bytes"], 4096 * 597_397)
        # Ceiling guard protects int8 active_slot
        with self.assertRaises(ValueError):
            backing_bytes_per_environment_v67(6, 20, q_cap=Q_V67_MAX + 1)
        with self.assertRaises(ValueError):
            network_kwargs(_scenario(), warp_q_cap=128)
        # Opt-in queue parity at q_cap=64
        m64 = radio_config_manifest(_scenario(), warp_q_cap=64)
        self.assertEqual(m64["network_kwargs"]["q_cap"], 64)
        self.assertTrue(m64["comparability"]["queue_capacities_match"])

    def test_v75_4_dynamic_mac_timing_opt_in_and_strict_default(self):
        s_custom = _scenario(mac_difs_us=70, mac_ack_timeout_us=200)
        # Strict default preserves v74 fail-closed behavior
        with self.assertRaises(ValueError):
            radio_config_manifest(s_custom)
        # Opt-in v75 dynamic MAC timing succeeds and propagates exact values
        m_dyn = radio_config_manifest(s_custom, allow_dynamic_mac_timing=True)
        self.assertTrue(m_dyn["dynamic_mac_timing_enabled"])
        self.assertEqual(m_dyn["warp_mac_effective"]["mac_difs_us"], 70)
        self.assertEqual(m_dyn["warp_mac_effective"]["mac_ack_timeout_us"], 200)
        proto = scenario_protocol_config(s_custom)
        self.assertEqual(proto.difs_us, 70)
        self.assertEqual(proto.ack_timeout_us, 200)
        kw = network_kwargs(s_custom, include_protocol=True)
        self.assertEqual(kw["protocol"].difs_us, 70)

    def test_v75_5_loader_accepts_null_bba_reliability_cap(self):
        cfg = load_scenario(SCENARIO, overrides=("bba_reliability_cap=null",))
        self.assertIsNone(cfg.bba_reliability_cap)
        cfg_num = load_scenario(SCENARIO, overrides=("bba_reliability_cap=0.75",))
        self.assertAlmostEqual(cfg_num.bba_reliability_cap, 0.75)

    def test_v75_default_v74_contract_remains_backward_compatible(self):
        m = radio_config_manifest(_scenario())
        self.assertEqual(m["comparability"]["cpu_queue_capacity"], 64)
        self.assertEqual(m["comparability"]["warp_queue_capacity"], Q_V67)
        self.assertFalse(m["comparability"]["queue_capacities_match"])
        self.assertFalse(m["comparability"]["event_parity_verified"])


if __name__ == "__main__":
    unittest.main()
