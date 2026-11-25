"""12-test Gate 0 verification suite for UAV-MARL + UavNetSim v5.2."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))

from v52_battery_7node_sim import (
    CANONICAL_BLACKOUT_CENTERS_2,
    CANONICAL_BUILDINGS_5,
    INF_TIME_US,
    INT32_MAX,
    PACKET_TTL_US,
    QUEUE_CAP_PKTS,
    SAFETY_HORIZON_US,
    SPEED_OF_LIGHT_V1,
    ZengRotaryWingParams,
    canonical_ds_combine,
    compute_link_budget_ranges_m,
    compute_v52_vram_gib,
    counts_to_canonical_ds,
    pack_entry_2x_uint32,
    run_7node_episode_comparison,
    run_seeded_3d_mc_canonical_5_2,
    segment_hits_cylinder_3d_fp32,
    step_energy_joules,
    unpack_entry_2x_uint32,
    zeng_propulsion_power_w,
)


class TestV52Gate0(unittest.TestCase):
    def test_01_n01_link_budget_self_consistency(self) -> None:
        lb_05 = compute_link_budget_ranges_m(tx_power_w=0.5, c_light=SPEED_OF_LIGHT_V1)
        lb_01 = compute_link_budget_ranges_m(tx_power_w=0.1, c_light=SPEED_OF_LIGHT_V1)
        self.assertAlmostEqual(lb_05["tx_power_dbm"], 26.99, places=2)
        self.assertAlmostEqual(lb_05["los_max_range_m"], 557.38, places=1)
        self.assertAlmostEqual(lb_05["nlos_max_range_m"], 26.96, places=1)
        self.assertAlmostEqual(lb_01["tx_power_dbm"], 20.00, places=2)
        self.assertAlmostEqual(lb_01["los_max_range_m"], 249.27, places=1)
        self.assertAlmostEqual(lb_01["nlos_max_range_m"], 15.76, places=1)

    def test_02_c02_canonical_5_buildings_2_blackouts(self) -> None:
        self.assertEqual(len(CANONICAL_BUILDINGS_5), 5)
        self.assertEqual(len(CANONICAL_BLACKOUT_CENTERS_2), 2)

    def test_03_zeng_2019_propulsion_curve_and_endurance(self) -> None:
        p_hover = zeng_propulsion_power_w(0.0)
        p_me = zeng_propulsion_power_w(10.21)
        p_vmax = zeng_propulsion_power_w(25.0)
        self.assertAlmostEqual(p_hover, 168.49, places=1)
        self.assertAlmostEqual(p_me, 126.01, places=1)
        self.assertAlmostEqual(p_vmax, 248.96, places=1)
        # U-shaped curve: hover > cruise min-power, vmax > hover
        self.assertGreater(p_hover, p_me)
        self.assertGreater(p_vmax, p_hover)
        # Endurance with E_max=180 kJ + 15W avionics: 682s @ 25 m/s to 1277s @ 10.21 m/s
        z = ZengRotaryWingParams()
        t_vmax = z.e_max_j / step_energy_joules(25.0, 0.0, 1.0, z)
        t_me = z.e_max_j / step_energy_joules(10.21, 0.0, 1.0, z)
        self.assertTrue(650.0 < t_vmax < 720.0)
        self.assertTrue(1240.0 < t_me < 1310.0)
        self.assertLess(t_me, 1500.0)  # H=1500s is a genuine non-binding safety ceiling

    def test_04_rf_tx_power_is_tiny_vs_propulsion(self) -> None:
        p_me = zeng_propulsion_power_w(10.21)
        # 3W PA+baseband at 20% TX duty cycle is <0.5% of propulsion -> explicit lambda_b mandatory
        rf_avg_w = 3.0 * 0.20
        self.assertLess(rf_avg_w / p_me, 0.005)

    def test_05_h1500_int32_clock_safety(self) -> None:
        self.assertEqual(SAFETY_HORIZON_US, 1_500_000_000)
        self.assertLess(SAFETY_HORIZON_US + PACKET_TTL_US, INF_TIME_US)
        self.assertLess(INF_TIME_US, INT32_MAX)

    def test_06_2x_uint32_no_overflow_at_h1500_with_aoi_ts(self) -> None:
        # 1500 > 1023 would overflow 10-bit 1x uint32; 2x uint32 handles up to 65535 + 20-bit ts
        for origin, cell, ts, n_p, n_n in [
            (0, 0, 0, 0, 0),
            (5, 399, 1500, 1500, 0),
            (3, 250, 1499, 1024, 475),
        ]:
            w0, w1 = pack_entry_2x_uint32(origin, cell, ts, n_p, n_n)
            self.assertEqual(unpack_entry_2x_uint32(w0, w1), (origin, cell, ts, n_p, n_n))

    def test_07_m11_queue_64_and_vram_budget(self) -> None:
        self.assertGreater(QUEUE_CAP_PKTS, 60)
        vram = compute_v52_vram_gib(4096, QUEUE_CAP_PKTS, 62, 8)
        self.assertAlmostEqual(vram["total_e4096_gib"], 1.173, places=2)
        self.assertAlmostEqual(vram["per_t4_e2048_gib"], 0.586, places=2)

    def test_08_b01_canonical_ds_exactness(self) -> None:
        # Compare closed-form counts_to_canonical_ds against step-by-step Dempster combination
        m = (0.0, 0.0, 1.0)
        for _ in range(3):
            m = canonical_ds_combine(m, (0.80, 0.0, 0.20))
        for _ in range(1):
            m = canonical_ds_combine(m, (0.0, 0.80, 0.20))
        m_closed = counts_to_canonical_ds(3, 1, 0.80, 0.80)
        for a, b in zip(m, m_closed):
            self.assertAlmostEqual(a, b, places=9)
        self.assertAlmostEqual(sum(m_closed), 1.0, places=9)

    def test_09_n02_3d_ray_cylinder_sloping_uav_gcs(self) -> None:
        # Building 0 at (280, 280, R=45, H=75); UAV(500,500,50) to GCS(50,50,15) passes right through
        self.assertTrue(segment_hits_cylinder_3d_fp32((500, 500, 50), (50, 50, 15), 280, 280, 45, 75))
        # Flying above roof (100m to 85m) -> MISS
        self.assertFalse(segment_hits_cylinder_3d_fp32((500, 500, 100), (50, 50, 85), 280, 280, 45, 75))

    def test_10_r01_r02_r03_empirical_episode_full_comm_beats_zero_comm(self) -> None:
        res_zero = run_7node_episode_comparison("zero_comm", seed=20261125, max_steps=300, tx_power_w=0.5)
        res_full = run_7node_episode_comparison("full_comm", seed=20261125, max_steps=300, tx_power_w=0.5)
        self.assertEqual(res_zero["gcs_tp_confirmed"], 0)
        self.assertEqual(res_zero["r_airtime"], 0.0)
        self.assertGreater(res_full["gcs_tp_confirmed"], 0)
        self.assertLess(res_full["r_airtime"], 0.0)  # lambda_b actively charged!
        self.assertGreater(res_full["total_return"], res_zero["total_return"])

    def test_11_battery_depletion_triggers_termination_before_h1500(self) -> None:
        # At 15 m/s cruise (138.55W + 15W = 153.55W), 180 kJ depletes at ~1173 s < 1500 s
        res = run_7node_episode_comparison("zero_comm", seed=20261125, max_steps=1500, tx_power_w=0.5)
        self.assertTrue(res["terminated"])
        self.assertFalse(res["truncated"])
        self.assertLess(res["steps_run"], 1250)

    def test_12_reproducible_canonical_5_2_mc(self) -> None:
        mc1 = run_seeded_3d_mc_canonical_5_2(seed=20261125, num_trials=1000, tx_power_w=0.5)
        mc2 = run_seeded_3d_mc_canonical_5_2(seed=20261125, num_trials=1000, tx_power_w=0.5)
        self.assertEqual(mc1, mc2)
        self.assertAlmostEqual(mc1["los_max_range_m"], 557.00, places=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
