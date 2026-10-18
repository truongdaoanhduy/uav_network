"""Scenario V3 freezes the large-report radio-energy study without mutating V2."""
from pathlib import Path
import unittest

from uav_network.configuration import load_config
from uav_network.experiments.comparison_metrics import evaluation_protocol


ROOT = Path(__file__).resolve().parents[1]
V2_HASH = "03f482e6bab87457c684d91eb653a11cb7ab61f56af752f74ad83e4868cc6e50"


class ScenarioV3Tests(unittest.TestCase):
    def test_v3_freezes_large_report_radio_study_and_v2_stays_legacy(self):
        v2 = load_config(ROOT / "configs/comparison-25k-scenario-v2.yaml")
        self.assertEqual((v2.network.report_bytes, v2.network.buffer_bytes), (1024, 3072))
        self.assertEqual((v2.network.pa_efficiency, v2.network.tx_circuit_power_w,
                          v2.network.rx_circuit_power_w), (1.0, 0.0, 0.0))
        self.assertEqual(evaluation_protocol(v2, [2000000044], {"next_seed": 50000})["scenario_hash"],
                         V2_HASH)

        v3 = load_config(ROOT / "configs/comparison-25k-scenario-v3.yaml")
        self.assertEqual((v3.world.num_uavs, v3.world.num_targets, v3.world.num_obstacles,
                          v3.world.max_steps), (6, 35, 25, 1800))
        self.assertEqual((v3.runtime.num_envs, v3.runtime.episodes, v3.runtime.seed), (4096, 25000, 44))
        self.assertEqual((v3.runtime.eval_seed, v3.runtime.eval_episodes), (2000000044, 30))
        self.assertEqual((v3.network.report_bytes, v3.network.buffer_bytes,
                          v3.network.payload_bytes, v3.network.max_queue_packets),
                         (1_000_000, 3_000_000, 1024, 64))
        self.assertEqual((v3.network.report_ttl_s, v3.network.bit_rate_bps,
                          v3.network.peer_range_m, v3.network.gcs_range_m),
                         (300.0, 2_000_000.0, 500.0, 500.0))
        self.assertEqual((v3.network.tx_power_min_w, v3.network.tx_power_max_w,
                          v3.network.tx_power_mapping), (0.001, 0.4, "log"))
        self.assertEqual(v3.network.max_retries, 5)
        self.assertAlmostEqual(v3.network.pa_efficiency, 0.35)
        self.assertAlmostEqual(v3.network.tx_circuit_power_w, 0.1)
        self.assertAlmostEqual(v3.network.rx_circuit_power_w, 0.1)
        self.assertEqual(v3.reward.propulsion_energy_per_kj, 0.01)
        self.assertEqual(v3.reward.communication_efficiency, 1.0)
        self.assertIn("scenario-v3", v3.tracking.tags)
        self.assertIn("scenario-v3", v3.tracking.group)
        self.assertIn("scenario-v3", v3.runtime.output_dir)
        self.assertNotEqual(evaluation_protocol(v3, [2000000044], {"next_seed": 50000})["scenario_hash"],
                            V2_HASH)

    def test_resource_estimator_is_linear_in_envs_not_report_fragments(self):
        from uav_network.experiments.state_resources import estimate_warp_state_bytes
        v2 = load_config(ROOT / "configs/comparison-25k-scenario-v2.yaml")
        v3 = load_config(ROOT / "configs/comparison-25k-scenario-v3.yaml")
        counts = [1, 64, 512, 4096]
        old = estimate_warp_state_bytes(v2, counts)
        new = estimate_warp_state_bytes(v3, counts)
        self.assertEqual(old["bytes_per_env"], new["bytes_per_env"])
        self.assertEqual(new["fragments_per_report"], 977)
        self.assertEqual(old["fragments_per_report"], 1)
        self.assertEqual(new["session_fragment_dense_bytes"], 0)
        for count in counts:
            self.assertEqual(new["estimated_bytes"][str(count)], new["bytes_per_env"] * count)


if __name__ == "__main__":
    unittest.main()
