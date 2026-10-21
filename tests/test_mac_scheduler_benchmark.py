"""Contract for reproducible MAC Scheduler Gen1/Gen2 benchmarks."""
from pathlib import Path
import unittest

from uav_network.configuration import load_config
from uav_network.experiments.mac_scheduler_benchmark import benchmark_mac_scheduler


ROOT = Path(__file__).resolve().parents[1]


def tiny_config():
    c = load_config(ROOT / "configs/comparison-25k-scenario-v3.yaml")
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.runtime.num_envs = 2
    c.world.num_uavs = 2
    c.world.num_targets = 2
    c.world.num_obstacles = 0
    c.world.max_steps = 20
    c.tracking.enabled = False
    return c.validate()


class MacSchedulerBenchmarkTests(unittest.TestCase):
    def test_rejects_invalid_scheduler_corpus_and_sizes(self):
        c = tiny_config()
        for kwargs, pattern in (
            ({"scheduler": "gen3", "corpus": "low_contention", "num_envs": 2, "steps": 1}, "scheduler"),
            ({"scheduler": "gen1", "corpus": "unknown", "num_envs": 2, "steps": 1}, "corpus"),
            ({"scheduler": "gen1", "corpus": "low_contention", "num_envs": 0, "steps": 1}, "num_envs"),
            ({"scheduler": "gen1", "corpus": "low_contention", "num_envs": 2, "steps": 0}, "steps"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, pattern):
                benchmark_mac_scheduler(c, **kwargs)

    def test_result_contract_and_corpus_labels(self):
        c = tiny_config()
        required = {
            "scheduler", "commit", "device", "num_envs", "steps", "corpus",
            "collector_seconds", "packet_step_seconds", "joint_transitions_per_second",
            "scheduler_metrics", "scientific_metrics", "gpu_memory", "simulator_error_count", "terminated_worlds",
        }
        rows = {}
        for corpus in ("low_contention", "active_streaming"):
            row = benchmark_mac_scheduler(c, num_envs=2, steps=1, scheduler="gen2", corpus=corpus)
            self.assertTrue(required <= row.keys())
            self.assertEqual(row["scheduler"], "gen2")
            self.assertEqual(row["corpus"], corpus)
            self.assertEqual(row["device"], "cpu")
            self.assertEqual(row["num_envs"], 2)
            self.assertEqual(row["steps"], 1)
            self.assertGreater(row["collector_seconds"], 0)
            self.assertGreaterEqual(row["packet_step_seconds"], 0)
            self.assertGreater(row["joint_transitions_per_second"], 0)
            self.assertEqual(row["simulator_error_count"], 0)
            self.assertEqual(set(row["scientific_metrics"]), {
                "generated_packets", "received_packets", "phy_successes", "phy_failures",
                "collisions", "energy_blocked_transmissions", "payload_bytes_received",
                "expired_packets", "reports_delivered", "targets_delivered",
                "communication_energy_j", "tx_energy_j", "rx_energy_j",
            })
            self.assertEqual(set(row["scheduler_metrics"]), {
                "attempted_chains", "completed_chains", "fused_events",
                "competitor_fallbacks", "boundary_fallbacks", "max_chain_length",
            })
            rows[corpus] = row
        self.assertNotEqual(rows["low_contention"]["corpus"], rows["active_streaming"]["corpus"])

    def test_gen1_metrics_remain_zero_and_commit_is_nonempty(self):
        row = benchmark_mac_scheduler(tiny_config(), num_envs=1, steps=1,
                                      scheduler="gen1", corpus="low_contention")
        self.assertTrue(row["commit"])
        self.assertTrue(all(value == 0 for value in row["scheduler_metrics"].values()))


if __name__ == "__main__":
    unittest.main()
