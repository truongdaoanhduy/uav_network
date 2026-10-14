"""Scientific mission metrics must count failed deliveries and real swept safety."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from uav_network.configuration import Config


class ComparisonMetricsTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("uav_network.experiments.comparison_metrics"),
                             "Missing mission-level comparison metrics")
        from uav_network.experiments import comparison_metrics
        return comparison_metrics

    def config(self):
        c = Config()
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 1
        c.world.max_steps, c.world.dt_s = 10, 1.0
        c.world.safety_distance_m, c.world.obstacle_clearance_m = 2.0, 1.0
        return c

    def test_undelivered_targets_are_censored_and_energy_denominator_excludes_gcs(self):
        m = self.module()
        c = self.config()
        trace = m.MissionMetrics(c, np.array([[0., 0.], [10., 0.]]),
                                 np.array([[5., 5., 1.]]), [True, False], [False, False])
        trace.observe(np.array([[1., 0.], [11., 0.]]), [True, True], [True, False], 2)
        values = trace.finish({"success": False, "energy": {"propulsion_energy_j": 100.},
                               "network": {"enabled": True, "uav_communication_energy_j": 20.,
                                           "gcs_communication_energy_j": 900.,
                                           "generated_packets": 4, "received_packets": 3,
                                           "report_e2e_delay_s": .5},
                               "motion_counts": {}, "blocked_moves": 0}, 7.0)
        self.assertEqual(values["gcs_delivery_fraction"], .5)
        self.assertEqual(values["mission_completion_time_capped_s"], 10.)
        self.assertEqual(values["target_delivery_time_capped_mean_s"], 6.)
        self.assertAlmostEqual(values["delivery_auc"], .4)
        self.assertEqual(values["uav_energy_j"], 120.)
        self.assertEqual(values["uav_j_per_gcs_delivery"], 120.)
        self.assertEqual(values["first_discovery_to_gcs_delay_delivered_mean_s"], 2.)
        self.assertEqual(values["report_delay_delivered_mean_s"], .5)
        self.assertEqual(values["first_gcs_delivery_observed_s"], [2., None])

    def test_zero_delivery_or_generation_has_null_conditional_metrics(self):
        m = self.module()
        trace = m.MissionMetrics(self.config(), np.array([[0., 0.], [10., 0.]]),
                                 np.array([[5., 5., 1.]]), [False, False], [False, False])
        trace.observe(np.array([[0., 0.], [10., 0.]]), [False, False], [False, False], 1)
        values = trace.finish({"success": False, "energy": {"propulsion_energy_j": 2.},
                               "network": {"enabled": True, "generated_packets": 0,
                                           "received_packets": 0, "report_e2e_delay_s": 0.},
                               "motion_counts": {}, "blocked_moves": 0}, 0.)
        for name in ("uav_j_per_gcs_delivery", "delivery_given_discovery_fraction",
                     "report_delay_delivered_mean_s", "packet_pdr_fraction"):
            self.assertIsNone(values[name], name)
        self.assertEqual(values["delivery_auc"], 0.)

    def test_swept_peer_and_obstacle_crossing_count_real_violations(self):
        m = self.module()
        trace = m.MissionMetrics(self.config(), np.array([[0., 0.], [10., 0.]]),
                                 np.array([[5., 0., 1.]]), [False, False], [False, False])
        trace.observe(np.array([[10., 0.], [0., 0.]]), [False, False], [False, False], 1)
        values = trace.finish({"success": False, "energy": {"propulsion_energy_j": 2.},
                               "network": {"enabled": False}, "motion_counts": {},
                               "blocked_moves": 100}, 0.)
        self.assertEqual(values["swept_peer_violation_pairs"], 1)
        self.assertEqual(values["swept_obstacle_violation_segments"], 2)
        self.assertEqual(values["safety_violation_steps"], 1)
        self.assertFalse(values["safe_episode"])
        self.assertEqual(values["path_length_m"], 20.)

    def test_paired_comparison_checks_scenario_seeds_and_reports_uncertainty(self):
        self.assertIsNotNone(importlib.util.find_spec("uav_network.experiments.comparison"),
                             "Missing paired evaluation comparison")
        from uav_network.experiments.comparison import compare_evaluations
        def report(algorithm, offset):
            return {"algorithm": algorithm, "protocol": {"scenario_hash": "same", "training_seed_overlap": False, "validation_seed_overlap": False},
                    "records": [{"seed": seed, "comparison_metrics": {
                        "gcs_delivery_fraction": value + offset,
                        "uav_energy_j": 100., "targets_delivered": 1}} for seed, value in ((20, .2), (21, .4), (22, .6))]}
        a, b = report("masac", .1), report("matd3", 0.)
        result = compare_evaluations(a, b, bootstrap_samples=100, seed=44)
        metric = result["metrics"]["gcs_delivery_fraction"]
        self.assertAlmostEqual(metric["difference_left_minus_right"], .1)
        self.assertEqual(metric["favored"], "left")
        self.assertEqual(metric["paired_episodes"], 3)
        b["records"][0]["seed"] = 99
        with self.assertRaisesRegex(ValueError, "seed"):
            compare_evaluations(a, b)
        b = report("matd3", 0.)
        b["protocol"]["scenario_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "scenario"):
            compare_evaluations(a, b)

    def test_validation_bank_reuse_and_resumed_unknown_history_are_exploratory(self):
        m = self.module()
        c = self.config()
        c.tracking.enabled, c.tracking.evaluation_interval = True, 1
        c.runtime.eval_seed, c.runtime.eval_episodes = 10044, 2
        protocol = m.evaluation_protocol(c, [10044, 10045], {"next_seed": 50, "episodes": 4})
        self.assertFalse(protocol["training_seed_overlap"])
        self.assertIn("validation_seed_overlap", protocol)
        self.assertTrue(protocol["validation_seed_overlap"])
        from uav_network.experiments.comparison import compare_evaluations
        report = {"algorithm": "masac", "protocol": protocol,
                  "records": [{"seed": seed, "comparison_metrics": {"gcs_delivery_fraction": .2,
                                "uav_energy_j": 100., "targets_delivered": 0}} for seed in (10044, 10045)]}
        self.assertEqual(compare_evaluations(report, report)["status"], "exploratory")
        c.runtime.warm_start_checkpoint = "previous-policy.pt"
        unknown = m.evaluation_protocol(c, [2000000044], {"next_seed": 100, "episodes": 8})
        self.assertIsNone(unknown["validation_seed_overlap"])
        c.runtime.seed = 99
        history = m.evaluation_protocol(c, [44], {"next_seed": 100, "episodes": 8, "records": [{"seed": 44}]})
        self.assertTrue(history["training_seed_overlap"])

    def test_single_episode_or_training_seed_overlap_is_exploratory(self):
        self.assertIsNotNone(importlib.util.find_spec("uav_network.experiments.comparison"),
                             "Missing paired evaluation comparison")
        from uav_network.experiments.comparison import compare_evaluations
        report = {"algorithm": "masac", "protocol": {"scenario_hash": "same", "training_seed_overlap": True, "validation_seed_overlap": False},
                  "records": [{"seed": 44, "comparison_metrics": {"gcs_delivery_fraction": .2,
                                "uav_energy_j": 100., "targets_delivered": 0}}]}
        result = compare_evaluations(report, report, bootstrap_samples=100)
        self.assertEqual(result["status"], "exploratory")
        self.assertIsNone(result["metrics"]["gcs_delivery_fraction"]["difference_ci95"])
        self.assertIsNone(result["pooled_uav_j_per_gcs_delivery"]["left"])


if __name__ == "__main__":
    unittest.main()
