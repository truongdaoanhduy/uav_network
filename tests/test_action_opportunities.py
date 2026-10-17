"""Joint action opportunities must be real, passive and identical across backends."""
import importlib.util
import unittest
import numpy as np

from uav_network.experiments.telemetry import diagnostic_rates, OutcomePool
from test_gpu_packets import config
import test_gpu_packets as packet_tests

@unittest.skipUnless(importlib.util.find_spec("warp"), "optional Warp backend required")
class ActionOpportunityTests(unittest.TestCase):
    def compare(self, positions, destination, expected, *, targets=None, obstacles=None):
        c = config(power_limited_admission=True)
        c.network.peer_range_m = c.network.gcs_range_m = 500
        c.network.tx_power_min_w, c.network.tx_power_max_w = .001, .4
        c.network.tx_power_mapping = "log"
        obstacles = obstacles or []
        c.world.num_obstacles = len(obstacles)
        helper = packet_tests.GPUPacketTests()
        native = helper.oracle(c, positions, [(destination, 0)], targets, obstacles)[0]["diagnostics"]
        wp, s, p, mac = helper.prepare(c, positions, targets, obstacles)
        warp = helper.step(c, s, p, mac, wp, destination)["diagnostics"]
        for name, count in expected.items():
            self.assertEqual(native.get(name), count, "native " + name)
            self.assertEqual(warp.get(name), count, "warp " + name)
        return native, warp

    def test_power_infeasible_nlos_is_not_a_gpu_receiver_opportunity(self):
        self.compare([[500,400],[950,950]], [0,0],
            {"action_receiver_eligible":0, "action_no_opportunity":2},
            obstacles=[[500,200,30]])

    def test_exact_silence_and_progress_peer_opportunities(self):
        cases = [
            ([[500,700],[500,350]], [0,0], None, (1,1,1)),
            ([[500,700],[500,350]], [1,0], None, (0,1,0)),
            ([[500,300],[500,150]], [0,0], None, (1,0,0)),
            ([[500,700],[500,950]], [0,0], None, (1,0,0)),
            ([[500,700],[500,350]], [0,0], [[100,100]], (0,0,0)),
        ]
        names=("action_eligible_silent","action_progress_peer_opportunity",
               "action_progress_peer_silent")
        for positions,destination,targets,counts in cases:
            with self.subTest(positions=positions,destination=destination,targets=targets):
                self.compare(positions,destination,dict(zip(names,counts)),targets=targets)

class OpportunityPoolingTests(unittest.TestCase):
    def test_joint_rates_are_pooled_by_opportunity_counts_and_empty_is_unknown(self):
        records=[]
        for eligible,joint,peer,peer_joint in [(1,1,1,1),(9,0,3,0)]:
            records.append({"return":0, "search_rate_percent":0,"delivery_rate_percent":0,
                         "success":False,"network":{"diagnostics":{
                "action_receiver_eligible":eligible,"action_eligible_silent":joint,
                "action_progress_peer_opportunity":peer,"action_progress_peer_silent":peer_joint}}})
        pool=OutcomePool()
        pool.add(records)
        result=pool.metrics()
        prefix="network/diagnostics/"
        self.assertEqual(result.get(prefix+"action_silent_given_eligible_fraction"),.1)
        self.assertEqual(result.get(prefix+"action_silent_given_eligible_fraction_count"),10)
        self.assertEqual(result.get(prefix+"action_silent_given_progress_peer_fraction"),.25)
        self.assertEqual(result.get(prefix+"action_silent_given_progress_peer_fraction_count"),4)
        empty=diagnostic_rates(dict(action_receiver_eligible=0,action_eligible_silent=0,
            action_progress_peer_opportunity=0,action_progress_peer_silent=0))
        self.assertIn("action_silent_given_eligible_fraction",empty)
        self.assertIsNone(empty["action_silent_given_eligible_fraction"])
        self.assertIsNone(empty["action_silent_given_progress_peer_fraction"])

if __name__=="__main__":
    unittest.main()
