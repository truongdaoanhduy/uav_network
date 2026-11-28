"""V112: exact YAML factorial and canonical TX/action contract, CPU-only.

This prevents future edits from silently changing extra parameters across
ablation pairs. It is a source-level contract, NOT evidence of PEER utility.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research/zhang_xue/src'),str(ROOT/'scripts/gpu')]
from zhang_xue.configuration.loader import load_scenario
from v112_paired_scenario_gates import CONFIGS,EDGES,verify_configs
from dataclasses import asdict

class V112FactorialContract(unittest.TestCase):
    def setUp(self):
        self.paths={k:ROOT/'research/zhang_xue/configs'/name for k,name in CONFIGS.items()}

    def test_each_pair_differs_in_exactly_one_field(self):
        mapping=verify_configs(self.paths)
        self.assertEqual(set(mapping),set(CONFIGS))
        for name,(left,right,field) in EDGES.items():
            a=asdict(load_scenario(self.paths[left]));b=asdict(load_scenario(self.paths[right]))
            self.assertEqual([x for x in a if a[x]!=b[x]],[field],name)

    def test_tx_power_not_a_learned_action_in_all_profiles(self):
        for name,path in self.paths.items():
            s=load_scenario(path)
            with self.subTest(name=name):
                self.assertEqual(s.comm_action_mode,'compact_3')
                self.assertFalse(s.enable_tx_power_control)
                self.assertAlmostEqual(s.tx_power_w,0.1)
                self.assertEqual(s.belief_inference_mode,'dempster_shafer')
                self.assertEqual(s.exploration_reward_mode,'first_visit')
                self.assertEqual(s.ds_aoi_eta,0)
                self.assertEqual(s.w_p2p,0)
                self.assertEqual(s.lambda_b,0)

if __name__=='__main__':unittest.main()
