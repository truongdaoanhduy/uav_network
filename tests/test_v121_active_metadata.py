"""The maintenance handoff must point to a runnable DS-only scenario."""
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ActiveMetadata(unittest.TestCase):
    def test_current_task_declares_loadable_ds_only_candidate(self):
        from zhang_xue.configuration.loader import load_scenario
        doc = (ROOT/'docs/superpowers/CURRENT_TASK.md').read_text()
        match = re.search(r'Active candidate YAML:\s*`?([^`\s]+\.yaml)',doc)
        self.assertIsNotNone(match)
        path = match.group(1).rstrip('.')
        cfg = load_scenario(ROOT/path)
        self.assertEqual(cfg.belief_inference_mode,'dempster_shafer')
        self.assertFalse(cfg.enable_peer_dissemination)
        self.assertFalse(cfg.enable_data_mule_forwarding)

    def test_isolated_rf_profile_differs_only_in_noise(self):
        from dataclasses import asdict
        from zhang_xue.configuration.loader import load_scenario
        folder=ROOT/'research/zhang_xue/configs'
        base=asdict(load_scenario(folder/'scenario_v103_ds_only_1km.yaml'))
        rf=asdict(load_scenario(folder/'scenario_v111_ds_rf_stress_no_peer_1km.yaml'))
        self.assertEqual([k for k in base if base[k]!=rf[k]],['mac_noise_power_w'])


if __name__=='__main__':
    unittest.main()
