"""V91->V106 DS-Only Boundary Contract: Warp runtime strictly rejects legacy Bayesian configs."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]

from zhang_xue.gpu.mission_v80 import GPUWorldV80Params

CFG = ROOT / "research/zhang_xue/configs"


class TestV91ToV106DsOnlyBoundary(unittest.TestCase):
    def test_warp_rejects_legacy_bayesian_yamls_fail_closed(self):
        for fname in ("scenario_v91_bayes_1km.yaml", "scenario_v95_bayes_shared_coverage_1km.yaml"):
            with self.subTest(yaml=fname), self.assertRaises(ValueError):
                GPUWorldV80Params.from_yaml(CFG / fname, envs=1, device="cpu", seed=44)

    def test_warp_kernels_and_params_contain_no_bayesian_runtime_path(self):
        k_src = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py").read_text(encoding="utf-8")
        self.assertNotIn("_fuse_belief_v91", k_src)
        self.assertNotIn("_bayes_from_counts_v91", k_src)
        d = GPUWorldV80Params.from_yaml(CFG / "scenario_v103_ds_only_1km.yaml", envs=1, device="cpu")
        self.assertFalse(hasattr(d, "bayesian_count_mode"))
        self.assertTrue(d.static_raw_count and d.ds_aoi_eta == 0.0)
        self.assertEqual(d.horizon, 900)


if __name__ == "__main__":
    unittest.main()
