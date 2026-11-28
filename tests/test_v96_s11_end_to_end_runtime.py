"""V96->V106 S11 End-to-End Runtime Contract (DS-only, fail-closed on legacy Bayes)."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src"), str(ROOT / "scripts/gpu")]

from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
from zhang_xue.gpu.state_v67 import HAS_WARP, wp
from v92_non_oracle_mission_probe import non_oracle_frontier_actions_np

CFG = ROOT / "research/zhang_xue/configs"


class TestV96S11EndToEndRuntime(unittest.TestCase):
    def test_legacy_bayesian_s11_yamls_fail_closed(self):
        for fname in ("scenario_v95_bayes_shared_coverage_1km.yaml", "scenario_v91_bayes_1km.yaml"):
            with self.subTest(yaml=fname), self.assertRaises(ValueError):
                GPUWorldV80Params.from_yaml(CFG / fname, envs=2, device="cpu", seed=44)

    @unittest.skipUnless(HAS_WARP, "Requires NVIDIA Warp")
    def test_ds_shared_coverage_runtime_reward_ledger_exact(self):
        devs = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for dev in devs:
            with self.subTest(device=dev):
                p = GPUWorldV80Params.from_yaml(
                    CFG / "scenario_v95_ds_shared_coverage_1km.yaml",
                    envs=4, device=dev, seed=44,
                )
                w = WarpParallelMissionV80(p)
                act_wp = wp.zeros((4, 6, 2), dtype=wp.float32, device=dev)
                dst_wp = wp.zeros((4, 6), dtype=wp.int32, device=dev)
                for _ in range(15):
                    cont, dest = non_oracle_frontier_actions_np(
                        w.buffers["obs"].numpy(), w.buffers["mask"].numpy(), v_max=p.v_max_mps
                    )
                    wp.copy(act_wp, wp.array(cont, dtype=wp.float32, device=dev))
                    wp.copy(dst_wp, wp.array(dest, dtype=wp.int32, device=dev))
                    w.step(act_wp, dst_wp)
                    wp.synchronize_device(dev)
                    c = w.buffers["reward_components"].numpy()
                    r = w.buffers["reward"].numpy()
                    self.assertLess(float(np.max(np.abs(c.sum(axis=1) - r))), 1e-5)


if __name__ == "__main__":
    unittest.main()
