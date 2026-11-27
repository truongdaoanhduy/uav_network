"""End-to-end S11 runtime gate: YAML -> load_scenario -> GPUWorldV80Params -> WarpParallelMissionV80."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src"), str(ROOT / "scripts/gpu")]

from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
from zhang_xue.gpu.state_v67 import HAS_WARP, wp
from v92_non_oracle_mission_probe import non_oracle_frontier_actions_np

CFG = ROOT / "research/zhang_xue/configs"


def _run_paired_s11(ds_yaml: Path, bayes_yaml: Path, *, device: str, envs: int = 4, steps: int = 30, seed: int = 44):
    s_ds, s_by = load_scenario(ds_yaml), load_scenario(bayes_yaml)
    assert s_ds.shared_coverage_reward is s_by.shared_coverage_reward
    p_ds = GPUWorldV80Params.from_yaml(ds_yaml, envs=envs, device=device, seed=seed)
    p_by = GPUWorldV80Params.from_yaml(bayes_yaml, envs=envs, device=device, seed=seed)
    assert p_ds.bayesian_count_mode is False and p_by.bayesian_count_mode is True

    w_ds, w_by = WarpParallelMissionV80(p_ds), WarpParallelMissionV80(p_by)
    act_wp = wp.zeros((envs, 6, 2), dtype=wp.float32, device=device)
    dst_wp = wp.zeros((envs, 6), dtype=wp.int32, device=device)
    idx = np.arange(6)

    cov_ds_all, cov_by_all = [], []
    for _ in range(steps):
        active = ~(w_ds.buffers["done"].numpy().astype(bool) | w_by.buffers["done"].numpy().astype(bool))
        if not active.any():
            break
        cont, dest = non_oracle_frontier_actions_np(
            w_by.buffers["obs"].numpy(), w_by.buffers["mask"].numpy(), v_max=p_by.v_max_mps
        )
        cont[~active], dest[~active] = 0.0, 0
        wp.copy(act_wp, wp.array(cont, dtype=wp.float32, device=device))
        wp.copy(dst_wp, wp.array(dest, dtype=wp.int32, device=device))

        w_ds.step(act_wp, dst_wp)
        w_by.step(act_wp, dst_wp)
        wp.synchronize_device(device)

        pos_ds = w_ds.network.data["ledger_pos"].numpy()[active][:, idx, idx, :]
        pos_by = w_by.network.data["ledger_pos"].numpy()[active][:, idx, idx, :]
        neg_ds = w_ds.network.data["ledger_neg"].numpy()[active][:, idx, idx, :]
        neg_by = w_by.network.data["ledger_neg"].numpy()[active][:, idx, idx, :]
        assert np.array_equal(pos_ds, pos_by), "Own positive sensing counts diverged"
        assert np.array_equal(neg_ds, neg_by), "Own negative sensing counts diverged"

        c_ds = w_ds.buffers["reward_components"].numpy()[active]
        c_by = w_by.buffers["reward_components"].numpy()[active]
        r_ds = w_ds.buffers["reward"].numpy()[active]
        r_by = w_by.buffers["reward"].numpy()[active]
        assert np.max(np.abs(c_ds.sum(axis=1) - r_ds)) < 1e-5
        assert np.max(np.abs(c_by.sum(axis=1) - r_by)) < 1e-5
        cov_ds_all.append(c_ds[:, 0].copy())
        cov_by_all.append(c_by[:, 0].copy())

    return np.concatenate(cov_ds_all), np.concatenate(cov_by_all)


@unittest.skipUnless(HAS_WARP, "Requires NVIDIA Warp")
class TestV96S11EndToEndRuntime(unittest.TestCase):
    def _devices(self) -> list[str]:
        devs = ["cpu"]
        if wp.is_cuda_available():
            devs.append("cuda:0")
        return devs

    def test_s11_shared_coverage_bit_identical_on_cpu_and_cuda(self):
        ds_cfg = CFG / "scenario_v95_ds_shared_coverage_1km.yaml"
        by_cfg = CFG / "scenario_v95_bayes_shared_coverage_1km.yaml"
        for dev in self._devices():
            with self.subTest(device=dev):
                cov_ds, cov_by = _run_paired_s11(ds_cfg, by_cfg, device=dev)
                self.assertGreater(float(cov_ds.sum()), 0.0)
                self.assertTrue(np.array_equal(cov_ds, cov_by))

    def test_s11_disabled_control_diverges_on_identical_counts(self):
        ds_cfg = CFG / "scenario_v89_static_1km.yaml"
        by_cfg = CFG / "scenario_v91_bayes_1km.yaml"
        for dev in self._devices():
            with self.subTest(device=dev):
                cov_ds, cov_by = _run_paired_s11(ds_cfg, by_cfg, device=dev, steps=10)
                self.assertGreater(float(cov_ds.sum()), 0.0)
                self.assertFalse(np.array_equal(cov_ds, cov_by))


if __name__ == "__main__":
    unittest.main()
