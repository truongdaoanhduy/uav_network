"""V79 audit & contract regression tests (YAML parity, selective reset safety, terminated/truncated)."""
import ast
from pathlib import Path
import unittest

from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_V77 = ROOT / "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml"
KERNELS_V79 = ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py"
RESET_OBS_V79 = ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_reset_obs.py"
MISSION_V79 = ROOT / "research/zhang_xue/src/zhang_xue/gpu/mission_v79.py"
RUNNER_V79 = ROOT / "scripts/gpu/run_v79_real_rollout.py"


class TestV79AuditAndParityContracts(unittest.TestCase):
    def test_v79_params_from_yaml_eliminates_12db_vs_20db_drift(self):
        cfg = load_scenario(SCENARIO_V77)
        params = GPUWorldV79Params.from_yaml(SCENARIO_V77, envs=4096, device="cuda:0", seed=44)
        self.assertAlmostEqual(params.per_building_loss_db, 12.0)
        self.assertAlmostEqual(params.per_building_loss_db, cfg.rf_per_building_loss_db)
        self.assertEqual(params.obstacle_model, cfg.rf_obstacle_model)
        self.assertEqual(params.queue_capacity, cfg.mac_queue_capacity)
        self.assertAlmostEqual(params.fp_hold, cfg.fp_hold)
        self.assertAlmostEqual(params.safety_failure_penalty, cfg.safety_failure_penalty)
        self.assertIs(params.enable_fp_unlatch, cfg.enable_fp_unlatch)
        self.assertAlmostEqual(params.ds_aoi_eta, cfg.ds_aoi_eta)

    def test_v79_params_zhang_max_disables_aoi_eta_and_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "ds_fusion_mode"):
            GPUWorldV79Params.from_yaml(
                SCENARIO_V77, scenario_overrides=("ds_fusion_mode=zhang_max",)
            )
        with self.assertRaises(ValueError):
            GPUWorldV79Params(queue_capacity=128).validate()
        with self.assertRaises(ValueError):
            GPUWorldV79Params(obstacle_model="invalid").validate()
        with self.assertRaises(ValueError):
            GPUWorldV79Params.from_yaml(
                SCENARIO_V77, scenario_overrides=("enable_tx_power_control=true",)
            )

    def test_v79_files_ast_valid_and_under_500_lines(self):
        for path in (KERNELS_V79, RESET_OBS_V79, MISSION_V79, RUNNER_V79):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            self.assertLessEqual(len(text.splitlines()), 500, f"{path.name} exceeded 500 lines")

    def test_v79_kernels_contain_all_p0_parity_and_reset_fixes(self):
        k_src = KERNELS_V79.read_text(encoding="utf-8") + "\n" + RESET_OBS_V79.read_text(encoding="utf-8")
        self.assertIn("def step_v67_masked", k_src)
        self.assertIn("if active_mask[e] == wp.uint8(0)", k_src)
        self.assertIn("net.mission_epoch_us[e] = net.now[e]", k_src)
        self.assertIn("net.gcs_last_rx_us[e, cid] - net.mission_epoch_us[e]", k_src)
        self.assertIn("ep_id[e] * 11939", k_src)
        self.assertIn("_segment_hits_cylinder_3d_v79", k_src)
        self.assertIn("_in_blackout_v67", k_src)
        self.assertIn("net.ack_origin_gen[e, u, dst, idx]", k_src)
        self.assertIn("terminated[e] = is_term", k_src)
        self.assertIn("truncated[e] = is_trunc", k_src)

    def test_v79_runner_separates_terminated_and_uses_gpu_autoreset(self):
        r_src = RUNNER_V79.read_text(encoding="utf-8")
        self.assertIn("replay.add_batch(old, state, action, rew, nextobs, nextstate, term)", r_src)
        self.assertIn("world.auto_reset_gpu()", r_src)
        self.assertIn("max_memory_allocated", r_src)


if __name__ == "__main__":
    unittest.main()
