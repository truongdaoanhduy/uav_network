"""V79.1 fail-closed YAML and log provenance contracts; CPU-only Python tests."""
import ast
from pathlib import Path
import unittest
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params

ROOT=Path(__file__).resolve().parents[1]
SCENARIO=ROOT/"research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml"

class TestHardening(unittest.TestCase):
    def test_supported_yaml_defaults(self):
        p=GPUWorldV79Params.from_yaml(SCENARIO,envs=4096,device="cuda:0")
        self.assertEqual(p.envs,4096)
        self.assertEqual(p.per_building_loss_db,12.0)
        self.assertEqual(p.horizon,900)
    def test_all_unsupported_model_changes_fail_closed(self):
        cases=[
            ("ds_fusion_mode=zhang_max","ds_fusion_mode"),
            ("evidence_calibration=sensor_likelihood_ratio","evidence_calibration"),
            ("num_targets=14","num_targets"),
            ("dt_s=0.5","dt_s"),
            ("fov_radius_m=75.0","fov_radius_m"),
            ("enable_3d_sensing_occlusion=false","enable_3d_sensing_occlusion"),
            ("enable_accel_energy=true","enable_accel_energy"),
            ("obs_pool_mode=mean","obs_pool_mode"),
            ("safety_failure_terminate=false","safety_failure_terminate"),
            ("network_substeps=2","network_substeps"),
            ("reward_scale_mode=per_target_norm","reward_scale_mode"),
            ("w_p2p=0.25","w_p2p"),
            ("mac_slot_us=25","mac_slot_us"),
            ("battery_termination_mode=any_depleted","battery_termination_mode"),
            ("gcs_target_exclusion_m=50.0","gcs_target_exclusion_m"),
        ]
        for override,name in cases:
            with self.subTest(field=name),self.assertRaisesRegex(ValueError,name):
                GPUWorldV79Params.from_yaml(SCENARIO,scenario_overrides=(override,))
    def test_invalid_numeric_physics_fail_early(self):
        for field,value in (("ds_mf",1.0),("pf",0.95),("noise_power_w",0.0),
                            ("tx_power_w",-0.1),("ds_aoi_eta",-1.0)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                GPUWorldV79Params.from_yaml(SCENARIO,**{field:value})
    def test_compatible_research_parameter_sweep(self):
        for value in (0,6,12,20):
            with self.subTest(rf=value):
                p=GPUWorldV79Params.from_yaml(SCENARIO,scenario_overrides=(f"rf_per_building_loss_db={value}.0",))
                self.assertEqual(p.per_building_loss_db,value)
    def test_unsafe_terminal_rollback_is_present(self):
        path=ROOT/"research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py"
        tree=ast.parse(path.read_text())
        defn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_move_v79")
        body=ast.get_source_segment(path.read_text(),defn)
        self.assertLess(body.find("if failed[e] != wp.uint8(0)"),body.find("# Phase 3: commit"))
        self.assertIn("return",body)
    def test_repeat_run_log_is_truncated(self):
        code=(ROOT/"scripts/gpu/run_v79_real_rollout.py").read_text()
        self.assertIn('.progress.jsonl").write_text("")',code)

if __name__=="__main__":unittest.main()
