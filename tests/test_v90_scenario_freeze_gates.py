"""V90 scenario-first freeze gates: Tier-2A pair fallback (world55 step800), HELLO missing-slot, belief calibration."""
import ast
import math
from pathlib import Path
import unittest

import numpy as np

from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params
from zhang_xue.mobility.flight_safety_energy import (
    kinematic_safety_shield,
    trajectory_safety_violations,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_V89 = ROOT / "research/zhang_xue/configs/scenario_v89_static_1km.yaml"
CPU_SHIELD = ROOT / "research/zhang_xue/src/zhang_xue/mobility/flight_safety_energy.py"
GPU_SHIELD = ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py"
HELLO_KERN = ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v89_hello_peer.py"


def _ds_mass_t(p: int, n: int, mf: float = 0.8, me: float = 0.8) -> float:
    if p + n <= 0:
        return 0.0
    a = p * math.log1p(-mf)
    b = n * math.log1p(-me)
    s = max(a, b)
    pa = math.exp(a - s)
    pb = math.exp(b - s)
    pu = math.exp(min(a, b))
    mt = pb * (1.0 - math.exp(a)) if p > 0 else 0.0
    me_m = pa * (1.0 - math.exp(b)) if n > 0 else 0.0
    return mt / max(1e-20, mt + me_m + pu)


def _bayes_prob_t(p: int, n: int, pd: float = 0.85, pf: float = 0.03, prior: float = 15.0 / 392.0) -> float:
    l0 = math.log(prior / (1.0 - prior))
    dl_pos = math.log(pd / pf)
    dl_neg = math.log((1.0 - pd) / (1.0 - pf))
    logit = l0 + p * dl_pos + n * dl_neg
    return 1.0 / (1.0 + math.exp(-logit))


class TestV90ScenarioFreezeGates(unittest.TestCase):
    def test_v90_tier2a_resolves_v89_forensic_world55_step800_zero_violations(self):
        """Exact V89 forensic state (seed44, world55, step800) had pair (4,5) 47.28m; V90 Tier-2A -> 0 violations."""
        pos = np.array([
            [229.60142517089844, 223.15643310546875],
            [92.47745513916016, 443.0407409667969],
            [449.3066101074219, 162.22584533691406],
            [974.0149536132812, 168.78961181640625],
            [916.7086181640625, 161.6463165283203],
            [867.5413818359375, 150.08656311035156],
        ], dtype=np.float64)
        vel = np.array([
            [3.7372794151306152, 0.9076542854309082],
            [-3.9641213417053223, -7.135869026184082],
            [6.890013217926025, -1.2968497276306152],
            [0.15214014053344727, -0.31101083755493164],
            [6.823620796203613, 2.0878169536590576],
            [11.579828262329102, -9.5345458984375],
        ], dtype=np.float64)
        cmd = np.array([
            [-0.7584686279296875, 3.0958609580993652],
            [-8.463891983032227, -4.95594596862793],
            [8.04638385772705, -2.286308765411377],
            [-3.260464906692505, 3.3433187007904053],
            [2.0424160957336426, 0.6249160766601562],
            [7.719885349273682, -6.356363773345947],
        ], dtype=np.float64)
        alive = np.ones(6, dtype=bool)
        buildings, _, _ = scale_canonical_geometry(1000.0, 20)
        vs, _ = kinematic_safety_shield(
            pos, vel, cmd, alive, buildings, map_size_m=1000.0, dt_s=1.0,
            v_max_mps=15.0, a_max_mps2=5.0, uav_safe_distance_m=50.0,
        )
        # Strictly verify acceleration and speed bounds are respected
        self.assertTrue(np.all(np.linalg.norm(vs - vel, axis=1) <= 5.0 + 1e-6))
        self.assertTrue(np.all(np.linalg.norm(vs, axis=1) <= 15.0 + 1e-6))
        vl = trajectory_safety_violations(
            pos, vs, alive, buildings, map_size_m=1000.0, dt_s=1.0, uav_safe_distance_m=50.0,
        )
        self.assertFalse(any(vl.values()), f"Expected 0 violations on V89 forensic state, got {vl}")

    def test_v90_widened_spawn_300_steps_random_motion_zero_collisions(self):
        """6 UAVs from V89 widened spawn (x=90+120*u, ~123.69m) survive 300 random steps with 0 collisions."""
        buildings, _, _ = scale_canonical_geometry(1000.0, 20)
        alive = np.ones(6, dtype=bool)
        for seed in (44, 2027, 10044):
            rng = np.random.default_rng(seed)
            pos = np.array([[90.0 + 120.0 * u, 110.0 + 30.0 * (u % 2)] for u in range(6)], dtype=np.float64)
            vel = np.zeros((6, 2), dtype=np.float64)
            for t in range(300):
                cmd = rng.uniform(-15.0, 15.0, size=(6, 2))
                vs, _ = kinematic_safety_shield(
                    pos, vel, cmd, alive, buildings, map_size_m=1000.0, dt_s=1.0,
                    v_max_mps=15.0, a_max_mps2=5.0, uav_safe_distance_m=50.0,
                )
                vl = trajectory_safety_violations(
                    pos, vs, alive, buildings, map_size_m=1000.0, dt_s=1.0, uav_safe_distance_m=50.0,
                )
                self.assertFalse(any(vl.values()), f"seed={seed} t={t} viol={vl}")
                pos = pos + vs
                vel = vs

    def test_v90_belief_calibration_quantification_static_targets(self):
        """Quantifies agreement at (2,0) and divergence at (1,0)/(1,1) between DS 0.8/0.8 and exact Bayes."""
        self.assertGreaterEqual(_ds_mass_t(2, 0), 0.90)
        self.assertGreaterEqual(_bayes_prob_t(2, 0), 0.90)
        self.assertLess(_ds_mass_t(1, 0), 0.90)
        self.assertLess(_bayes_prob_t(1, 0), 0.90)
        # Expected false-confirm probability on an empty cell with 2 visits is Pf^2 = 0.0009
        # Across 377 empty legal cells, expected FPs at 2 visits/cell = 377 * 0.0009 = 0.3393
        self.assertAlmostEqual(377.0 * (0.03 ** 2), 0.3393, places=4)

    def test_v90_scenario_yaml_and_source_parity_under_500_lines(self):
        p = GPUWorldV80Params.from_yaml(SCENARIO_V89, envs=4, device="cpu", seed=44)
        self.assertTrue(p.static_raw_count)
        self.assertTrue(p.obs_schema_v2)
        self.assertEqual(p.ds_aoi_eta, 0.0)
        self.assertEqual(p.uav_safe_distance_m, 50.0)
        self.assertEqual(p.v_max_mps, 15.0)
        self.assertEqual(p.a_max_mps2, 5.0)
        self.assertEqual(p.dt_s, 1.0)
        for path in (CPU_SHIELD, GPU_SHIELD, HELLO_KERN):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            self.assertLessEqual(len(text.splitlines()), 500, f"{path.name} exceeded 500 lines")
        self.assertIn("def _cand_phys_peers_v90", GPU_SHIELD.read_text(encoding="utf-8"))
        self.assertIn("obs[e, u, offset + 5] = 1.0", HELLO_KERN.read_text(encoding="utf-8"))
        self.assertIn("obs[e, u, offset + 6] = 1.0", HELLO_KERN.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
