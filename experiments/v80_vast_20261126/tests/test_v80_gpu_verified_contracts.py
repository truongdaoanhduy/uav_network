"""GPU V80 regression gates. Source-of-truth numerical parity and fail-closed dt assumptions."""
import math
import os
from pathlib import Path
import unittest
import numpy as np
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params

ROOT=Path(__file__).resolve().parents[1]
YAML=ROOT/'research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml'


class TestV80ScientificContracts(unittest.TestCase):
    def test_unsupported_dt_is_rejected_instead_of_silent_network_divergence(self):
        with self.assertRaisesRegex(ValueError,'dt_s'):
            GPUWorldV80Params.from_yaml(YAML,envs=4,device='cpu',dt_s=0.5)

    def test_real_cpu_warp_overflow_is_caught_at_first_step(self):
        import warp as wp
        from zhang_xue.gpu.mission_v80 import WarpParallelMissionV80
        from scripts.gpu.run_v80_real_rollout import require_no_warp_overflow
        p=GPUWorldV80Params.from_yaml(YAML,envs=4,device='cpu',seed=44,horizon=2)
        w=WarpParallelMissionV80(p)
        flag=w.network.data["overflow"]
        require_no_warp_overflow(w,1)
        bad=np.zeros(flag.shape,dtype=np.uint8)
        bad.flat[0]=1
        wp.copy(flag,wp.array(bad,dtype=wp.uint8,device='cpu'))
        with self.assertRaisesRegex(RuntimeError,'vector_step=1'):
            require_no_warp_overflow(w,1)

    def test_noise_model_reference_is_not_same_as_paper(self):
        params=GPUWorldV80Params.from_yaml(YAML,envs=4,device='cpu')
        noise_paper=4.0e-11 # Zhou et al., UavNetSim-v1, ICCC 2025, Table I page 4
        expected=10.0*math.log10(noise_paper/params.noise_power_w)
        self.assertAlmostEqual(expected,19.596,delta=.01)

    @unittest.skipUnless(os.environ.get('RUN_V80_CUDA_GATE')=='1','CUDA gate opt-in')
    def test_cuda_cpu_kinematic_shield_parity_4worlds_16steps(self):
        import warp as wp
        from zhang_xue.gpu.mission_v80 import WarpParallelMissionV80
        from zhang_xue.mobility.flight_safety_energy import kinematic_safety_shield,trajectory_safety_violations
        dev=os.environ.get('V80_CUDA_DEVICE','cuda:1')
        p=GPUWorldV80Params.from_yaml(YAML,envs=4,device=dev,seed=44,horizon=900)
        w=WarpParallelMissionV80(p)
        rng=np.random.default_rng(219)
        worst=0.0
        max_accel=0.0
        missed=0
        for _ in range(16):
            old=w.buffers['positions'].numpy().copy()
            oldvel=w.buffers['velocities'].numpy().copy()
            bat=w.buffers['battery'].numpy().copy()
            acts=rng.uniform(-1,1,size=(4,6,2)).astype('float32')
            ref=[]
            for e in range(4):
                v,_=kinematic_safety_shield(old[e,:6,:2],oldvel[e],acts[e]*p.v_max_mps,bat[e]>0,w.buildings,
                      map_size_m=1000,dt_s=1,v_max_mps=p.v_max_mps,a_max_mps2=p.a_max_mps2,uav_safe_distance_m=p.uav_safe_distance_m)
                ref.append(v)
            w.step(wp.array(acts,dtype=wp.float32,device=dev),wp.zeros((4,6),dtype=wp.int32,device=dev))
            actual=w.buffers['velocities'].numpy()
            failures=w.buffers['failed'].numpy()
            worst=max(worst,float(np.max(np.abs(actual-np.asarray(ref)))))
            max_accel=max(max_accel,float(np.max(np.linalg.norm(actual-oldvel,axis=-1))))
            for e in range(4):
                physical=trajectory_safety_violations(old[e,:6,:2],actual[e],bat[e]>0,w.buildings,
                         map_size_m=1000,dt_s=1,uav_safe_distance_m=p.uav_safe_distance_m)
                missed+=int(any(physical.values()) and failures[e]==0)
            w.auto_reset_gpu()
        self.assertLess(worst,1e-3,f'CPU-Warp velocity parity drift {worst}')
        self.assertLess(max_accel,5.001)
        self.assertEqual(missed,0,'GPU safety flag missed a CPU-detected swept physical violation')
