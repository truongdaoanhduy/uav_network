"""Integration: a CPU rollout must not claim CUDA execution."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class TestV79StatusMatchesRealDevice(unittest.TestCase):
    def test_cpu_run_reports_cpu_not_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result.json'
            env=os.environ.copy()
            env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(ROOT/'research/zhang_xue/src')])
            env['OMP_NUM_THREADS']='2'
            cmd=[sys.executable,'scripts/gpu/run_v79_real_rollout.py',
                 '--algorithm','matd3','--envs','2','--steps','2','--horizon','2',
                 '--batch-size','4','--hidden','16','--world-device','cpu',
                 '--learner-device','cpu','--allow-experimental-physics',
                 '--output',str(output)]
            run=subprocess.run(cmd,cwd=ROOT,env=env,text=True,
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout[-1500:])
            d=json.loads(output.read_text())
            self.assertEqual(d['status'],'V79_CPU_ROLLOUT_PASS_VERIFY_PARITY_AND_CONVERGENCE')
            self.assertEqual(d['episodes_ended_total'],2)
            self.assertEqual(d['world_steps'],2)
            self.assertEqual(d['updates'],1)  # first step collects 2 < batch=4
            self.assertEqual(d['cpu_warp_parity_proven'],False)
            self.assertEqual(d['gpu_training_converged'],False)

if __name__=='__main__':unittest.main()
