"""Bounded CUDA correctness check; no training or experiment mutation."""
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = 'c05806a'
root = Path('/tmp/uav-local-report-audit')
output = Path('/kaggle/working')
subprocess.run(['git','clone','--quiet','https://github.com/truongdaoanhduy/uav_network.git',str(root)],check=True)
subprocess.run(['git','checkout','--quiet',COMMIT],cwd=root,check=True)
subprocess.run(['bash','setup.sh','--system-site-packages'],cwd=root,
               env=dict(os.environ,PYTHON=sys.executable),check=True)
python = str(root / '.venv/bin/python')
subprocess.run([python,'-m','pip','install','-q','warp-lang==1.18.0','pytest','pytest-subtests'],check=True)
env = dict(os.environ,PYTHONPATH=str(root),MPLBACKEND='Agg',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
probe = subprocess.run([python,'-c',"import torch,json; assert torch.cuda.is_available(); print(json.dumps({'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'torch':torch.__version__}))"],cwd=root,env=env,text=True,capture_output=True,check=True)
tests = ['tests/test_decentralized_observations.py','tests/test_gpu_physics.py','tests/test_gpu_packets.py','tests/test_gpu_vector.py']
run = subprocess.run([python,'-m','pytest','-q',*tests],cwd=root,env=env,text=True,capture_output=True,timeout=600)
(output / 'local-report-cuda.log').write_text(run.stdout + run.stderr)
receipt = {'source_commit':COMMIT,'scope':'native/Warp sensing locality and physics/packet/vector regression; no production training','hardware':json.loads(probe.stdout.strip().splitlines()[-1]),'tests':tests,'exit_code':run.returncode,'tail':(run.stdout+run.stderr)[-3500:]}
(output / 'local-report-cuda.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)
raise SystemExit(run.returncode)
