PHASE='red'
COMMIT='6e0ed9bd5c45dfa2d7a7af86a8042a9966122468'
TEST='"""Sparse checkpoints must exclude unused replay storage and retain exact resume."""\nimport io\nfrom types import SimpleNamespace\n\nimport pytest\nimport torch\n\nfrom uav_network.experiments.pql_replay import MeasuredReplay\n\n\ndef replay_with_rows(capacity, device, rows):\n    replay = MeasuredReplay(capacity, (2, 3), 4, seed=44, device=device)\n    for offset in range(0, rows, 4):\n        count = min(4, rows - offset)\n        observation = torch.full((count, 2, 3), float(offset), device=device)\n        state = torch.full((count, 4), float(offset), device=device)\n        action = SimpleNamespace(continuous=torch.zeros((count, 2, 3), device=device),\n                                 destination=torch.zeros((count, 2), dtype=torch.long, device=device))\n        replay.add_batch(observation, state, action, torch.full((count,), float(offset), device=device),\n                         observation + 1, state + 1, torch.zeros(count, device=device))\n    if rows:\n        replay.sample(min(3, len(replay)))\n    return replay\n\n\ndef available(device):\n    if device.startswith("cuda") and (not torch.cuda.is_available()\n                                     or int(device.split(":")[1]) >= torch.cuda.device_count()):\n        pytest.skip("requested real CUDA device unavailable")\n\n\ndef serialize(replay):\n    stream = io.BytesIO()\n    torch.save(replay.state_dict(), stream)\n    return stream.getvalue()\n\n\n@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])\n@pytest.mark.parametrize("rows", [0, 3])\ndef test_sparse_checkpoint_size_does_not_grow_with_unused_capacity(device, rows):\n    available(device)\n    small = serialize(replay_with_rows(16, device, rows))\n    large = serialize(replay_with_rows(2048, device, rows))\n    # Both contain the same 0/3 valid transitions; capacity metadata needs only bytes.\n    assert len(large) <= len(small) + 2048, (len(small), len(large))\n\n\n@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])\n@pytest.mark.parametrize("rows", [3, 20])\ndef test_serialized_replay_preserves_next_samples_and_measurements(device, rows):\n    available(device)\n    original = replay_with_rows(16, device, rows)\n    saved = torch.load(io.BytesIO(serialize(original)), map_location="cpu", weights_only=True)\n    restored = MeasuredReplay(16, (2, 3), 4, seed=99, device=device)\n    restored.load_state_dict(saved)\n    assert restored.index == original.index\n    assert restored.size == original.size\n    assert restored.stats() == original.stats()\n    for _ in range(2):\n        expected, actual = original.sample(3), restored.sample(3)\n        for name in expected:\n            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)\n    assert restored.stats() == original.stats()\n'

import os,json,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
root=Path('/tmp/uav-pql-sparse-source');out=Path('/kaggle/working')
r={'phase':PHASE,'source_commit':COMMIT,'started_utc':datetime.now(timezone.utc).isoformat(),'stages':[],'status':'starting','scope':'No W&B/HF writes or deletion; bounded CUDA verification only'}
def save(): (out/'pql-sparse-receipt.json').write_text(json.dumps(r,indent=2))
save()
try:
 subprocess.run(['git','clone','--quiet','https://github.com/truongdaoanhduy/uav_network.git',str(root)],check=True)
 subprocess.run(['git','checkout','--quiet',COMMIT],cwd=root,check=True)
 (root/'tests/test_pql_replay_serialization.py').write_text(TEST)
 if PHASE=='green':
  subprocess.run(['bash','setup.sh','--system-site-packages','--cloud'],cwd=root,env=dict(os.environ,PYTHON=sys.executable),check=True)
  python=str(root/'.venv/bin/python')
  subprocess.run([python,'-m','pip','install','-q','pytest','pytest-subtests','warp-lang==1.18.0'],check=True)
 else:
  python=sys.executable
  subprocess.run([python,'-m','pip','install','-q','pytest'],check=True)
 env=dict(os.environ,PYTHONPATH=str(root)+':'+str(root/'tests'),WANDB_MODE='disabled',HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
 for k in ['HF_TOKEN','WANDB_API_KEY','HUGGING_FACE_HUB_TOKEN']:env.pop(k,None)
 hw=subprocess.run([python,'-c',"import json,torch;assert torch.cuda.is_available() and torch.cuda.device_count()==2;print(json.dumps({'devices':[torch.cuda.get_device_name(i) for i in range(2)],'torch':torch.__version__,'cuda':torch.version.cuda}))"],env=env,text=True,capture_output=True,check=True)
 r['hardware']=json.loads(hw.stdout.strip().splitlines()[-1]);save()
 stages=[('sparse-regression',[python,'-m','pytest','-q','tests/test_pql_replay_serialization.py','--junitxml='+str(out/'sparse.xml')],600)]
 if PHASE=='green':stages += [('full-suite',[python,'-m','pytest','-q','-ra','tests','--junitxml='+str(out/'full-suite.xml')],3600),('full4096-pql',[python,'-m','pytest','-q','-s','tests/test_cuda_exact_resume.py::CudaFull4096ResumeTests::test_masac_pql','tests/test_cuda_exact_resume.py::CudaFull4096ResumeTests::test_matd3_pql','--junitxml='+str(out/'full4096.xml')],2400)]
 for name,command,timeout in stages:
  r['active_stage']=name;save();start=time.monotonic()
  stage_env=dict(env,UAV_CUDA_FULL4096='1') if name=='full4096-pql' else env
  with (out/(name+'.log')).open('w') as log:p=subprocess.run(command,cwd=root,env=stage_env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
  r['stages'].append({'name':name,'exit_code':p.returncode,'seconds':time.monotonic()-start,'tail':(out/(name+'.log')).read_text(errors='replace')[-4000:]});save()
  if PHASE=='red':
   suite=ET.parse(out/'sparse.xml').getroot().find('testsuite');assert p.returncode==1 and suite.attrib['failures']=='6' and suite.attrib['errors']=='0' and suite.attrib['skipped']=='0', 'Unexpected RED result'
  elif p.returncode:raise RuntimeError('Stage failed: '+name)
 r['status']='expected_red' if PHASE=='red' else 'passed';r['completed_utc']=datetime.now(timezone.utc).isoformat();save();print(json.dumps(r),flush=True)
except BaseException as e:
 r.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());save();print(json.dumps(r),flush=True);raise
