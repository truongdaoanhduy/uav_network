import json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/tmp/uav-task7-smoke-source')
COMMIT='0c80b7d4c7016412d86883ea4e39eca5931811c9'
subprocess.run(['git','clone','--quiet','https://github.com/truongdaoanhduy/uav_network.git',str(ROOT)],check=True)
subprocess.run(['git','checkout','--quiet',COMMIT],cwd=ROOT,check=True)
subprocess.run(['bash','setup.sh','--system-site-packages'],cwd=ROOT,env=dict(os.environ,PYTHON=sys.executable),check=True)
python=str(ROOT/'.venv/bin/python')
subprocess.run([python,'-m','pip','install','-q','warp-lang==1.18.0','pytest','pytest-subtests'],check=True)
script=Path('/kaggle/working/smoke.py')
script.write_text('''import json,sys,subprocess,os
from pathlib import Path
import torch,warp
from torch._inductor import config as inductor_config
from uav_network.configuration import Config
from uav_network.cloud.execution_check import check_execution
torch.set_num_threads(1);inductor_config.compile_threads=1
out=Path('/kaggle/working')
result={'source_commit':'0c80b7d4c7016412d86883ea4e39eca5931811c9','scope':'Bounded correctness smoke; not canonical 4096-world memory, speed or quality evidence','versions':{'torch':torch.__version__,'warp':warp.__version__},'cuda_devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'cases':{}}
for flag in ['compile_models','fused_optimizers']:
    c=Config();c.world.num_uavs=2;c.world.num_targets=2;c.world.num_obstacles=0;c.world.max_steps=10;c.world.dt_s=.01
    c.runtime.environment_backend='warp_packet';c.runtime.device='cuda';c.runtime.replay_device='cpu';c.runtime.replay_cpu_fraction=.75
    c.runtime.num_envs=4;c.runtime.torch_threads=1;c.runtime.vectorize_critics=True;c.runtime.amp_mode='off';c.runtime.compile_models=False;c.runtime.fused_optimizers=False
    setattr(c.runtime,flag,True)
    c.masac.hidden_sizes=[16,16];c.masac.replay_capacity=64;c.masac.batch_size=8;c.masac.learning_starts=0;c.masac.updates_per_step=1
    c.tracking.enabled=False;c.validate()
    report=check_execution(c,out/(flag+'-execution-check.json'))
    assert report['status']=='passed' and report['replay_layout']=='hybrid'
    assert report['replay_cpu_bytes']>0 and report['replay_hot_bytes']>0
    assert report['devices']['world']=='cuda:0' and report['devices']['learner']=='cuda:1'
    assert report['critic_updates']>0 and report['actor_updates']>0
    result['cases'][flag]=report
    (out/'task7-acceleration-smoke.json').write_text(json.dumps(result,indent=2))
    print('PREFLIGHT',flag,json.dumps(report),flush=True)
test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_memory_probe.py::test_cuda_probe_adapter_executes_fp16_compiled_critic_and_real_steps'],capture_output=True,text=True)
(out/'task7-base-probe-cuda.log').write_text(test.stdout+test.stderr)
result['base_probe_cuda_exit_code']=test.returncode;result['base_probe_cuda_tail']=(test.stdout+test.stderr)[-4000:]
(out/'task7-acceleration-smoke.json').write_text(json.dumps(result,indent=2))
print('BASE_PROBE_CUDA',test.returncode,result['base_probe_cuda_tail'],flush=True)
raise SystemExit(test.returncode)
''')
subprocess.run([python,str(script)],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',WANDB_MODE='offline'),check=True)
