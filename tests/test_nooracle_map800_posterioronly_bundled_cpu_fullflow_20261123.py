"""Real CPU compiled source train->v11 checkpoint->heldout evaluation, no network."""
import ast,base64,io,re,tarfile,subprocess,os,sys,json
from pathlib import Path

from scripts.build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123 import (
    CONFIG,build_notebook)


def test_exact_kaggle_embedded_source_can_train_and_evaluate_posterior_only(tmp_path):
    nb=build_notebook()
    m=re.search(r'payload = base64\.b85decode\(("[^"]+")\)',nb.cells[2].source)
    archive=base64.b85decode(ast.literal_eval(m.group(1)))
    root=tmp_path/'source';root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name.startswith(('uav_network/','configs/','tests/','scripts/','third_party/','pyproject.toml'))
            path=root/member.name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
    assert (root/CONFIG).is_file()
    runner='''import json
from pathlib import Path
import torch
from uav_network.configuration import load_config
from uav_network.experiments.training import train
from uav_network.experiments.evaluation import evaluate
from uav_network.environment.observations import observation_version
p=Path('/tmp/noop')
c=load_config(__CONFIG__)
c.runtime.device='cpu';c.runtime.num_envs=2;c.runtime.episodes=2
c.world.max_steps=12;c.matd3.learning_starts=4;c.matd3.batch_size=16
c.matd3.replay_capacity=1024;c.matd3.updates_per_step=1
c.runtime.output_dir=__OUTPUT__
c.tracking.enabled=False;c.tracking.deferred_gpu_eval_enabled=False
c.validate()
assert observation_version(c)==11
result=train(c)
assert result['episodes']==2 and result['episode_budget_complete'],result
checkpoint=result['checkpoint']
assert Path(checkpoint).is_file(),checkpoint
eval_result=evaluate(checkpoint,episodes=2,seed=2000000044,output_dir=__EVAL__)
assert eval_result['episodes']==2
a={'actual_training_episodes':result['episodes'],'validation_episodes':eval_result['episodes'],
   'checkpoint_exists':True,'observation_version':11,'source':'kaggle_embedded_tar'}
print('BUNDLED_POSTERIORONLY_CPU_FLOW_JSON',json.dumps(a),flush=True)
'''
    source=runner.replace('__CONFIG__',repr(CONFIG)).replace('__OUTPUT__',repr(str(tmp_path/'train'))).replace('__EVAL__',repr(str(tmp_path/'eval')))
    p=subprocess.run([sys.executable,'-c',source],cwd=root,
       env={**os.environ,'PYTHONPATH':str(root),'TORCHINDUCTOR_COMPILE_THREADS':'1',
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
       capture_output=True,text=True,timeout=95)
    assert p.returncode==0, (p.stdout[-2100:],p.stderr[-2200:])
    key='BUNDLED_POSTERIORONLY_CPU_FLOW_JSON '
    rows=[json.loads(line[len(key):]) for line in p.stdout.splitlines() if line.startswith(key)]
    assert len(rows)==1
    assert rows[0]['actual_training_episodes']==2
    assert rows[0]['validation_episodes']==2
    assert rows[0]['checkpoint_exists']
