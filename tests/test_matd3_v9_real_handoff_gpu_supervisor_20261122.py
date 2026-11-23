"""Real CPU snapshot bank through the *actual* one-session phase-2 supervisor.

GPU subprocess and W&B network are faked; no synthetic result is reported as
real validation. This test targets the missing handoff that killed 504d1b89.
"""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

from uav_network.experiments.one_session import run


class FakeWandbRun:
    def __init__(self):
        self.summary={}
        self.finished=None
    def finish(self,exit_code):self.finished=exit_code


class FakeMonitor:
    def __init__(self,**kw):self.kw=kw
    def start(self):return self
    def stop(self,**kw):self.stopped=kw


class FakeStreamer:
    def __init__(self,owner,rows,folder,**kw):
        self.index=len(rows)
        self.finished=True
    def poll(self):pass


def test_phase2_consumes_real_immutable_actor_bank_and_reaches_test_completion(tmp_path,monkeypatch):
    original=Path('/tmp/uav_matd3_v9_cpu_train_val100_20261122/run_01_seed44')
    if not (original/'deferred-gpu-eval/manifest.json').is_file():
        import pytest
        pytest.skip('Real CPU train/Val100 smoke bank not present on this system')
    folder=tmp_path/'train/matd3';folder.mkdir(parents=True)
    shutil.copytree(original/'deferred-gpu-eval',folder/'deferred-gpu-eval')
    shutil.copy2(original/'summary.json',folder/'summary.json')
    spec={'stage':'train','runs':{'matd3':{'id':'aabbcc99'}},'training_only':True,
          'gpus_per_run':2,'require_full_episode_budget':True,
          'entity':'owner','project':'paper','gpu_eval_candidate_limit':2,
          'validation_step_targets':None,
          'dense_eval_schedule':{'final_test_scenarios':4000}}
    spec_file=tmp_path/'job.json';spec_file.write_text(json.dumps(spec))
    out=tmp_path/'gpu_batch_eval'
    fake_wandb=FakeWandbRun()
    def subprocess_eval(args):
        out.mkdir(exist_ok=True)
        (out/'report.json').write_text(json.dumps({
            'validation_completed':2,'best_global_step':80,
            'best_checkpoint':str(folder/'deferred-gpu-eval/eval_step_000000000080_policy_33.pt'),
            'publication':{'hf_best':'verified'},
            'final_test':{'episodes':4000,'delivery_rate_percent':33.0}}))
        return SimpleNamespace(poll=lambda:0)
    import sys
    sdk=SimpleNamespace(init=lambda **kw:fake_wandb,
                        Api=lambda **kw:object())
    monkeypatch.setenv('UAV_DIAGNOSTIC_STORE','off')
    with patch.dict(sys.modules,{'wandb':sdk}),\
         patch('uav_network.experiments.one_session.subprocess.Popen',side_effect=subprocess_eval),\
         patch('uav_network.experiments.one_session_wandb.EvalConvergenceStreamer',FakeStreamer),\
         patch('uav_network.experiments.one_session_wandb.previously_logged_steps',return_value=set()),\
         patch('uav_network.experiments.one_session_wandb.log_final_test'),\
         patch('uav_network.experiments.post_training_runtime_heartbeat.PostTrainingRuntimeHeartbeat',FakeMonitor):
        status=run(spec_file,policies_per_batch=2,max_eval_worlds_per_gpu=256,
                   max_test_worlds_per_gpu=256,final_test_batch_size=0,work_dir=tmp_path)
    assert status['snapshots']>=2
    assert status['validation_candidates']==2
    assert status['gpu_val_status']=='complete'
    assert status['gpu_test_status']=='complete'
    assert status['test_episodes']==4000
    assert fake_wandb.summary['fullflow/train_complete'] is True
    assert fake_wandb.summary['fullflow/validation_complete'] is True
    assert fake_wandb.summary['fullflow/test_complete'] is True
    assert fake_wandb.summary['fullflow/status']=='finished'
    assert fake_wandb.finished==0
