"""Real MATD3 train -> deferred snapshot handoff, online scalar data simulated."""
import json
from pathlib import Path
from types import SimpleNamespace

from uav_network.configuration import load_config
from uav_network.experiments.training import train


class FakeOnlineRun:
    def __init__(self):
        self.offline=False
        self.id='aabbcc99'
        self.entity='owner'
        self.project='paper'
        self.url='https://wandb.ai/owner/paper/runs/aabbcc99'
        self.summary={}
        self.logged=[]
        self.saved=[]
        self.finish_code=None
    def define_metric(self,*args,**kwargs):pass
    def save(self,p,**kwargs):self.saved.append(Path(p).name)
    def log(self,values,**kwargs):self.logged.append(dict(values))
    def finish(self,exit_code=0):self.finish_code=exit_code


def test_actual_matd3_training_completes_without_early_best_and_produces_dense_wandb(tmp_path,monkeypatch):
    import sys
    from unittest.mock import patch
    fake=FakeOnlineRun()
    sdk=SimpleNamespace(init=lambda **kw:fake,Settings=lambda **kw:kw,
                        Artifact=lambda *args,**kwargs:None)
    c=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    c.runtime.device='cpu';c.runtime.num_envs=2;c.runtime.episodes=4
    c.world.max_steps=20
    c.matd3.learning_starts=4;c.matd3.batch_size=16
    c.matd3.replay_capacity=1024;c.matd3.updates_per_step=4
    c.runtime.output_dir=str(tmp_path)
    c.runtime.checkpoint_interval=2;c.runtime.log_interval_steps=8
    c.tracking.enabled=True
    c.tracking.deferred_gpu_eval_collection_stride=4
    c.tracking.deferred_gpu_eval_keep_all=True
    c.tracking.online_gpu_eval_enabled=False
    c.runtime.max_train_seconds=0
    c.tracking.entity='owner';c.tracking.project='paper'
    c.tracking.checkpoint_store='huggingface'
    c.tracking.checkpoint_repo='owner/repo'
    monkeypatch.setenv('UAV_TRAINING_ONLY','true')
    monkeypatch.setenv('UAV_DEFERRED_EXPORT_HF','false')
    monkeypatch.setenv('UAV_DIAGNOSTIC_STORE','off')
    monkeypatch.setenv('UAV_WANDB_COMPACT_LOGS','true')
    monkeypatch.setenv('UAV_WANDB_EPISODE_LOG_EVERY','2')
    monkeypatch.setenv('UAV_TRAIN_EPISODE_RETURN_LOG_EVERY','2')
    monkeypatch.setenv('WANDB_RUN_ID','aabbcc99')
    # Avoid removing modules lazily imported by torch.manual_seed when the
    # W&B mock is torn down; PyTorch C++ library registration survives.
    monkeypatch.setitem(sys.modules,'wandb',sdk)
    with patch('uav_network.experiments.tracking._authentication',return_value=('owner','paper')):
        result=train(c)
    assert result['episode_budget_complete'] and result['episodes']==4
    assert result['checkpoint_artifact'] is None
    assert result['training_completed_awaiting_best_actor'] is True
    assert fake.summary['fullflow/status']=='awaiting_gpu_validation'
    assert fake.summary['fullflow/train_complete'] is True
    assert fake.finish_code==0
    assert any('train/step_reward_mean' in row for row in fake.logged)
    assert any('train_progress/discovery_rate_percent' in row for row in fake.logged)
    assert any('train_loss/critic_loss' in row for row in fake.logged)
    assert any('training/discovery_auc' in row for row in fake.logged)
    assert any('training/coverage_auc' in row for row in fake.logged)
    assert any('train_cohort/discovery_auc_mean' in row for row in fake.logged)
    steps=[row['global_step'] for row in fake.logged if 'train/step_reward_mean' in row]
    assert steps==sorted(steps) and all(n%2==0 for n in steps)
    assert all('episode/index' not in row for row in fake.logged if 'train_loss/critic_loss' in row)
    manifest=Path(result['deferred_gpu_evaluation']['manifest'])
    data=json.loads(manifest.read_text())
    assert data['count']>0 and data['collected']==data['count']
