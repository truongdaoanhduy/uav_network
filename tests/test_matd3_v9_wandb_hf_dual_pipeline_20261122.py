"""Run true MATD3 CPU training with *both* W&B panels and HF forensic archiving.

Only the network transports are fake; Warp, collector, learner, telemetry,
artifact creation, and SHA256 integrity checks run for real. No remote upload.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import sys
import tarfile

from uav_network.configuration import load_config
from uav_network.experiments.training import train
from uav_network.experiments.storage_retention import verify_archive_parts

class FakeOnlineRun:
    def __init__(self):
        self.offline=False;self.id='aabbcc99';self.entity='owner';self.project='paper'
        self.url='https://wandb.ai/owner/paper/runs/aabbcc99'
        self.summary={};self.logged=[];self.saved=[];self.finish_code=None
    def define_metric(self,*args,**kwargs):pass
    def save(self,p,**kwargs):self.saved.append(Path(p).name)
    def log(self,values,**kwargs):self.logged.append(dict(values))
    def finish(self,exit_code=0):self.finish_code=exit_code



class VerifiedFakeHub:
    instances=[]
    def __init__(self,config,output,run_id,*,verify_download=True,**kwargs):
        self.output=Path(output)
        self.run_id=run_id
        self.url='https://huggingface.co/datasets/fake/audit/tree/main/runs/'+run_id
        self.repo_id='fake/audit'
        self.part_paths=[]
        self.manifests=[]
        self.verify_download=verify_download
        self.__class__.instances.append(self)
    def save(self,filename,*,base_path,policy):
        assert Path(base_path)==self.output
        assert policy=='now'
        path=Path(filename)
        assert path.is_file() and path.suffix=='.gz'
        if path.name.startswith('diagnostics-'):
            with tarfile.open(path,'r:gz') as tar:
                for member in tar:
                    assert member.isfile()
                    with tar.extractfile(member) as source:
                        digest=hashlib.sha256(source.read()).hexdigest()
                    assert len(digest)==64
            self.part_paths.append(path)
        elif path.name=='manifest.json.gz':
            self.manifests.append(path)
        return 'runs/'+self.run_id+'/'+path.relative_to(self.output).as_posix()
    def stats(self):
        return {'storage/hf_verified_uploaded_files':len(self.part_paths)+len(self.manifests),
                'storage/hf_readback_sha256_enabled':1}


def test_simultaneous_real_train_produces_paper_wandb_and_reconstructable_public_hf(tmp_path,monkeypatch):
    VerifiedFakeHub.instances.clear()
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
    c.tracking.enabled=True;c.tracking.deferred_gpu_eval_collection_stride=4
    c.tracking.deferred_gpu_eval_keep_all=True;c.tracking.online_gpu_eval_enabled=False
    c.runtime.max_train_seconds=0
    c.tracking.entity='owner';c.tracking.project='paper'
    c.tracking.checkpoint_store='huggingface';c.tracking.checkpoint_repo='fake/repo'
    for key,value in {'UAV_TRAINING_ONLY':'true','UAV_DEFERRED_EXPORT_HF':'false',
                      'UAV_DIAGNOSTIC_STORE':'huggingface_public','UAV_FAST_TRAINING':'false',
                      'UAV_WANDB_PLOTS_ONLY':'true','UAV_WANDB_COMPACT_LOGS':'true',
                      'UAV_WANDB_EPISODE_LOG_EVERY':'2',
                      'UAV_TRAIN_EPISODE_RETURN_LOG_EVERY':'2',
                      'WANDB_RUN_ID':'aabbcc99',
                      'UAV_FORENSIC_WORLD_IDS':'0,1',
                      'UAV_FORENSIC_STRIDE':'32',
                      'UAV_FORENSIC_FULL_BELIEF':'true',
                      'UAV_FORENSIC_MAX_RAW_MIB':'384'}.items():monkeypatch.setenv(key,value)
    # Do not use patch.dict(sys.modules) here: its teardown removes modules
    # imported by torch.manual_seed, but PyTorch's C++ library registrations
    # persist. A second MATD3 training test would then re-register
    # `_inductor_test` and fail with duplicate TORCH_LIBRARY namespace.
    monkeypatch.setitem(sys.modules,'wandb',sdk)
    with patch('uav_network.experiments.tracking._authentication',return_value=('owner','paper')), \
         patch('uav_network.experiments.tracking.PublicHubDiagnosticUploader',VerifiedFakeHub):
        result=train(c)
    assert result['episode_budget_complete'] and result['episodes']==4
    assert fake.finish_code==0
    assert fake.summary.get('fullflow/status')=='awaiting_gpu_validation'
    assert fake.summary.get('forensic_storage',{}).get('backend')=='huggingface_public_dataset'
    assert any('episode/index' in row and 'episode/return' in row for row in fake.logged)
    assert any('training/discovery_rate_percent' in row for row in fake.logged)
    assert any('training/gcs_delivery_rate_percent' in row for row in fake.logged)
    assert any('training/discovery_auc' in row for row in fake.logged)
    assert any('train_progress/discovery_rate_percent' in row for row in fake.logged)
    assert any('train_loss/critic_loss' in row for row in fake.logged)
    assert any('train_cohort/discovery_auc_mean' in row for row in fake.logged)
    assert not any('forensic-transitions.jsonl' in row for row in fake.logged)
    assert VerifiedFakeHub.instances
    hub=VerifiedFakeHub.instances[0]
    assert hub.part_paths and hub.manifests
    manifest=verify_archive_parts(tmp_path)
    assert manifest['full_reconstruction_verified'] is True
    assert len(manifest['parts'])>=1
    assert all(v==0 for v in manifest['unarchived_tail_bytes'].values())
    names={m['name'] for part in manifest['parts'] for m in part['members']}
    assert {'episodes.jsonl','training-step-reward.jsonl','training-step-progress.jsonl',
            'training-step-optimizer.jsonl','training-events.jsonl','run-provenance.jsonl'}.issubset(names)
    assert sum(1 for x in (tmp_path/'episodes.jsonl').open())==4
    traces=[json.loads(row) for row in (tmp_path/'forensic-transitions.jsonl').read_text().splitlines()]
    assert traces, 'Selected-world pre/post trace must be recorded'
    assert {r['world_index'] for r in traces}>={0,1}
    assert all('world_before' in r and 'world_after' in r and
               'reward_accounting' in r for r in traces)


def test_native_dashboard_has_original_episode_and_global_step_axes_with_five_sections():
    """Exact native scalar panels, not images, and no fake episode for losses."""
    import pytest
    pytest.importorskip('wandb_workspaces')
    from uav_network.experiments.wandb_dualaxis_baseline_workspace import (
        new_workspace,EPISODE,VECTOR,LEARNER,EVAL,TEST)
    view=new_workspace('aabbcc99',variant='V9 READ ONLY MODEL TEST')
    assert [len(section.panels) for section in view.sections]==[
        len(EPISODE),len(VECTOR),len(LEARNER),len(EVAL),len(TEST)]
    assert sum(len(section.panels) for section in view.sections)==48
    for section,expected_x in zip(view.sections[:4],[
        'episode/index','global_step','global_step',
        'gpu_eval_periodic/checkpoint_global_step']):
        for panel in section.panels:
            assert getattr(panel.x,'name',panel.x)==expected_x
            assert panel.smoothing_type=='average'
            assert panel.y and all(isinstance(y,str) for y in panel.y)
    assert all('episode/index' not in str(panel.x) for panel in view.sections[2].panels)
    assert 'gpu_final/' in str(view.sections[-1].panels[0].metric)
