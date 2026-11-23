"""Real compiled CPU MATD3 TX08 train; only W&B and HF transports are fakes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,sys,tarfile

import torch

from uav_network.configuration import load_config
from uav_network.experiments.training import train
from uav_network.experiments.storage_retention import verify_archive_parts
from scripts.launch_nooracle_map800_tx08_training_with_hf_forensics_20261123 import CONFIG,forensic_environment


class FakeRun:
    def __init__(self):
        self.offline=False;self.id='tx08proof99';self.entity='owner';self.project='paper'
        self.url='https://wandb.ai/owner/paper/runs/tx08proof99'
        self.summary={};self.logged=[];self.saved=[];self.finish_code=None
    def define_metric(self,*args,**kwargs):pass
    def save(self,path,**kwargs):self.saved.append(Path(path).name)
    def log(self,data,**kwargs):self.logged.append(dict(data))
    def finish(self,exit_code=0):self.finish_code=exit_code


class FakeVerifiedHF:
    instances=[]
    def __init__(self,config,output,run_id,*,verify_download=True,**kwargs):
        self.output=Path(output);self.run_id=run_id
        self.url='https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/runs/'+run_id
        self.repo_id='duy95/uav-search-diagnostics'
        self.verify_download=verify_download
        self.uploads=[];self.__class__.instances.append(self)
    def save(self,filename,*,base_path,policy):
        p=Path(filename)
        assert Path(base_path)==self.output and policy=='now'
        assert p.exists() and p.suffix=='.gz'
        if p.name.endswith('.tar.gz'):
            with tarfile.open(p,mode='r:gz') as archive:
                for member in archive:
                    assert member.isfile() and member.size>=0
                    assert len(hashlib.sha256(archive.extractfile(member).read()).hexdigest())==64
        self.uploads.append(p.name)
        return 'runs/'+self.run_id+'/'+str(p.relative_to(self.output))
    def stats(self):
        return {'storage/hf_verified_uploaded_files':len(self.uploads),
                'storage/hf_readback_sha256_enabled':int(self.verify_download)}


def test_tx08_compiled_real_train_emits_deep_rf_and_optimizer_logs_to_fake_public_hf(tmp_path,monkeypatch):
    FakeVerifiedHF.instances.clear()
    fake=FakeRun()
    fake_sdk=SimpleNamespace(init=lambda **kw:fake,Settings=lambda **kw:kw,Artifact=lambda *args,**kwargs:None)
    c=load_config(CONFIG)
    c.runtime.device='cpu';c.runtime.num_envs=2;c.runtime.episodes=2
    c.world.max_steps=12
    c.matd3.learning_starts=4;c.matd3.batch_size=16
    c.matd3.replay_capacity=1024;c.matd3.updates_per_step=1
    c.runtime.output_dir=str(tmp_path)
    c.runtime.checkpoint_interval=2;c.runtime.log_interval_steps=8
    c.tracking.deferred_gpu_eval_collection_stride=4
    c.tracking.deferred_gpu_eval_keep_all=True
    c.tracking.entity='owner';c.tracking.project='paper'
    c.tracking.checkpoint_repo='fake/test'
    c.tracking.resumable_checkpoint_interval_seconds=0
    c.validate()
    for key,value in forensic_environment({'WANDB_RUN_ID':'tx08proof99'}).items():
        if key in ('UAV_DEFERRED_EXPORT_HF',):value='false' # no remote checkpoint upload in smoke test
        monkeypatch.setenv(key,value)
    monkeypatch.setenv('TORCHINDUCTOR_COMPILE_THREADS','1')
    torch.set_num_threads(1)
    monkeypatch.setitem(sys.modules,'wandb',fake_sdk)
    with patch('uav_network.experiments.tracking._authentication',return_value=('owner','paper')),\
         patch('uav_network.experiments.tracking.PublicHubDiagnosticUploader',FakeVerifiedHF):
        result=train(c)
    assert result['episodes']==2 and result['episode_budget_complete']
    assert fake.finish_code==0
    assert FakeVerifiedHF.instances and FakeVerifiedHF.instances[0].uploads
    assert fake.summary['forensic_storage']['backend']=='huggingface_public_dataset'
    assert any('network/diagnostics/tx_power_histogram_edges_w/bin_4' in d for d in fake.logged)
    traces=[json.loads(x) for x in (tmp_path/'forensic-transitions.jsonl').read_text().splitlines()]
    assert traces
    assert {x['world_index'] for x in traces}=={0,1}
    for row in traces:
        for phase in ('world_before','world_after'):
            for key in ('tx_energy','rx_energy','pkt_power','tx_power','diagnostic_counts','power_histogram'):
                assert key in row[phase],(phase,key)
        assert row['actual_actor_action_destination']
        assert row['actual_actor_action_continuous_xyz_rf']
    manifest=verify_archive_parts(tmp_path)
    assert manifest['full_reconstruction_verified'] is True
    sources={m['name'] for part in manifest['parts'] for m in part['members']}
    assert {'episodes.jsonl','forensic-transitions.jsonl','forensic-minibatches.jsonl',
            'training-step-reward.jsonl','training-step-optimizer.jsonl','run-provenance.jsonl'}<=sources
