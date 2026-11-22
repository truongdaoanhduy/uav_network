"""No-oracle run must not lose raw scientific evidence before HF upload."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import os
import sys

from uav_network.configuration import Config
from uav_network.experiments.tracking import OnlineTracker


class _FakeWandbRun:
    offline=False
    id='forensic1'
    entity='owner'
    project='paper'
    url='https://wandb.ai/owner/paper/runs/forensic1'

    def __init__(self):
        self.summary={}
        self.saved=[]
    def define_metric(self,*args,**kwargs):pass
    def save(self,filename,**kw):self.saved.append(Path(filename).name)
    def log(self,*args,**kwargs):pass
    def finish(self,exit_code=0):pass


class _FakeUploader:
    def __init__(self,config,output,run_id,*,verify_download=True):
        self.output=Path(output)
        self.uploads=[]
        self.repo_id='owner/uav-search-diagnostics'
        self.url='https://huggingface.co/datasets/owner/uav-search-diagnostics/tree/main/runs/'+run_id
    def save(self,filename,*,base_path,policy):
        assert Path(filename).is_file()
        assert policy=='now'
        self.uploads.append(Path(filename).name)
    def stats(self):return {'storage/hf_uploaded_count':len(self.uploads)}


def _config(path):
    c=Config()
    c.tracking.enabled=True
    c.tracking.credentials_file=str(Path(path)/'unused-credentials.json')
    c.tracking.entity='owner'
    c.tracking.project='paper'
    c.tracking.checkpoint_store='huggingface'
    c.tracking.checkpoint_repo='owner/checkpoint-repo'
    return c


def test_public_diagnostics_are_wired_and_periodically_uploaded_not_only_finalized():
    """Old 18d had archiver/uploader; v7 omitted both => this must RED before fix."""
    with TemporaryDirectory() as root:
        run=_FakeWandbRun()
        sdk=SimpleNamespace(init=lambda **kw:run,Settings=lambda **kw:kw)
        env={'WANDB_API_KEY':'fake-test','UAV_DIAGNOSTIC_STORE':'huggingface_public',
             'UAV_FAST_TRAINING':'false','UAV_HF_VERIFY_PUBLIC_READBACK':'false'}
        with patch.dict(sys.modules,{'wandb':sdk}),\
             patch.dict(os.environ,env),\
             patch('uav_network.experiments.tracking.PublicHubDiagnosticUploader',_FakeUploader):
            t=OnlineTracker(_config(root),root)
            try:
                assert t.public_diagnostics is not None
                assert t.compressed_journals is not None
                # Force archive tick without an external clock or network.
                t.compressed_journals.interval=0.
                t._record_event('progress',{'completed_episodes':1,'global_step':4096})
                w=t.compressed_journals
                assert w.parts_registered>=1,(w.parts_created,w.parts_registered,w.pending,w.upload_errors,w._new_bytes(), w.interval, t.public_diagnostics.uploads)
                assert t.public_diagnostics.uploads
                assert all(x.startswith('diagnostics-') and x.endswith('.tar.gz')
                           for x in t.public_diagnostics.uploads)
            finally:t.close()
            assert 'manifest.json.gz' in t.public_diagnostics.uploads


def test_forensic_writes_disabled_by_default_to_protect_fast_training():
    with TemporaryDirectory() as root:
        run=_FakeWandbRun()
        sdk=SimpleNamespace(init=lambda **kw:run,Settings=lambda **kw:kw)
        with patch.dict(sys.modules,{'wandb':sdk}),patch.dict(os.environ,{
            'WANDB_API_KEY':'fake-test','UAV_DIAGNOSTIC_STORE':'off'}):
            t=OnlineTracker(_config(root),root)
            try:
                assert getattr(t,'public_diagnostics',None) is None
                assert getattr(t,'compressed_journals',None) is None
            finally:t.close()
