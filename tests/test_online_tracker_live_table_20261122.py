"""W&B Runs Runtime/Episodes must work while MATD3 is actually training."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from uav_network.configuration import Config
from uav_network.experiments.tracking import OnlineTracker


class FakeRun:
    offline = False
    id = 'table001'
    entity = 'owner'
    project = 'project'
    url = 'https://wandb.ai/owner/project/runs/table001'

    def __init__(self):
        self.summary={}
        self.logged=[]
        self.exit_code=None
    def define_metric(self,*args,**kwargs):pass
    def save(self,*args,**kwargs):pass
    def log(self,values):self.logged.append(dict(values))
    def finish(self,exit_code=0):self.exit_code=exit_code


def new_tracker(run,directory):
    config=Config()
    config.runtime.episodes=20000
    config.tracking.enabled=True
    config.tracking.checkpoint_store='wandb'
    config.tracking.credentials_file=str(Path(directory)/'absent-file.json')
    sdk=SimpleNamespace(init=lambda **kwargs:run,Settings=lambda **kwargs:kwargs)
    return config,sdk


def test_live_heartbeat_publishes_runtime_and_actual_episodes_before_first_reset():
    run=FakeRun()
    with TemporaryDirectory() as directory:
        config,sdk=new_tracker(run,directory)
        with patch.dict('sys.modules',{'wandb':sdk}),patch.dict(os.environ,{
            'WANDB_API_KEY':'fake','WANDB_ENTITY':'owner','WANDB_PROJECT':'project'}):
            tracker=OnlineTracker(config,directory)
            try:
                assert run.summary['Episodes']=='0 / 20000'
                assert isinstance(run.summary['Runtime'],str) and run.summary['Runtime']
                assert tracker.live_runtime.thread.is_alive()
                with patch('uav_network.experiments.tracking.episode_metrics',return_value={}):
                    tracker.episode({'episode':7})
                tracker.live_runtime.publish()
                assert run.summary['Episodes']=='7 / 20000'
                assert (Path(directory)/'runtime-heartbeat.json').is_file()
                assert run.summary['runtime_is_live'] is True
            finally:
                tracker.close()
            assert tracker.live_runtime.thread.is_alive() is False
            assert run.summary['Episodes']=='7 / 20000'
            assert run.summary['runtime_status']=='finished'
            assert run.exit_code==0


def test_failed_online_tracker_stops_heartbeat_and_does_not_forge_episodes():
    run=FakeRun()
    with TemporaryDirectory() as directory:
        config,sdk=new_tracker(run,directory)
        with patch.dict('sys.modules',{'wandb':sdk}),patch.dict(os.environ,{
            'WANDB_API_KEY':'fake','WANDB_ENTITY':'owner','WANDB_PROJECT':'project'}):
            tracker=OnlineTracker(config,directory,initial_progress={'episodes':11})
            assert run.summary['Episodes']=='11 / 20000'
            tracker.close(failed=True)
            assert tracker.live_runtime.thread.is_alive() is False
            assert run.summary['runtime_status']=='failed'
            assert run.summary['Episodes']=='11 / 20000'
            assert run.exit_code==1
