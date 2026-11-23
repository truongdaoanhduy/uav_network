"""Kaggle V7 crash reproduction: train-only completion precedes GPU validation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from uav_network.configuration import Config
from uav_network.experiments.tracking import OnlineTracker


def _tracker(path, *, deferred):
    t=OnlineTracker.__new__(OnlineTracker)
    c=Config()
    c.tracking.deferred_gpu_eval_enabled=deferred
    c.tracking.deferred_gpu_eval_keep_all=True
    t.config=c
    t.output=Path(path)
    t.run=SimpleNamespace(summary={})
    t.records=[]
    t.initial_step=0
    t.latest_evidence={}
    t.learning_phase='learning'
    t._record_event=Mock()
    t.checkpoint=Mock(side_effect=AssertionError('MUST defer best upload until validation'))
    t.identity=Mock(return_value={'wandb_run_id':'deadbeef'})
    return t


def _result(path):
    root=Path(path)
    folder=root/'deferred-gpu-eval';folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps({'count':2,'collected':2,'pruned':0,'keep_all_distinct_versions':True}))
    (folder/'deferred-gpu-eval.tar.gz').write_bytes(b'recovery archive fixture')
    return {'global_step':29368320,'episodes':20000,'updates':28556,
            'episode_budget_complete':True,'stop_reason':'episode_budget',
            'checkpoint':str(root/'checkpoint.pt'),
            'checkpoint_selection':{'kind':'none','reason':'No valid periodic validation completed'},
            'deferred_gpu_evaluation':{
                'snapshot_count':2,'manifest':str(folder/'manifest.json'),
                'archive':str(folder/'deferred-gpu-eval.tar.gz')},
            'environment_state_saved':False}


def test_finished_training_hands_off_complete_actor_archive_before_heldout_validation():
    with TemporaryDirectory() as d,patch.dict('os.environ',{'UAV_TRAINING_ONLY':'true'}):
        tracker=_tracker(d,deferred=True)
        result=tracker.complete(_result(d))
        tracker.checkpoint.assert_not_called()
        assert result['checkpoint_artifact'] is None
        assert result['training_completed_awaiting_best_actor'] is True
        assert tracker.run.summary['fullflow/train_complete'] is True
        assert tracker.run.summary['checkpoint_selection']['kind']=='pending_gpu_validation'
        assert tracker.run.summary['episode_budget_complete'] is True


def test_training_only_without_valid_snapshot_archive_does_not_claim_success():
    with TemporaryDirectory() as d,patch.dict('os.environ',{'UAV_TRAINING_ONLY':'true'}):
        tracker=_tracker(d,deferred=True)
        result=_result(d)
        Path(result['deferred_gpu_evaluation']['archive']).unlink()
        with pytest.raises((ValueError,FileNotFoundError),match='archive|snapshot'):
            tracker.complete(result)
        tracker.checkpoint.assert_not_called()


def test_ordinary_non_deferred_tracking_still_publishes_final_checkpoint():
    with TemporaryDirectory() as d,patch.dict('os.environ',{'UAV_TRAINING_ONLY':'false'}):
        t=_tracker(d,deferred=False)
        t.checkpoint.side_effect=None
        t.checkpoint.return_value='owner/main/checkpoint:v1'
        result=_result(d)
        assert t.complete(result)['checkpoint_artifact']=='owner/main/checkpoint:v1'
        t.checkpoint.assert_called_once_with(result['checkpoint'],result,final=True)
