from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json
from unittest.mock import patch
from types import SimpleNamespace
import torch
import numpy as np
from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv, Action

class DiagnosticsTests(unittest.TestCase):
    def test_reward_components_match_real_episode_return(self):
        c=Config();c.network.enabled=False;c.world.max_steps=4
        env=SwarmSearchEnv(c);env.reset(seed=44);total=0
        try:
            for _ in range(4):
                _,r,_,_,info=env.step(Action(np.zeros((6,3)),np.zeros(6,dtype=int)));total+=r
            self.assertAlmostEqual(sum(info['reward_components'].values()),total)
            self.assertEqual(info['action_counts']['silent'],24)
        finally:env.close()

    def test_warnings_keep_evidence_without_claiming_a_root_cause(self):
        from uav_network.experiments.diagnostics import TrainingDiagnostics
        with TemporaryDirectory() as folder:
            c=Config();c.tracking.diagnostic_window=2
            d=TrainingDiagnostics(c,Path(folder))
            for i in range(2):
                values=d.episode({'episode':i+1,'global_step':(i+1)*10,'return':-5,
                                 'search_rate_percent':80,'delivery_rate_percent':0,'success':False})
            self.assertEqual(values['diagnostics/delivery_gap_warning'],1)
            self.assertTrue((Path(folder)/'diagnostics.jsonl').is_file())

    def test_real_training_logs_q_gradients_and_last_actual_td3_actor_update(self):
        from test_continuation import small_config
        from uav_network.experiments.training import train
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as folder:
                metrics=[]
                tracker=SimpleNamespace(learner=lambda values, step: metrics.append(values),
                    episode=lambda *args: None, diagnostic=lambda *args: None,
                    progress=lambda *args, **kwargs: None, failure=lambda *args: None,
                    identity=lambda: {"wandb_run_id":"test"}, complete=lambda *args: {},
                    close=lambda **kwargs: None)
                with patch("uav_network.experiments.training.start_tracking", return_value=tracker):
                    train(small_config(Path(folder)/algorithm,algorithm))
                self.assertTrue(metrics)
                self.assertTrue(all(np.isfinite(list(m.values())).all() for m in metrics))
                self.assertIn("critic_grad_norm",metrics[-1])
                self.assertIn("td_error_abs_mean",metrics[-1])
                self.assertIn("actor_loss",metrics[-1])
                if algorithm=="matd3":
                    self.assertEqual(metrics[-1]["actor_update_step"],2)

    def test_real_nonfinite_model_preserves_failed_replay_batch_and_traceback(self):
        from test_continuation import small_config
        from uav_network.algorithms.checkpoints import make_learner
        from uav_network.environment.observations import observation_dim,state_dim
        from uav_network.experiments.training import train
        with TemporaryDirectory() as directory:
            folder=Path(directory)/"failed"
            c=small_config(folder)
            learner=make_learner(c,observation_dim(c),state_dim(c))
            with torch.no_grad():next(learner.critics.parameters()).fill_(float("nan"))
            progress={"episodes":0,"global_step":0,"collection_steps":0,"next_seed":44,"records":[]}
            with patch("uav_network.experiments.training.restore_learner",return_value=(learner,progress)):
                with self.assertRaises(FloatingPointError):train(c)
            error=json.loads((folder/"failure.json").read_text())
            self.assertIn("Non-finite learner loss",error["traceback"])
            self.assertEqual(error["context"]["episode_seeds"],[44,45])
            self.assertTrue((folder/"failed_batch.pt").is_file())
            self.assertTrue(error["context"]["nonfinite_parameter_names"])
            self.assertTrue((folder/"failure_state.pt").is_file())

    def test_missing_checkpoint_and_worker_startup_preserve_failure_evidence(self):
        from test_continuation import small_config
        from uav_network.experiments.training import train
        with TemporaryDirectory() as directory:
            folder=Path(directory)/"missing"
            c=small_config(folder);c.runtime.warm_start_checkpoint=str(Path(directory)/"absent.pt")
            with self.assertRaises(FileNotFoundError):train(c)
            self.assertEqual(json.loads((folder/"failure.json").read_text())["error_type"],"FileNotFoundError")
            folder=Path(directory)/"worker"
            with patch("uav_network.experiments.training.make_vector",side_effect=OSError("worker start failed")):
                with self.assertRaises(OSError):train(small_config(folder))
            self.assertEqual(json.loads((folder/"failure.json").read_text())["error_type"],"OSError")

if __name__=='__main__':unittest.main()
