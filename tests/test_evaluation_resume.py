"""Resumed CPU sessions retain every evaluation episode and deterministic seed."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from uav_network.configuration import Config
from uav_network.experiments.training import train
from uav_network.experiments import evaluation
from uav_network.cloud.visualize import build_visualization_spec
from test_cloud_continuation import parent_spec, learner_result


class EvaluationResumeTests(unittest.TestCase):
    def checkpoint(self, root):
        c = Config()
        c.runtime.output_dir = str(root / "train")
        c.runtime.algorithm = "matd3"
        c.runtime.device = c.runtime.replay_device = "cpu"
        c.runtime.episodes = 1
        c.world.max_steps = 3
        c.world.num_targets, c.world.num_obstacles = 1, 0
        c.matd3.hidden_sizes, c.matd3.replay_capacity = [8, 8], 8
        c.matd3.batch_size = c.matd3.learning_starts = 2
        return train(c)["checkpoint"]

    def test_paused_evaluation_resumes_same_seed_prefix_without_rerunning_completed_episodes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root)
            saved = []
            try:
                with self.assertRaises(evaluation.EvaluationPaused) as caught:
                    evaluation.evaluate(checkpoint, episodes=3, seed=100, output_dir=root/"first",
                        should_pause=lambda: bool(saved), on_checkpoint=lambda records: saved.append(list(records)))
            except AttributeError as error:
                self.fail(f"CPU session progress cannot be resumed: {error}")
            self.assertEqual([r["seed"] for r in caught.exception.records], [100])
            seen = []
            resumed = evaluation.evaluate(checkpoint, episodes=3, seed=100, output_dir=root/"resumed",
                completed_records=caught.exception.records, on_episode=lambda r: seen.append(r["seed"]))
            full = evaluation.evaluate(checkpoint, episodes=3, seed=100, output_dir=root/"full")
            self.assertEqual(seen, [101, 102])
            self.assertEqual(resumed["records"], full["records"])
            self.assertEqual(resumed["protocol"], full["protocol"])

    def test_wrong_seed_prefix_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root)
            record = evaluation.evaluate(checkpoint, episodes=1, seed=123, output_dir=root/"one")["records"]
            try:
                with self.assertRaisesRegex(ValueError, "prefix"):
                    evaluation.evaluate(checkpoint, episodes=3, seed=100,
                        completed_records=record, output_dir=root/"bad")
            except TypeError as error:
                self.fail(f"CPU resume validation is absent: {error}")

    def test_cpu_session_timeout_can_cover_full_evaluation(self):
        spec = build_visualization_spec(parent_spec(), learner_result(complete=True))
        self.assertEqual(spec["cpu_eval"]["episodes"], 2)
        self.assertEqual(spec["timeout_seconds"], 39600)


def test_evaluation_reports_real_inner_progress_without_changing_results(tmp_path):
    checkpoint=EvaluationResumeTests().checkpoint(tmp_path)
    progress=[]
    observed=evaluation.evaluate(checkpoint,episodes=1,seed=100,output_dir=tmp_path/'observed',on_progress=progress.append)
    plain=evaluation.evaluate(checkpoint,episodes=1,seed=100,output_dir=tmp_path/'plain')
    assert observed['records']==plain['records']
    assert progress[0]=={'status':'episode_started','seed':100,'step':0}
    assert progress[-1]['status']=='episode_complete' and progress[-1]['step']==3
