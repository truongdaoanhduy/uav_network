"""CPU visualization progress and publication-only recovery survive session boundaries."""
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_evaluation_resume as evaluation_tests
from uav_network.cloud import visualize
from uav_network.experiments import evaluation


class VisualizationResumeTests(unittest.TestCase):
    def spec(self, config):
        return {"entity": "owner", "project": "main", "account": "selected",
                "config": str(config), "overrides": [], "commit": "a"*40,
                "kernel_ref": "owner/cpu", "parent_kernel_ref": "owner/gpu", "training_mode": "base",
                "runs": {"matd3": {"id": "cpu1", "name": "CPU"}},
                "artifacts": {"matd3": "owner/main/model:v1"}, "training_status": {"matd3": {"episodes": 1}},
                "session_resumable": True, "timeout_seconds": 39600,
                "cpu_eval": {"episodes": 3, "seed": 10044, "frame_stride": 2, "fps": 5}}

    def tracker(self):
        run = SimpleNamespace(summary={}, log=lambda *a, **k: None, use_artifact=lambda *a, **k: None,
                              save=lambda *a, **k: None)
        return SimpleNamespace(run=run, close=lambda **kw: None,
                               identity=lambda: {"wandb_run_id": "cpu1", "wandb_url": "https://wandb.ai/owner/main/runs/cpu1"})

    def test_cpu_job_writes_portable_prefix_and_next_session_finishes_without_repeating_it(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = evaluation_tests.EvaluationResumeTests().checkpoint(root)
            spec = self.spec(root/"train/resolved_config.yaml")
            actual = evaluation.evaluate
            seen = []
            def pause_after_one(*args, **kwargs):
                self.assertIn("on_checkpoint", kwargs, "CPU runner does not persist completed evaluation records")
                kwargs["should_pause"] = lambda: bool(seen)
                original = kwargs["on_episode"]
                kwargs["on_episode"] = lambda r: (seen.append(r["seed"]), original(r))
                return actual(*args, **kwargs)
            with patch("uav_network.experiments.tracking.start_tracking", side_effect=lambda *a: self.tracker()), \
                 patch("uav_network.cloud.remote.prepare_checkpoints", return_value={"matd3": checkpoint}), \
                 patch.object(visualize, "evaluate", side_effect=pause_after_one):
                result = visualize.run_visualization(spec, root/"first")
            self.assertEqual(result, {})
            progress = json.loads((root/"first/evaluation-progress.json").read_text())
            self.assertEqual(progress["records"]["matd3"][0]["seed"], 10044)
            self.assertEqual(progress["artifacts"], spec["artifacts"])
            spec["completed_records"] = progress["records"]
            resumed_seeds = []
            def resume(*args, **kwargs):
                original = kwargs["on_episode"]
                kwargs["on_episode"] = lambda r: (resumed_seeds.append(r["seed"]), original(r))
                return actual(*args, **kwargs)
            with patch("uav_network.experiments.tracking.start_tracking", side_effect=lambda *a: self.tracker()), \
                 patch("uav_network.cloud.remote.prepare_checkpoints", return_value={"matd3": checkpoint}), \
                 patch.object(visualize, "evaluate", side_effect=resume), \
                 patch.object(visualize, "publish_visualization_media", return_value={}):
                result = visualize.run_visualization(spec, root/"second")
            self.assertEqual(resumed_seeds, [10045, 10046])
            self.assertEqual(len(result["matd3"]["records"]), 3)

    def test_publication_only_attempt_loads_existing_assets_without_evaluating_again(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"assets"; source.mkdir()
            (source/"final_episode.png").write_bytes(b"image")
            (source/"final_episode.mp4").write_bytes(b"video")
            (source/"metrics.json").write_text("{}")
            result = {"algorithm": "matd3", "episodes": 3, "device": "cpu", "records": [],
                      "render_path": str(source/"final_episode.png"), "animation_path": str(source/"final_episode.mp4"),
                      "mean_return": 0, "success_rate": 0, "mean_search_rate_percent": 0,
                      "mean_delivery_rate_percent": 0}
            spec = self.spec("configs/kaggle-smoke.yaml")
            spec["publication_source_ref"] = "owner/old-cpu"
            result["checkpoint_artifact"] = spec["artifacts"]["matd3"]
            (source/"metrics.json").write_text(json.dumps(result))
            def download(args, account, **kw):
                destination = Path(args[args.index("-p")+1])/"visualize/matd3"
                shutil.copytree(source, destination)
                return ""
            try:
                with patch("uav_network.experiments.tracking.start_tracking", side_effect=lambda *a: self.tracker()), \
                     patch.dict(os.environ, {"KAGGLE_API_TOKEN": "test-token"}), \
                     patch.object(visualize, "run_kaggle", side_effect=download), \
                     patch.object(visualize, "evaluate", side_effect=AssertionError("repeated evaluation")), \
                     patch.object(visualize, "publish_visualization_media", return_value={"wandb": {"status": "ok"}}):
                    got = visualize.run_visualization(spec, root/"retry")
            except AttributeError as error:
                self.fail(f"Publication-only recovery is missing: {error}")
            self.assertEqual(got["matd3"]["publication"]["wandb"]["status"], "ok")
