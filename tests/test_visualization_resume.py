"""CPU visualization progress and publication-only recovery survive session boundaries."""
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_evaluation_resume as evaluation_tests
from uav_network.cloud import visualize
from uav_network.experiments import evaluation


@unittest.skipIf(torch.cuda.is_available(), reason="requires a CPU-only host")
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
                               diagnostic=lambda values,step: run.log({"global_step":step, **values}),
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

    def test_cpu_failure_retains_traceback_and_source_artifact(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);tracker=self.tracker();failures=[]
            tracker.failure=lambda path,step: failures.append(Path(path))
            spec=self.spec("configs/kaggle-smoke.yaml")
            with patch("uav_network.experiments.tracking.start_tracking",return_value=tracker), \
                 patch("uav_network.cloud.remote.prepare_checkpoints",side_effect=RuntimeError("checkpoint download broke")):
                with self.assertRaisesRegex(RuntimeError,"CPU visualization failures"):
                    visualize.run_visualization(spec,root)
            path=root/"tracking/matd3/failure.json"
            assert path.is_file() and failures==[path]
            report=json.loads(path.read_text())
            assert report["error_type"]=="RuntimeError"
            assert report["context"]["checkpoint_artifact"]=="owner/main/model:v1"
            assert "checkpoint download broke" in report["traceback"]


    def test_resumed_immediate_pause_keeps_completed_prefix_visible(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); checkpoint=evaluation_tests.EvaluationResumeTests().checkpoint(root)
            spec=self.spec(root/"train/resolved_config.yaml")
            spec["completed_records"]={"matd3":[{"episode":1,"seed":10044}]}
            tracker=self.tracker(); logged=[]; tracker.run.log=logged.append
            def pause(*args, **kwargs):
                kwargs["should_pause"]=lambda:True
                return evaluation.evaluate(*args, **kwargs)
            with patch("uav_network.experiments.tracking.start_tracking",return_value=tracker), \
                 patch("uav_network.cloud.remote.prepare_checkpoints",return_value={"matd3":checkpoint}), \
                 patch.object(visualize,"evaluate",side_effect=pause):
                result=visualize.run_visualization(spec,root/"resumed")
            assert result=={}
            assert logged[0]["evaluation/episodes_completed"]==1
            assert logged[-1]["evaluation_status/status"]=="paused"
            assert logged[-1]["evaluation/episodes_completed"]==1

    def test_nonresumable_failure_preserves_completed_cpu_episodes(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); spec=self.spec("configs/kaggle-smoke.yaml")
            spec["session_resumable"]=False
            tracker=self.tracker(); failures=[]
            tracker.failure=lambda path,step:failures.append((step,json.loads(Path(path).read_text())))
            def fail_after_episode(*args, **kwargs):
                kwargs["on_episode"]({"episode":1,"return":2.,"search_rate_percent":80.,
                                        "delivery_rate_percent":70.,"success":False})
                raise RuntimeError("evaluation failed after one completed episode")
            with patch("uav_network.experiments.tracking.start_tracking",return_value=tracker), \
                 patch("uav_network.cloud.remote.prepare_checkpoints",return_value={"matd3":root/"policy.pt"}), \
                 patch.object(visualize,"evaluate",side_effect=fail_after_episode):
                with self.assertRaisesRegex(RuntimeError,"CPU visualization failures"):
                    visualize.run_visualization(spec,root)
            assert failures[0][0]==1
            assert failures[0][1]["context"]["completed_eval_episodes"]==1

    def test_secondary_reporting_error_with_closed_stdout_keeps_original_failure(self):
        from io import StringIO
        with TemporaryDirectory() as directory:
            root=Path(directory); spec=self.spec("configs/kaggle-smoke.yaml")
            closed=StringIO();closed.close()
            with patch("uav_network.experiments.tracking.start_tracking",return_value=self.tracker()), \
                 patch("uav_network.cloud.remote.prepare_checkpoints",side_effect=RuntimeError("download failure")), \
                 patch("uav_network.experiments.diagnostics.failure_report",side_effect=OSError("disk unavailable")), \
                 patch("sys.stdout",closed):
                with self.assertRaisesRegex(RuntimeError,"CPU visualization failures"):
                    visualize.run_visualization(spec,root)
            assert json.loads((root/"visualization-failures.json").read_text())["matd3"]["error_type"]=="RuntimeError"

    def test_close_failure_does_not_replace_original_cpu_error(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); spec=self.spec("configs/kaggle-smoke.yaml")
            tracker=self.tracker();tracker.failure=lambda *args:None
            tracker.close=lambda **kwargs:(_ for _ in ()).throw(OSError("finish unavailable"))
            with patch("uav_network.experiments.tracking.start_tracking",return_value=tracker), \
                 patch("uav_network.cloud.remote.prepare_checkpoints",side_effect=ValueError("original download failure")):
                with self.assertRaisesRegex(RuntimeError,"CPU visualization failures"):
                    visualize.run_visualization(spec,root)
            report=json.loads((root/"tracking/matd3/failure.json").read_text())
            assert report["error_type"]=="ValueError"
            assert "original download failure" in report["traceback"]
            assert json.loads((root/"visualization-failures.json").read_text())["matd3"]["error_type"]=="ValueError"
