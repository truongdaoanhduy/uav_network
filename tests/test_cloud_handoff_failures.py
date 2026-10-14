"""Completed PQL waves survive failure to schedule a later learner."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uav_network.cloud import remote, visualize
from uav_network.configuration import Config


class CloudHandoffFailureTests(unittest.TestCase):
    def test_later_deadline_failure_retains_completed_wave_and_cpu_submission(self):
        self.check_completed_wave(RuntimeError("No safe training time remains before the kernel ceiling"),
                                  fail_at_start=False)

    def test_later_process_start_failure_retains_completed_wave_and_cpu_submission(self):
        self.check_completed_wave(OSError("Learner process cannot start"), fail_at_start=True)

    def check_completed_wave(self, failure, fail_at_start):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            c = Config()
            c.pql.enabled = True
            spec = {"stage": "train", "config": "unused.yaml", "overrides": [],
                    "cpu_visualize": True, "commit": "a" * 40, "kernel_ref": "owner/parent",
                    "account": "test", "credential_dataset": "owner/private-wandb",
                    "entity": "owner", "project": "project",
                    "runs": {"masac": {"id": "first", "name": "MASAC"},
                             "matd3": {"id": "second", "name": "MATD3"}}}
            summary = {"algorithm": "masac", "episodes": 4, "global_step": 16, "updates": 7,
                       "tracking_status": "finished", "checkpoint_artifact": "owner/project/model:v1"}
            launches = []
            process = SimpleNamespace(returncode=0, poll=lambda: 0)

            def start(args, **kwargs):
                if launches:
                    raise failure
                launches.append(args)
                folder = output / "train" / "masac"
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text(json.dumps(summary))
                return process

            def submit(args, account, **kwargs):
                self.assertEqual(json.loads((output / "job-result.json").read_text()), {"masac": summary})
                return "Kernel version 1 successfully pushed. https://www.kaggle.com/code/owner/parent-cpu"

            limits = [1, 1] if fail_at_start else [1, failure]
            with (
                patch.object(remote, "Path", side_effect=lambda path: output if path == "/kaggle/working" else Path(path)),
                patch.object(remote, "load_config", return_value=c),
                patch.object(remote.torch.cuda, "is_available", return_value=True),
                patch.object(remote.torch.cuda, "device_count", return_value=2),
                patch.object(remote, "hardware_info", return_value={}),
                patch.object(remote.subprocess, "run"),
                patch.object(remote, "pql_benchmarks", return_value={}),
                patch.object(remote, "learner_time_limit", side_effect=limits),
                patch.object(remote.subprocess, "Popen", side_effect=start),
                patch.object(remote, "stop_process_tree"),
                patch.object(visualize, "verify_credential_dataset"),
                patch.object(visualize, "run_kaggle", side_effect=submit),
                patch.dict("os.environ", {"KAGGLE_API_TOKEN": "test-kaggle-token",
                                         "WANDB_API_KEY": "test-wandb-key"}),
            ):
                with self.assertRaises(type(failure)) as raised:
                    remote.run_job(spec)
            self.assertIs(raised.exception, failure)
            self.assertTrue((output / "job-result.json").is_file(),
                            "Scheduling failure dropped the completed learner summary")
            self.assertEqual(json.loads((output / "job-result.json").read_text()), {"masac": summary})
            scheduling = json.loads((output / "job-scheduling-failure.json").read_text())
            self.assertEqual(scheduling, {"algorithm": "matd3", "error_type": type(failure).__name__})
            child = json.loads((output / "cpu-visualization.json").read_text())
            self.assertEqual(child["artifacts"], {"masac": summary["checkpoint_artifact"]})
            metadata = json.loads((output / "cpu-kernel" / "kernel-metadata.json").read_text())
            self.assertFalse(metadata["enable_gpu"])
            self.assertEqual(metadata["dataset_sources"], ["owner/private-wandb"])


if __name__ == "__main__":
    unittest.main()
