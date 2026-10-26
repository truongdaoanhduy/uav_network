"""Actual PQL policy artifacts run the saved Warp backend on CPU and compare results."""
import importlib.util
import json
import pytest
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uav_network.configuration import Config
from uav_network.experiments.training import train
from uav_network.cloud.visualize import run_visualization


@unittest.skipUnless(importlib.util.find_spec("warp"), "Warp optional dependency unavailable")
class CPUHandoffComparisonTests(unittest.TestCase):
    def test_real_pql_policies_keep_backend_and_generate_cpu_paired_metrics(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            trained = {}
            for algorithm in ("masac", "matd3"):
                c = Config()
                c.world.max_steps = 8
                c.runtime.algorithm, c.runtime.environment_backend = algorithm, "warp_packet"
                c.runtime.num_envs = c.runtime.episodes = 4
                c.runtime.device = c.runtime.replay_device = "cpu"
                c.runtime.output_dir = str(root / algorithm)
                c.pql.enabled, c.pql.mode = True, "synchronous"
                c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
                c.pql.warmup_collection_steps, c.pql.critic_sample_ratio = 2, 2
                for settings in (c.masac, c.matd3):
                    settings.hidden_sizes, settings.batch_size, settings.replay_capacity = [16, 16], 8, 256
                trained[algorithm] = train(c)

            def artifact(ref, **kwargs):
                algorithm = ref.split(":")[0].split("/")[-1]
                def download(root):
                    destination = Path(root) / "checkpoint.pt"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(trained[algorithm]["checkpoint"], destination)
                    return str(destination)
                return SimpleNamespace(get_entry=lambda name: SimpleNamespace(download=download))

            spec = {"entity": "owner", "project": "project", "kernel_ref": "owner/cpu",
                    "artifacts": {a: f"owner/project/{a}:v1" for a in trained},
                    "runs": {a: {"id": a + "1", "name": a.upper() + "-CPU"} for a in trained},
                    "cpu_eval": {"episodes": 2, "seed": 2000000044, "frame_stride": 5, "fps": 10},
                    "training_status": {a: {"episode_budget_complete": True} for a in trained}}
            api = SimpleNamespace(artifact=artifact)
            run = SimpleNamespace(summary={}, log=lambda *a,**k:None, save=lambda *a,**k:None,
                                  use_artifact=lambda *a,**k:None)
            tracker = SimpleNamespace(run=run, identity=lambda:{}, close=lambda failed=False:None)
            with patch.dict(sys.modules, {"wandb": SimpleNamespace(Api=lambda: api)}), \
                 patch("uav_network.experiments.tracking.start_tracking", return_value=tracker), \
                 patch("uav_network.cloud.visualize.publish_visualization_media",
                       return_value={"wandb": {"status": "ok"},
                                     "huggingface": {"status": "ok"}}), \
                 patch("uav_network.experiments.checkpoint_store.storage_credentials", side_effect=lambda *a, **k:
                     pytest.fail("Render-only test attempted live HF publication")), \
                 patch("torch.cuda.is_available", return_value=False):
                results = run_visualization(spec, root / "cpu")
            for algorithm, result in results.items():
                self.assertEqual(result["device"], "cpu")
                self.assertEqual(result["environment_backend"], "warp_packet")
                self.assertEqual(result["animation"]["sampled_steps"], [0, 5, 8])
                self.assertTrue(Path(result["animation_path"]).is_file())
                self.assertFalse(result["protocol"]["training_seed_overlap"])
            compared = json.loads((root / "cpu/visualize/comparison.json").read_text())
            self.assertEqual(compared["paired_episodes"], 2)
            self.assertEqual(compared["equal_environment_interaction_budget"], True)
            self.assertTrue((root / "cpu/visualize/comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
