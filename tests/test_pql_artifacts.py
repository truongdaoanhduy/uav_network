"""Final cloud artifacts retain a real resumable state alongside light policies."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from uav_network.configuration import Config
from uav_network.cloud.remote import prepare_checkpoints
from uav_network.experiments.tracking import OnlineTracker


class PQLArtifactTests(unittest.TestCase):
    def test_pql_downloads_full_resume_state_and_baseline_downloads_light_policy(self):
        config = Config()
        config.pql.enabled = True
        entries = []
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            def entry(name):
                entries.append(name)
                def download(root):
                    path = Path(root) / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {"algorithm": "masac", "progress": {"episodes": 1}}
                    if name == "resume.pt":
                        payload.update(version=3, training={"optimizers": {"critic": {}}, "collections": 32})
                    else:
                        payload["version"] = 2
                    torch.save(payload, path)
                    return str(path)
                return SimpleNamespace(download=download)
            api = SimpleNamespace(artifact=lambda *args, **kwargs: SimpleNamespace(get_entry=entry))
            spec = {"warm_start_artifacts": {"masac": "owner/project/model:final"}}
            with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=lambda: api)}):
                full = prepare_checkpoints(spec, folder / "pql", config=config)
                config.pql.enabled = False
                light = prepare_checkpoints(spec, folder / "baseline", config=config)
            self.assertEqual(entries, ["resume.pt", "checkpoint.pt"])
            self.assertIn("training", torch.load(full["masac"], weights_only=True))
            self.assertNotIn("training", torch.load(light["masac"], weights_only=True))

    def test_only_final_pql_artifact_includes_resume_and_legacy_artifacts_stay_small(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            policy, resume = output / "final.pt", output / "resume_latest.pt"
            torch.save({"version": 2}, policy)
            torch.save({"version": 3, "training": {"collections": 4}}, resume)
            (output / "episodes.jsonl").touch()
            artifacts = []
            def artifact(*args, **kwargs):
                files = {}
                def add_file(path, name):
                    self.assertTrue(Path(path).is_file())
                    files[name] = torch.load(path, weights_only=True)
                value = SimpleNamespace(add_file=add_file, files=files)
                artifacts.append(value)
                return value
            tracker = OnlineTracker.__new__(OnlineTracker)
            tracker.config, tracker.output = Config(), output
            tracker.config.pql.enabled = True
            tracker.wandb = SimpleNamespace(Artifact=artifact)
            tracker.run = SimpleNamespace(id="test", save=lambda *args, **kwargs: None,
                log_artifact=lambda *args, **kwargs: SimpleNamespace(wait=lambda **kwargs: None, qualified_name="model:final"))
            state = {"episodes": 4, "global_step": 16, "resume_checkpoint": str(resume)}
            tracker.checkpoint(policy, state, final=True)
            self.assertEqual(artifacts[-1].files["resume.pt"]["version"], 3)
            tracker.checkpoint(policy, state, final=False)
            self.assertEqual(set(artifacts[-1].files), {"checkpoint.pt"})
            tracker.config.pql.enabled = False
            tracker.checkpoint(policy, state, final=True)
            self.assertEqual(set(artifacts[-1].files), {"checkpoint.pt"})


if __name__ == "__main__":
    unittest.main()
