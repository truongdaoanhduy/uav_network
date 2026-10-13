"""Real native checkpoint rollout keeps sampled and terminal animation frames."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from uav_network.configuration import Config
from uav_network.experiments.evaluation import evaluate
from uav_network.experiments.training import train


class CPUVisualizationTests(unittest.TestCase):
    def test_real_cpu_checkpoint_animation_keeps_terminal_frame_and_policy(self):
        with TemporaryDirectory() as directory:
            c = Config()
            c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 2, 1, 7
            c.runtime.algorithm, c.runtime.device, c.runtime.replay_device = "matd3", "cpu", "cpu"
            c.runtime.num_envs, c.runtime.episodes = 2, 2
            c.runtime.output_dir = str(Path(directory) / "train")
            c.matd3.hidden_sizes, c.matd3.replay_capacity = [32, 32], 32
            c.matd3.batch_size = c.matd3.learning_starts = 4
            trained = train(c)
            checkpoint = Path(trained["checkpoint"])
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            try:
                report = evaluate(checkpoint, episodes=1, seed=10044,
                                  output_dir=Path(directory) / "visualize", animate=True,
                                  frame_stride=3, fps=10, video_format="gif")
            except TypeError as error:
                self.fail(f"CPU animation interface is missing: {error}")
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(report["records"][0]["seed"], 10044)
            self.assertEqual(report["animation"]["sampled_steps"], [0, 3, 6, 7])
            self.assertTrue(Path(report["render_path"]).is_file())
            with Image.open(report["animation_path"]) as animation:
                self.assertEqual(animation.n_frames, 4)
                self.assertGreater(animation.width, 100)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)

    def test_invalid_animation_options_fail_before_reading_checkpoint(self):
        for options in ({"frame_stride": 0}, {"fps": 0}, {"video_format": "avi"}):
            with self.subTest(options=options):
                try:
                    with self.assertRaises(ValueError):
                        evaluate("missing-checkpoint.pt", animate=True, **options)
                except TypeError as error:
                    self.fail(f"Animation option validation is missing: {error}")


if __name__ == "__main__":
    unittest.main()
