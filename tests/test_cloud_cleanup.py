"""SIGTERM closes real native simulator workers and marks tracking as failed."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest

from uav_network.cloud.remote import stop_process_tree
from uav_network.experiments.lifecycle import graceful_termination


class CleanupTests(unittest.TestCase):
    def test_termination_handler_runs_finally_and_restores_previous_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        cleanup = []
        with self.assertRaises(KeyboardInterrupt):
            with graceful_termination():
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                finally:
                    cleanup.append(True)
        self.assertEqual(cleanup, [True])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_terminating_real_training_closes_native_workers_and_tracker(self):
        script = '''import json, multiprocessing, sys
from pathlib import Path
from uav_network.configuration import Config
import uav_network.experiments.training as training
root=Path(sys.argv[1])
class Tracker:
    def progress(self, *args, **kwargs):
        (root/'workers.json').write_text(json.dumps([p.pid for p in multiprocessing.active_children()]))
    def episode(self, record): pass
    def complete(self, result): return {}
    def close(self, failed=False): (root/'closed.json').write_text(json.dumps({'failed':failed}))
training.start_tracking=lambda config, output: Tracker()
c=Config(); c.runtime.device='cpu'; c.runtime.num_envs=2; c.runtime.episodes=100000
c.runtime.output_dir=str(root/'train'); c.world.num_targets=1; c.world.num_obstacles=0
c.masac.hidden_sizes=[8,8]; c.masac.batch_size=8; c.masac.replay_capacity=32
training.train(c)
'''
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "process.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-c", script, directory],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    deadline = time.monotonic() + 15
                    while not (root / "workers.json").exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertTrue((root / "workers.json").is_file())
                    workers = json.loads((root / "workers.json").read_text())
                    self.assertEqual(len(workers), 2)
                    stop_process_tree(process, grace_seconds=10)
                    self.assertTrue(json.loads((root / "closed.json").read_text())["failed"])
                    self.assertIsNotNone(process.poll())
                    for pid in workers:
                        self.assertFalse(Path(f"/proc/{pid}").exists())
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


if __name__ == "__main__":
    unittest.main()
