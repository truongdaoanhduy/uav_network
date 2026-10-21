"""The large production preflight must fail before allocating on a CPU host."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from uav_network.configuration import Config


class ExecutionCheckTests(unittest.TestCase):
    def test_counter_total_records_compile_break_reasons(self):
        from uav_network.cloud.execution_check import _counter_total
        self.assertEqual(_counter_total({"reason-a": 2, "reason-b": 3}), 5)
        self.assertEqual(_counter_total({}), 0)

    def test_cuda_guard_precedes_world_and_replay_allocation(self):
        from uav_network.cloud.execution_check import check_execution
        config = Config()
        config.runtime.amp_mode = "fp16"
        config.runtime.compile_models = config.runtime.fused_optimizers = True
        with TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False), patch(
                "uav_network.environment.vector.make_vector") as allocate:
            with self.assertRaisesRegex(RuntimeError, "two T4"):
                check_execution(config, Path(directory) / "evidence.json")
            allocate.assert_not_called()
