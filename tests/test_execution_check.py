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

    def test_each_requested_acceleration_runs_execution_preflight(self):
        from types import SimpleNamespace
        from uav_network.cloud import remote
        for flag in (None, "amp_mode", "compile_models", "fused_optimizers"):
            c = Config()
            if flag:
                setattr(c.runtime, flag, "fp16" if flag == "amp_mode" else True)
            commands = []
            reports = {"masac": {"results": [{"status": "passed"}]}}
            spec = {"config": "configs/example.yaml", "overrides": ["runtime.num_envs=2"],
                    "runs": {"masac": {"id": "example"}}, "stage": "benchmark"}
            with self.subTest(flag=flag), TemporaryDirectory() as directory, \
                 patch.object(remote.torch.cuda, "is_available", return_value=True), \
                 patch.object(remote, "hardware_info", return_value={}), \
                 patch.object(remote, "load_config", return_value=c), \
                 patch.object(remote, "prepare_checkpoints", return_value={}), \
                 patch.object(remote, "base_benchmarks", return_value=reports), \
                 patch.object(remote, "Path", return_value=Path(directory)), \
                 patch.object(remote.subprocess, "run", side_effect=lambda args, **kwargs:
                              commands.append(args) or SimpleNamespace(returncode=0)):
                self.assertEqual(remote.run_job(spec), reports)
                checks = [args for args in commands if "uav_network.cloud.execution_check" in args]
                self.assertEqual(len(checks), int(flag is not None))
                if checks:
                    self.assertEqual(checks[0][checks[0].index("--algorithm")+1], "masac")
                    self.assertIn("runtime.num_envs=2", checks[0])

    def test_execution_preflight_builds_configured_hybrid_replay(self):
        # Execute the actual allocation expression with CUDA storage mapped to
        # CPU; the production two-T4 guard is tested separately.
        import ast
        import inspect
        import torch
        from types import SimpleNamespace
        from uav_network.cloud import execution_check
        from uav_network.environment.observations import observation_dim, state_dim
        from uav_network.experiments.pql_replay import MeasuredReplay, make_replay
        tree = ast.parse(inspect.getsource(execution_check.check_execution))
        expression = next(node.value for node in ast.walk(tree)
                          if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "replay" for t in node.targets))
        class CPUStorage(ast.NodeTransformer):
            def visit_Constant(self, node):
                return ast.copy_location(ast.Constant("cpu"), node) if node.value == "cuda:1" else node
        expression = CPUStorage().visit(expression)
        c = Config(); c.masac.replay_capacity = 64
        c.runtime.replay_device = "cpu"; c.runtime.replay_cpu_fraction = .75
        replay = eval(compile(ast.fix_missing_locations(ast.Expression(expression)),
                              execution_check.__file__, "eval"),
                      dict(cfg=c, learner=SimpleNamespace(device=torch.device("cpu")),
                           observation_dim=observation_dim, state_dim=state_dim,
                           MeasuredReplay=MeasuredReplay, make_replay=make_replay))
        saved = replay.state_dict()
        self.assertEqual(saved.get("layout"), "hybrid")
        self.assertEqual(saved["cpu_capacity"], 48)
        self.assertEqual(replay.hot_capacity, 16)

    def test_preflight_accepts_one_supported_cuda_board_and_uses_its_device(self):
        import torch
        from types import SimpleNamespace
        from uav_network.cloud.execution_check import check_execution
        # Load consumers before patching constructors; imports must not retain
        # these test doubles in the real trainer or worker modules.
        from uav_network.experiments import training, pql_system
        class AllocationReached(RuntimeError):
            pass
        config = Config(); config.runtime.device = "cuda"
        vector = SimpleNamespace(reset=lambda **kwargs: ({}, {}), close=lambda: None,
                                 device=torch.device("cuda:0"))
        def learner_allocation(config, observation_dim, state_dim, device):
            self.assertEqual(str(device), "cuda:0")
            raise AllocationReached("selected single CUDA device")
        with TemporaryDirectory() as directory, \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(torch.cuda, "get_device_name", return_value="Tesla P100"), \
             patch.object(torch.cuda, "init"), \
             patch.object(torch.cuda, "reset_peak_memory_stats"), \
             patch("uav_network.environment.vector.make_vector", return_value=vector), \
             patch("uav_network.algorithms.checkpoints.make_learner", side_effect=learner_allocation):
            with self.assertRaisesRegex(AllocationReached, "selected single CUDA device"):
                check_execution(config, Path(directory) / "evidence.json")

    def test_cuda_guard_precedes_world_and_replay_allocation(self):
        from uav_network.cloud.execution_check import check_execution
        config = Config()
        config.runtime.amp_mode = "fp16"
        config.runtime.compile_models = config.runtime.fused_optimizers = True
        with TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False), patch(
                "uav_network.environment.vector.make_vector") as allocate:
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                check_execution(config, Path(directory) / "evidence.json")
            allocate.assert_not_called()

    def test_preflight_initializes_cuda_before_resetting_peak_memory_stats(self):
        import torch
        from uav_network.cloud.execution_check import check_execution
        class ResetReached(RuntimeError):
            pass
        state = {"initialized": False}
        def initialize():
            state["initialized"] = True
        def reset(device):
            self.assertTrue(state["initialized"], "CUDA context must be initialized before resetting peak stats")
            raise ResetReached(str(device))
        config = Config(); config.runtime.device = "cuda"
        with TemporaryDirectory() as directory, \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(torch.cuda, "init", side_effect=initialize), \
             patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=reset):
            with self.assertRaisesRegex(ResetReached, "0"):
                check_execution(config, Path(directory) / "evidence.json")
