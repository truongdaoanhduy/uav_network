"""Production training-only must not change the semantics of legacy preflight tests."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from uav_network.cloud import remote
from uav_network.configuration import Config


def test_preflight_disables_training_only_only_in_test_children(monkeypatch):
    monkeypatch.setenv("UAV_TRAINING_ONLY", "true")
    config = Config()
    config.pql.enabled = True
    seen = []

    class ReachedPQLPreflight(Exception):
        pass

    def inspect_subprocess(command, **kwargs):
        assert command[:3] == [remote.sys.executable, "-m", "unittest"]
        assert command[command.index("-p") + 1] in {
            "test_cuda.py", "test_cuda_scenario_v2.py",
            "test_training_audit_regressions.py", "test_pql_training.py"}
        assert os.environ["UAV_TRAINING_ONLY"] == "true"
        assert kwargs["env"]["UAV_TRAINING_ONLY"] == "false"
        seen.append(command[command.index("-p") + 1])
        if seen[-1] == "test_pql_training.py":
            raise ReachedPQLPreflight

    with TemporaryDirectory() as temp:
        directory = Path(temp)
        source_path = remote.Path
        with (patch.object(remote, "Path", side_effect=lambda name: directory if name == "/kaggle/working" else source_path(name)),
              patch.object(remote, "load_config", return_value=config),
              patch.object(remote.torch.cuda, "is_available", return_value=True),
              patch.object(remote, "hardware_info", return_value={}),
              patch.object(remote.subprocess, "run", side_effect=inspect_subprocess)):
            with pytest.raises(ReachedPQLPreflight):
                remote.run_job({"stage": "train", "config": "unused",
                                "overrides": [], "runs": {"masac": {}},
                                "training_only": True})
    assert seen == ["test_cuda.py", "test_cuda_scenario_v2.py",
                    "test_training_audit_regressions.py", "test_pql_training.py"]
