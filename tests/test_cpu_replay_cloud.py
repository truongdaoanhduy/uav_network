"""The reviewed CPU-replay probe launches through the standard redacted cloud path."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from uav_network.configuration import load_config
from uav_network.cloud import cli, remote


def test_cli_accepts_exact_cpu_replay_diagnostic_mode():
    def launch(args, credentials):
        assert args.stage == "benchmark"
        assert args.algorithm == "masac"
        assert args.diagnostic == "cpu-replay-1m"
        return {"diagnostic": args.diagnostic}
    with patch.object(cli, "load_credentials", return_value=None), \
         patch.object(cli, "launch", side_effect=launch):
        result = cli.main(["launch", "--stage", "benchmark", "--algorithm", "masac",
                           "--diagnostic", "cpu-replay-1m",
                           "--credential-dataset", "owner/private"])
    assert result == {"diagnostic": "cpu-replay-1m"}


def test_launch_marks_diagnostic_spec_and_rechecks_idle_account():
    config = load_config("configs/comparison-20k-steps1800-scenario-v3.yaml")
    account = SimpleNamespace(name="account_27", enabled=True, token="test-token")
    credentials = SimpleNamespace(accounts=[account], wandb_key="test-wandb",
                                  entity="truongdaoanhduy-uit", project="uav_search_target")
    args = SimpleNamespace(
        account="account_27", required_gpu_hours=1.5, credential_dataset="owner/private",
        credential_timeout_seconds=120, config="configs/comparison-20k-steps1800-scenario-v3.yaml",
        stage="benchmark", algorithm="masac", warm_start_artifact=[], overrides=[],
        auto_continue=False, cpu_visualize=False, compare_legacy=False,
        gpus_per_run=2, accelerator="NvidiaTeslaT4", timeout_seconds=5400,
        key_file="unused", diagnostic="cpu-replay-1m")
    captured = {}
    def kernel_script(spec):
        captured.update(spec)
        return "pass\n"
    def kaggle(args, account, timeout=45):
        if args[:2] == ["kernels", "list"]:
            return json.dumps([{"ref": "owner/old"}])
        assert args[:2] == ["kernels", "push"]
        return "https://www.kaggle.com/code/owner/new-probe"
    with TemporaryDirectory() as folder, \
         patch.object(cli, "ROOT", Path(folder)), \
         patch.object(cli.subprocess, "check_output", side_effect=["", "a" * 40]), \
         patch.object(cli, "account_quotas", return_value=[{"name":"account_27","ok":True,"remaining_hours":26.23}]), \
         patch.object(cli, "select_account", return_value=account), \
         patch.object(cli, "resolve_username", return_value="owner"), \
         patch.object(cli, "verify_credential_dataset"), \
         patch.object(cli, "load_config", return_value=config), \
         patch.object(cli, "kernel_script", side_effect=kernel_script), \
         patch.object(cli, "kernel_metadata", return_value={}), \
         patch.object(cli, "run_kaggle", side_effect=kaggle), \
         patch.object(cli, "kernel_status", return_value="complete"), \
         patch.object(cli, "pushed_kernel_ref", return_value="owner/new-probe"):
        result = cli.launch(args, credentials)
    assert captured["diagnostic"] == "cpu-replay-1m"
    assert captured["runs"].keys() == {"masac"}
    assert captured["config"] == "configs/comparison-20k-steps1800-scenario-v3.yaml"
    assert result["kernel_ref"] == "owner/new-probe"


def test_remote_cpu_replay_diagnostic_runs_reviewed_module_and_preserves_result():
    spec = {"commit": "b" * 40, "diagnostic": "cpu-replay-1m",
            "runs": {"masac": {"id": "run123", "name": "DIAG"}}}
    with TemporaryDirectory() as folder:
        root = Path(folder)
        def child(args, **kwargs):
            assert args[:3] == [remote.sys.executable, "-m", "uav_network.experiments.cpu_replay_probe"]
            assert kwargs["env"]["WANDB_RUN_ID"] == "run123"
            assert kwargs["env"]["PROBE_COMMIT"] == spec["commit"]
            out = Path(args[args.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "cpu-replay-result.json").write_text(json.dumps({"status":"passed","schema":1}))
        with patch.object(remote.subprocess, "run", side_effect=child):
            result = remote.cpu_replay_diagnostic(spec, root)
        assert result["status"] == "passed"
        assert json.loads((root / "job-result.json").read_text())["status"] == "passed"


def test_cli_accepts_exact_hybrid_replay_diagnostic_mode():
    def launch(args, credentials):
        assert args.stage == "benchmark"
        assert args.algorithm == "masac"
        assert args.diagnostic == "hybrid-replay-1m"
        return {"diagnostic": args.diagnostic}
    with patch.object(cli, "load_credentials", return_value=None), \
         patch.object(cli, "launch", side_effect=launch):
        result = cli.main(["launch", "--stage", "benchmark", "--algorithm", "masac",
                           "--diagnostic", "hybrid-replay-1m",
                           "--credential-dataset", "owner/private"])
    assert result == {"diagnostic": "hybrid-replay-1m"}


def test_remote_hybrid_replay_diagnostic_selects_hybrid_layout():
    spec = {"commit": "c" * 40, "diagnostic": "hybrid-replay-1m",
            "runs": {"masac": {"id": "run456", "name": "DIAG-HYBRID"}}}
    with TemporaryDirectory() as folder:
        root = Path(folder)
        def child(args, **kwargs):
            assert args[:3] == [remote.sys.executable, "-m", "uav_network.experiments.cpu_replay_probe"]
            assert args[args.index("--layout") + 1] == "hybrid"
            out = Path(args[args.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "cpu-replay-result.json").write_text(json.dumps({"status":"passed","schema":1,"layout":"hybrid"}))
        with patch.object(remote.subprocess, "run", side_effect=child):
            result = remote.cpu_replay_diagnostic(spec, root)
        assert result["status"] == "passed"
        assert result["layout"] == "hybrid"
