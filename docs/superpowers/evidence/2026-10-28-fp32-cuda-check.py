"""Run canonical FP32 execution checks in isolated CUDA child processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


def check_case(algorithm, mode, output):
    import torch
    from uav_network.configuration import load_config
    from uav_network.cloud.acceleration import apply_measured_acceleration_policy
    from uav_network.cloud.execution_check import check_execution
    from uav_network.algorithms import checkpoints
    from uav_network.experiments import pql_system
    from uav_network.environment.observations import observation_dim

    canonical = Path("configs/comparison-20k-hybrid75-1kb.yaml").resolve()
    initial = [f"runtime.algorithm={algorithm}", f"pql.enabled={str(mode == 'pql').lower()}"]
    config = load_config(canonical, initial)
    overrides = apply_measured_acceleration_policy(
        canonical, "train", algorithm, config, initial, canonical)
    config = load_config(canonical, overrides)
    assert config.runtime.amp_mode == "off"
    assert not any(x.startswith("runtime.amp_mode=") for x in overrides)
    assert torch.get_default_dtype() == torch.float32
    assert not torch.is_autocast_enabled("cuda")
    assert torch.cuda.device_count() == 2
    owned = []
    original = checkpoints.make_learner

    def capture(*args, **kwargs):
        learner = original(*args, **kwargs)
        owned.append(learner)
        return learner

    report = {}
    try:
        with patch.object(checkpoints, "make_learner", capture), patch.object(
                pql_system, "make_learner", capture):
            report = check_execution(config, output)
        dtypes = []
        for learner in owned:
            parameters = list(learner.parameters())
            gradients = [p.grad for p in parameters if p.grad is not None]
            assert parameters and gradients
            assert all(p.dtype == torch.float32 for p in parameters + gradients)
            assert all(bool(torch.isfinite(p).all()) for p in parameters + gradients)
            assert not learner.precision.enabled and learner.precision.scalers == {}
            # A separate encoder spot check does not instrument the timed preflight.
            with torch.no_grad(), torch.cuda.device(learner.device):
                value = learner.actors[0].encoder(torch.zeros(
                    1, observation_dim(config), device=learner.device))
            assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
            dtypes.append(dict(parameter_dtypes=sorted({str(p.dtype) for p in parameters}),
                gradient_dtypes=sorted({str(g.dtype) for g in gradients}),
                gradient_tensor_count=len(gradients), encoder_spot_check_dtype=str(value.dtype),
                scalers_empty=True))
        assert owned and report["amp_autocast_fp16"] == 0
        report.update(dtype_checks=dtypes, default_dtype=str(torch.get_default_dtype()),
                      no_amp_override=True, dtype_checks_passed=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        Path(output).write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    if len(sys.argv) > 1:
        check_case(sys.argv[1], sys.argv[2], sys.argv[3])
        return
    import wandb
    spec = json.loads(Path("/kaggle/working/diagnostic-spec.json").read_text())
    folder = Path("/kaggle/working/fp32-check")
    folder.mkdir(exist_ok=True)
    run = wandb.init(entity=spec["entity"], project=spec["project"],
        id=spec["wandb_run_id"], resume="never", mode="online",
        name="CHECK-FP32-CANONICAL-" + spec["wandb_run_id"],
        job_type="fp32-diagnostic", tags=["diagnostic", "not-production", "amp-off"],
        config={"stage": "diagnostic", **spec})
    reports = []
    failed = True
    try:
        for algorithm in ("masac", "matd3"):
            for mode in ("base", "pql"):
                output = folder / (algorithm + "-" + mode + ".json")
                status = subprocess.run([sys.executable, __file__, algorithm, mode, str(output)],
                    check=False, timeout=1500)
                report = json.loads(output.read_text()) if output.exists() else {
                    "status": "failed", "error_type": "ChildProducedNoReport"}
                reports.append(report)
                run.save(str(output), base_path=str(folder), policy="now")
                run.log({"diagnostic/cases_completed": len(reports),
                         "diagnostic/last_case_passed": int(report.get("dtype_checks_passed", False))})
                if status.returncode or not report.get("dtype_checks_passed"):
                    raise RuntimeError("FP32 CUDA case failed: " + algorithm + "-" + mode)
        failed = False
    finally:
        summary = {"source_commit": spec["commit"], "kernel_ref": spec["kernel_ref"],
            "status": "failed" if failed else "passed", "production_training": False,
            "cases": reports}
        output = folder / "result.json"
        output.write_text(json.dumps(summary, indent=2, allow_nan=False))
        run.save(str(output), base_path=str(folder), policy="now")
        run.summary.update({"diagnostic/all_fp32_cases_passed": not failed,
                            "diagnostic/case_count": len(reports)})
        run.finish(exit_code=int(failed))


if __name__ == "__main__":
    main()
