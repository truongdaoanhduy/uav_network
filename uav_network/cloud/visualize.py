"""A private CPU-only job evaluates uploaded immutable policy checkpoints."""
import argparse
import json
import os
from pathlib import Path
import re

import torch

from .client import pushed_kernel_ref, run_kaggle, verify_credential_dataset
from .credentials import Account, Credentials
from .kernel import kernel_metadata, kernel_script
from ..experiments.evaluation import evaluate
from ..experiments.hardware import hardware_info


def build_visualization_spec(parent, results):
    artifacts, statuses = {}, {}
    for algorithm, result in results.items():
        artifact = result.get("checkpoint_artifact")
        prefix = re.escape(parent["entity"] + "/" + parent["project"] + "/")
        if (algorithm not in {"masac", "matd3"} or algorithm not in parent["runs"]
                or result.get("tracking_status") != "finished" or not isinstance(artifact, str)
                or not re.fullmatch(prefix + r"[\w.-]+:v[0-9]+", artifact)):
            raise ValueError("CPU visualization requires each successful learner's uploaded immutable checkpoint version")
        artifacts[algorithm] = artifact
        statuses[algorithm] = {key: result.get(key) for key in
                               ("episodes", "global_step", "episode_budget_complete", "stop_reason")}
    if not artifacts:
        raise ValueError("CPU visualization needs at least one completed learner checkpoint")
    return {"stage": "visualize", "commit": parent["commit"], "entity": parent["entity"],
            "project": parent["project"],
            "credential_dataset": parent.get("cpu_credential_dataset", parent["credential_dataset"]),
            "kernel_ref": parent["kernel_ref"] + "-cpu-visualize", "account": parent["account"],
            "parent_kernel_ref": parent["kernel_ref"], "artifacts": artifacts,
            "training_status": statuses, "cpu_eval": parent.get("cpu_eval", {
                "episodes": 1, "seed": 10044, "frame_stride": 5, "fps": 10}),
            "timeout_seconds": 3600}


def submit_visualization(parent, results, output, account=None, credentials=None):
    spec = build_visualization_spec(parent, results)
    if account is None:
        token = os.environ.get("KAGGLE_API_TOKEN")
        if not token:
            raise RuntimeError("Autonomous CPU launch requires its private Kaggle credential")
        account = Account(parent["account"], token)
    if credentials is None:
        credentials = Credentials([account], os.environ["WANDB_API_KEY"], parent["entity"], parent["project"])
    username = parent["kernel_ref"].split("/", 1)[0]
    verify_credential_dataset(account, credentials, spec["credential_dataset"], username, only_selected=True)
    folder = Path(output) / "cpu-kernel"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.py").write_text(kernel_script(spec))
    (folder / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(spec), indent=2))
    response = run_kaggle(["kernels", "push", "-p", str(folder), "--timeout", "3600"], account, timeout=50)
    ref = pushed_kernel_ref(response)
    manifest = {**spec, "kernel_ref": ref, "kernel_url": "https://www.kaggle.com/code/" + ref}
    (Path(output) / "cpu-visualization.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"cpu_visualization_url": manifest["kernel_url"]}), flush=True)
    return manifest


def run_visualization(spec, output=Path("/kaggle/working")):
    if torch.cuda.is_available():
        raise RuntimeError("Visualization must run in a separate CPU-only session")
    from .remote import prepare_checkpoints
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "hardware.json").write_text(json.dumps(hardware_info(), indent=2))
    settings = spec["cpu_eval"]
    results, failures = {}, {}
    for algorithm, artifact in spec["artifacts"].items():
        try:
            checkpoint = prepare_checkpoints({"warm_start_artifacts": {algorithm: artifact}}, output)[algorithm]
            results[algorithm] = evaluate(checkpoint, episodes=settings["episodes"], seed=settings["seed"],
                                          output_dir=output / "visualize" / algorithm, animate=True,
                                          frame_stride=settings["frame_stride"], fps=settings["fps"])
            results[algorithm]["checkpoint_artifact"] = spec["artifacts"][algorithm]
            results[algorithm]["training_status"] = spec["training_status"][algorithm]
        except Exception as error:
            failures[algorithm] = {"error_type": type(error).__name__}
        (output / "visualization-result.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    if failures:
        (output / "visualization-failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError("CPU visualization failures: " + ", ".join(failures))
    print(json.dumps({a: {k: v[k] for k in ("device", "render_path", "animation_path")} for a, v in results.items()}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    run_visualization(json.loads(Path(args.spec).read_text()))


if __name__ == "__main__":
    main()
