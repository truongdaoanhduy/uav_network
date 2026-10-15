"""A private CPU-only job evaluates uploaded immutable policy checkpoints."""
import argparse
import json
import os
from pathlib import Path
import re
from uuid import uuid4

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
    from .continuation import kernel_ref_for_segment
    cpu_ref = kernel_ref_for_segment(parent.get("root_kernel_ref", parent["kernel_ref"]), 0, cpu=True)
    runs = {a: {"id": uuid4().hex[:8],
                "name": f"{a.upper()}-CPU-visualize-{parent['runs'][a]['id']}-{uuid4().hex[:6]}"}
            for a in artifacts}
    return {"stage": "visualize", "runs": runs,
            "config": parent.get("config"), "overrides": parent.get("overrides", []),
            **{k: parent[k] for k in ("storage_credential_dataset", "checkpoint_repo") if k in parent}, "commit": parent["commit"], "entity": parent["entity"],
            "project": parent["project"],
            "credential_dataset": parent.get("cpu_credential_dataset", parent["credential_dataset"]),
            "kernel_ref": cpu_ref, "account": parent["account"],
            "parent_kernel_ref": parent["kernel_ref"],
            "root_kernel_ref": parent.get("root_kernel_ref", parent["kernel_ref"]),
            "continuation_index": parent.get("continuation_index", 0),
            "requested_episodes": parent.get("requested_episodes"),
            "training_mode": parent.get("training_mode", "unknown"), "artifacts": artifacts,
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
        tracker = None
        failed = True
        try:
            from ..configuration import load_config
            from ..experiments.tracking import start_tracking
            config = load_config(spec.get("config"), spec.get("overrides", []))
            config.runtime.algorithm = algorithm
            config.runtime.device = config.runtime.replay_device = "cpu"
            config.runtime.num_envs = 1
            config.runtime.episodes = settings["episodes"]
            config.pql.enabled = False
            config.tracking.enabled = True
            config.tracking.entity, config.tracking.project = spec["entity"], spec["project"]
            identity = spec.get("runs", {}).get(algorithm, {"id": uuid4().hex[:8],
                "name": algorithm.upper() + "-CPU-visualize"})
            config.tracking.run_name = identity["name"]
            tracking_output = output / "tracking" / algorithm
            tracking_output.mkdir(parents=True, exist_ok=True)
            previous_id = os.environ.get("WANDB_RUN_ID")
            os.environ["WANDB_RUN_ID"] = identity["id"]
            try:
                tracker = start_tracking(config, tracking_output)
            finally:
                if previous_id is None:
                    os.environ.pop("WANDB_RUN_ID", None)
                else:
                    os.environ["WANDB_RUN_ID"] = previous_id
            tracker.run.summary.update({"stage": "visualize", "execution_device": "cpu",
                "parent_kernel_ref": spec.get("parent_kernel_ref"),
                "checkpoint_artifact": artifact,
                "visualization_output_url": "https://www.kaggle.com/code/" + spec["kernel_ref"] + "/output"})
            tracker.run.log({"global_step": 0, "evaluation/episodes_completed": 0})
            def record_episode(record):
                tracker.run.log({"global_step": record["episode"],
                    "evaluation/episodes_completed": record["episode"],
                    "evaluation/return": record["return"],
                    "evaluation/search_rate_percent": record["search_rate_percent"],
                    "evaluation/delivery_rate_percent": record["delivery_rate_percent"],
                    "evaluation/success": int(record["success"])})
            inputs = {"warm_start_artifacts": {algorithm: artifact}}
            checkpoint = prepare_checkpoints(inputs, output)[algorithm]
            resolved = inputs.get("resolved_warm_start_artifacts", {}).get(algorithm, artifact)
            tracker.run.use_artifact(resolved, type="model")
            results[algorithm] = evaluate(checkpoint, episodes=settings["episodes"], seed=settings["seed"],
                                          output_dir=output / "visualize" / algorithm, animate=True,
                                          frame_stride=settings["frame_stride"], fps=settings["fps"],
                                          on_episode=record_episode)
            results[algorithm]["checkpoint_artifact"] = artifact
            results[algorithm]["training_status"] = spec["training_status"][algorithm]
            results[algorithm].update(tracker.identity())
            tracker.run.summary.update({"eval/" + k: results[algorithm][k] for k in
                ("mean_return", "success_rate", "mean_search_rate_percent", "mean_delivery_rate_percent")})
            tracker.run.summary["evaluation/episodes_completed"] = results[algorithm]["episodes"]
            tracker.run.save(str(output / "visualize" / algorithm / "metrics.json"),
                             base_path=str(output), policy="now")
            failed = False
        except Exception as error:
            failures[algorithm] = {"error_type": type(error).__name__}
        finally:
            if tracker is not None:
                tracker.close(failed=failed)
        (output / "visualization-result.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    if "masac" in results and "matd3" in results:
        from ..experiments.comparison import compare_evaluations, save_comparison
        try:
            save_comparison(compare_evaluations(results["masac"], results["matd3"]), output / "visualize")
        except Exception as error:
            failures["comparison"] = {"error_type": type(error).__name__}
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
