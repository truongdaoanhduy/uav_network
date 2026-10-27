"""Generate a public-code-only bootstrap for an explicitly private GPU kernel."""
import json


def kernel_metadata(spec):
    datasets = [spec["credential_dataset"]]
    if spec.get("storage_credential_dataset"):
        datasets.append(spec["storage_credential_dataset"])
    needs_launch_token = (spec.get("requires_kaggle_token") or
                          (spec.get("cpu_visualize") and spec.get("orchestration") != "github_actions"))
    if needs_launch_token:
        datasets.append(spec["pipeline_credential_dataset"])
    return {"id": spec["kernel_ref"], "title": spec["kernel_ref"].split("/")[-1].replace("-", " "),
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": spec.get("stage") not in {"visualize", "archive", "continuation-watch"}, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": datasets,
            "competition_sources": [], "kernel_sources": [], "model_sources": []}


def kernel_script(spec):
    # Only a SHA, dataset reference and public experiment metadata are embedded.
    script = '''import json
import os
from pathlib import Path
import subprocess
import sys
import time

SPEC = __SPEC__
if SPEC["stage"] in {"visualize", "archive"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["UAV_JOB_STARTED_AT"] = str(time.monotonic())
os.environ["UAV_JOB_STARTED_AT_UTC"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
os.environ["UAV_JOB_SUBMITTED_AT_UTC"] = SPEC.get("submitted_at_utc", "unknown")
os.environ["UAV_KAGGLE_ACCOUNT_ALIAS"] = SPEC.get("account", "unknown")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["WANDB_ENTITY"] = SPEC["entity"]
os.environ["WANDB_PROJECT"] = SPEC["project"]
os.environ["WANDB_MODE"] = "online"
os.environ["UAV_STAGE"] = SPEC["stage"]
os.environ["WANDB_SILENT"] = "true"
os.environ["UAV_GIT_COMMIT"] = SPEC["commit"]
os.environ["UAV_KAGGLE_ACCOUNT"] = SPEC["kernel_ref"].split("/", 1)[0]
os.environ["UAV_KAGGLE_KERNEL"] = SPEC["kernel_ref"]
os.environ["UAV_TRAINING_MODE"] = SPEC.get("training_mode", "unknown")
os.environ["UAV_AUTO_CONTINUE"] = "true" if SPEC.get("auto_continue") else "false"
os.environ["UAV_ROOT_KAGGLE_KERNEL"] = SPEC.get("root_kernel_ref", SPEC["kernel_ref"])
os.environ["UAV_CONTINUATION_INDEX"] = str(SPEC.get("continuation_index", 0))
if "requested_episodes" in SPEC:
    os.environ["UAV_EPISODE_BUDGET"] = str(SPEC["requested_episodes"])
else:
    os.environ.pop("UAV_EPISODE_BUDGET", None)
root = Path("/tmp/uav_network")
subprocess.run(["git", "clone", "--quiet", "https://github.com/truongdaoanhduy/uav_network.git", str(root)], check=True)
subprocess.run(["git", "checkout", "--quiet", SPEC["commit"]], cwd=root, check=True)
setup_env = dict(os.environ, PYTHON=sys.executable)
subprocess.run(["bash", "setup.sh", "--system-site-packages", "--cloud"], cwd=root, env=setup_env, check=True)
inputs = Path("/kaggle/input")
candidates = list(inputs.rglob("wandb_api_key.txt")) + list(inputs.rglob("wandb_secret.json"))
if len(candidates) != 1:
    raise RuntimeError("Expected one private W&B credential file")
credential = candidates[0]
key = credential.read_text().strip() if credential.suffix == ".txt" else json.loads(credential.read_text())["api_key"]
if not key:
    raise RuntimeError("Empty private W&B credential")
os.environ["WANDB_API_KEY"] = key
if SPEC.get("storage_credential_dataset"):
    storage_tokens = list(inputs.rglob("hf_token.txt"))
    if len(storage_tokens) != 1 or not storage_tokens[0].read_text().strip():
        raise RuntimeError("Expected one private checkpoint storage credential")
    os.environ["HF_TOKEN"] = storage_tokens[0].read_text().strip()
    os.environ["HF_REPO_ID"] = SPEC["checkpoint_repo"]
needs_launch_token = (SPEC.get("requires_kaggle_token") or
                      (SPEC.get("cpu_visualize") and SPEC.get("orchestration") != "github_actions"))
if needs_launch_token:
    tokens = list(inputs.rglob("uav_cpu_launch_token.txt"))
    if len(tokens) != 1 or not tokens[0].read_text().strip():
        raise RuntimeError("Expected one private CPU launch credential")
    os.environ["KAGGLE_API_TOKEN"] = tokens[0].read_text().strip()
spec_file = Path("/kaggle/working/job.json")
spec_file.write_text(json.dumps(SPEC, indent=2))
python = root / ".venv/bin/python"
# The remote runner uses CUDA_VISIBLE_DEVICES per independent learner.
runner = {"visualize": "uav_network.cloud.visualize", "archive": "uav_network.cloud.archive"}.get(
    SPEC["stage"], "uav_network.cloud.remote")
subprocess.run([str(python), "-m", runner, "--spec", str(spec_file)], cwd=root, check=True)
'''
    return script.replace("__SPEC__", repr(spec))
