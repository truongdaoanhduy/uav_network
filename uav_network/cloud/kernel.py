"""Generate a public-code-only bootstrap for an explicitly private GPU kernel."""
import json


def kernel_metadata(spec):
    datasets = [spec["credential_dataset"]]
    if spec.get("cpu_visualize"):
        datasets.append(spec["pipeline_credential_dataset"])
    return {"id": spec["kernel_ref"], "title": spec["kernel_ref"].split("/")[-1].replace("-", " "),
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": spec.get("stage") != "visualize", "enable_tpu": False,
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
if SPEC["stage"] == "visualize":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["UAV_JOB_STARTED_AT"] = str(time.monotonic())
os.environ["UAV_JOB_STARTED_AT_UTC"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
os.environ["UAV_JOB_SUBMITTED_AT_UTC"] = SPEC.get("submitted_at_utc", "unknown")
os.environ["UAV_KAGGLE_ACCOUNT_ALIAS"] = SPEC.get("account", "unknown")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["WANDB_ENTITY"] = SPEC["entity"]
os.environ["WANDB_PROJECT"] = SPEC["project"]
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_SILENT"] = "true"
os.environ["UAV_GIT_COMMIT"] = SPEC["commit"]
os.environ["UAV_KAGGLE_ACCOUNT"] = SPEC["kernel_ref"].split("/", 1)[0]
os.environ["UAV_KAGGLE_KERNEL"] = SPEC["kernel_ref"]
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
if SPEC.get("cpu_visualize"):
    tokens = list(inputs.rglob("uav_cpu_launch_token.txt"))
    if len(tokens) != 1 or not tokens[0].read_text().strip():
        raise RuntimeError("Expected one private CPU launch credential")
    os.environ["KAGGLE_API_TOKEN"] = tokens[0].read_text().strip()
spec_file = Path("/kaggle/working/job.json")
spec_file.write_text(json.dumps(SPEC, indent=2))
python = root / ".venv/bin/python"
# The remote runner uses CUDA_VISIBLE_DEVICES per independent learner.
runner = "uav_network.cloud.visualize" if SPEC["stage"] == "visualize" else "uav_network.cloud.remote"
subprocess.run([str(python), "-m", runner, "--spec", str(spec_file)], cwd=root, check=True)
'''
    return script.replace("__SPEC__", repr(spec))
