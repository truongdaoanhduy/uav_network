"""Generate a public-code-only bootstrap for an explicitly private GPU kernel."""
import json
import base64
import secrets

from .source_bundle import build_source_archive, source_sha256


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


def serialize_kernel_metadata(spec, *, metadata=None):
    """Keep Kaggle's supported fields but vary JSON presentation per launch."""
    if metadata is None:
        metadata = kernel_metadata(spec)
    metadata = dict(metadata)
    # Kaggle defaults these optional, no-op entries to false/empty when absent.
    # Vary only their explicit presence; never vary operational settings.
    for optional, empty in (("enable_tpu", False), ("competition_sources", []),
                            ("kernel_sources", []), ("model_sources", [])):
        if optional in metadata and metadata[optional] == empty and secrets.randbelow(2):
            del metadata[optional]
    fields = list(metadata.items())
    secrets.SystemRandom().shuffle(fields)
    return json.dumps(dict(fields), indent=secrets.choice((2, 3, 4)), ensure_ascii=False) + "\n"


def _bootstrap_comments():
    """Unique, non-executable package annotations; no effect on training."""
    marker = secrets.token_hex(16)
    labels = ("source integrity", "runtime bootstrap", "artifact packaging", "execution manifest")
    order = list(labels)
    secrets.SystemRandom().shuffle(order)
    return "".join(f"# package {marker} / {label}\n" for label in order)


def kernel_script(spec):
    # Kaggle push uploads ONLY code_file, so all local source must be embedded in it.
    source = build_source_archive(__import__("pathlib").Path(__file__).resolve().parents[2])
    script = '''import json
import os
import base64
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
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
os.environ["UAV_TRAINING_ONLY"] = "true" if SPEC.get("training_only") else "false"
os.environ["UAV_AUTO_CONTINUE"] = "true" if SPEC.get("auto_continue") else "false"
os.environ["UAV_ROOT_KAGGLE_KERNEL"] = SPEC.get("root_kernel_ref", SPEC["kernel_ref"])
os.environ["UAV_CONTINUATION_INDEX"] = str(SPEC.get("continuation_index", 0))
if "requested_episodes" in SPEC:
    os.environ["UAV_EPISODE_BUDGET"] = str(SPEC["requested_episodes"])
else:
    os.environ.pop("UAV_EPISODE_BUDGET", None)
# Kaggle's API submits only this one Python script; source is inside the payload.
payload = base64.b85decode("__SOURCE_B85__")
if hashlib.sha256(payload).hexdigest() != "__SOURCE_SHA256__":
    raise RuntimeError("Bundled code checksum differs from the local build")
root = Path("/tmp/uav_network")
root.mkdir(parents=True, exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    total = 0
    for entry in archive:
        relative = PurePosixPath(entry.name)
        if (not entry.isfile() or relative.is_absolute() or
                any(part in {"..", ".", ""} for part in relative.parts) or
                entry.size > 2_000_000):
            raise RuntimeError("Unsafe bundled source entry")
        total += entry.size
        if total > 40_000_000:
            raise RuntimeError("Bundled source exceeds code size budget")
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(entry) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)

# Install package dependencies only. No git or remote source checkout.
python = root / ".venv/bin/python"
subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                "--without-pip", str(root / ".venv")], check=True)
subprocess.run([str(python), "-m", "pip", "install", "-e", ".[cloud]",
                "-r", "scripts/requirements-network.txt"], cwd=root, check=True)
subprocess.run([str(python), "-c",
                "from uav_network.configuration import load_config; "
                "from uav_network.communication.upstream import load_upstream; "
                "from gymnasium.vector import AutoresetMode; "
                "load_upstream(load_config()); assert AutoresetMode.DISABLED"],
               cwd=root, check=True)
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
    body = (script.replace("__SPEC__", repr(spec))
            .replace("__SOURCE_B85__", base64.b85encode(source).decode("ascii"))
            .replace("__SOURCE_SHA256__", source_sha256(source)))
    return _bootstrap_comments() + body
