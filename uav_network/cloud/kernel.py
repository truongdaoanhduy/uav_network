"""Generate a public-code-only bootstrap for an explicitly private GPU kernel."""
import json
import base64
import secrets
import re

from .source_bundle import build_source_archive, source_sha256


# Presentation-only keys. Kaggle CLI ignores unknown fields, but they remain
# in local kernel-metadata.json. The generated kernel.py comments are uploaded.
_PROFILE_EXTRAS = {
    "masac": {"policy_note_stochastic": "layout-only", "entropy_display": {"enabled": True, "kind": "annotation"}},
    "matd3": {"policy_note_deterministic": "layout-only", "twin_critic_display": {"enabled": True, "kind": "annotation"}},
    "both": {"policy_note_multi": "layout-only"},
    "generic": {"policy_note_generic": "layout-only"},
}


def _metadata_algorithm(spec):
    runs = spec.get("runs")
    if isinstance(runs, dict):
        names = sorted(set(runs) & {"masac", "matd3"})
        if len(names) == 1:
            return names[0]
        if len(names) > 1:
            return "both"
    declared = spec.get("algorithm")
    if declared in {"masac", "matd3", "both"}:
        return declared
    reference = str(spec.get("kernel_ref", "")).lower()
    for name in ("masac", "matd3"):
        if re.search(r"(?:^|[^a-z])" + name + r"(?:$|[^a-z])", reference):
            return name
    return "generic"


def validate_metadata_extras(value):
    """Validate extra *annotations*, not training overrides or Kaggle controls."""
    if not isinstance(value, dict) or any(k not in {"common", "masac", "matd3", "both", "generic"} for k in value):
        raise ValueError("Metadata extras must be a map with common/masac/matd3/both/generic sections")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 8192:
        raise ValueError("Metadata extras exceed 8 KiB")
    def validate(node, depth=0):
        if depth > 6:
            raise ValueError("Metadata extras nested too deeply")
        if isinstance(node, dict):
            for k, v in node.items():
                if not isinstance(k, str) or len(k) > 80 or re.search(r"(?:secret|password|api.?key|credential|auth.?token|hf.?token)", k, re.I):
                    raise ValueError("Metadata extras contain an invalid or secret-like key")
                validate(v, depth + 1)
        elif isinstance(node, list):
            for item in node:
                validate(item, depth + 1)
        elif node is None or type(node) in (str, int, float, bool):
            if isinstance(node, str) and len(node) > 1000:
                raise ValueError("Metadata annotation value too long")
        else:
            raise ValueError("Metadata extras must contain only JSON values")
    for section in value.values():
        if not isinstance(section, dict):
            raise ValueError("Each metadata extras section must be a key-value map")
        validate(section)
    return value


def _custom_metadata_map(spec, algorithm):
    custom = validate_metadata_extras(spec.get("metadata_extras", {}))
    return {**custom.get("common", {}), **custom.get(algorithm, {})}


def _packaging_annotations(spec):
    algorithm = _metadata_algorithm(spec)
    return {
        "algorithm_profile": algorithm,
        "nonce": secrets.token_hex(16),
        "display_only": True,
        "profile_fields": _PROFILE_EXTRAS[algorithm],
        "custom": _custom_metadata_map(spec, algorithm),
    }


def kernel_metadata(spec):
    datasets = [spec["credential_dataset"]]
    if spec.get("storage_credential_dataset"):
        datasets.append(spec["storage_credential_dataset"])
    needs_launch_token = (spec.get("requires_kaggle_token") or
                          (spec.get("cpu_visualize") and spec.get("orchestration") != "github_actions"))
    if needs_launch_token:
        datasets.append(spec["pipeline_credential_dataset"])
    metadata = {"id": spec["kernel_ref"], "title": spec["kernel_ref"].split("/")[-1].replace("-", " "),
                "code_file": "kernel.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": spec.get("stage") not in {"visualize", "archive", "continuation-watch"}, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": datasets,
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    annotations = _packaging_annotations(spec)
    metadata["x_uav_annotations"] = annotations
    # Multiple arbitrary, inert key-value pairs. Never alter supported Kaggle fields.
    for _ in range(secrets.randbelow(7) + 5):
        metadata["x_package_" + secrets.token_hex(6)] = secrets.choice(
            (secrets.token_hex(8), secrets.randbelow(10_000_000), True, False))
    return metadata


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


def _bootstrap_comments(spec):
    """Unique, non-executable package annotations; no effect on training."""
    marker = secrets.token_hex(16)
    labels = ("source integrity", "runtime bootstrap", "artifact packaging", "execution manifest")
    order = list(labels)
    secrets.SystemRandom().shuffle(order)
    info = _packaging_annotations(spec)
    annotation = json.dumps(info, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return ("".join(f"# package {marker} / {label}\n" for label in order)
            + f"# package {marker} / metadata-map {annotation}\n")


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
    # User-defined display annotations must never enter the executable job spec.
    runtime_spec = {key: value for key, value in spec.items() if key != "metadata_extras"}
    body = (script.replace("__SPEC__", repr(runtime_spec))
            .replace("__SOURCE_B85__", base64.b85encode(source).decode("ascii"))
            .replace("__SOURCE_SHA256__", source_sha256(source)))
    return _bootstrap_comments(spec) + body
