"""Credential-free public Kaggle gate for Framefree CUDA exactness."""
import json
from pathlib import Path
import re


_COMMIT = re.compile(r"[0-9a-f]{40}")
_REF = re.compile(r"[\w-]+/[\w-]+")


def _validate_commit(commit):
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("Framefree CUDA gate requires an exact 40-character Git commit SHA")
    return commit


def kernel_metadata(kernel_ref):
    if not isinstance(kernel_ref, str) or not _REF.fullmatch(kernel_ref):
        raise ValueError("Framefree CUDA gate requires owner/kernel-slug")
    return {"id": kernel_ref, "title": kernel_ref.split("/", 1)[1].replace("-", " "),
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": False, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [], "model_sources": []}


def kernel_script(commit):
    commit = _validate_commit(commit)
    return f'''import json, pathlib, subprocess, sys, time

COMMIT = {commit!r}
REPO = "https://github.com/truongdaoanhduy/uav_network.git"
work = pathlib.Path("/kaggle/working/uav_network")
result = {{"commit": COMMIT, "started_at": time.time()}}

def run(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, check=True)

run(["nvidia-smi"])
run(["git", "clone", "--filter=blob:none", REPO, str(work)])
run(["git", "checkout", COMMIT], cwd=work)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work, text=True).strip()
if actual != COMMIT:
    raise RuntimeError(f"commit mismatch: {{actual}}")
run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[gpu]", "pytest"], cwd=work)
import torch
result.update(torch_cuda=torch.cuda.is_available(), gpu_count=torch.cuda.device_count(),
              gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
if not result["torch_cuda"]:
    result.update(pytest_returncode=-1, pytest_stdout="", pytest_stderr="CUDA unavailable")
else:
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_framefree_cuda.py"]
    proc = subprocess.run(cmd, cwd=work, text=True, capture_output=True)
    print(proc.stdout, flush=True)
    print(proc.stderr, file=sys.stderr, flush=True)
    result.update(pytest_returncode=proc.returncode, pytest_stdout=proc.stdout[-12000:],
                  pytest_stderr=proc.stderr[-6000:])
result["finished_at"] = time.time()
pathlib.Path("/kaggle/working/framefree_cuda_result.json").write_text(json.dumps(result, indent=2))
if result["pytest_returncode"]:
    raise SystemExit(result["pytest_returncode"])
'''


def prepare_gate(output, kernel_ref, commit):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "kernel.py").write_text(kernel_script(commit))
    (output / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(kernel_ref), indent=2))
    return output



def verify_remote(kernel_ref, commit, output, runner=None):
    """Download and validate a public Framefree CUDA gate result."""
    import subprocess
    if not isinstance(kernel_ref, str) or not _REF.fullmatch(kernel_ref):
        raise ValueError("Framefree CUDA gate requires owner/kernel-slug")
    commit = _validate_commit(commit)
    runner = runner or subprocess.run
    status = runner(["kaggle", "kernels", "status", kernel_ref], text=True, capture_output=True, check=True)
    if "complete" not in status.stdout.lower():
        raise RuntimeError("Framefree CUDA gate kernel is not complete")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runner(["kaggle", "kernels", "output", kernel_ref, "-p", str(output),
            "--file-pattern", r"(^|.*/)framefree_cuda_result\.json$", "-q", "-o"],
           text=True, capture_output=True, check=True)
    matches = list(output.rglob("framefree_cuda_result.json"))
    if len(matches) != 1:
        raise RuntimeError("Framefree CUDA gate output is missing or ambiguous")
    return validate_result(json.loads(matches[0].read_text()), commit)

def validate_result(payload, commit):
    commit = _validate_commit(commit)
    if (not isinstance(payload, dict) or payload.get("commit") != commit
            or payload.get("torch_cuda") is not True
            or type(payload.get("gpu_count")) is not int or payload["gpu_count"] < 1
            or type(payload.get("pytest_returncode")) is not int or payload["pytest_returncode"] != 0):
        raise ValueError("Framefree CUDA gate result is incomplete, stale, or failed")
    return payload
