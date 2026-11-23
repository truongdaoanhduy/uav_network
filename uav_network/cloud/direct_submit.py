"""Save a Kaggle script directly through the official SDK, with no metadata file.

Called as a subprocess with the selected account's token in the environment.
Input is a small nonsecret JSON job definition on stdin; output is a receipt.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys


def prepare_save_request(job, *, request_type=None):
    """Build the exact Kaggle API payload from the job, never from metadata JSON."""
    if request_type is None:
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
        request_type = ApiSaveKernelRequest
    ref = job["kernel_ref"]
    if not re.fullmatch(r"[\w-]+/[\w-]+", ref) or len(ref) > 50:
        raise ValueError("Invalid Kaggle kernel reference")
    path = Path(job["code_file"])
    if path.name != "kernel.py" or not path.is_file():
        raise ValueError("Expected the generated kernel.py source file")
    sources = job["dataset_sources"]
    if not isinstance(sources, list) or not sources or any(
            not isinstance(x, str) or not re.fullmatch(r"[\w-]+/[\w-]+", x)
            for x in sources):
        raise ValueError("Expected valid private credential dataset references")
    timeout = job["timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 39600:
        raise ValueError("Kaggle session timeout must be between 1 and 39600 seconds")

    request = request_type()
    request.slug = ref
    request.new_title = ref.split("/", 1)[1].replace("-", " ")
    request.text = path.read_text()
    request.language = "python"
    request.kernel_type = "script"
    request.is_private = True
    request.enable_gpu = bool(job["enable_gpu"])
    request.enable_tpu = False
    request.enable_internet = True
    request.dataset_data_sources = sources
    request.competition_data_sources = []
    request.kernel_data_sources = []
    request.model_data_sources = []
    request.category_ids = []
    request.session_timeout_seconds = timeout
    if request.enable_gpu:
        accelerator = job.get("accelerator", "NvidiaTeslaT4")
        if accelerator != "NvidiaTeslaT4":
            raise ValueError("Unsupported accelerator for current UAV launcher")
        if not hasattr(request, "machine_shape"):
            raise RuntimeError("Installed Kaggle SDK cannot explicitly request GPU T4x2: install kaggle>=2.2,<3")
        request.machine_shape = accelerator
    return request


def submit_direct(job, *, api_factory=None, request_type=None):
    """Save and run the notebook using the authenticated official Kaggle SDK."""
    if api_factory is None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api_factory = KaggleApi
    api = api_factory()
    api.authenticate()
    request = prepare_save_request(job, request_type=request_type)
    with api.build_kaggle_client() as kaggle:
        result = kaggle.kernels.kernels_api_client.save_kernel(request)
    if result is None:
        raise RuntimeError("Kaggle save_kernel returned no response")
    if getattr(result, "error", None):
        raise RuntimeError("Kaggle save_kernel returned an error")
    for name in ("invalidDatasetSources", "invalidCompetitionSources", "invalidKernelSources",
                 "invalidModelSources"):
        if getattr(result, name, None):
            raise RuntimeError("Kaggle save_kernel rejected attached input sources")
    url = getattr(result, "url", None)
    if not isinstance(url, str):
        raise RuntimeError("Kaggle did not confirm a notebook URL")
    match = re.fullmatch(r"https://(?:www\.)?kaggle\.com/code/([\w-]+/[\w-]+)", url.rstrip("/"))
    if not match or match.group(1) != job["kernel_ref"]:
        raise RuntimeError("Kaggle returned an unexpected notebook URL/reference")
    return {"url": url, "ref": match.group(1), "version": getattr(result, "versionNumber", None)}


def main():
    data = json.load(sys.stdin)
    # Keep stdout machine-readable even when the SDK prints status warnings.
    with redirect_stdout(io.StringIO()):
        answer = submit_direct(data)
    print(json.dumps(answer))


if __name__ == "__main__":
    main()
