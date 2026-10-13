"""Hashes of scientific outcomes and weights; runtime/PIDs are excluded."""
from hashlib import sha256
import json
import platform
from pathlib import Path

import numpy as np
import torch


def scientific_record(record):
    if isinstance(record, dict):
        return {k: scientific_record(v) for k, v in record.items()
                if k not in {"worker_pid", "elapsed_s", "timestamp_utc"}}
    if isinstance(record, list):
        return [scientific_record(v) for v in record]
    return record


def reproducibility_result(learner, records):
    model_hash = sha256()
    for name, tensor in sorted(learner.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        model_hash.update(name.encode())
        model_hash.update(str((value.dtype, tuple(value.shape))).encode())
        model_hash.update(value.numpy().tobytes())
    payload = json.dumps(scientific_record(records), sort_keys=True, separators=(",", ":"), allow_nan=False)
    config = learner.config.to_dict()
    settings = {k: config[k] for k in ("world", "sensor", "network", "energy", "reward", learner.algorithm)}
    settings["runtime"] = {k: v for k, v in config["runtime"].items()
                           if k not in {"output_dir", "warm_start_checkpoint"}}
    checkpoint = config["runtime"]["warm_start_checkpoint"]
    settings["warm_start_sha256"] = (None if checkpoint == "none" else
                                      sha256(Path(checkpoint).read_bytes()).hexdigest())
    hardware = {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "cuda_build": torch.version.cuda,
                "device": str(learner.device),
                "gpu": torch.cuda.get_device_name(learner.device) if learner.device.type == "cuda" else None}
    return {"training_seed": learner.config.runtime.seed,
            "episode_hash": sha256(payload.encode()).hexdigest(),
            "model_hash": model_hash.hexdigest(),
            "settings_hash": sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
            "hardware": hardware,
            "scope": "Exact repetition requires identical source/settings/platform and completed episode budget"}
