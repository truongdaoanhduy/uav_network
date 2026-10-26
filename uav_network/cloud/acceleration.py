"""Measured acceleration defaults for the canonical production cloud profile."""
from pathlib import Path


_ACCELERATION_KEYS = (
    "runtime.amp_mode",
    "runtime.compile_models",
    "runtime.fused_optimizers",
)


def _override_keys(overrides):
    return {item.partition("=")[0].strip() for item in overrides if "=" in item}


def apply_measured_acceleration_policy(config_path, stage, algorithm, config, overrides, canonical_path):
    """Return cloud overrides with measured defaults added only when applicable."""
    result = list(overrides)
    if stage not in {"train", "validate"}:
        return result
    if Path(config_path).resolve() != Path(canonical_path).resolve():
        return result
    if config.pql.enabled and algorithm == "both":
        raise ValueError("Measured PQL production policy requires separate MASAC and MATD3 launches")
    if algorithm not in {"masac", "matd3", "both"}:
        raise ValueError("Measured acceleration policy requires masac, matd3, or both")

    compile_models = not (config.pql.enabled and algorithm == "matd3")
    desired = {
        "runtime.amp_mode": '"fp16"',
        "runtime.compile_models": str(compile_models).lower(),
        "runtime.fused_optimizers": "false",
    }
    present = _override_keys(result)
    for key in _ACCELERATION_KEYS:
        if key not in present:
            result.append(f"{key}={desired[key]}")
    return result
