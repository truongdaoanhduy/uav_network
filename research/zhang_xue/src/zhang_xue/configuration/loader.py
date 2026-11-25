"""Typed scenario configuration from YAML plus explicit CLI overrides.

Only declared EnvConfigV54 parameters are permitted. No eval, no implicit
addition of source constants, and no silent coercion of strings to numbers.
The research_3000 profile is intentionally unsupported by the 20x20 kernel.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from zhang_xue.environment.configuration import EnvConfigV54, make_v57_config


def _scalar(value: Any, expected: Any, key: str) -> Any:
    if isinstance(value, (list, tuple, set, dict)):
        raise ValueError(f"{key} must be a scalar, not {type(value).__name__}")
    if expected is None:
        if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
            return None if value is None else float(value)
        raise ValueError(f"{key} must be a floating-point number or null")
    if isinstance(expected, bool):
        if type(value) is not bool:
            raise ValueError(f"{key} must be true/false")
        return value
    if isinstance(expected, str):
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value
    if isinstance(expected, int):
        if type(value) is not int:
            raise ValueError(f"{key} must be an integer")
        return value
    if isinstance(expected, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{key} must be a number")
        return float(value)
    raise ValueError(f"Unsupported configuration field {key}")


def load_scenario(
    path: str | Path | None = None,
    overrides: tuple[str, ...] = (),
    *,
    preset: str = "current",
) -> EnvConfigV54:
    if preset != "current":
        raise ValueError("preset must be current")
    reference = make_v57_config()
    values = asdict(reference)
    data = {}
    if path is not None:
        source = Path(path)
        parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Scenario YAML must contain a mapping")
        data = parsed
    for setting in overrides:
        key, sep, raw = setting.partition("=")
        if not sep:
            raise ValueError(f"Expected --set key=value, got {setting!r}")
        data[key] = yaml.safe_load(raw)
    unknown = set(data) - set(values)
    if unknown:
        raise ValueError(f"Unsupported scenario fields: {', '.join(sorted(unknown))}")
    for key, value in data.items():
        values[key] = _scalar(value, values[key], key)
    return EnvConfigV54(**values)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pinned 7-radio CPU scenario research runner")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preset", choices=("current",), default="current")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--communication", choices=("silent", "gcs"), default="gcs")
    parser.add_argument("--dry-run", action="store_true")
    return parser
