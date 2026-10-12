"""Legacy key.txt schema, secret-safe values and GPU quota selection."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re


@dataclass
class Account:
    name: str
    token: str = field(repr=False)
    enabled: bool = True


@dataclass
class Credentials:
    accounts: list[Account]
    wandb_key: str = field(repr=False)
    entity: str
    project: str
    min_gpu_hours: float = .25


def load_credentials(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("Cannot read key.txt as JSON") from error
    items = data.get("kaggle", {}).get("accounts", [])
    if not isinstance(items, list) or not items:
        raise ValueError("key.txt needs a kaggle.accounts list")
    accounts = []
    for index, item in enumerate(items):
        if (not isinstance(item, dict) or not isinstance(item.get("token"), str)
                or not item["token"] or type(item.get("enabled", True)) is not bool):
            raise ValueError(f"Invalid Kaggle account at index {index}")
        name = item.get("name", f"account_{index + 1:02d}")
        if not isinstance(name, str) or not name:
            raise ValueError("Account names must be nonempty strings")
        accounts.append(Account(name, item["token"], item.get("enabled", True)))
    w = data.get("wandb", {})
    if any(not isinstance(w.get(k), str) or not w[k] for k in ("api_key", "entity", "project")):
        raise ValueError("key.txt needs wandb.api_key/entity/project")
    hours = data.get("kaggle", {}).get("selection", {}).get("min_gpu_remaining_hours", .25)
    if not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours < 0:
        raise ValueError("Minimum GPU hours must be finite and nonnegative")
    return Credentials(accounts, w["api_key"], w["entity"], w["project"], float(hours))


def parse_quota(payload):
    def rows(value):
        if isinstance(value, dict):
            if str(value.get("resource", "")).upper() == "GPU":
                yield value
            for child in value.values():
                yield from rows(child)
        elif isinstance(value, list):
            for child in value:
                yield from rows(child)

    row = next(rows(payload), None)
    if row is None:
        raise ValueError("Quota response has no GPU resource")
    result = {}
    for name in ("used", "remaining", "total"):
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*h?\s*", str(row.get(name, "")))
        if not match:
            raise ValueError("Quota hours could not be parsed")
        result[f"{name}_hours"] = float(match.group(1))
    return result


def select_account(credentials, quotas, required_hours=None, requested=None):
    required = credentials.min_gpu_hours if required_hours is None else required_hours
    if not math.isfinite(required) or required < 0:
        raise ValueError("Required GPU hours must be finite and nonnegative")
    eligible = {a.name: a for a in credentials.accounts if a.enabled and (not requested or a.name == requested)}
    candidates = [q for q in quotas if q.get("ok") and q.get("name") in eligible
                  and q.get("remaining_hours", 0) > 0 and q["remaining_hours"] >= required]
    if not candidates:
        raise RuntimeError("No enabled account has enough verified GPU quota")
    chosen = max(candidates, key=lambda q: q["remaining_hours"])
    return eligible[chosen["name"]]
