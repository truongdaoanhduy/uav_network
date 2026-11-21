"""Validation-only Pareto checkpoint archive. Does not modify actor/critic algorithms."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path

METRICS = (
    ("discovery_rate_percent", 1),
    ("gcs_delivery_rate_percent", 1),
    ("discovery_auc", 1),
    ("delivery_auc", 1),
    ("energy_per_delivered_target_j", -1),
    ("gcs_delivery_time_capped_mean_s", -1),
)

def _values(result):
    m = result.get("mission_metrics", {})
    vals = []
    for key, sign in METRICS:
        val = m.get(key)
        if type(val) not in (float, int) or not math.isfinite(val):
            return None
        if key == "energy_per_delivered_target_j" and val <= 0:
            return None
        vals.append(float(val) * sign)
    return tuple(vals)

def dominates(a, b):
    """True only if a is no worse than b on all objectives and better on one."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

def select(results):
    """Pure selector; validation only. Stable tie-breaking by checkpoint step."""
    if not results:
        raise ValueError("Empty validation results")
    candidates = []
    by_metric = {}
    identity = None
    for index, result in enumerate(results):
        src = result.get("source", {})
        if src.get("kind") != "periodic_validation":
            raise ValueError("Unexpected non-validation candidate")
        marker = (result.get("algorithm"), result.get("protocol", {}).get("scenario_hash"),
                  tuple(result.get("protocol", {}).get("seeds", [])))
        if identity is None:
            identity = marker
        if identity != marker:
            raise ValueError("Validation protocol mismatch between checkpoints")
        sha = src.get("checkpoint_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("Missing checkpoint SHA256")
        record = {"index": index, "sha256": sha, "episode": src["episodes"],
                  "global_step": src["global_step"],
                  "metrics": {key: (float(v) if type(v) in (int,float) and math.isfinite(v) else None)
                              for key, _ in METRICS
                              for v in [result.get("mission_metrics", {}).get(key)]}}
        vals = _values(result)
        if vals is not None:
            record["score"] = vals
            candidates.append(record)
        # Per-metric champions may be defined even if another metric is missing.
        for key, sign in METRICS:
            v = record["metrics"].get(key)
            if type(v) not in (int, float) or not math.isfinite(v):
                continue
            if key == "energy_per_delivered_target_j" and v <= 0:
                continue
            prev = by_metric.get(key)
            if prev is None or sign * v > sign * prev["value"]:
                by_metric[key] = {"sha256": sha, "index": index,
                                  "episode": src["episodes"], "value": v}
    # Equal vectors: keep the first one, not duplicates of the same result.
    frontier = []
    for i, x in enumerate(candidates):
        other_is_better = any(
            dominates(y["score"], x["score"]) or
            (i > j and y["score"] == x["score"])
            for j, y in enumerate(candidates) if i != j
        )
        if not other_is_better:
            frontier.append(x)
    # If multiple models tie for a metric, prefer an equivalent champion on
    # the non-dominated frontier, avoiding redundant storage of dominated ones.
    for key, selected in tuple(by_metric.items()):
        eligible = [item for item in frontier
                    if item["metrics"].get(key) == selected["value"]]
        if eligible:
            choice = min(eligible, key=lambda item: item["index"])
            by_metric[key] = {"sha256": choice["sha256"], "index": choice["index"],
                              "episode": choice["episode"], "value": selected["value"]}
    return {"version": 1, "criteria": [{"name": k, "direction": "max" if sign > 0 else "min"}
                                      for k, sign in METRICS],
            "evaluation_only": True,
            "count_valid": len(candidates), "count_excluded": len(results) - len(candidates),
            "pareto": frontier, "best_per_metric": by_metric}

def save_archive(results, snapshot_dir, output_dir):
    """Copy only selected immutable actor snapshots; verify source paths and SHA256."""
    summary = select(results)
    root = Path(snapshot_dir).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    keep = {r["sha256"]: r["index"] for r in summary["pareto"]}
    keep.update({r["sha256"]: r["index"] for r in summary["best_per_metric"].values()})
    stored = []
    for digest, index in keep.items():
        src = Path(results[index]["source"]["checkpoint"]).resolve()
        if not src.is_relative_to(root):
            raise ValueError("Checkpoint path escapes verified snapshot directory")
        if not src.is_file():
            raise FileNotFoundError(src)
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Checkpoint SHA256 mismatch: " + src.name)
        dest = out / (digest + ".pt")
        if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
            tmp = dest.with_suffix(".pt.tmp")
            try:
                tmp.write_bytes(data)
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
        stored.append({"sha256": digest, "file": dest.name, "episode":
                       results[index]["source"]["episodes"]})
    summary["stored_models"] = stored
    path = out / "pareto_manifest.json"
    tmp = out / "pareto_manifest.json.tmp"
    try:
        tmp.write_text(json.dumps(summary, indent=2, allow_nan=False))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return summary
