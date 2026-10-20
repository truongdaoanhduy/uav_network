"""Resumable paired communication ablations of one immutable saved policy.

These controls are fixed-policy evaluations, not retrained baselines. Movement
components are preserved per action; battery/delivery feedback can change later
policy actions and physical trajectories.
"""
from __future__ import annotations

from functools import partial
import hashlib
import json
import math
import os
import re
from pathlib import Path
import tempfile

import numpy as np
import torch

from ..communication.link_budget import power_fraction
from ..environment.env import Action
from .comparison_metrics import evaluation_protocol, summarize_metrics
from .evaluation import load_evaluation_policy, make_evaluation_env, run_episode
from .telemetry import OutcomePool

CONTROLS_VERSION = 1
MODES = ("learned", "direct_only", "fixed_power")
TRACE_SEMANTICS = (
    "Successful whole-report custody transfers after complete reassembly; "
    "destination num_uavs is GCS; power_w is last received fragment TX power; "
    "positions are at reception time. This is not a per-packet log."
)


def _fixed_power(config, power_w):
    power = config.network.tx_power_max_w if power_w is None else float(power_w)
    if not math.isfinite(power) or not config.network.tx_power_min_w <= power <= config.network.tx_power_max_w:
        raise ValueError("fixed_power_w must be finite and inside the configured physical TX bounds")
    return power


def communication_control(action, observation, config, mode="learned", fixed_power_w=None):
    """Copy an Action, changing only the requested communication component."""
    if mode not in MODES:
        raise ValueError(f"Unknown communication control: {mode}")
    result = Action(np.array(action.continuous, copy=True), np.array(action.destination, copy=True))
    result.validate(config.world.num_uavs)
    if mode == "direct_only":
        n = config.world.num_uavs
        masks = np.asarray(observation)[:, -(n + 1):]
        peer = (result.destination > 0) & (result.destination < n)
        result.destination[peer] = np.where(masks[peer, n] > 0, n, 0)
    elif mode == "fixed_power":
        result.continuous[:, 2] = 2.0 * power_fraction(_fixed_power(config, fixed_power_w), config.network) - 1.0
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _checkpoint_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _summary(protocol, records):
    by_mode = {}
    for mode in protocol["modes"]:
        selected = [r for r in records if r["mode"] == mode]
        pool = OutcomePool()
        pool.add(selected)
        energy = sum(r["comparison_metrics"]["uav_energy_j"] for r in selected)
        delivered = sum(r["comparison_metrics"]["targets_delivered"] for r in selected)
        radio = sum(r["comparison_metrics"]["uav_radio_energy_j"] for r in selected)
        diagnostics = [r.get("network", {}).get("diagnostics") for r in selected]
        report_counts_available = bool(selected) and all(d is not None for d in diagnostics)
        by_mode[mode] = {
            "complete": len(selected) == len(protocol["seeds"]),
            "episodes": len(selected), "pooled": pool.metrics(),
            "metric_summary": summarize_metrics(selected),
            "total_mission_uav_energy_j": energy,
            "total_mission_uav_radio_energy_j": radio,
            "uav_radio_energy_j_per_delivered_report": radio / delivered if delivered else None,
            "total_delivered_reports": delivered,
            "mission_energy_j_per_delivered_report": energy / delivered if delivered else None,
            "energy_numerator_semantics": "Total UAV propulsion plus UAV TX/RX radio energy over every completed mission, including failed missions; excludes GCS energy.",
            "successful_direct_gcs_reports": sum(d["direct_gcs_reports"] for d in diagnostics) if report_counts_available else None,
            "successful_relayed_gcs_reports": sum(d["relayed_gcs_reports"] for d in diagnostics) if report_counts_available else None,
        }
    expected = len(protocol["seeds"]) * len(protocol["modes"])
    return {"schema": 1, "protocol": protocol, "complete": len(records) == expected,
            "expected_cases": expected, "completed_cases": len(records),
            "status": "complete" if len(records) == expected else "partial",
            "by_mode": by_mode, "records": records}


def run_relay_audit_cases(checkpoint, output_dir, cases, on_progress=None):
    """Run selected cases inside an existing full audit protocol.

    This is a parallelization helper: it never creates or changes the protocol.
    Every produced case therefore belongs to the exact fingerprint established
    by :func:`run_relay_audit`.
    """
    checkpoint = Path(checkpoint)
    output = Path(output_dir)
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if not isinstance(protocol, dict) or protocol.get("schema") != 1:
        raise ValueError("Existing audit protocol is invalid")
    checkpoint_hash = _checkpoint_hash(checkpoint)
    if protocol.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Checkpoint does not match existing audit protocol")

    learner, progress, source = load_evaluation_policy(checkpoint)
    config = learner.config
    torch.set_num_threads(config.runtime.torch_threads)
    seeds = list(protocol.get("seeds", []))
    modes = list(protocol.get("modes", []))
    power = _fixed_power(config, protocol.get("fixed_power_w"))
    expected = {
        "schema": 1, "controls_version": CONTROLS_VERSION,
        "checkpoint_sha256": checkpoint_hash, "config": config.to_dict(),
        "algorithm": learner.algorithm,
        "evaluation": evaluation_protocol(config, seeds, progress),
        "seeds": seeds, "modes": modes, "fixed_power_w": power,
        "trace": bool(protocol.get("trace")),
        "training_budget": {"episodes_completed": progress.get("episodes"),
                            "global_step": progress.get("global_step"),
                            "episodes_target": config.runtime.episodes,
                            "episode_budget_complete": progress.get("episodes") == config.runtime.episodes},
        "scope": "Fixed-policy communication ablations, not retrained baselines; per-action movement is preserved but later trajectories may diverge.",
        "trace_semantics": TRACE_SEMANTICS,
    }
    if source is not None:
        expected["source"] = source
    expected["fingerprint"] = _digest(expected)
    if expected != protocol:
        raise ValueError("Existing audit protocol/checkpoint is incompatible")

    requested = list(cases)
    if len(set(requested)) != len(requested):
        raise ValueError("Requested audit cases must be unique")
    allowed_seeds, allowed_modes = set(seeds), set(modes)
    for seed, mode in requested:
        if seed not in allowed_seeds or mode not in allowed_modes:
            raise ValueError("Requested audit case is outside the existing protocol")

    completed = skipped = 0
    for seed, mode in requested:
        path = output / "cases" / f"{mode}-{seed}.json"
        try:
            case = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            case = {}
        if isinstance(case, dict) and case.get("complete") is True:
            if (case.get("fingerprint") != protocol["fingerprint"] or case.get("mode") != mode
                    or case.get("seed") != seed or not isinstance(case.get("record"), dict)):
                raise ValueError(f"Incompatible completed audit case: {path}")
            skipped += 1
            continue

        env = make_evaluation_env(config)
        def notify(value):
            value = {**value, "mode": mode, "fingerprint": protocol["fingerprint"]}
            if on_progress is not None:
                on_progress(value)
        try:
            record, _, _, events = run_episode(
                env, learner, seed=seed,
                action_control=partial(communication_control, mode=mode, fixed_power_w=power),
                trace=bool(protocol["trace"]), on_progress=notify)
            record["mode"] = mode
            if protocol["trace"]:
                record["communication_trace"] = {"semantics": TRACE_SEMANTICS, "events": events}
            _atomic_json(path, {"complete": True, "fingerprint": protocol["fingerprint"],
                                "mode": mode, "seed": seed, "record": record})
            completed += 1
        finally:
            env.close()
    return {"completed_cases": completed, "skipped_cases": skipped,
            "requested_cases": len(requested), "fingerprint": protocol["fingerprint"]}


def run_relay_audit(checkpoint, seeds=None, modes=MODES, fixed_power_w=None, output_dir=None,
                    trace=False, on_progress=None):
    """Evaluate a complete paired matrix, reusing only matching completed cases.

    Default bank is 30 consecutive saved eval_seed values. Explicit banks contain
    1..30 unique unsigned 32-bit seeds. A changed bank/control/checkpoint requires
    another output directory.
    """
    checkpoint = Path(checkpoint)
    checkpoint_hash = _checkpoint_hash(checkpoint)
    learner, progress, source = load_evaluation_policy(checkpoint)
    config = learner.config
    torch.set_num_threads(config.runtime.torch_threads)
    seeds = list(range(config.runtime.eval_seed, config.runtime.eval_seed + 30)) if seeds is None else list(seeds)
    modes = list(modes)
    if (not 1 <= len(seeds) <= 30 or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must contain 1..30 unique unsigned 32-bit integers")
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("modes must be unique, nonempty learned/direct_only/fixed_power controls")
    power = _fixed_power(config, fixed_power_w)
    if _checkpoint_hash(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading")
    protocol = {
        "schema": 1, "controls_version": CONTROLS_VERSION, "checkpoint_sha256": checkpoint_hash,
        "config": config.to_dict(), "algorithm": learner.algorithm,
        "evaluation": evaluation_protocol(config, seeds, progress),
        "seeds": seeds, "modes": modes, "fixed_power_w": power, "trace": bool(trace),
        "training_budget": {"episodes_completed": progress.get("episodes"),
                            "global_step": progress.get("global_step"),
                            "episodes_target": config.runtime.episodes,
                            "episode_budget_complete": progress.get("episodes") == config.runtime.episodes},
        "scope": "Fixed-policy communication ablations, not retrained baselines; per-action movement is preserved but later trajectories may diverge.",
        "trace_semantics": TRACE_SEMANTICS,
    }
    if source is not None:
        protocol["source"] = source
    protocol["fingerprint"] = _digest(protocol)
    output = Path(output_dir) if output_dir else checkpoint.parent / "relay_audit"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing audit protocol/checkpoint is incompatible; use another output directory")
    else:
        # Single-writer recovery: process loss before initial replace can leave
        # only our NamedTemporaryFile staging files, including partial writes.
        staging = list(output.iterdir())
        for path in staging:
            if (not re.fullmatch(r"protocol\.json\.[a-z0-9_]{8}\.tmp", path.name)
                    or path.is_symlink() or not path.is_file()):
                raise ValueError("Nonempty audit output has no protocol; use another output directory")
            try:
                staged_protocol = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if staged_protocol != protocol:
                raise ValueError("Existing staged audit protocol/checkpoint is incompatible")
        _atomic_json(protocol_path, protocol)
        for path in staging:
            path.unlink()
    records = []
    pending = []
    for seed in seeds:
        for mode in modes:
            path = output / "cases" / f"{mode}-{seed}.json"
            try:
                case = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                case = {}
            if not isinstance(case, dict):
                case = {}
            if case.get("complete") is True:
                if (case.get("fingerprint") != protocol["fingerprint"] or case.get("mode") != mode
                        or case.get("seed") != seed or not isinstance(case.get("record"), dict)):
                    raise ValueError(f"Incompatible completed audit case: {path}")
                records.append(case["record"])
            else:
                pending.append((seed, mode, path))
    _atomic_json(output / "summary.json", _summary(protocol, records))
    for seed, mode, path in pending:
        env = make_evaluation_env(config)
        def notify(value):
            value = {**value, "mode": mode, "fingerprint": protocol["fingerprint"]}
            if value["status"] != "running" or value["step"] % 10 == 0:
                _atomic_json(output / "progress.json", value)
            if on_progress is not None:
                on_progress(value)
        try:
            record, _, _, events = run_episode(
                env, learner, seed=seed, action_control=partial(
                    communication_control, mode=mode, fixed_power_w=power),
                trace=trace, on_progress=notify)
            record["mode"] = mode
            if trace:
                record["communication_trace"] = {"semantics": TRACE_SEMANTICS, "events": events}
            case = {"complete": True, "fingerprint": protocol["fingerprint"],
                    "mode": mode, "seed": seed, "record": record}
            _atomic_json(path, case)
            records.append(record)
            _atomic_json(output / "summary.json", _summary(protocol, records))
        finally:
            env.close()
    return _summary(protocol, records)
