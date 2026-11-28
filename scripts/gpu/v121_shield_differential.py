#!/usr/bin/env python3
"""Common-prestate CPU/Warp shield differential, not stochastic world parity.

Each reference starts from the actual Warp pre-step state. Nothing is reset,
patched, or copied back to the mission after construction. Host readbacks are
intentional validation work and must not be reported as training throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]
VELOCITY_ATOL_MPS = 1.0e-3
POSITION_ATOL_M = 1.0e-3
MOTIONS = ("random", "converge", "boundary")


def _snapshot(world):
    names = ("positions", "velocities", "battery", "failed", "safety_reason",
             "intervention", "candidates", "v_safe", "episode_step", "episode_id",
             "done", "termination_reason")
    return {name: world.buffers[name].numpy().copy() for name in names}


def _actions(pre, step, motion, held_random):
    """Mixed rotates each world through all three motions in 60-step blocks."""
    pos = pre["positions"][:, :6, :2]
    actions = held_random.copy()
    labels = []
    for e in range(len(pos)):
        mode = MOTIONS[(e + step // 60) % 3] if motion == "mixed" else motion
        labels.append(mode)
        if mode == "random":
            continue
        if mode == "converge":
            goals = np.repeat(pos[e].mean(axis=0, keepdims=True), 6, axis=0)
        else:
            goals = np.asarray(((0., 0.), (1000., 0.), (1000., 1000.),
                                (0., 1000.), (0., 500.), (1000., 500.)))
            goals = np.roll(goals, e % 6, axis=0)
        delta = goals - pos[e]
        actions[e] = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1.)
    return actions.astype(np.float32), labels


def _reference(pre, e, action, world):
    from zhang_xue.learners.actor_adapter import scale_normalized_xy_to_mps
    from zhang_xue.mobility.flight_safety_energy import (
        kinematic_safety_shield, trajectory_safety_violations,
    )

    p = world.params
    pos = pre["positions"][e, :6, :2].astype(np.float64)
    vel = pre["velocities"][e].astype(np.float64)
    alive = pre["battery"][e] > 0.
    command = scale_normalized_xy_to_mps(action, v_max_mps=p.v_max_mps)
    if pre["failed"][e]:
        return {"already_failed": True, "command_mps": command,
                "v_safe_mps": None, "intervention": np.zeros(6),
                "violations": None, "failed": True,
                "committed_velocity_mps": vel, "committed_position_xy_m": pos}
    safe, intervention = kinematic_safety_shield(
        pos, vel, command, alive, world.buildings, map_size_m=1000., dt_s=p.dt_s,
        v_max_mps=p.v_max_mps, a_max_mps2=p.a_max_mps2,
        uav_safe_distance_m=p.uav_safe_distance_m,
    )
    violations = trajectory_safety_violations(
        pos, safe, alive, world.buildings, map_size_m=1000., dt_s=p.dt_s,
        uav_safe_distance_m=p.uav_safe_distance_m,
    )
    failed = any(violations.values())
    # Active CPU safety_failure_terminate=True rolls back the complete flight step.
    committed_vel = vel if failed else np.where(alive[:, None], safe, 0.)
    committed_pos = pos if failed else pos + safe * p.dt_s
    return {"already_failed": False, "command_mps": command,
            "v_safe_mps": safe, "intervention": intervention,
            "violations": violations, "failed": failed,
            "committed_velocity_mps": committed_vel,
            "committed_position_xy_m": committed_pos}


def _max_vector_error(a, b):
    return float(np.max(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)))


def _measure(ref, post, e):
    arrays = [post[k][e] for k in ("positions", "velocities", "v_safe", "battery")]
    finite = all(np.isfinite(a).all() for a in arrays)
    safe_error = (0. if ref["already_failed"] else
                  _max_vector_error(ref["v_safe_mps"], post["v_safe"][e]))
    velocity_error = _max_vector_error(ref["committed_velocity_mps"], post["velocities"][e])
    position_error = _max_vector_error(ref["committed_position_xy_m"], post["positions"][e, :6, :2])
    failure_mismatch = bool(ref["failed"]) != bool(post["failed"][e])
    errors = {"proposal_velocity_mps": safe_error,
              "committed_velocity_mps": velocity_error, "motion_m": position_error,
              "failure_mismatch": failure_mismatch, "nonfinite": not finite}
    errors["diverged"] = (not finite or failure_mismatch
                          or safe_error > VELOCITY_ATOL_MPS
                          or velocity_error > VELOCITY_ATOL_MPS
                          or position_error > POSITION_ATOL_M)
    return errors


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


def _counterexample(step, e, label, action, pre, ref, post, errors):
    return {"step": step + 1, "world": e, "motion": label,
            "action_normalized_xy": action,
            "prestate": {k: pre[k][e] for k in (
                "positions", "velocities", "battery", "failed", "episode_step",
                "episode_id", "done", "termination_reason")},
            "cpu_reference": ref,
            "warp_actual": {k: post[k][e] for k in post}, "errors": errors}


def run(args):
    # Deferred so --help and importing pure reporting helpers need no Warp/Torch.
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80

    p = GPUWorldV80Params.from_yaml(
        args.scenario, envs=args.envs, device=args.device, seed=args.seed,
    )
    if p.strict_braking_terminate:
        raise ValueError("CPU reference uses physical swept detector, not strict braking termination")
    world = WarpParallelMissionV80(p)
    rng = np.random.default_rng(args.seed)
    held = np.zeros((args.envs, 6, 2), dtype=np.float32)
    # No application sends: HELLO/network still run; motion parity does not need DATA.
    discrete = wp.array(np.zeros((args.envs, 6), dtype=np.int32), dtype=wp.int32, device=args.device)
    counters = {name: 0 for name in (
        "compared_world_steps", "eligible_shield_world_steps", "already_failed_world_steps",
        "post_terminal_world_steps", "divergent_world_steps", "failure_mismatches",
        "nonfinite_world_steps", "cpu_intervened_uav_steps", "warp_intervened_uav_steps",
        "cpu_failure_events", "warp_failure_events", "warp_committed_intervention_world_steps",
    )}
    maxima = {"proposal_velocity_mps": 0., "committed_velocity_mps": 0., "motion_m": 0.}
    by_motion = {mode: {"compared": 0, "eligible": 0, "divergent": 0,
                        "cpu_intervened_uav_steps": 0, "warp_intervened_uav_steps": 0}
                 for mode in MOTIONS}
    cpu_failed, warp_failed = set(), set()
    first_terminal = [None] * args.envs
    log_path = args.output.with_suffix(".counterexamples.jsonl")
    trace_path = args.output.with_suffix(".steps.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    initial_episode = world.buffers["episode_id"].numpy().copy()
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log, trace_path.open("w", encoding="utf-8") as trace:
        for step in range(args.steps):
            pre = _snapshot(world)
            if step % 8 == 0:
                held = rng.uniform(-1., 1., (args.envs, 6, 2)).astype(np.float32)
            actions, labels = _actions(pre, step, args.motion, held)
            references = [_reference(pre, e, actions[e], world) for e in range(args.envs)]
            world.step(wp.array(actions, dtype=wp.float32, device=args.device), discrete)
            post = _snapshot(world)
            if not np.array_equal(post["episode_id"], initial_episode):
                raise RuntimeError("Unexpected episode reset invalidates common-prestate audit")
            for e, ref in enumerate(references):
                errors = _measure(ref, post, e)
                mode = by_motion[labels[e]]
                counters["compared_world_steps"] += 1
                counters["post_terminal_world_steps"] += int(bool(pre["done"][e]))
                counters["already_failed_world_steps"] += int(ref["already_failed"])
                mode["compared"] += 1
                if post["done"][e] and first_terminal[e] is None:
                    first_terminal[e] = step + 1
                cpu_interventions, warp_interventions = 0, 0
                if not ref["already_failed"]:
                    counters["eligible_shield_world_steps"] += 1
                    mode["eligible"] += 1
                    cpu_interventions = int(np.count_nonzero(ref["intervention"] * p.v_max_mps > VELOCITY_ATOL_MPS))
                    warp_interventions = int(np.count_nonzero(np.linalg.norm(
                        post["candidates"][e] - post["v_safe"][e], axis=1) > VELOCITY_ATOL_MPS))
                    counters["cpu_intervened_uav_steps"] += cpu_interventions
                    counters["warp_intervened_uav_steps"] += warp_interventions
                    mode["cpu_intervened_uav_steps"] += cpu_interventions
                    mode["warp_intervened_uav_steps"] += warp_interventions
                    if ref["failed"]:
                        counters["cpu_failure_events"] += 1
                        cpu_failed.add(e)
                    if post["failed"][e]:
                        counters["warp_failure_events"] += 1
                        warp_failed.add(e)
                    elif warp_interventions:
                        counters["warp_committed_intervention_world_steps"] += 1
                for key in maxima:
                    if np.isfinite(errors[key]):
                        maxima[key] = max(maxima[key], errors[key])
                counters["failure_mismatches"] += int(errors["failure_mismatch"])
                counters["nonfinite_world_steps"] += int(errors["nonfinite"])
                transition = {"step": step + 1, "world": e, "motion": labels[e],
                              "eligible_shield_prestate": not ref["already_failed"],
                              "pre_terminal": bool(pre["done"][e]),
                              "cpu_failed": bool(ref["failed"]), "warp_failed": bool(post["failed"][e]),
                              "warp_safety_reason": int(post["safety_reason"][e]),
                              "cpu_intervention_sum": float(np.sum(ref["intervention"])),
                              "warp_intervention_buffer": float(post["intervention"][e]),
                              "cpu_proposal_intervened_uavs": cpu_interventions,
                              "warp_proposal_intervened_uavs": warp_interventions, "errors": errors}
                trace.write(json.dumps(_json_safe(transition), allow_nan=False) + "\n")
                if errors["diverged"]:
                    counters["divergent_world_steps"] += 1
                    mode["divergent"] += 1
                    record = _counterexample(step, e, labels[e], actions[e], pre, ref, post, errors)
                    log.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")
                    log.flush()
            if (step + 1) % 30 == 0:
                trace.flush()
                print(f"shield step {step + 1}/{args.steps}: divergences={counters['divergent_world_steps']}", flush=True)
    sources = ("scripts/gpu/v121_shield_differential.py",
               "research/zhang_xue/src/zhang_xue/learners/actor_adapter.py",
               "research/zhang_xue/src/zhang_xue/mobility/flight_safety_energy.py",
               "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py",
               "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py",
               "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v96_backup.py",
               "research/zhang_xue/src/zhang_xue/gpu/mission_v80.py")
    summary = {
        "gate": "V121_COMMON_PRESTATE_SHIELD_DIFFERENTIAL",
        "status": "PASS_SHIELD_ONLY" if not counters["divergent_world_steps"] else "FAIL_SHIELD_DIFFERENTIAL",
        "scope": "CPU shield proposal, swept physical detector and fail-closed flight commit from actual Warp prestates",
        "not_tested": ["full stochastic sensor/network world parity", "trained policy behavior", "energy parity"],
        "scenario": str(args.scenario), "scenario_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "source_sha256": {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in sources},
        "device": args.device, "envs": args.envs, "steps": args.steps, "seed": args.seed,
        "motion": args.motion, "motion_schedule": "mixed: world-offset random/converge/boundary, rotate every60 steps; random held8 steps",
        "episode_handling": "no auto-reset; failed worlds retain physical state; post-terminal samples counted separately",
        "tolerances": {"velocity_max_uav_l2_mps": VELOCITY_ATOL_MPS,
                       "position_max_uav_l2_m": POSITION_ATOL_M, "relative_tolerance": 0.,
                       "intervention_activity_candidate_to_safe_mps": VELOCITY_ATOL_MPS,
                       "failure_mismatch_allowed": False},
        "intervention_definition": "uav counts use candidate-to-shield proposal changes, including rejected proposals; committed world count excludes failed steps; actual intervention buffer logged per step",
        "cpu_detector": "trajectory_safety_violations; GPU extra viability/braking flags remain visible differences",
        "cpu_precision": "FP64 reference from exact FP32 Warp prestate values", "warp_precision": "FP32",
        "first_terminal_step": first_terminal, "cpu_failed_worlds": sorted(cpu_failed),
        "warp_failed_worlds": sorted(warp_failed), "counters": counters, "max_errors": maxima,
        "by_motion": by_motion, "counterexamples_jsonl": str(log_path), "steps_jsonl": str(trace_path),
        "runtime_versions": {"warp": getattr(wp, "__version__", "unknown"), "numpy": np.__version__},
        "buildings": world.buildings, "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(args.output, summary)
    print(json.dumps({"status": summary["status"], "output": str(args.output), **counters}), flush=True)
    return 1 if counters["divergent_world_steps"] else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", type=Path, default=ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml")
    ap.add_argument("--envs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--motion", choices=("mixed", *MOTIONS), default="mixed")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if not 1 <= args.envs <= 64 or not 1 <= args.steps <= 900:
        ap.error("bounded audit requires envs in1..64 and steps in1..900")
    try:
        return run(args)
    except Exception as exc:
        _write_json(args.output, {"gate": "V121_COMMON_PRESTATE_SHIELD_DIFFERENTIAL",
                                 "status": "ERROR", "error": repr(exc),
                                 "scope": "shield differential was not completed"})
        print(f"Shield differential error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
