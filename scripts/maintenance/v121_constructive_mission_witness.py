#!/usr/bin/env python3
"""ORACLE_VALIDATION_ONLY: execute a constructive mission, never a baseline policy.

True target cells are read once for offline assignment and waypoint planning.
All movement, sensing, DS evidence, radio delivery, battery use and termination
are produced by the unchanged environment. Failures remain in the evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/zhang_xue/src"))
from zhang_xue.communication.common import segment_hits_cylinder_3d_fp32
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv

LABEL = "ORACLE_VALIDATION_ONLY"
PROFILES = {
    "base": "scenario_v103_ds_only_1km.yaml",
    "rf_only": "scenario_v111_ds_rf_stress_no_peer_1km.yaml",
}


def distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def segment_clear(a, b, circles, *, escape=False):
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    den = dx * dx + dy * dy
    for x, y, radius in circles:
        rx, ry = ax - x, ay - y
        if escape and rx * rx + ry * ry < radius * radius:
            # A peer may have approached since the last plan: permit retreat.
            if rx * dx + ry * dy >= 0 and distance(b, (x, y)) > math.hypot(rx, ry):
                continue
        tau = max(0.0, min(1.0, -(rx * dx + ry * dy) / max(1e-12, den)))
        if (rx + tau * dx) ** 2 + (ry + tau * dy) ** 2 <= radius * radius:
            return False
    return True


class GridPlanner:
    """A* on a 25m grid with swept obstacle clearance and visible sensor goals."""
    def __init__(self, cfg, buildings):
        self.cfg = cfg
        self.spacing = 25.0
        self.side = int(round(cfg.map_size_m / self.spacing)) + 1
        self.buildings = buildings
        # Full-speed stopping distance plus the environment's 2m building margin.
        margin = 2.0 + cfg.v_max_mps ** 2 / (2.0 * cfg.a_max_mps2) + 5.0
        self.static = [(x, y, r + margin) for x, y, r, _ in buildings]
        self.boundary = 2.0 + cfg.v_max_mps ** 2 / (2.0 * cfg.a_max_mps2) + 5.0
        self.nodes = {(x, y): (x * self.spacing, y * self.spacing)
                      for x in range(self.side) for y in range(self.side)
                      if self.legal((x * self.spacing, y * self.spacing), self.static)}

    def legal(self, xy, circles):
        x, y = xy
        return (self.boundary <= x <= self.cfg.map_size_m - self.boundary
                and self.boundary <= y <= self.cfg.map_size_m - self.boundary
                and all(distance(xy, (cx, cy)) > r for cx, cy, r in circles))

    def plan(self, start, goal, peers=(), *, sensing=False):
        peer_radius = self.cfg.uav_safe_distance_m + 27.5
        circles = self.static + [(float(x), float(y), peer_radius) for x, y in peers]
        valid = {node for node, xy in self.nodes.items() if self.legal(xy, circles)}
        if sensing:
            goals = {node for node in valid
                     if distance(self.nodes[node], goal) <= self.cfg.fov_radius_m - 5.0
                     and not any(segment_hits_cylinder_3d_fp32(
                         (*self.nodes[node], self.cfg.uav_alt_m), (*goal, 0.0), *b)
                         for b in self.buildings)}
        else:
            goals = {node for node in valid
                     if distance(self.nodes[node], goal) <= self.spacing * 1.5
                     and segment_clear(self.nodes[node], goal, circles)}
        if not goals:
            return []
        entry = sorted(valid, key=lambda n: distance(start, self.nodes[n]))
        origin = next((n for n in entry
                       if segment_clear(start, self.nodes[n], circles, escape=True)), None)
        if origin is None:
            return []
        radius = self.cfg.fov_radius_m - 5.0 if sensing else self.spacing * 1.5
        heuristic = lambda n: max(0.0, distance(self.nodes[n], goal) - radius)
        frontier = [(heuristic(origin), origin)]
        costs, previous = {origin: 0.0}, {}
        found = None
        while frontier:
            _, node = heapq.heappop(frontier)
            if node in goals:
                found = node
                break
            x, y = node
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
                nxt = (x + dx, y + dy)
                if nxt not in valid or not segment_clear(self.nodes[node], self.nodes[nxt], circles):
                    continue
                new_cost = costs[node] + self.spacing * math.hypot(dx, dy)
                if new_cost < costs.get(nxt, float("inf")):
                    costs[nxt], previous[nxt] = new_cost, node
                    heapq.heappush(frontier, (new_cost + heuristic(nxt), nxt))
        if found is None:
            return []
        route = [self.nodes[found]]
        while found != origin:
            found = previous[found]
            route.append(self.nodes[found])
        route.reverse()
        if not sensing:
            route.append(tuple(map(float, goal)))
        # Remove grid turns only when the whole swept segment retains clearance.
        smooth, current = [], start
        while route:
            index = next((i for i in range(len(route) - 1, -1, -1)
                          if segment_clear(current, route[i], circles, escape=True)), 0)
            smooth.append(route[index])
            current, route = route[index], route[index + 1:]
        return smooth


def assign_targets(staging, targets, vmax):
    """Offline greedy minimum estimated completion time, including return cost."""
    queues = [[] for _ in staging]
    ends, elapsed = np.asarray(staging).copy(), np.zeros(len(staging))
    remaining = set(targets)
    while remaining:
        _, actor, cid = min(
            (elapsed[u] + distance(ends[u], targets[cid]) / vmax + 4.0
             + 0.25 * distance(targets[cid], staging[u]) / vmax, u, cid)
            for u in range(len(staging)) for cid in remaining)
        elapsed[actor] += distance(ends[actor], targets[cid]) / vmax + 4.0
        ends[actor] = targets[cid]
        queues[actor].append(cid)
        remaining.remove(cid)
    return queues


def source_manifest():
    paths = sorted((ROOT / "research/zhang_xue/src").rglob("*.py")) + [Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashes, hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def run_world(profile, seed, steps, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{profile}_seed{seed}"
    trace_path = out / f"{stem}_steps.jsonl"
    scenario = ROOT / "research/zhang_xue/configs" / PROFILES[profile]
    cfg = load_scenario(scenario)
    if (cfg.enable_peer_dissemination or cfg.enable_data_mule_forwarding
            or cfg.belief_inference_mode != "dempster_shafer"
            or cfg.enable_tx_power_control or cfg.tx_power_w != 0.1):
        raise ValueError("Witness requires unchanged DS-only pure-GCS fixed-TX profiles")
    manifest, source_sha = source_manifest()
    env = ZhangXue7NodePacketEnv(cfg, seed=seed)
    _, _, mask = env.reset(seed=seed)
    staging = env.positions[:env.actor_n, :2].copy()
    # Sole read of target truth for offline assignment and sensor-viewpoint plans.
    target_ids = [int(cid) for cid in np.flatnonzero(env.true_targets)]
    targets = {cid: ((cid % env.grid_w + .5) * env.cell_size_m,
                     (cid // env.grid_w + .5) * env.cell_size_m) for cid in target_ids}
    queues = assign_targets(staging, targets, cfg.v_max_mps)
    assignment = [q.copy() for q in queues]
    planner = GridPlanner(cfg, env.buildings)
    paths, keys = [[] for _ in queues], [None] * env.actor_n
    collected, arrived, stagnation = [], [False] * env.actor_n, np.zeros(env.actor_n, dtype=int)
    info, error, terminated, truncated = {}, None, False, False
    safety_ever, energy_ever, finite = False, False, True
    started = time.monotonic()
    with trace_path.open("w", encoding="utf-8") as trace:
        trace.write(json.dumps({"kind":"metadata", "label":LABEL, "profile":profile,
                                "seed":seed, "assignment":assignment, "targets":targets,
                                "staging":staging.tolist(), "source_sha256":source_sha}) + "\n")
        for step in range(1, steps + 1):
            command = np.zeros((env.actor_n, 2), dtype=float)
            for u in range(env.actor_n):
                while queues[u]:
                    cid = queues[u][0]
                    pos = int(env.net.ledger_pos[u, u, cid])
                    neg = int(env.net.ledger_neg[u, u, cid])
                    if pos - neg < 2:
                        break
                    collected.append({"actor":u,"cell":cid,"step":step-1,"positive":pos,"negative":neg})
                    queues[u].pop(0)
                cid = queues[u][0] if queues[u] else None
                goal = targets[cid] if cid is not None else staging[u]
                if cid is None and distance(env.positions[u], goal) < 3.0:
                    arrived[u] = True
                    continue
                arrived[u] = False
                key = cid if cid is not None else "return"
                if key != keys[u] or step % 8 == u or stagnation[u] >= 5 or not paths[u]:
                    peers = [env.positions[v, :2] for v in range(env.actor_n) if v != u]
                    paths[u] = planner.plan(env.positions[u, :2], goal, peers, sensing=cid is not None)
                    keys[u], stagnation[u] = key, 0
                while paths[u] and distance(env.positions[u], paths[u][0]) < 3.0:
                    paths[u].pop(0)
                if paths[u]:
                    delta = np.asarray(paths[u][0]) - env.positions[u, :2]
                    dist = float(np.linalg.norm(delta))
                    speed = min(cfg.v_max_mps, .8 * math.sqrt(2 * cfg.a_max_mps2 * dist), dist / cfg.dt_s)
                    command[u] = delta * speed / max(1e-12, dist)
            before = env.positions[:env.actor_n, :2].copy()
            try:
                _, _, mask, reward, terminated, truncated, info = env.step(command, mask[:,1].astype(np.int32))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                trace.write(json.dumps({"kind":"exception","step":step,"error":error}) + "\n")
                break
            moved = np.linalg.norm(env.positions[:env.actor_n, :2] - before, axis=1)
            stagnation = np.where((moved < .5) & (np.linalg.norm(command, axis=1) > 1), stagnation + 1, 0)
            safety_ever |= bool(info.get("safety_violation", False))
            energy_ever |= bool(np.any(env.battery_j <= 0))
            finite &= bool(np.isfinite(env.positions).all() and np.isfinite(env.battery_j).all() and math.isfinite(reward))
            record = {"kind":"step","label":LABEL,"step":step,
                      "positions_m":env.positions.tolist(),"velocity_mps":env.velocities.tolist(),
                      "command_velocity_mps":command.tolist(),"battery_j":env.battery_j.tolist(),
                      "safety_violation":info.get("safety_violation",False),
                      "safety_violations":info.get("safety_violations",{}),
                      "gcs_tp_ever":info.get("gcs_tp_confirmed",0),
                      "gcs_tp_active":info.get("gcs_tp_active_confirmed",0),
                      "gcs_fp_ever":info.get("gcs_fp_ever_confirmed",0),
                      "gcs_fp_active":info.get("gcs_fp_latched",0),
                      "queue_lengths":[len(r.queue) for r in env.net.radios],
                      "network_stats":dict(env.net.stats),"remaining_goals":queues,
                      "returned_to_staging":arrived,"terminated":terminated,"truncated":truncated}
            trace.write(json.dumps(record, allow_nan=False) + "\n")
            if step % 50 == 0 or terminated or truncated:
                trace.flush()
                print(json.dumps({"label":LABEL,"profile":profile,"seed":seed,"step":step,
                                  "gcs_tp":info.get("gcs_tp_confirmed",0),"remaining_goals":sum(map(len,queues)),
                                  "returned":sum(arrived),"elapsed_s":round(time.monotonic()-started,2)}), flush=True)
            if terminated or truncated or safety_ever or energy_ever or not finite:
                break
    success = bool(info.get("gcs_tp_confirmed", 0) == cfg.num_targets and not safety_ever
                   and not energy_ever and finite and error is None and env.t <= steps)
    result = {"label":LABEL,"profile":profile,"seed":seed,"success_witness":success,
              "claim":"Actual simulator witness for this seed only; no formal bound or policy baseline",
              "steps":env.t,"max_steps":steps,"reset_count":1,"gcs_tp_ever":info.get("gcs_tp_confirmed",0),
              "gcs_tp_active":info.get("gcs_tp_active_confirmed",0),"gcs_fp_ever":info.get("gcs_fp_ever_confirmed",0),
              "safety_failure":safety_ever,"energy_failure":energy_ever,"all_finite":finite,"error":error,
              "terminated":terminated,"truncated":truncated,"assignment":assignment,"collected_goals":collected,
              "remaining_goals":queues,"returned_to_staging":arrived,"battery_j":env.battery_j.tolist(),
              "final_positions_m":env.positions.tolist(),"network_stats":dict(env.net.stats),
              "scenario":str(scenario.relative_to(ROOT)),"scenario_sha256":hashlib.sha256(scenario.read_bytes()).hexdigest(),
              "source_sha256":source_sha,"source_manifest":manifest,"trace":str(trace_path),
              "elapsed_s":time.monotonic()-started}
    (out / f"{stem}_result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"label":LABEL,"profile":profile,"seed":seed,"finished":True,
                      "success_witness":success,"steps":env.t,"gcs_tp":result["gcs_tp_ever"],"error":error}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[44,45,46,47,48])
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--workers", type=int, choices=[1,2], default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/v121_scenario_audit/constructive_witness")
    args = parser.parse_args()
    if not 1 <= args.steps <= 900:
        parser.error("--steps must be 1..900; no resets or horizon padding")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    jobs = [(p,s,args.steps,str(args.output_dir)) for p in args.profiles for s in args.seeds]
    results = []
    if args.workers == 1:
        results = [run_world(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_world,*job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    summary = {"label":LABEL,"all_success_witnesses":all(r["success_witness"] for r in results),
               "claim":"Finite empirical feasibility examples only. Failures retained. No formal bound.",
               "worlds":len(results),"results":[{k:r[k] for k in ("profile","seed","success_witness","steps","gcs_tp_ever","safety_failure","energy_failure","error","scenario_sha256","source_sha256")} for r in results]}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)
    return 0 if summary["all_success_witnesses"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
