"""Read-only 10-run W&B experiment audit, preserving exact episode histories.

The fetched run IDs are pinned in the resulting JSON. No training, checkpoint,
W&B run, or source configuration is modified. Never export credentials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

SCENARIO_FIELDS = ("world", "sensor", "network", "energy", "reward")
TRACKING_FIELDS = ("evaluation_interval_steps", "periodic_evaluation_episodes",
                   "periodic_evaluation_seed", "evaluation_interval", "evaluation_timeout_seconds")
RUNTIME_FIELDS = (
    "seed", "episodes", "num_envs", "deterministic",
    "environment_backend", "replay_device", "replay_cpu_fraction",
    "critic_chunk_size", "vectorize_critics", "compile_models",
    "amp_mode", "fused_optimizers",
)
METRIC_KEYS = (
    "episode/index", "episode/return", "outcome/search_rate_percent",
    "outcome/delivery_rate_percent", "global_step", "episode/length",
)
EVAL_KEYS = (
    "eval/mean_delivery_rate_percent", "eval/mean_search_rate_percent",
    "eval/mean_return", "evaluation/checkpoint_global_step", "global_step",
)
CASE_SPEC = {
    1: (True, 512, 8, True),
    2: (True, 512, 16, True),
    3: (False, 512, 16, True),
    4: (False, 512, 16, False),
    5: (True, 256, 8, True),
}
PAIR_DESIGNS = ((1, 2, "updates8_vs16"), (1, 5, "width512_vs256"),
                (2, 3, "shared_vs_independent"), (3, 4, "actor_layernorm_vs_plain"))
MILESTONES = (4096, 8192, 10000, 11675, 12000, 15000, 16384, 20000)


def stable_hash(data):
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), default=str).encode()
    ).hexdigest()


def run_case(config):
    rt = config.get("runtime", {})
    algorithm = rt.get("algorithm")
    if algorithm not in ("masac", "matd3"):
        raise ValueError(f"unexpected algorithm: {algorithm}")
    settings = config.get(algorithm, {})
    shared = (settings.get("share_actor_parameters"),
              settings.get("share_critic_parameters"))
    if shared[0] is not shared[1]:
        raise ValueError("Actor/critic sharing mismatch")
    hidden = settings.get("hidden_sizes", [])
    if len(hidden) != 2 or hidden[0] != hidden[1]:
        raise ValueError("Unsupported hidden widths")
    spec = (shared[0], int(hidden[0]), int(settings["updates_per_step"]),
            bool(settings.get("actor_layer_norm", False)))
    matches = [index for index, item in CASE_SPEC.items() if item == spec]
    if len(matches) != 1:
        raise ValueError(f"Unrecognized case: {spec}")
    return algorithm, matches[0]


def extract_episode_history(run):
    rows = []
    for record in run.scan_history(keys=list(METRIC_KEYS), page_size=2500):
        if record.get("episode/index") is None:
            continue
        if any(record.get(key) is None for key in METRIC_KEYS):
            raise ValueError(f"{run.id}: episode missing required metrics")
        rows.append(tuple(float(record[key]) for key in METRIC_KEYS))
    if not rows:
        raise ValueError(f"{run.id}: no episode data")
    result = np.asarray(rows, dtype=np.float64)
    result = result[np.argsort(result[:, 0], kind="stable")]
    idx = result[:, 0].astype(np.int64)
    if not np.array_equal(idx, np.arange(1, idx[-1] + 1)):
        raise ValueError(f"{run.id}: missing/duplicate completed episodes")
    if not np.isfinite(result).all():
        raise ValueError(f"{run.id}: non-finite episode history")
    return result


def extract_eval_history(run):
    df = run.history(samples=10000, keys=list(EVAL_KEYS))
    items = []
    for record in df.to_dict("records"):
        if any(record.get(key) is None for key in EVAL_KEYS):
            continue
        record = {k: float(record[k]) for k in EVAL_KEYS}
        if not all(np.isfinite(v) for v in record.values()):
            continue
        items.append({
            "checkpoint_global_step": int(record["evaluation/checkpoint_global_step"]),
            "logged_global_step": int(record["global_step"]),
            "delivery": record["eval/mean_delivery_rate_percent"],
            "search": record["eval/mean_search_rate_percent"],
            "return": record["eval/mean_return"],
        })
    items.sort(key=lambda x: x["checkpoint_global_step"])
    distinct = {}
    for item in items:
        # One entry per checkpoint; latest log of the same eval wins.
        distinct[item["checkpoint_global_step"]] = item
    return list(distinct.values())


def training_slice(rows, episode, tail=1024):
    episode = int(episode)
    if episode < tail or episode > len(rows):
        return None
    subset = rows[episode - tail:episode]
    assert subset[-1, 0] == episode
    return {
        "episode": episode, "window": tail,
        "return": float(subset[:, 1].mean()),
        "return_std": float(subset[:, 1].std()),
        "search": float(subset[:, 2].mean()),
        "delivery": float(subset[:, 3].mean()),
        "global_step_at_episode": int(rows[episode - 1, 4]),
        "episode_length_mean": float(subset[:, 5].mean()),
        "cumulative_return": float(rows[:episode, 1].mean()),
        "cumulative_search": float(rows[:episode, 2].mean()),
        "cumulative_delivery": float(rows[:episode, 3].mean()),
    }


def matched_pair(a, b):
    budget = min(len(a["rows"]), len(b["rows"]))
    window = min(1024, budget)
    left = training_slice(a["rows"], budget, window)
    right = training_slice(b["rows"], budget, window)
    ea = {item["checkpoint_global_step"]: item for item in a["eval_history"]}
    eb = {item["checkpoint_global_step"]: item for item in b["eval_history"]}
    common = sorted(set(ea) & set(eb))
    v = None
    if common:
        c = common[-1]
        v = {"checkpoint_global_step": c, "left": ea[c], "right": eb[c],
             "delta_delivery_left_minus_right": ea[c]["delivery"] - eb[c]["delivery"],
             "delta_return_left_minus_right": ea[c]["return"] - eb[c]["return"]}
    return {
        "episode_cutoff": budget, "window": window,
        "left": left, "right": right,
        "delta_delivery_left_minus_right": left["delivery"] - right["delivery"],
        "delta_return_left_minus_right": left["return"] - right["return"],
        "validation_at_latest_shared_checkpoint": v,
    }


def build_snapshot(runs, dataset):
    fingerprints = {v["scenario_hash"] for v in dataset.values()}
    source_commits = {v["source_git_commit"] for v in dataset.values()}
    evaluation_hashes = {v["evaluation_protocol_hash"] for v in dataset.values()}
    if len(source_commits) != 1 or None in source_commits:
        raise ValueError("Code revision unknown/different: reject unified source comparison")
    if len(evaluation_hashes) != 1:
        raise ValueError("Evaluation seed bank/cadence differs")
    if len(fingerprints) != 1:
        raise ValueError("Scenario/reward/sensing/network/energy config differs: reject cross-run comparison")
    expected = {(algorithm, case) for algorithm in ("matd3", "masac") for case in CASE_SPEC}
    if set(dataset) != expected:
        raise ValueError(f"Matrix mismatch {set(dataset)^expected}")
    common_ep = min(len(d["rows"]) for d in dataset.values())
    intersection = set.intersection(
        *(set(i["checkpoint_global_step"] for i in d["eval_history"])
          for d in dataset.values())
    ) if dataset else set()
    common_eval = max(intersection) if intersection else None
    pairwise = {}
    for algorithm in ("matd3", "masac"):
        for l, r, desc in PAIR_DESIGNS:
            pairwise[f"{algorithm}_{l:02d}_vs_{r:02d}_{desc}"] = matched_pair(
                dataset[(algorithm, l)], dataset[(algorithm, r)])
    for case in CASE_SPEC:
        pairwise[f"cross_algorithm_case_{case:02d}"] = matched_pair(
            dataset[("matd3", case)], dataset[("masac", case)])
    return {
        "capture_date_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "10 newest W&B runs by created_at, exact 5x2 35T25O BASE matrix",
        "project": "truongdaoanhduy-uit/uav_search_target",
        "protocol": {
            "scenario_sha256": next(iter(fingerprints)),
            "source_git_commit": next(iter(source_commits)),
            "evaluation_protocol_sha256": next(iter(evaluation_hashes)),
            "observation_version": next(iter({v["observation_version"] for v in dataset.values()})),
            "all_run_common_episode_cutoff": common_ep,
            "all_run_shared_eval_checkpoint_global_step": common_eval,
            "training_window": "last 1024 COMPLETED TRAINING EPISODES before inclusive episode cutoff",
            "episode_metrics_not_fixed_policy": True,
            "evaluation_metrics_fixed_seed_bank_and_policy": True,
            "early_stopping_not_imputed": True,
            "best_checkpoint_not_used_as_fair_same_budget_ranking": True,
        },
        "runs": [{
            "algorithm": alg, "case": case, "id": d["id"],
            "name": d["name"], "created_at": d["created_at"],
            "state": d["state"], "configured_episodes": d["planned_episodes"],
            "observed_episodes": len(d["rows"]),
            "last_episode_global_step": int(d["rows"][-1, 4]),
            "summary_global_step": d["summary_global_step"],
            "best_checkpoint_step": d["best_step"],
            "best_validation": d["best_validation"],
            "best_checkpoint_storage": d["best_checkpoint_storage"],
            "runtime_seconds": d["runtime_seconds"],
            "scenario_sha256": d["scenario_hash"],
            "runtime_protocol_sha256": d["runtime_hash"],
            "case_fingerprint_sha256": d["case_fingerprint"],
            "source_git_commit": d["source_git_commit"],
            "evaluation_protocol": d["evaluation_protocol"],
            "scientific_config": d["scientific_config"],
            "training_milestones": {
                str(m): training_slice(d["rows"], m)
                for m in MILESTONES if m <= len(d["rows"])
            },
            "training_at_common_episode": training_slice(d["rows"], common_ep),
            "validation": d["eval_history"],
            "validation_at_common_checkpoint": next((
                item for item in d["eval_history"]
                if item["checkpoint_global_step"] == common_eval
            ), None),
            "url": f"https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/{d['id']}",
        } for (alg, case), d in sorted(dataset.items())],
        "pairwise": pairwise,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="truongdaoanhduy-uit/uav_search_target")
    parser.add_argument("--key-file", default="/home/aduy/Documents/NCKH/uav_network/key.txt")
    parser.add_argument("--output-dir", default="outputs/wandb-ten-20261108")
    parser.add_argument("--snapshot", default="docs/methods/evidence/wandb_latest_ten_2026-11-08.json")
    parser.add_argument("--ids", nargs="*", default=[],
                        help="Optional pinned run IDs; default selects newest ten (capture only)")
    args = parser.parse_args()
    import wandb
    from uav_network.configuration import load_config
    from uav_network.experiments.dashboard import _authentication
    profile = load_config("configs/comparison-independent-35t25o-h512-xyz-sensing.yaml",
                          [f"tracking.credentials_file={args.key_file}"])
    entity, project = _authentication(profile)
    if f"{entity}/{project}" != args.project:
        raise ValueError("W&B destination does not match requested project")
    api = wandb.Api(timeout=35)
    records = (
        [api.run(f"{args.project}/{rid}") for rid in args.ids]
        if args.ids else
        list(r for idx, r in enumerate(api.runs(args.project, order="-created_at", per_page=10)) if idx < 10)
    )
    if len(records) != 10:
        raise ValueError(f"Expected ten runs, got {len(records)}")
    seen = set()
    data = {}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for r in records:
        cfg = r.config or {}
        algorithm, case = run_case(cfg)
        if (algorithm, case) in seen:
            raise ValueError(f"Duplicate experiment case {algorithm} {case}")
        seen.add((algorithm, case))
        if cfg.get("pql", {}).get("enabled", False):
            raise ValueError("PQL must not enter BASE comparison")
        scenario = {key: cfg.get(key) for key in SCENARIO_FIELDS}
        scientific = {key: cfg.get(key) for key in SCENARIO_FIELDS}
        scientific["continuous_dim"] = cfg.get("continuous_dim")
        scientific["motion_dim"] = cfg.get("motion_dim")
        protocol = {k: cfg.get("runtime", {}).get(k) for k in RUNTIME_FIELDS}
        eval_protocol = {k: cfg.get("tracking", {}).get(k) for k in TRACKING_FIELDS}
        eval_protocol.update({
            "eval_seed": cfg.get("runtime", {}).get("eval_seed"),
            "eval_episodes": cfg.get("runtime", {}).get("eval_episodes")})
        source_commit = cfg.get("source_git_commit")
        case_fingerprint = stable_hash({
            "source_git_commit": source_commit,
            "observation_version": cfg.get("observation_version"),
            "scenario": scientific, "runtime": protocol,
            "evaluation": eval_protocol,
            "algorithm": algorithm, "active_algorithm": cfg.get(algorithm),
            "training_mode": cfg.get("training_mode"),
            "pql": cfg.get("pql", {}).get("enabled")})
        cache = output / f"{r.id}_episode_history.npz"
        if cache.exists():
            with np.load(cache) as archived:
                rows = archived["rows"]
        else:
            rows = extract_episode_history(r)
            np.savez_compressed(cache, rows=rows)
        evaluations = extract_eval_history(r)
        summ = dict(r.summary._json_dict)
        if len(rows) != int(summ.get("episodes_completed", len(rows))):
            raise ValueError(f"{r.id}: W&B summary episodes mismatch exact history")
        data[(algorithm, case)] = {
            "rows": rows, "id": r.id, "name": r.name,
            "state": r.state, "created_at": r.created_at,
            "planned_episodes": int(cfg.get("runtime", {}).get("episodes", 0)),
            "summary_global_step": summ.get("global_step"),
            "runtime_seconds": summ.get("_runtime"),
            "best_step": summ.get("best/checkpoint_global_step"),
            "best_validation": {
                "delivery": summ.get("best/mean_delivery_rate_percent"),
                "search": summ.get("best/mean_search_rate_percent"),
                "return": summ.get("best/mean_return"),
            },
            "best_checkpoint_storage": {
                "backend": summ.get("best_checkpoint_storage.backend"),
                "repo_id": summ.get("best_checkpoint_storage.repo_id"),
                "revision": summ.get("best_checkpoint_storage.revision"),
                "sha256": summ.get("best_checkpoint_storage.sha256"),
                "checkpoint_config_sha256": summ.get("checkpoint_selection.config_sha256"),
                "episode": summ.get("best/checkpoint_episode"),
            },
            "scientific_config": {
                "world": {k: cfg.get("world", {}).get(k) for k in
                          ("num_uavs", "num_targets", "num_obstacles", "map_size_m", "max_steps")},
                "actor_layer_norm": cfg[algorithm].get("actor_layer_norm"),
                "share_actor_parameters": cfg[algorithm].get("share_actor_parameters"),
                "share_critic_parameters": cfg[algorithm].get("share_critic_parameters"),
                "hidden_sizes": cfg[algorithm].get("hidden_sizes"),
                "updates_per_step": cfg[algorithm].get("updates_per_step"),
                "batch_size": cfg[algorithm].get("batch_size"),
                "replay_capacity": cfg[algorithm].get("replay_capacity"),
                "seed": cfg.get("runtime", {}).get("seed"),
            },
            "scenario_hash": stable_hash(scientific),
            "runtime_hash": stable_hash(protocol),
            "evaluation_protocol": eval_protocol,
            "evaluation_protocol_hash": stable_hash(eval_protocol),
            "source_git_commit": source_commit,
            "observation_version": cfg.get("observation_version"),
            "case_fingerprint": case_fingerprint,
            "eval_history": evaluations,
        }
        print("CAPTURED", algorithm, case, r.id, "ep", len(rows),
              "eval", len(evaluations), "state", r.state, flush=True)
    snapshot = build_snapshot(records, data)
    target = Path(args.snapshot)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    csv_target = output / "matched_10_case_comparison.csv"
    with csv_target.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "algorithm", "case", "run_id", "observed_episodes", "common_episode",
            "common_window_return", "common_window_search", "common_window_delivery",
            "common_eval_step", "common_eval_return", "common_eval_search", "common_eval_delivery",
            "best_eval_delivery", "best_checkpoint_step",
        ])
        writer.writeheader()
        for d in snapshot["runs"]:
            t = d["training_at_common_episode"]
            ev = d["validation_at_common_checkpoint"] or {}
            writer.writerow({
                "algorithm": d["algorithm"], "case": d["case"],
                "run_id": d["id"], "observed_episodes": d["observed_episodes"],
                "common_episode": t["episode"],
                "common_window_return": t["return"], "common_window_search": t["search"],
                "common_window_delivery": t["delivery"],
                "common_eval_step": ev.get("checkpoint_global_step"),
                "common_eval_return": ev.get("return"),
                "common_eval_search": ev.get("search"),
                "common_eval_delivery": ev.get("delivery"),
                "best_eval_delivery": d["best_validation"]["delivery"],
                "best_checkpoint_step": d["best_checkpoint_step"],
            })
    print("CAPTURE_COMPLETE", "all episode", snapshot["protocol"]["all_run_common_episode_cutoff"],
          "eval checkpoint", snapshot["protocol"]["all_run_shared_eval_checkpoint_global_step"],
          "json", target, "csv", csv_target, flush=True)


if __name__ == "__main__":
    main()
