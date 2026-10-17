"""Paired checkpoint comparisons; uncertainty covers test scenarios, not training seeds."""
import csv
import json
from pathlib import Path
import numpy as np

from .comparison_metrics import METRICS


def compare_evaluations(left, right, bootstrap_samples=2000, seed=44):
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be an integer >=100")
    protocols = [r.get("protocol", {}) for r in (left, right)]
    if not protocols[0].get("scenario_hash") or protocols[0]["scenario_hash"] != protocols[1].get("scenario_hash"):
        raise ValueError("Comparison requires the same resolved scenario and environment backend")
    rows = []
    for report in (left, right):
        records = report.get("records", [])
        if not records or any("comparison_metrics" not in r for r in records):
            raise ValueError("Re-evaluate checkpoints to create mission comparison metrics")
        if any(type(r.get("seed")) is not int for r in records):
            raise ValueError("Evaluation seeds must be integers")
        by_seed = {r["seed"]: r["comparison_metrics"] for r in records}
        if len(by_seed) != len(records):
            raise ValueError("Evaluation seeds must be unique")
        rows.append(by_seed)
    if rows[0].keys() != rows[1].keys():
        raise ValueError("Paired comparison requires exactly the same evaluation seeds")
    seeds = sorted(rows[0])
    n = len(seeds)
    rng = np.random.default_rng(seed)
    indices = rng.integers(n, size=(bootstrap_samples, n))
    overlap = [p.get("training_seed_overlap") for p in protocols]
    validation_overlap = [p.get("validation_seed_overlap") for p in protocols]
    exploratory = n < 2 or any(value is not False for value in overlap + validation_overlap)
    result = {"status": "exploratory" if exploratory else "paired_scenario_comparison",
              "left_algorithm": left["algorithm"], "right_algorithm": right["algorithm"],
              "paired_episodes": n, "seeds": seeds, "bootstrap_samples": bootstrap_samples,
              "bootstrap_seed": seed, "confidence": .95, "training_seed_overlap": overlap,
              "validation_seed_overlap": validation_overlap,
              "scenario_hash": protocols[0]["scenario_hash"],
              "training_global_steps": [p.get("training_global_step") for p in protocols],
              "uncertainty_scope": "scenario variation conditional on these checkpoints; independent training runs still required",
              "overall_winner": None, "metrics": {}}
    budgets = result["training_global_steps"]
    result["equal_environment_interaction_budget"] = budgets[0] == budgets[1] if None not in budgets else None
    for name, (unit, preferred) in METRICS.items():
        pairs = [(rows[0][s].get(name), rows[1][s].get(name)) for s in seeds]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        values = np.asarray(pairs, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite metric: {name}")
        diff = values[:, 0] - values[:, 1]
        count = len(diff)
        interval = None
        if count > 1:
            # Conditional metrics can omit failures; expose their paired count.
            sample = indices if count == n else rng.integers(count, size=(bootstrap_samples, count))
            interval = list(map(float, np.quantile(diff[sample].mean(axis=1), [.025, .975])))
        favored = "inconclusive"
        if interval and not exploratory and preferred in {"higher", "lower"}:
            signed = np.asarray(interval) * (1 if preferred == "higher" else -1)
            if signed.min() > 1e-12:
                favored = "left"
            elif signed.max() < -1e-12:
                favored = "right"
        result["metrics"][name] = {"left_mean": float(values[:, 0].mean()),
                                   "right_mean": float(values[:, 1].mean()),
                                   "difference_left_minus_right": float(diff.mean()),
                                   "difference_ci95": interval, "paired_episodes": count,
                                   "conditional_subset": count != n, "unit": unit,
                                   "preferred": preferred, "favored": favored}
    energy = np.array([[rows[side][s]["uav_energy_j"] for s in seeds] for side in (0, 1)])
    deliveries = np.array([[rows[side][s]["targets_delivered"] for s in seeds] for side in (0, 1)])
    denominator = deliveries.sum(axis=1)
    pooled = {"left": float(energy[0].sum() / denominator[0]) if denominator[0] else None,
              "right": float(energy[1].sum() / denominator[1]) if denominator[1] else None,
              "unit": "J/delivered target", "difference_ci95": None}
    if n > 1 and denominator.min() > 0:
        den = deliveries[:, indices].sum(axis=2)
        valid = (den > 0).all(axis=0)
        pooled["valid_bootstrap_samples"] = int(valid.sum())
        # Undefined zero-delivery replicates are disclosed rather than replaced with zero.
        if valid.all():
            ratios = energy[:, indices].sum(axis=2) / den
            pooled["difference_ci95"] = list(map(float, np.quantile(ratios[0] - ratios[1], [.025, .975])))
    result["pooled_uav_j_per_gcs_delivery"] = pooled
    return result


def save_comparison(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    fields = ("metric", "left_mean", "right_mean", "difference_left_minus_right", "difference_ci95",
              "paired_episodes", "conditional_subset", "unit", "preferred", "favored")
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for name, values in report["metrics"].items():
            writer.writerow({"metric": name, **values})
    return report


def compare_checkpoints(left, right, output, episodes=30, seed=2000000044,
                        bootstrap_samples=2000, animate=False):
    from .evaluation import evaluate
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Comparison folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reports = [evaluate(path, episodes=episodes, seed=seed, output_dir=output / side, animate=animate)
               for path, side in ((left, "left"), (right, "right"))]
    result = compare_evaluations(*reports, bootstrap_samples=bootstrap_samples)
    result["evaluations"] = {side: str(output / side / "metrics.json") for side in ("left", "right")}
    return save_comparison(result, output)
