"""Compare MATD3/MASAC runs without mismatching evaluation seeds or budgets.

Exact matched evaluation provenance is required. A run with only a final
summary cannot be treated as comparable to an earlier active checkpoint.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from .cause_evidence_report import analyze, finite


SCIENTIFIC_KEYS = (
    "algorithm", "pql_enabled", "batch", "updates_per_collection",
    "num_envs", "replay_capacity", "replay_cpu_fraction", "shared_actor",
    "shared_critic", "actor_layernorm", "grad_clip_mode", "grad_clip_norm",
    "output_normalization", "vectorize_critics", "critic_chunk_size",
    "amp_mode", "compile_models", "training_seed",
    "targets", "uavs", "map_size_m", "max_steps", "motion_safety",
    "verified_empty_belief", "belief_patch_cells", "belief_coarse_cells",
    "report_buffer_bytes", "report_payload_bytes", "report_ttl_s",
    "power_control", "receiver_admission_mask",
    "reward_unsafe_action_coefficient", "reward_shield_correction_coefficient",
    "reward_shield_intervention_coefficient",
    "reward_information_gain_coefficient", "reward_investigation_coefficient",
)


def _checkpoints(folder):
    rows = []
    for path in sorted((Path(folder) / "evaluations").glob("*/metrics.json")):
        try:
            data = json.loads(path.read_text())
            protocol = data.get("protocol")
            if not isinstance(protocol, dict):
                continue
            step = protocol.get("training_global_step")
            seeds = protocol.get("seeds")
            scenario = protocol.get("scenario_hash")
            n = data.get("episodes")
            if (type(step) is not int or type(n) is not int or n < 1
                    or not isinstance(scenario, str) or len(scenario) != 64
                    or not isinstance(seeds, list) or len(seeds) != n
                    or not all(type(seed) is int for seed in seeds)
                    or not all(finite(data.get(x)) for x in (
                        "mean_delivery_rate_percent", "mean_search_rate_percent",
                        "mean_return", "success_rate"))):
                continue
            if protocol.get("training_seed_overlap") is True:
                # This is *training*-exposed; include with explicit warning.
                leakage = "training_seed_overlap"
            elif protocol.get("validation_seed_overlap") is True:
                leakage = "validation_selection_seed_overlap"
            else:
                leakage = "no_overlap_reported" if (
                    protocol.get("training_seed_overlap") is False and
                    protocol.get("validation_seed_overlap") is False) else "unverified_overlap"
            rows.append({
                "global_step": step, "n_eval_episodes": n,
                "scenario_hash": scenario, "seeds": seeds,
                "delivery_percent": float(data["mean_delivery_rate_percent"]),
                "search_percent": float(data["mean_search_rate_percent"]),
                "mean_return": float(data["mean_return"]),
                "success_fraction": float(data["success_rate"]),
                "seed_exposure": leakage,
                "checkpoint_source": str(path),
                "checkpoint_sha256": data.get("policy_provenance", {}).get("checkpoint_sha256"),
            })
        except (OSError, ValueError, TypeError):
            continue
    return rows


def compare(named_folders):
    if len(named_folders) != len(set(named_folders)):
        raise ValueError("Duplicate run labels")
    individual = {}
    for name, folder in named_folders.items():
        if not Path(folder).is_dir():
            raise NotADirectoryError(folder)
        report = analyze(folder)
        config = report.get("configuration", {})
        individual[name] = {
            "source_folder": str(folder),
            "evidence_coverage": report["coverage"],
            "episodes": report["episodes"],
            "configuration": config,
            "performance": report.get("performance"),
            "evaluations": _checkpoints(folder),
        }
    paired = []
    for left, right in combinations(individual, 2):
        a = individual[left]["configuration"].get("fields", {})
        b = individual[right]["configuration"].get("fields", {})
        missing = [k for k in SCIENTIFIC_KEYS if a.get(k) is None or b.get(k) is None]
        diffs = [k for k in SCIENTIFIC_KEYS if k not in missing and a[k] != b[k]]
        # User-facing configuration attribution is conditional on full source
        # files and scenario equivalence; missing config is not 'identical'.
        paired.append({
            "left": left, "right": right,
            "scientific_controls_changed": diffs,
            "scientific_controls_unknown": missing,
            "single_dimension_ablation": len(diffs) == 1 and not missing,
            "changed_dimension": diffs[0] if len(diffs) == 1 and not missing else None,
            "interpretation": ("Single configured dimension changed, but seed/initialization and source equivalence still require verification"
                               if len(diffs) == 1 and not missing else
                               "Not an isolated config intervention; confounders or missing controls remain"),
        })
    groups = defaultdict(list)
    for name, row in individual.items():
        for item in row["evaluations"]:
            # Exact match on scientific scenario, held-out or validation seed
            # bank, and amount of collected experience. Do not nearest-match.
            groups[(item["global_step"], item["scenario_hash"],
                    tuple(item["seeds"]))].append((name, item))
    matched = []
    for (step, scenario, seeds), rows in sorted(groups.items()):
        if len({name for name, _ in rows}) < 2:
            continue
        matched.append({
            "global_step": step, "scenario_hash": scenario,
            "evaluation_first_seed": seeds[0], "evaluation_last_seed": seeds[-1],
            "evaluation_episodes": len(seeds),
            "checkpoint_samples": [
                {"run": name, "delivery_percent": row["delivery_percent"],
                 "search_percent": row["search_percent"],
                 "mean_return": row["mean_return"],
                 "success_fraction": row["success_fraction"],
                 "seed_exposure": row["seed_exposure"],
                 "checkpoint_sha256": row["checkpoint_sha256"],
                 "source": row["checkpoint_source"]}
                for name, row in rows],
            "purpose": "Exact matched evaluation sample; 5 reused validation seeds are not independent test evidence",
        })
    return {
        "schema": 1, "comparison_policy": "exact scenario, eval seed bank, and global_step; never interpolate",
        "runs": individual, "configured_pairs": paired,
        "matched_evaluations": matched,
        "matched_evaluation_count": len(matched),
        "limitations": [
            "No exact checkpoint comparison if evaluation records or seed provenance are missing",
            "Runtime/LN/sharing controls without source code SHA do not confirm identical implementation",
            "Repeated use of the same 5 validation seeds biases checkpoint selection",
            "A single training seed is not evidence of cross-seed algorithm robustness",
            "High throughput and delivery are different objectives",
            "A numeric Q/CBF symptom does not establish its causal effect on delivery",
        ],
    }


def render(result):
    lines = ["# Cross-run UAV study: matched budget and seed evidence", "",
             f"Matched evaluations: **{result['matched_evaluation_count']}**", "",
             "## Configured comparisons", "",
             "| Run A | Run B | Scientific parameters changed | Isolated? |",
             "|---|---|---|---|"]
    for row in result["configured_pairs"]:
        changed = ", ".join(row["scientific_controls_changed"]) or "none confirmed"
        if row["scientific_controls_unknown"]:
            changed += f"; {len(row['scientific_controls_unknown'])} unknown"
        lines.append(f"| {row['left']} | {row['right']} | {changed} | {row['single_dimension_ablation']} |")
    lines += ["", "## Exactly matched evaluation banks", ""]
    if not result["matched_evaluations"]:
        lines.append("No exact common scenario + global_step + seed bank is available; comparison unavailable, not zero.")
    for sample in result["matched_evaluations"]:
        lines.append(f"### {sample['global_step']:,} joint transitions, seeds {sample['evaluation_first_seed']}–{sample['evaluation_last_seed']}")
        lines.append("| Run | Delivery % | Search % | Success fraction | Seed exposure |")
        lines.append("|---|---:|---:|---:|---|")
        for x in sample["checkpoint_samples"]:
            lines.append(f"| {x['run']} | {x['delivery_percent']:.3f} | {x['search_percent']:.3f} | {x['success_fraction']:.3f} | {x['seed_exposure']} |")
        lines.append("")
    lines += ["", "## Scientific limitations", ""]
    lines += ["- " + text for text in result["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True,
                    help="Run name=LOCAL_OUTPUT_FOLDER; provide at least two")
    ap.add_argument("--output", required=True,
                    help="Output folder for JSON/Markdown; must be empty")
    args = ap.parse_args(argv)
    if len(args.run) < 2:
        ap.error("Pass two or more --run name=path arguments")
    runs = {}
    for value in args.run:
        if "=" not in value:
            ap.error("--run must be name=path")
        name, folder = value.split("=", 1)
        if not name.strip() or not folder.strip() or name in runs:
            ap.error("--run names must be nonempty and unique")
        runs[name] = folder
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        ap.error("Output path must be empty")
    result = compare(runs)
    output.mkdir(parents=True, exist_ok=True)
    (output/"matched_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (output/"matched_comparison.md").write_text(render(result))
    print(json.dumps({"runs": len(runs), "matched_evaluations": result["matched_evaluation_count"],
                      "output": str(output)}))


if __name__ == "__main__":
    main()
