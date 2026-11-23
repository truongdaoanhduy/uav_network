"""Offline report/figures/registry from an immutable 10-run W&B JSON snapshot.

No access to W&B, secrets or private training code is needed to regenerate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = {
    1: "Shared · H512 · U8 · LN",
    2: "Shared · H512 · U16 · LN",
    3: "Independent · H512 · U16 · LN",
    4: "Independent · H512 · U16 · NoLN",
    5: "Shared · H256 · U8 · LN",
}


def fmt(value, digits=1):
    return f"{value:.{digits}f}" if value is not None else "—"


def plot_eval_curves(snapshot, algorithm, output):
    fig, ax = plt.subplots(figsize=(11.5, 6.3), layout="constrained")
    runs = [r for r in snapshot["runs"] if r["algorithm"] == algorithm]
    for record in sorted(runs, key=lambda r: r["case"]):
        x = np.array([e["checkpoint_global_step"] for e in record["validation"]]) / 1e6
        y = np.array([e["delivery"] for e in record["validation"]])
        ax.plot(x, y, marker="o", markersize=4, linewidth=1.7,
                label=f"{record['case']:02d}: {CASES[record['case']]} (ep {record['observed_episodes']:,})")
    common = snapshot["protocol"]["all_run_shared_eval_checkpoint_global_step"] / 1e6
    ax.axvline(common, linestyle="--", linewidth=1, label=f"All-10 matched checkpoint {common:.2f}M")
    ax.set_title(f"{algorithm.upper()} BASE — fixed-policy validation delivery, seed bank 1000000044–48")
    ax.set_xlabel("Checkpoint global transitions (millions)")
    ax.set_ylabel("Delivery rate (%)")
    ax.set_ylim(0, 70)
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(output, dpi=165)
    plt.close(fig)


def plot_10_case_common_eval(snapshot, output):
    runs = snapshot["runs"]
    labels = [f"{'M' if r['algorithm']=='matd3' else 'S'}{r['case']:02d}" for r in runs]
    values = [r["validation_at_common_checkpoint"]["delivery"] for r in runs]
    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    bars = ax.bar(labels, values)
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, value + .6,
                f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(values) + 12)
    ax.set_ylabel("Validation delivery (%)")
    ax.set_title("All ten models at exactly 7,372,800 checkpoint transitions (5 episodes)")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.savefig(output, dpi=165)
    plt.close(fig)


def rows_for_table(snapshot):
    lines = []
    for run in sorted(snapshot["runs"], key=lambda r: (0 if r["algorithm"]=="matd3" else 1, r["case"])):
        ev = run["validation_at_common_checkpoint"]
        tr = run["training_at_common_episode"]
        url = run["url"]
        lines.append(
            f"| [{run['algorithm'].upper()} {run['case']:02d}]({url}) "
            f"| {CASES[run['case']]} | {run['observed_episodes']:,} / {run['configured_episodes']:,} "
            f"| {fmt(run['runtime_seconds']/3600,2)} "
            f"| {fmt(tr['delivery'],2)}% | {fmt(ev['delivery'],2)}% "
            f"| {fmt(run['best_validation']['delivery'],2)}% |"
        )
    return "\n".join(lines)


def pairwise_table(snapshot, algorithm):
    lines = []
    for prefix in (f"{algorithm}_01_vs_02", f"{algorithm}_01_vs_05",
                   f"{algorithm}_02_vs_03", f"{algorithm}_03_vs_04"):
        name = next(k for k in snapshot["pairwise"] if k.startswith(prefix))
        pair = snapshot["pairwise"][name]
        ev = pair["validation_at_latest_shared_checkpoint"]
        comparison = name.split("_vs_")
        rcase = comparison[1][:2]
        lcase = prefix.split("_")[1]
        lines.append(
            f"| {lcase} vs {rcase} | {pair['episode_cutoff']:,} "
            f"| {fmt(pair['delta_delivery_left_minus_right'],2)} "
            f"| {fmt(pair['delta_return_left_minus_right'],1)} "
            f"| {(fmt(ev['checkpoint_global_step']/1e6,3)+'M') if ev else '—'} "
            f"| {fmt(ev['delta_delivery_left_minus_right'],2) if ev else '—'} "
            f"| {fmt(ev['delta_return_left_minus_right'],1) if ev else '—'} |"
        )
    return "\n".join(lines)


def registry(snapshot):
    p = snapshot["protocol"]
    return {
        "type": "uav_marl_completed_experiment_registry",
        "captured_at_utc": snapshot["capture_date_utc"],
        "source_code_commit": p["source_git_commit"],
        "scenario_sha256": p["scenario_sha256"],
        "evaluation_protocol_sha256": p["evaluation_protocol_sha256"],
        "observations_version": p["observation_version"],
        "run_count": len(snapshot["runs"]),
        "full_20000": sum(r["observed_episodes"] >= 20000 for r in snapshot["runs"]),
        "partial": sum(r["observed_episodes"] < 20000 for r in snapshot["runs"]),
        "do_not_repeat_identical_training": True,
        "reuse_rule": (
            "Existing run is reusable if source behavior, scenario/world/generation, "
            "sensing/belief, communications/energy, reward/CBF, action/observation, "
            "algorithm/architecture/optimizer/replay, seed and evaluation protocol "
            "remain equivalent. Comment-only commits are not scientific changes. "
            "Do not re-launch completed case solely because a new chat starts."
        ),
        "invalidate_if_scientific_change": [
            "Policy/critic learning equations, hybrid Gumbel or action semantics",
            "Actor/critic sharing, hidden width, LayerNorm, optimizer schedule U8/U16",
            "Scenario/map, targets/obstacles, generation/sensor/belief",
            "Reward, CBF/safety, energy, networking or observation/global state",
            "Replay semantics, batch size/capacity, seed or evaluation seed/metric protocol",
        ],
        "partial_handling": (
            "A finished W&B run with <20000 episodes is valid partial evidence, "
            "not a completed 20k training. Do not restart automatically; only "
            "resume/extend after explicit research need and checkpoint verification."
        ),
        "runs": [
            {
                "algorithm": r["algorithm"], "case": r["case"], "run_id": r["id"],
                "run_url": r["url"], "case_name": CASES[r["case"]],
                "observed_episodes": r["observed_episodes"],
                "target_episodes": r["configured_episodes"],
                "result_status": ("complete_20000" if r["observed_episodes"] >= 20000
                                  else "partial_evidence"),
                "source_git_commit": r["source_git_commit"],
                "case_fingerprint_sha256": r["case_fingerprint_sha256"],
                "scenario_sha256": r["scenario_sha256"],
                "runtime_protocol_sha256": r["runtime_protocol_sha256"],
                "best_delivery_pct": r["best_validation"]["delivery"],
                "best_checkpoint_storage": r["best_checkpoint_storage"],
                "common_eval_delivery_pct": r["validation_at_common_checkpoint"]["delivery"],
                "common_eval_step": r["validation_at_common_checkpoint"]["checkpoint_global_step"],
            } for r in snapshot["runs"]
        ],
        "primary_result": {
            "algorithm": "matd3", "case": 1,
            "reason": "Best observed fixed-policy delivery among the ten run histories; "
                      "the same U8 H512 Shared MATD3 finished 20k episodes.",
        },
        "efficient_td3_option": {"algorithm": "matd3", "case": 5},
        "completed_sac_option": {"algorithm": "masac", "case": 5},
        "snapshot": "docs/methods/evidence/wandb_latest_ten_2026-11-08.json",
        "hf_verification": "docs/methods/evidence/wandb_ten_hf_checkpoint_verification_2026-11-08.json",
        "report": "docs/methods/WANDB_10_LATEST_35T25O_COMPARISON_2026-11-08.md",
        "note": "One training seed and 5-episode validation; not a multi-seed significance claim.",
    }


def render(snapshot):
    protocol = snapshot["protocol"]
    common_ep = protocol["all_run_common_episode_cutoff"]
    common_eval = protocol["all_run_shared_eval_checkpoint_global_step"]
    all_full = sum(r["observed_episodes"] >= r["configured_episodes"] for r in snapshot["runs"])
    main_td3 = next(r for r in snapshot["runs"] if r["algorithm"] == "matd3" and r["case"] == 1)
    main_u16 = next(r for r in snapshot["runs"] if r["algorithm"] == "matd3" and r["case"] == 2)
    main_fast = next(r for r in snapshot["runs"] if r["algorithm"] == "matd3" and r["case"] == 5)
    main_sac = next(r for r in snapshot["runs"] if r["algorithm"] == "masac" and r["case"] == 5)
    sac_large = next(r for r in snapshot["runs"] if r["algorithm"] == "masac" and r["case"] == 1)
    pair12 = next(v for k,v in snapshot["pairwise"].items() if k.startswith("matd3_01_vs_02"))
    pair23 = next(v for k,v in snapshot["pairwise"].items() if k.startswith("matd3_02_vs_03"))
    pair_sac = next(v for k,v in snapshot["pairwise"].items() if k.startswith("masac_01_vs_05"))
    v12 = pair12["validation_at_latest_shared_checkpoint"]
    v23 = pair23["validation_at_latest_shared_checkpoint"]
    vsac = pair_sac["validation_at_latest_shared_checkpoint"]

    return f"""# Live W&B comparison — ten 35-target/25-obstacle BASE runs (captured {snapshot['capture_date_utc']})

## Provenance and zero-retraining rule

This is a **read-only snapshot** from the project
[truongdaoanhduy-uit/uav_search_target](https://wandb.ai/truongdaoanhduy-uit/uav_search_target),
not a fresh training run. Ten most recently created runs match the original
five architecture/update cases for each algorithm: 6 UAVs, 1km, 35 targets,
25 obstacles, 1,500 maximum steps/episode, seed 44, BASE/no PQL,
4,096 GPU vector environments, batch 8,192, replay 1M,
T4×2 notebook recipe, no Critic chunking/AMP/compile/fused optimizers,
actor-only optional LayerNorm.

**Source commit logged by all ten W&B runs:** `{protocol['source_git_commit']}`.
**Common scenario fingerprint SHA-256:** `{protocol['scenario_sha256']}`.
**Common evaluation protocol SHA-256:** `{protocol['evaluation_protocol_sha256']}`.
**Observation version:** {protocol['observation_version']}.
All ten share the same scientific environment fingerprint and the same
tracked runtime protocol. Every individual case has its own full
`case_fingerprint_sha256` in the [registry](evidence/wandb_10_case_reuse_registry_2026-11-08.json).
The source was the embedded notebook snapshot at the reported commit,
not the later comment/test-only audit commit.

**Never duplicate these experiments merely because a new chat/session starts.**
Reuse the recorded W&B results and checkpoint references as far as they go.
A source comment/documentation change does not invalidate the data.
If any scientifically relevant configuration changes, start a separately named
cohort; if a partial case is explicitly needed at greater horizon,
verify resume/checkpoint availability before extending instead of
restarting by default.

## Exact completed progress

- {len(snapshot['runs'])} W&B runs report state `finished` but only **{all_full}/10 actually reached 20,000 completed episodes**.
- **{10-all_full}/10** are partial (time-limited and/or early-stopped); their
  W&B `finished` flag is *not* a guarantee of full target completion.
- **All-10 matched training episode budget:** {common_ep:,}.
  Training statistic = mean over completed episodes {common_ep-1023:,}–{common_ep:,}.
- **All-10 common fixed-checkpoint validation budget:** {common_eval:,} global transitions.
  `eval/mean_*` results use 5 fixed-seed validation episodes.
- `episode/return` and `outcome/*` are **in-sample completed training episodes**,
  not deterministic policy validation. Equal episodes does not imply equal
  transitions because episode length varies. We therefore report both
  matched-episode windows **and** matched-checkpoint validation.
- `best/mean_*` is selected over different available time horizons. It is
  useful for recovering the strongest saved checkpoint but is **not**
  the fair, equal-budget cross-run ranking.

| Run (W&B link) | Architecture | Completed ep | W&B runtime (h) | Training delivery at ep {common_ep:,}, last 1024 | Validation delivery at {common_eval/1e6:.2f}M transitions | Best validation delivery (unmatched horizon) |
|---|---|---:|---:|---:|---:|---:|
{rows_for_table(snapshot)}

Each table value can be traced to the [immutable JSON snapshot](evidence/wandb_latest_ten_2026-11-08.json).
The **W&B runtime** is the recorded run lifetime, not independently
measured GPU-kernel-only compute time.

## Fair matched comparisons within each algorithm

Each difference below means **left case minus right case**.
Positive delivery pp or return indicates that the left case scored
higher under the stated measurement. Training values use the final 1,024
completed episodes up to the pair's minimum observed episode. Validation
values use the latest **identical checkpoint global-step** available to both.
Do not mix these two estimators.

### MATD3

| Cases | Matched ep | Training Δdelivery (pp) | Training Δreturn | Matched eval checkpoint | Validation Δdelivery (pp) | Validation Δreturn |
|---|---:|---:|---:|---:|---:|---:|
{pairwise_table(snapshot, 'matd3')}

At equal 20,000 completed episodes, **MATD3 case 01 (Shared H512 U8 LN)**
beats **case 02 (Shared H512 U16 LN)** in the trailing training window,
and at shared checkpoint {v12['checkpoint_global_step']:,} transitions:
delivery advantage **{v12['delta_delivery_left_minus_right']:+.2f} pp**
and return advantage **{v12['delta_return_left_minus_right']:+.1f}**.
Both achieved the 20k target; W&B lifetime was
{main_td3['runtime_seconds']/3600:.2f} h versus
{main_u16['runtime_seconds']/3600:.2f} h. Thus U16 offers no demonstrated
benefit here and costs more wall-clock time.

The strongest MATD3 independent/shared controlled pair is
**case 02 (Shared H512 U16 LN) vs case 03 (Independent H512 U16 LN)**.
At shared checkpoint {v23['checkpoint_global_step']:,},
Shared is **{v23['delta_delivery_left_minus_right']:+.2f} pp** higher in
delivery rate. Independent case 04 without LayerNorm sometimes beats
independent case 03 but **both remain far below the Shared variants**.
Do not infer from this single seed that independent architectures
are intrinsically incapable of MARL.

**MATD3 case 05 (Shared H256 U8 LN)** achieves the 20k target in
{main_fast['runtime_seconds']/3600:.2f} h, versus case 01 in
{main_td3['runtime_seconds']/3600:.2f} h. It was better in early
same-step validation ({main_fast['validation_at_common_checkpoint']['delivery']:.2f}%
vs {main_td3['validation_at_common_checkpoint']['delivery']:.2f}% at
{common_eval:,}), but its best recorded checkpoint reaches
{main_fast['best_validation']['delivery']:.2f}% compared with
{main_td3['best_validation']['delivery']:.2f}% for H512.
H256 U8 is a useful throughput/early-learning option, not the
highest observed best-checkpoint option.

### MASAC

| Cases | Matched ep | Training Δdelivery (pp) | Training Δreturn | Matched eval checkpoint | Validation Δdelivery (pp) | Validation Δreturn |
|---|---:|---:|---:|---:|---:|---:|
{pairwise_table(snapshot, 'masac')}

The **only MASAC run to finish 20k episodes** was case 05:
Shared H256 U8 LN ({main_sac['observed_episodes']:,} episodes,
{main_sac['runtime_seconds']/3600:.2f} h, best delivery
{main_sac['best_validation']['delivery']:.2f}%).
Shared H512 U8 LN (case 01) reached
{sac_large['observed_episodes']:,} episodes; on an episode-matched
comparison it has {pair_sac['delta_delivery_left_minus_right']:+.2f} pp
higher training delivery and at matched checkpoint
{vsac['checkpoint_global_step']:,} has
{vsac['delta_delivery_left_minus_right']:+.2f} pp higher validation
delivery. It is more expensive and incomplete, so it should
**not** be dismissed as inherently worse, nor declared better at 20k.

MASAC U16 cases 02–04 each stopped after 4,096 completed episodes.
The evidence suffices to document early dynamics and poor throughput
but **does not establish final convergence superiority/inferiority**.
Do not prescribe another identical 20k fresh run from these data;
only extend a stored checkpoint when a new, explicit question warrants it.

## Cross-algorithm interpretation

The most defensible completed-case comparison is **case 05**, because
both algorithms completed 20k with matching Shared H256 U8 LN architecture
and scientific scenario. At the pair's latest common checkpoint
22,118,400 transitions, MATD3 delivery exceeds MASAC by
**{snapshot['pairwise']['cross_algorithm_case_05']['validation_at_latest_shared_checkpoint']['delta_delivery_left_minus_right']:+.2f} pp**.
MASAC reaches its strongest validation only later at
{main_sac['best_checkpoint_step']:,} transitions; comparing that later
52.00% against MATD3 at an earlier checkpoint is not fair.
The source algorithm remains the causal difference only to the extent
the distinct algorithms also have intrinsically different update cadence.

Do not use MASAC case 01's 11,675 episodes versus MATD3 case 01's
20,000 episodes to claim a final-outcome advantage. At shared episode and
checkpoint budgets, their relative performance varies across learning phases.

## Recommendations based only on these recorded runs

1. **Primary MATD3 baseline:** Case 01, Shared Actor/Critic, H512,
   U8, actor LayerNorm, default initialization. Strongest observed
   validated delivery: {main_td3['best_validation']['delivery']:.2f}%.
   Fixed-policy validation at its last logged checkpoint:
   {main_td3['validation'][-1]['delivery']:.2f}%.
2. **Compute-efficient MATD3 baseline:** Case 05, Shared H256 U8 LN;
   substantially faster 20k completion with solid early validation
   and best recorded delivery {main_fast['best_validation']['delivery']:.2f}%.
3. **Completed MASAC baseline:** Case 05, Shared H256 U8 LN; preserve
   the 20k run and its best checkpoint. Case 01 H512 U8 shows
   promise but is partial, so keep it as a separate research candidate.
4. **Do not use U16 as an automatic improvement:** The matched MATD3
   U8/U16 comparison favors U8 in this specific 35T25O/seed44 setup.
   MASAC U16 evidence is too incomplete to claim long-horizon quality.
5. **Do not re-run these exact cases.** If algorithm code, reward/CBF,
   observation/action, scenario generation, training budget, update rules
   or evaluation protocol changes materially, create a new versioned
   cohort with a new fingerprint. Reusing these already-observed runs
   does **not** require launching new Kaggle training.

The recommendation is conditional on a **single training seed** and
5 periodic validation episodes, without statistical significance
across training seeds. Any future paper-level robustness claim
would require separately justified evaluation; this registry is
not that claim.

## Figures and reproduction artifacts

- [MATD3 delivery validation versus checkpoint transitions](evidence/wandb_ten_matd3_validation_2026-11-08.png)
- [MASAC delivery validation versus checkpoint transitions](evidence/wandb_ten_masac_validation_2026-11-08.png)
- [All 10 at identical 7.3728M checkpoint](evidence/wandb_ten_all_common_validation_2026-11-08.png)
- [Immutable complete metric snapshot](evidence/wandb_latest_ten_2026-11-08.json)
- [Case reuse registry](evidence/wandb_10_case_reuse_registry_2026-11-08.json),
  which records the W&B-reported best checkpoint repository, immutable
  Hugging Face revision and file SHA-256 for all ten cases. This is
  provenance from W&B metadata, not proof of current artifact availability.
- [Pinned HF checkpoint verification](evidence/wandb_ten_hf_checkpoint_verification_2026-11-08.json):
  all 10 Hugging Face commit revisions and the corresponding
  `uav-network/<run_id>/best/checkpoint.pt` tree entries were accessible
  on 2026-11-08. This checks repository metadata only; large checkpoint
  tensors were not downloaded or independently hashed.
- Offline compressed per-episode histories stay at
  `outputs/wandb-ten-20261108/*_episode_history.npz` on the user's
  sandbox laptop. These are local archival data, **not** tracked Git files.
- Script for an explicit refresh: `PYTHONPATH=. python scripts/audit_latest_ten_wandb_baselines.py --ids <ten pinned IDs>`.
  Regenerate report (without any W&B access):
  `python scripts/render_wandb_ten_case_registry.py`.

These reports contain no API keys. **Do not delete W&B source runs or
saved best checkpoints** before a verified backup/reproducibility check.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="docs/methods/evidence/wandb_latest_ten_2026-11-08.json")
    parser.add_argument("--report", default="docs/methods/WANDB_10_LATEST_35T25O_COMPARISON_2026-11-08.md")
    parser.add_argument("--evidence-dir", default="docs/methods/evidence")
    args = parser.parse_args()
    snapshot = json.loads(Path(args.snapshot).read_text())
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    for algorithm in ("matd3", "masac"):
        plot_eval_curves(snapshot, algorithm, evidence / f"wandb_ten_{algorithm}_validation_2026-11-08.png")
    plot_10_case_common_eval(snapshot, evidence / "wandb_ten_all_common_validation_2026-11-08.png")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(snapshot), encoding="utf-8")
    registry_path = evidence / "wandb_10_case_reuse_registry_2026-11-08.json"
    registry_path.write_text(json.dumps(registry(snapshot), ensure_ascii=False, indent=2) + "\n")
    print("RENDERED", report, registry_path)
    print("MATCHED", snapshot["protocol"]["all_run_common_episode_cutoff"],
          snapshot["protocol"]["all_run_shared_eval_checkpoint_global_step"])


if __name__ == "__main__":
    main()
