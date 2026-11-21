"""Render a real, checkpoint-based OpenAI Spinning Up-style validation figure.

Only computed from observed Validation episodes (16 fixed seeds per checkpoint).
An 11-checkpoint centered moving average is used for readability. No shaded
std band unless there are independent TRAINING seeds, which this one-run figure
does not claim to provide. Held-out Test is intentionally never plotted on
the Validation learning curve.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os

from .wandb_comparison_report_20261115 import load_completed, METRICS

TITLES = {
    "search_rate_percent": ("Discovery rate", "%"),
    "delivery_rate_percent": ("GCS delivery rate", "%"),
    "discovery_auc": ("Discovery AUC", "normalized"),
    "delivery_auc": ("Delivery AUC", "normalized"),
    "energy_per_delivered_target_j": ("Energy per delivery", "J / target"),
    "gcs_delivery_time_capped_mean_s": ("Capped GCS delivery time", "seconds"),
    "map_coverage_percent": ("Map coverage", "%"),
    "coverage_auc": ("Coverage AUC", "fraction"),
    "mission_success_rate_percent": ("Mission success rate", "%"),
}


def smoothed(values, width=11):
    import numpy as np
    a = np.asarray(values, dtype=float)
    if a.ndim != 1:
        raise ValueError("Only one-dimensional observed metric series")
    finite = np.isfinite(a)
    window = np.ones(min(width, len(a)), dtype=float)
    sums = np.convolve(np.where(finite, a, 0.0), window, "same")
    counts = np.convolve(finite.astype(float), window, "same")
    return np.divide(sums, counts, out=np.full_like(sums, np.nan),
                     where=counts > 0)


def draw(work, spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result = load_completed(work, spec)  # refuses incomplete / non-heldout test
    values = []
    x = np.asarray([r["source"]["global_step"] for r in result["validation"]],
                   dtype=float)
    for row in result["validation"]:
        metric_values = {}
        from .wandb_comparison_report_20261115 import comparison_values
        metric_values.update(comparison_values(row))
        values.append(metric_values)

    fig, axs = plt.subplots(3, 3, figsize=(14.0, 10.0), dpi=135)
    fig.patch.set_facecolor("white")
    color = "#3867ae" if result["algorithm"] == "masac" else "#c66d29"
    for ax, (metric, (label, unit)) in zip(axs.flat, TITLES.items()):
        y = np.asarray([row[metric] if row[metric] is not None else np.nan
                        for row in values], dtype=float)
        avg = smoothed(y, 11)
        ax.set_facecolor("white")
        ax.plot(x, y, color=color, alpha=.19, lw=.85, label="Raw checkpoint")
        ax.plot(x, avg, color=color, lw=2.0, label="Moving mean (11 checkpoints)")
        if np.isfinite(y).any():
            ix = int(np.flatnonzero(np.isfinite(y))[-1])
            ax.scatter([x[ix]], [avg[ix]], s=21, color=color, zorder=5)
        ax.set_title(label, fontsize=10.5, loc="left", pad=9, fontweight="semibold")
        ax.set_ylabel(unit, fontsize=8)
        ax.set_xlabel("Environment transitions", fontsize=8)
        ax.grid(alpha=.19, lw=.65)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6,6))
    fig.suptitle(f"{result['algorithm'].upper()} | Validation learning curves",
                 x=.035, y=.99, ha="left", fontweight="bold", fontsize=17)
    fig.text(.035,.945,"Evaluation of saved policies during the training trajectory  |  "
             "x = joint environment transitions  |  16 fixed scenarios per checkpoint",
             ha="left",fontsize=9,color="#515e6a")
    fig.subplots_adjust(left=.065,right=.985,top=.90,bottom=.07,
                        wspace=.38,hspace=.48)
    out=Path(work)/"gpu_batch_eval"
    png=out/"spinningup_validation_curves.png"
    svg=out/"spinningup_validation_curves.svg"
    fig.savefig(png,bbox_inches="tight",dpi=180)
    fig.savefig(svg,bbox_inches="tight")
    plt.close(fig)
    return result, png, svg


def publish(work, spec):
    data,png,svg=draw(work,spec)
    import wandb
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Authenticated W&B connection required for publishing")
    run_id=data["run_id"]
    with wandb.init(entity=spec["entity"],project=spec["project"],
                    id=run_id,resume="must",reinit="finish_previous",
                    mode="online") as run:
        run.log({"paper/validation_learning_curves":wandb.Image(str(png),
            caption="Deterministic held-out validation across checkpoint history; "
                    "11-checkpoint moving mean; training seed 44 (no fake SD band)")})
        run.summary["paper/validation_curve_checkpoint_count"]=len(data["validation"])
    receipt={"status":"published","run":run_id,"plot_png":str(png),
             "plot_svg":str(svg),
             "validation_checkpoints":len(data["validation"])}
    (Path(work)/"gpu_batch_eval/spinningup_plot_receipt.json").write_text(
        json.dumps(receipt,indent=2))
    print("SPINNINGUP_PAPER_PLOT_READY",json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--spec",type=Path,required=True)
    parser.add_argument("--work",type=Path,default=Path("/kaggle/working"))
    args=parser.parse_args()
    publish(args.work,json.loads(args.spec.read_text()))


if __name__=="__main__":
    main()
