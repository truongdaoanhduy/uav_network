"""Algorithm-independent, one-metric-per-image paper-style charts for W&B.

Each chart uses ONLY the episode history logged by its owning run. No other
algorithm's data is fetched, and no Eval/Test metrics are relabelled as Training.
Shaded bands show observed local within-run variability, NOT uncertainty across
independent training seeds. Missing data remain missing; never interpolate
across a large unobserved episode gap.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import re


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _eligible(key):
    if key == "episode/return":
        return "reward", "Episodic Return", 1.
    if key == "episode/length":
        return "episode_length", "Episode Length (steps)", 1.
    if key.startswith("training/"):
        metric = key[len("training/"):]
        # Don't duplicate aliases when canonical names exist.
        if metric in {"search_rate_percent", "delivery_rate_percent"}:
            return None
        # Metric names are dynamic: new algorithms may emit new outcomes.
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_/]{0,95}", metric):
            return None
        label = metric.replace("_", " ").replace("/", " / ").title()
        scale = 1e-3 if metric.endswith("_energy_j") or metric == "energy_per_delivered_target_j" else 1.
        if scale != 1.:
            label = label.replace(" J", "") + " (kJ)"
        return metric.replace("/", "_"), label, scale
    if key.startswith("learner/") and "loss" in key.lower():
        name = key.split("/", 1)[-1]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,65}", name):
            return name, name.replace("_", " ").title(), 1.
    return None


class EpisodePlotCollector:
    """Collect W&B-bound numeric scalars without changing their semantics.

    A learner metric logged between episodes uses the most recent known
    completed training episode as its X-coordinate.
    """

    def __init__(self, algorithm):
        self.algorithm = str(algorithm).upper()
        self.current_episode = None
        self.values = defaultdict(list)
        self.training_samples = set()

    def observe(self, values):
        candidate = values.get("episode/index")
        if candidate is None:
            candidate = values.get("train/episodes_completed")
        if _is_finite_number(candidate) and float(candidate) >= 0:
            self.current_episode = int(candidate)
        current = self.current_episode
        if current is None or current < 1:
            return
        for metric, value in values.items():
            if _eligible(metric) is None or not _is_finite_number(value):
                continue
            self.values[metric].append((current, float(value)))
            if metric == "episode/return":
                self.training_samples.add(current)


def _gaussian(values, sigma):
    import numpy as np

    n = len(values)
    if n < 2:
        return np.asarray(values, dtype=float)
    radius = min(n - 1, max(2, int(round(sigma * 4))))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    values = np.asarray(values, dtype=float)
    weights = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(weights, kernel, mode="valid")


def _segments(x):
    import numpy as np

    if len(x) < 2:
        return [slice(0, len(x))]
    gaps = np.diff(x)
    normal = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1.
    # Never draw a line through a run section for which data was not logged.
    breaks = np.flatnonzero(gaps > max(64., normal * 4)) + 1
    boundaries = [0, *breaks.tolist(), len(x)]
    return [slice(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if a < b]


def _samples(points):
    """Sort and mean duplicate episode observations (e.g. multiple learner logs)."""
    import numpy as np

    bins = defaultdict(list)
    for episode, value in points:
        if _is_finite_number(episode) and _is_finite_number(value):
            bins[int(episode)].append(float(value))
    x = np.asarray(sorted(bins), dtype=float)
    y = np.asarray([float(np.mean(bins[int(e)])) for e in x], dtype=float)
    return x, y


def render_wandb_episode_plots(collector, output, *, max_charts=80):
    """Render one PNG per scalar metric and one algorithm per PNG.

    Returns paths for the caller to log to its own active W&B run. No
    W&B credentials, API calls or global workspace mutations are required.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = {"MASAC": "#d0a338", "MATD3": "#7965b9", "MAPPO": "#3d8cc9"}
    color = colors.get(collector.algorithm, "#4c79b5")
    result = {
        "algorithm": collector.algorithm, "x_axis": "Training Episode",
        "training_samples": len(collector.training_samples),
        "charts": {}, "samples_by_metric": {}, "segments_by_metric": {},
        "shade_definition": "observed within-run rolling SD; NOT across-seed SD",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.facecolor": "#ebebf3", "figure.facecolor": "white",
        "axes.edgecolor": "#ebebf3", "axes.axisbelow": True,
        "grid.color": "white", "grid.linewidth": 1.3,
        "axes.grid": True, "svg.fonttype": "none",
    })

    candidates = sorted(collector.values, key=lambda k: (k != "episode/return", k))
    for source_key in candidates:
        item = _eligible(source_key)
        if not item:
            continue
        name, label, scale = item
        x, y = _samples(collector.values[source_key])
        if len(x) < 5:
            continue
        # Skip repeated aliases or alternate naming of the exact same metric.
        key = f"paper_style/{name}"
        if key in result["charts"]:
            continue
        if len(result["charts"]) >= max_charts:
            break
        y = y * scale
        segments = _segments(x)
        fig, ax = plt.subplots(figsize=(10.5, 6), dpi=175)
        for sl in segments:
            xx, yy = x[sl], y[sl]
            if len(xx) < 2:
                ax.scatter(xx, yy, color=color, alpha=.5, s=9, zorder=3)
                continue
            # Smooth within observed contiguous segments only.
            sigma = max(.85, min(10., len(xx) / 5))
            center = _gaussian(yy, sigma)
            band_sigma = max(1., min(17., len(xx) / 4))
            broad_mean = _gaussian(yy, band_sigma)
            var = np.maximum(0., _gaussian(yy * yy, band_sigma) - broad_mean * broad_mean)
            sd = np.sqrt(var)
            lower, upper = center - sd, center + sd
            if name.endswith("_percent") or name == "mission_success_rate_percent":
                lower = np.maximum(0., lower)
                upper = np.minimum(100., upper)
            if name.endswith("_auc") or name.endswith("_fraction"):
                lower = np.maximum(0., lower)
                upper = np.minimum(1., upper)
            ax.plot(xx, yy, color=color, alpha=.1, linewidth=.68, zorder=2)
            ax.fill_between(xx, lower, upper, color=color, alpha=.18,
                            linewidth=0, zorder=1)
            ax.plot(xx, center, color=color, linewidth=2.45, zorder=4)
        ax.plot([], [], color=color, linewidth=2.5, label=collector.algorithm.lower())
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.12),
                  frameon=False, fontsize=13)
        ax.set_xlabel("Training Episode", fontsize=14)
        ax.set_ylabel(label, fontsize=14)
        ax.set_xlim(left=0, right=max(10., x.max() * 1.01))
        ax.tick_params(length=0, colors="#484855")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(top=.84, bottom=.20, left=.135, right=.97)
        fig.text(.5, .09, "Actual Training episodes  •  Smoothed mean  •  Within-run rolling SD",
                 ha="center", color="#60606b", fontsize=9.3)
        fig.text(.5, .056, "One training seed; shaded area is NOT between-seed uncertainty",
                 ha="center", color="#82828b", fontsize=8.7, style="italic")
        filename = name + ".png"
        path = output / filename
        fig.savefig(path, dpi=175, facecolor="white")
        plt.close(fig)
        result["charts"][key] = str(path)
        result["samples_by_metric"][key] = len(x)
        result["segments_by_metric"][key] = len(segments)
    return result


def publish_to_active_run(tracker):
    """Log charts into the exact W&B run that owns the training history."""
    collector = getattr(tracker, "_episode_plotter", None)
    if collector is None:
        return 0
    result = render_wandb_episode_plots(collector, tracker.output / "paper_style_charts")
    if not result["charts"]:
        return 0
    # 1 run = 1 algorithm; 1 image = 1 metric. Never combine runs.
    payload = {name: tracker.wandb.Image(path)
               for name, path in result["charts"].items()}
    tracker.run.log(payload)
    tracker.run.summary.update({
        "paper_style/plot_count": len(payload),
        "paper_style/x_axis": "Training Episode",
        "paper_style/shaded_band": result["shade_definition"],
        "paper_style/episode_samples": result["training_samples"],
        "paper_style/plot_status": "completed",
    })
    return len(payload)
