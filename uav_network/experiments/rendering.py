"""Static mission preview with paths, obstacles, camera FOV and task status."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def render_episode(env, paths, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    w = env.config.world
    for x, y, radius in env.obstacles:
        ax.add_patch(Circle((x, y), radius, color="slategray", alpha=.8))
    found = env.reports.known.any(axis=0)
    for label, mask, color in [("Hidden", ~found, "gray"),
                               ("Found, awaiting GCS", found & ~env.reports.delivered, "orange"),
                               ("Delivered", env.reports.delivered, "green")]:
        ax.scatter(*env.targets[mask].T, marker="*", s=110, c=color, label=label)
    paths = np.asarray(paths)
    colors = plt.get_cmap("tab10")
    for i, xy in enumerate(env.positions):
        color = colors(i % 10)
        ax.plot(paths[:, i, 0], paths[:, i, 1], color=color, linewidth=1.5)
        ax.scatter(*xy, color=color, marker="^", s=60)
        ax.text(xy[0] + 8, xy[1] + 8, str(i), fontsize=9)
        ax.add_patch(Circle(xy, env.config.sensor.footprint_radius_m, color=color, alpha=.08))
    ax.scatter(*env.gcs, color="black", marker="s", s=80, label="GCS")
    info = env.info()
    ax.set(xlim=(0, w.map_size_m), ylim=(0, w.map_size_m), aspect="equal",
           xlabel="x (m)", ylabel="y (m)",
           title=f"{w.num_uavs}-UAV search | found {info['targets_found']}/{w.num_targets} | "
                 f"delivered {info['targets_delivered']}/{w.num_targets}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return str(output)
