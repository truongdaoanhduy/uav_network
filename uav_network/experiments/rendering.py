"""Static mission preview with paths, obstacles, camera FOV and task status."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter, writers


def _draw(ax, env, paths, positions, battery, found, delivered, title):
    w = env.config.world
    for x, y, radius in env.obstacles:
        ax.add_patch(Circle((x, y), radius, color="slategray", alpha=.8))
    for label, mask, color in [("Hidden", ~found, "gray"),
                               ("Found, awaiting GCS", found & ~delivered, "orange"),
                               ("Delivered", delivered, "green")]:
        ax.scatter(*env.targets[mask].T, marker="*", s=110, c=color, label=label)
    paths = np.asarray(paths)
    colors = plt.get_cmap("tab10")
    for i, xy in enumerate(positions):
        color = colors(i % 10)
        ax.plot(paths[:, i, 0], paths[:, i, 1], color=color, linewidth=1.5)
        ax.scatter(*xy, color=color, marker="^", s=60)
        ax.text(xy[0] + 8, xy[1] + 8, f"{i}: {battery[i]:.0%}", fontsize=8)
        ax.add_patch(Circle(xy, env.config.sensor.footprint_radius_m, color=color, alpha=.08))
    ax.scatter(*env.gcs, color="black", marker="s", s=80, label="GCS")
    ax.set(xlim=(0, w.map_size_m), ylim=(0, w.map_size_m), aspect="equal",
           xlabel="x (m)", ylabel="y (m)",
           title=title)
    ax.legend(loc="upper right", fontsize=8)


def render_episode(env, paths, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    info, w = env.info(), env.config.world
    try:
        _draw(ax, env, paths, env.positions, env.battery_fraction,
              env.reports.known.any(axis=0), env.reports.delivered,
              f"{w.num_uavs}-UAV search | found {info['targets_found']}/{w.num_targets} | "
              f"delivered {info['targets_delivered']}/{w.num_targets}")
        fig.tight_layout()
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
    return str(output)


def animation_frame(env):
    return {"step": env.steps, "positions": env.positions.copy(),
            "battery": env.battery_fraction.copy(),
            "found": env.reports.known.any(axis=0).copy(),
            "delivered": env.reports.delivered.copy()}


def render_animation(env, paths, frames, output, fps=10, video_format="auto"):
    video_format = ("mp4" if writers.is_available("ffmpeg") else "gif") if video_format == "auto" else video_format
    if video_format == "mp4" and not writers.is_available("ffmpeg"):
        raise RuntimeError("MP4 needs ffmpeg; select gif or auto")
    output = Path(output).with_suffix("." + video_format)
    writer = (FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
              if video_format == "mp4" else PillowWriter(fps=fps))
    fig, ax = plt.subplots(figsize=(8, 8))
    paths, w = np.asarray(paths), env.config.world
    try:
        with writer.saving(fig, str(output), dpi=100):
            for frame in frames:
                ax.clear()
                step = frame["step"]
                title = (f"step {step}/{w.max_steps}, t={step * w.dt_s:g}s | "
                         f"found {frame['found'].sum()}/{w.num_targets} | "
                         f"GCS delivered {frame['delivered'].sum()}/{w.num_targets}")
                _draw(ax, env, paths[:step + 1], frame["positions"], frame["battery"],
                      frame["found"], frame["delivered"], title)
                fig.tight_layout()
                writer.grab_frame()
    finally:
        plt.close(fig)
    return str(output)
