"""Mission paths and opt-in completed report-hop visualization."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter, writers


COMMUNICATION_WINDOW_S = 20.


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
        ax.plot(paths[:, i, 0], paths[:, i, 1], color=color, linewidth=1.2, alpha=.7)
        ax.scatter(*xy, color=color, marker="^" if battery[i] > 0 else "X", s=60)
        label_xy = np.clip(xy + .008 * w.map_size_m, .012 * w.map_size_m, .915 * w.map_size_m)
        ax.text(*label_xy, f"{i}: {battery[i]:.0%}", fontsize=8,
                bbox=dict(facecolor="white", alpha=.65, edgecolor="none", pad=.5))
        ax.add_patch(Circle(xy, env.config.sensor.footprint_radius_m, color=color, alpha=.06))
    ax.scatter(*env.gcs, color="black", marker="s", s=80, label="GCS")
    ax.set(xlim=(0, w.map_size_m), ylim=(0, w.map_size_m), aspect="equal",
           xlabel="x (m)", ylabel="y (m)", title=title)
    ax.legend(loc="upper right", fontsize=8)


def _draw_communication(ax, events, now, num_uavs, window_s=COMMUNICATION_WINDOW_S):
    """Show completed whole-report hops at their event-time positions."""
    recent = [event for event in events
              if event.get("event") == "report_received"
              and now - window_s < event["time_s"] <= now]
    for event in recent:
        color = "#087f23" if event["destination"] == num_uavs else "#007cba"
        source, destination = event["source_xy"], event["destination_xy"]
        ax.annotate("", xy=destination, xytext=source,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=13),
                    zorder=6)
        midpoint = (np.asarray(source) + np.asarray(destination)) / 2
        ax.text(*midpoint, f"T{event['target']} {event['power_w']:.2f} W",
                fontsize=7, color=color, ha="center", zorder=7,
                bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=1))
    return recent


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
    info = env.info()
    energy, network = info.get("energy", {}), info.get("network", {})
    drain = getattr(env, "drain_communication_events", lambda: [])
    return {"step": env.steps, "positions": env.positions.copy(),
            "battery": env.battery_fraction.copy(),
            "found": env.reports.known.any(axis=0).copy(),
            "delivered": env.reports.delivered.copy(),
            "communication_events": drain(),
            "buffer_reports": info.get("buffer_reports", 0),
            "active_uavs": energy.get("active_uavs", int((env.battery_fraction > 0).sum())),
            "flight_energy_j": energy.get("propulsion_energy_j", 0.),
            "radio_energy_j": network.get("uav_communication_energy_j", 0.)}


def _communication_panel(ax, frame, events, recent, num_uavs):
    ax.clear()
    ax.axis("off")
    hops = sum(event["destination"] != num_uavs for event in events)
    gcs = len(events) - hops
    lines = ["COMMUNICATION", "", "Blue arrow: report to UAV", "Green arrow: report to GCS",
             f"Arrows: last {COMMUNICATION_WINDOW_S:g} simulation s",
             "Coordinates at reception time", "Power: last received fragment", "",
             f"Completed hops: {len(events)}", f"  Relay: {hops} | GCS receipts: {gcs}",
             f"Buffered reports: {frame.get('buffer_reports', 0)}",
             f"Active UAVs: {frame.get('active_uavs', num_uavs)}/{num_uavs}",
             f"Flight: {frame.get('flight_energy_j', 0.) / 1000:.1f} kJ",
             f"UAV radio: {frame.get('radio_energy_j', 0.):.5f} J", "", "LATEST COMPLETED HOPS"]
    for event in recent[-6:]:
        destination = "GCS" if event["destination"] == num_uavs else f"U{event['destination']}"
        lines.append(f"{event['time_s']:.1f}s U{event['source']} -> {destination}")
        lines.append(f"  T{event['target']} | hop {event['hops']} | {event['power_w']:.3f} W")
    if not recent:
        lines.append("No completed hop in this window")
    lines.extend(["", "Thin colored paths: flight history", "X: battery exhausted",
                  "Found target != delivered to GCS"])
    ax.text(0, 1, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=9,
            family="monospace", linespacing=1.45)


def render_animation(env, paths, frames, output, fps=10, video_format="auto"):
    video_format = ("mp4" if writers.is_available("ffmpeg") else "gif") if video_format == "auto" else video_format
    if video_format == "mp4" and not writers.is_available("ffmpeg"):
        raise RuntimeError("MP4 needs ffmpeg; select gif or auto")
    output = Path(output).with_suffix("." + video_format)
    writer = (FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
              if video_format == "mp4" else PillowWriter(fps=fps))
    fig, (ax, panel) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [3, 1]})
    paths, w = np.asarray(paths), env.config.world
    events = []
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
                events.extend(frame.get("communication_events", []))
                recent = _draw_communication(ax, events, step * w.dt_s, w.num_uavs)
                _communication_panel(panel, frame, events, recent, w.num_uavs)
                fig.tight_layout()
                writer.grab_frame()
    finally:
        plt.close(fig)
    return str(output)
