"""Keep only bounded generated episode policies; never delete resume or input files."""
from pathlib import Path
import re
import shutil


def prune_episode_checkpoints(output, keep=2):
    paths = [p for p in Path(output).iterdir() if
             re.fullmatch(r"episode_[0-9]{6,}\.pt", p.name) and p.is_file() and not p.is_symlink()]
    for path in sorted(paths, key=lambda p: int(p.stem.split("_")[1]))[:-keep]:
        path.unlink()


def disk_metrics(output):
    output = Path(output)
    usage = shutil.disk_usage(output)
    checkpoint_bytes = sum(p.stat().st_size for p in output.rglob("*")
                           if p.is_file() and (p.name.endswith(".pt") or p.name.endswith(".pt.tmp")))
    return {"storage/disk_free_bytes": usage.free, "storage/disk_used_bytes": usage.used,
            "storage/checkpoint_bytes": checkpoint_bytes}
