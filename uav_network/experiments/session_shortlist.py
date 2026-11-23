"""Pre-validation shortlist for a bounded single Kaggle session.

Uses only checkpoint metadata recorded during training; never looks at Validation
or Final Test results to select candidates. Originals are retained untouched.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path


def _uniform(indices, budget):
    """Chronological quantiles with endpoints and no repeated index."""
    indices = list(indices)
    if budget <= 0:
        return []
    if len(indices) <= budget:
        return indices
    if budget == 1:
        return [indices[-1]]
    return [indices[(i * (len(indices) - 1)) // (budget - 1)]
            for i in range(budget)]


def select_candidates(rows, limit=32):
    """Select at most N immutable actors before any validation metrics exist.

    Include the first checkpoint, final checkpoint, latest training-best
    candidate and time-spaced periodic checkpoints. Fill unused slots with
    equally spaced remaining versions. No validation/test peeking.
    """
    if limit < 2:
        raise ValueError("At least two candidate slots required")
    if not rows:
        raise ValueError("No Actor checkpoints")
    if len(rows) <= limit:
        return list(rows)

    needed = {0, len(rows) - 1}
    best = [i for i, r in enumerate(rows)
            if "training_metric_best_candidate" in r.get("reasons", [])]
    if best:
        needed.add(best[-1])
    periodic = [i for i,r in enumerate(rows)
                if "periodic" in r.get("reasons", [])]
    # Reserve mandatory anchors and fill the other slots with periodic
    # temporal quantiles, distributing across the training run.
    periodic = [i for i in periodic if i not in needed]
    needed.update(_uniform(periodic, limit - len(needed)))
    remain = [i for i in range(len(rows)) if i not in needed]
    needed.update(_uniform(remain, limit - len(needed)))
    assert len(needed) == limit
    return [rows[i] for i in sorted(needed)]


def select_step_targets(rows, step_targets):
    """Select *actual* policy versions on a prespecified global-step grid.

    No resampling, no interpolation, no evaluation peeking: missing checkpoints
    inside the observed horizon are errors. A terminal checkpoint is always
    added even when 20k episodes end off-grid. Future targets are ignored.
    """
    if not rows or int(rows[0]["global_step"]) != 0:
        raise ValueError("The step-zero initial Actor checkpoint is required")
    if (not isinstance(step_targets, (list, tuple)) or not step_targets or
            type(step_targets[0]) is not int or step_targets[0] != 0 or
            any(type(step) is not int or step < 0 for step in step_targets) or
            any(a >= b for a,b in zip(step_targets, step_targets[1:]))):
        raise ValueError("Step targets must be a strictly increasing list starting at zero")
    by_step = {int(row["global_step"]): row for row in rows}
    if any(a >= b for a,b in zip(by_step, list(by_step)[1:])):
        raise ValueError("Original checkpoint rows must be ordered by true global_step")
    final = int(rows[-1]["global_step"])
    selected = []
    for step in step_targets:
        if step > final:
            break
        if step not in by_step:
            raise ValueError(f"Missing exact checkpoint at requested global_step={step}")
        selected.append(by_step[step])
    if not selected or int(selected[-1]["global_step"]) != final:
        selected.append(rows[-1])
    # Include the strongest training-derived Actor candidate as an EXTRA real
    # policy for Validation, not as an automatic final/test winner. This does
    # not displace the predeclared, evenly-spaced evaluation curve.
    best_rows = [row for row in rows
                 if "training_metric_best_candidate" in row.get("reasons", [])]
    if best_rows:
        selected.append(best_rows[-1])
    return [by_step[step] for step in sorted({int(row["global_step"]) for row in selected})]


def create_shortlist(source, destination, *, limit=32, step_targets=None):
    """Verified, hardlinked snapshot view for a time-bounded GPU evaluation.

    The original immutable manifest/files remain the complete historical
    archive. The new manifest explicitly records the sample selection.
    """
    from .deferred_gpu_evaluation import verify_manifest
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Shortlist may not overwrite original snapshot bank")
    manifest = verify_manifest(source)
    if destination.exists():
        raise FileExistsError("Shortlist already exists; refusing stale reuse")
    rows = (select_step_targets(manifest["snapshots"], step_targets) if step_targets is not None
            else select_candidates(manifest["snapshots"], limit=limit))
    destination.mkdir(parents=True)
    try:
        for row in rows:
            original = source / row["file"]
            os.link(original, destination / row["file"])
        reduced = dict(manifest)
        reduced["snapshots"] = rows
        reduced["count"] = len(rows)
        reduced["collected"] = len(rows)
        reduced["pruned"] = 0
        reduced["keep_all_distinct_versions"] = True
        reduced["shortlist_provenance"] = {
            "method": ("prespecified_global_step_grid_plus_actual_final" if step_targets is not None
                           else "train_metadata_only_uniform_periodic_plus_last_trainingbest_final"),
            "requested_global_step_targets": step_targets,
            "source_count": manifest["count"],
            "selected_count": len(rows),
            "source_manifest_sha256": sha256((source/"manifest.json").read_bytes()).hexdigest(),
            "source_manifest": str(source/"manifest.json"),
            "original_archive_untouched": True,
            "validation_metrics_used_for_selection": False,
        }
        (destination/"manifest.json").write_text(json.dumps(reduced,indent=2,allow_nan=False))
        verify_manifest(destination)
        return reduced
    except Exception:
        # This creates an entirely new folder. It never removes source files.
        for f in destination.iterdir():
            f.unlink()
        destination.rmdir()
        raise
