"""Immutable Actor-only snapshots for deferred GPU validation.

Default mode is bounded: prune when candidate capacity fills and uniformly
select a subset at finalization. Explicit opt-in keep_all mode NEVER thins
checkpoint candidates and also stores best training-metric candidate labels.
No target critics, optimizer, replay, trajectory or credentials are saved.
"""
from __future__ import annotations
import hashlib
import json
import tarfile
from pathlib import Path

from .evaluation_snapshots import save_evaluation_snapshot


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class DeferredSnapshotBank:
    def __init__(self, config, output):
        self.config = config
        self.folder = Path(output) / "deferred-gpu-eval"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.target = int(config.tracking.deferred_gpu_eval_target)
        self.online = bool(config.tracking.online_gpu_eval_enabled)
        self.keep_all = bool(config.tracking.deferred_gpu_eval_keep_all)
        self.limit = int(config.tracking.deferred_gpu_eval_candidate_limit)
        self.stride = int(config.tracking.deferred_gpu_eval_collection_stride)
        self.next_collection = self.stride
        self.rows = []
        self.saved = 0
        self.pruned = 0
        if (self.stride < 1 or
            (self.keep_all and (self.target != 0 or self.limit != 0)) or
            (not self.keep_all and (self.target < 1 or self.limit < 2 * self.target))):
            raise ValueError("Invalid deferred GPU evaluation snapshot capacity or stride")

    def capture(self, actors, actor_updates, *, episodes, global_step,
                collection_steps, next_seed, force=False, reason=None,
                training_metric_candidate=None):
        reason = reason or ("final" if force else "periodic")
        if reason not in {"periodic", "final", "training_metric_best_candidate"}:
            raise ValueError("Unsupported snapshot collection reason")
        if training_metric_candidate is not None and reason != "training_metric_best_candidate":
            raise ValueError("Training metric metadata requires training candidate reason")
        if self.keep_all and self.rows and self.rows[-1]["policy_version"] == int(actor_updates):
            # The same immutable Actor can satisfy more than one capture reason.
            # Retain one file, mark ALL reasons so there is no duplicate validation.
            row = self.rows[-1]
            new_label = reason not in row.setdefault("reasons", [])
            if new_label:
                row["reasons"].append(reason)
            if training_metric_candidate is not None:
                row["training_metric_candidate"] = training_metric_candidate
            if new_label or training_metric_candidate is not None:
                with (self.folder / "capture-journal.jsonl").open("a") as journal:
                    journal.write(json.dumps({"event": "label_existing_policy",
                        "file": row["file"], "reason": reason,
                        "training_metric_candidate": training_metric_candidate},
                        allow_nan=False) + "\n")
            return False
        if self.online:
            # Live GPU worker is reading immutable Actor files in parallel.
            # Never prune/rewrite queued files: reserve one final Actor slot
            # and keep the entire queue capped at `target` real checkpoints.
            if not force and len(self.rows) >= self.target - 1:
                return False
            if self.rows and self.rows[-1]["policy_version"] == int(actor_updates):
                return False
            if int(actor_updates) <= 0:
                return False
            if len(self.rows) >= self.target:
                return False
        if not force and collection_steps < self.next_collection:
            return False
        if self.rows and self.rows[-1]["collection_steps"] == collection_steps:
            return False
        if not force:
            self.next_collection = collection_steps + self.stride
        progress = {"episodes": int(episodes), "global_step": int(global_step),
                    "collection_steps": int(collection_steps), "next_seed": int(next_seed)}
        path = save_evaluation_snapshot(
            self.config, self.folder, actors, int(actor_updates), progress)
        record = {"file": path.name, "episodes": int(episodes),
                  "global_step": int(global_step), "collection_steps": int(collection_steps),
                  "policy_version": int(actor_updates), "reasons": [reason]}
        if training_metric_candidate is not None:
            record["training_metric_candidate"] = training_metric_candidate
        self.rows.append(record)
        self.saved += 1
        if self.keep_all:
            # Crash-recovery journal; a new Actor snapshot is immutable and
            # saved atomically before this record becomes visible.
            with (self.folder / "capture-journal.jsonl").open("a") as journal:
                journal.write(json.dumps({**record, "sha256": sha256(path)}, allow_nan=False) + "\n")
        if not self.online and not self.keep_all and len(self.rows) >= self.limit:
            # Preserve first/last and near-uniform intermediate policies.
            old = self.rows
            keep_idx = set(range(0, len(old), 2)) | {len(old) - 1}
            self.rows = [row for i, row in enumerate(old) if i in keep_idx]
            for i, row in enumerate(old):
                if i not in keep_idx:
                    (self.folder / row["file"]).unlink()
                    self.pruned += 1
            self.stride *= 2
            self.next_collection = collection_steps + self.stride
        return True

    def write_training_best_alias(self, candidate):
        """Keep a named convenience alias; NOT the final Validation best.pt.

        Every source Actor is already retained as an immutable snapshot.
        This alias always points to the strongest *training-derived candidate*,
        and is never fed directly into the held-out Test.
        """
        if not self.keep_all or not self.rows:
            raise RuntimeError("Training-best alias requires keep-all actor snapshots")
        import shutil
        row = self.rows[-1]
        if "training_metric_best_candidate" not in row.get("reasons", []):
            raise RuntimeError("Training candidate was not captured or labeled")
        if row.get("training_metric_candidate") != candidate:
            raise RuntimeError("Training candidate metadata disagrees with recorded snapshot")
        src = self.folder / row["file"]
        alias = self.folder / "training_metric_best.pt"
        tmp = self.folder / "training_metric_best.pt.tmp"
        shutil.copyfile(src, tmp)
        tmp.replace(alias)
        info = {
            "kind": "training_metric_candidate_only",
            "source_file": row["file"],
            "source_policy_version": row["policy_version"],
            "source_global_step": row["global_step"],
            "source_sha256": sha256(src),
            "alias_sha256": sha256(alias),
            "candidate": candidate,
            "may_select_final_best": False,
        }
        output = self.folder / "training_metric_best.json"
        temporary = self.folder / "training_metric_best.json.tmp"
        temporary.write_text(json.dumps(info, indent=2, allow_nan=False))
        temporary.replace(output)
        return info

    def finalize(self):
        if not self.rows:
            manifest = {"schema": 1, "kind": "deferred_gpu_eval_actor_snapshots",
                        "target": ("all" if self.keep_all else self.target), "count": 0, "snapshots": []}
        else:
            # Keep all distinct policy versions and prefer checkpoints spread
            # across *actual training collection steps*, never interpolate.
            candidates = []
            for row in self.rows:
                if candidates and row["policy_version"] == candidates[-1]["policy_version"]:
                    prev = candidates.pop()  # retain latest actual checkpoint for this version
                    if prev["file"] != row["file"]:
                        if self.keep_all:
                            raise RuntimeError("keep_all capture produced duplicate policy versions")
                        (self.folder / prev["file"]).unlink()
                        self.pruned += 1
                candidates.append(row)
            if not self.keep_all and len(candidates) > self.target:
                # Equal temporal quantiles, including earliest and latest.
                k = len(candidates)
                selected_indices = {(i * (k - 1)) // (self.target - 1)
                                    for i in range(self.target)} if self.target > 1 else {k - 1}
                chosen = [r for i, r in enumerate(candidates) if i in selected_indices]
            else:
                chosen = candidates
            chosen_names = {row["file"] for row in chosen}
            for row in candidates:
                if row["file"] not in chosen_names:
                    (self.folder / row["file"]).unlink()
                    self.pruned += 1
            self.rows = chosen
            manifest = {"schema": 1, "kind": "deferred_gpu_eval_actor_snapshots",
                        "target": ("all" if self.keep_all else self.target), "count": len(chosen), "collected": self.saved,
                        "pruned": self.pruned, "keep_all_distinct_versions": self.keep_all, "final_stride": self.stride,
                        "validation_seed_first": self.config.tracking.periodic_evaluation_seed,
                        "validation_episodes": self.config.tracking.periodic_evaluation_episodes,
                        "snapshots": [{**row, "sha256": sha256(self.folder / row["file"])}
                                      for row in chosen]}
        path = self.folder / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        temporary.replace(path)
        return manifest

    def bundle(self):
        manifest = self.finalize()
        archive = self.folder.parent / "deferred-gpu-eval.tar.gz"
        temp = archive.with_suffix(".tar.gz.tmp")
        with tarfile.open(temp, "w:gz", compresslevel=1) as out:
            out.add(self.folder / "manifest.json", arcname="deferred-gpu-eval/manifest.json")
            for row in manifest["snapshots"]:
                out.add(self.folder / row["file"],
                        arcname="deferred-gpu-eval/" + row["file"])
            if self.keep_all:
                for optional in ("training_metric_best.pt", "training_metric_best.json",
                                 "capture-journal.jsonl"):
                    local = self.folder / optional
                    if local.is_file():
                        out.add(local, arcname="deferred-gpu-eval/" + optional)
        temp.replace(archive)
        return archive, manifest


def publish_bundle(archive, *, run_id, repo_id="duy95/uav-search-diagnostics"):
    """Best effort, only if explicitly opted in and HF_TOKEN is granted."""
    import os
    if os.environ.get("UAV_DEFERRED_EXPORT_HF", "false").lower() != "true":
        return None
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Deferred HF export requires granted HF_TOKEN")
    if not run_id or not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid W&B run identifier for immutable HF export")
    from huggingface_hub import HfApi
    remote = "deferred-gpu-eval/" + run_id + "/" + archive.name
    HfApi(token=token).upload_file(
        path_or_fileobj=str(archive), path_in_repo=remote,
        repo_id=repo_id, repo_type="dataset",
        commit_message="Immutable deferred GPU evaluation actors " + run_id)
    return {"repo_id": repo_id, "repo_type": "dataset", "path": remote,
            "sha256": sha256(archive)}
