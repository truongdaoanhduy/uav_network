"""Low-storage diagnostic evidence: immutable incremental .tar.gz on Hugging Face.

The old implementation uploaded full .jsonl.gz at finish and additionally
registered several uncompressed .jsonl files as W&B live files. That doubled
storage and lost final-only evidence if Kaggle killed the job. This implementation
packages only NEW complete JSONL lines in immutable gzip-compressed tar parts,
one upload per part, periodically and on exit. W&B numerical charts are
independent and continue to update online; raw journal archives are public
Hugging Face dataset files, never W&B Files.

No raw .jsonl files are uploaded to W&B by this module. Original JSONL remains
on disk for existing training/evaluation/resume/diagnostic consumers.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path
import shutil
import tarfile
import time


ARCHIVE_SOURCES = (
    "training-step-reward.jsonl",
    "training-step-progress.jsonl",
    "training-step-optimizer.jsonl",
    "training-events.jsonl",
    "failure-events.jsonl",  # sanitized traceback and failure context, not bare secrets
    "diagnostics.jsonl",
    "cohorts.jsonl",
    "episodes.jsonl",
    "action-cbf-samples.jsonl",
    "mission-cause-snapshots.jsonl",
    "training-health-snapshots.jsonl",
    "learner-layer-snapshots.jsonl",
    "root-cause-timeline.jsonl",
    "forensic-minibatches.jsonl",
    "forensic-metric-integrity.jsonl",
    "forensic-transitions.jsonl",
    "run-provenance.jsonl",
    "evaluations/events.jsonl",
    "thread-stacks.log",
)

DEFAULT_INTERVAL_SECONDS = 300  # W&B charts continue live; archives every ~5 min.
DEFAULT_MAX_RAW_BYTES = 8 * 1024 * 1024  # One bounded, immutable compressed part.


class _SliceReader(io.RawIOBase):
    """Read precisely a source file byte range; don't load journals into RAM."""

    def __init__(self, file, size):
        self.file = file
        self.remaining = size

    def readable(self):
        return True

    def read(self, size=-1):
        if not self.remaining:
            return b""
        size = min(self.remaining, 1024 * 1024 if size is None or size < 0 else size)
        data = self.file.read(size)
        self.remaining -= len(data)
        return data


def _complete_line_end(source: Path, start: int, max_length: int) -> int:
    """Find last complete JSONL newline before cap; leave incomplete rows alone."""
    try:
        size = source.stat().st_size
    except OSError:
        return start
    if size <= start:
        return start
    cap = min(size, start + max_length)
    with source.open("rb") as file:
        end = cap
        while end > start:
            left = max(start, end - 8192)
            file.seek(left)
            block = file.read(end - left)
            position = block.rfind(b"\n")
            if position >= 0:
                return left + position + 1
            end = left
        # A single JSONL line above cap is unexpected; do not upload partial
        # invalid JSON. It remains local and can be inspected if necessary.
    return start


class CompressedDiagnosticArchive:
    """Incremental, atomic and retryable gzip archives with no raw W&B uploads."""

    def __init__(self, output, run, *, interval_seconds=DEFAULT_INTERVAL_SECONDS,
                 max_raw_bytes=DEFAULT_MAX_RAW_BYTES, now=None):
        self.output = Path(output)
        self.folder = self.output / "diagnostic-archives"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.now = now or time.monotonic
        self.interval = float(interval_seconds)
        self.max_raw_bytes = int(max_raw_bytes)
        if self.interval < 0 or self.max_raw_bytes < 1024:
            raise ValueError("Invalid archive interval or maximum raw chunk")
        self.last_upload = self.now()
        self.offsets = {name: 0 for name in ARCHIVE_SOURCES}
        self.pending = []
        self.parts_created = self.parts_registered = 0
        self.raw_bytes_archived = self.gz_bytes_created = 0
        self.upload_errors = 0
        self.consecutive_upload_failures = 0
        self.next_upload_attempt_at = 0.0

    def _retry_delay(self, error):
        # Honor Hub 429 throttling without blocking the MATD3 learner.
        response = getattr(error, "response", None)
        if getattr(response, "status_code", None) == 429 or "429" in str(error):
            import re
            headers = getattr(response, "headers", {}) or {}
            hint = headers.get("Retry-After") or headers.get("retry-after")
            if not hint:
                match = re.search(r"Retry after (\d+) seconds", str(error))
                hint = match.group(1) if match else None
            try:
                wait = float(hint) if hint else 300.
            except (TypeError, ValueError):
                wait = 300.
            return max(300., wait + 2.)
        return float(min(30, 2 ** min(self.consecutive_upload_failures, 5)))

    def _deltas(self):
        result = {}
        remaining = self.max_raw_bytes
        for name in ARCHIVE_SOURCES:
            if remaining < 1024:
                break
            source = self.output / name
            start = self.offsets[name]
            if not source.is_file():
                continue
            end = _complete_line_end(source, start, remaining)
            if end > start:
                result[name] = (start, end)
                remaining -= end - start
        return result

    def _new_bytes(self):
        # Cheap stat calls; avoids compressing very small logs on every event.
        return sum(max(0, (self.output / name).stat().st_size - self.offsets[name])
                   for name in ARCHIVE_SOURCES if (self.output / name).is_file())

    def _create_part(self, deltas):
        index = self.parts_created + 1
        path = self.folder / f"diagnostics-{index:06d}.tar.gz"
        tmp = path.with_suffix(".tar.gz.tmp")
        try:
            with tarfile.open(tmp, "w:gz", compresslevel=6) as archive:
                for name, (start, end) in deltas.items():
                    info = tarfile.TarInfo(name)
                    info.size = end - start
                    info.mtime = 0
                    info.mode = 0o600
                    with (self.output / name).open("rb") as src:
                        src.seek(start)
                        archive.addfile(info, _SliceReader(src, info.size))
            # Verify gzip/tar integrity before registering the W&B upload.
            with tarfile.open(tmp, "r:gz") as checked:
                members = checked.getmembers()
                if len(members) != len(deltas) or sorted(m.name for m in members) != sorted(deltas):
                    raise IOError("Incomplete diagnostic archive")
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        for name, (_, end) in deltas.items():
            self.offsets[name] = end
        self.parts_created += 1
        self.raw_bytes_archived += sum(end-start for start,end in deltas.values())
        self.gz_bytes_created += path.stat().st_size
        self.pending.append(path)
        return path

    def _publish_pending(self):
        if self.run is None:
            # Offline smoke runs have no W&B client; retain immutable
            # verified archives locally, with no attempted remote upload.
            self.pending.clear()
            self.last_upload = self.now()
            return
        if self.now() < self.next_upload_attempt_at:
            self.last_upload = self.now()
            return
        still_pending = []
        for index, path in enumerate(self.pending):
            try:
                # A successful transfer is acknowledged exactly once.
                self.run.save(str(path), base_path=str(self.output), policy="now")
                self.parts_registered += 1
                self.consecutive_upload_failures = 0
            except Exception as error:
                # Stop immediately on 429. The prior implementation retried
                # every pending archive on each tick, causing thousands of
                # API calls and hours of wall-clock learner stalls.
                self.upload_errors += 1
                self.consecutive_upload_failures += 1
                self.next_upload_attempt_at = self.now() + self._retry_delay(error)
                still_pending.extend(self.pending[index:])
                break
        else:
            self.next_upload_attempt_at = 0.
        self.pending = still_pending
        self.last_upload = self.now()

    def tick(self, *, final=False):
        """Best-effort compression at cadence; force all remaining on final."""
        if not final and self.now() - self.last_upload < self.interval \
                and self._new_bytes() < self.max_raw_bytes:
            return 0
        created = 0
        # Finish can involve multiple parts, each bounded by max raw bytes.
        while True:
            delta = self._deltas()
            if not delta:
                break
            self._create_part(delta)
            created += 1
            if not final:
                break
        if created or self.pending or final:
            self._publish_pending()
        return created

    def stats(self):
        return {
            "storage/diagnostic_gz_parts_created":self.parts_created,
            "storage/diagnostic_gz_parts_registered":self.parts_registered,
            "storage/diagnostic_gz_pending_parts":len(self.pending),
            "storage/diagnostic_raw_bytes_archived":self.raw_bytes_archived,
            "storage/diagnostic_gz_bytes_created":self.gz_bytes_created,
            "storage/diagnostic_compression_ratio":self.raw_bytes_archived / max(1,self.gz_bytes_created),
            "storage/diagnostic_upload_errors":self.upload_errors,
            "storage/diagnostic_local_only":int(self.run is None),
        }


def upload_compressed_journals(tracker):
    """Finalize any remaining incremental archives without duplicate uploads."""
    writer = getattr(tracker, "compressed_journals", None)
    if writer is None:
        return
    try:
        writer.tick(final=True)
        # Retry previously failed remote parts; readback verification happens
        # inside PublicHubDiagnosticUploader.save before a part is acknowledged.
        if writer.pending and writer.run is not None:
            import time
            for attempt in range(2):
                time.sleep(1 + attempt)
                writer.tick(final=True)
                if not writer.pending:
                    break
        if getattr(tracker, "run", None) is not None:
            tracker.run.summary.update(writer.stats())
            manifest=verify_archive_parts(writer.output)
            tracker.run.summary.update({
                "storage/archive_local_byte_reconstruction_verified":
                    bool(manifest["full_reconstruction_verified"]),
                "storage/archive_part_count":len(manifest["parts"]),
                "storage/archive_unarchived_tail_bytes":
                    sum(max(0,int(v)) for v in manifest["unarchived_tail_bytes"].values()),
                "storage/hf_incomplete":bool(writer.pending),
            })
            destination=getattr(tracker,"public_diagnostics",None)
            # Do not upload a misleading final manifest if remote parts are
            # missing; local manifest remains for recovery from Kaggle outputs.
            if destination is not None and not writer.pending:
                try:
                    destination.save(str(writer.folder/"manifest.json.gz"),
                                     base_path=str(writer.output),policy="now")
                except Exception:
                    tracker.run.summary["storage/hf_manifest_upload_failed"] = True
                    tracker.run.summary["storage/hf_incomplete"] = True
                tracker.run.summary.update(destination.stats())
    except Exception as error:
        if getattr(tracker, "run", None) is not None:
            tracker.run.summary["storage/diagnostic_finalize_error_type"] = type(error).__name__
        if getattr(tracker, "journal", None) is not None:
            tracker.journal._warn(error)


def save_small_compressed_file(run, path, *, base_path):
    """Upload a single JSON/text/log diagnostic as .gz, never the raw file.

    Intended for immutable per-checkpoint evaluation and rare failure reports,
    not repeatedly growing raw journals (those belong in the tar.gz archiver).
    """
    source = Path(path)
    if not source.is_file():
        return None
    target = source.with_name(source.name + ".gz")
    tmp = target.with_name(target.name + ".tmp")
    try:
        with source.open("rb") as src, gzip.open(tmp,"wb",compresslevel=6) as dst:
            shutil.copyfileobj(src,dst,length=1024*1024)
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    run.save(str(target), base_path=str(base_path), policy="now")
    return target


def verify_archive_parts(output):
    """Read every member and compare archived content with source byte slices.

    The returned metadata is deterministic and suitable for a postmortem
    audit; unarchived tail bytes are reported explicitly. No network I/O.
    """
    from hashlib import sha256
    import json
    base = Path(output)
    folder = base/"diagnostic-archives"
    archives = sorted(folder.glob("diagnostics-*.tar.gz"))
    offsets = {name:0 for name in ARCHIVE_SOURCES}
    descriptors = []
    for archive in archives:
        members = []
        with tarfile.open(archive,"r:gz") as tar:
            for member in tar:
                if not member.isfile() or member.name not in offsets:
                    raise ValueError("Unexpected or non-file diagnostic member")
                with tar.extractfile(member) as stream:
                    payload = stream.read()
                source = base/member.name
                with source.open("rb") as stream:
                    stream.seek(offsets[member.name])
                    expected=stream.read(len(payload))
                if payload != expected:
                    raise IOError(f"Archive byte mismatch: {archive.name}:{member.name}")
                offsets[member.name] += len(payload)
                members.append({"name":member.name,"bytes":len(payload),
                                "sha256":sha256(payload).hexdigest()})
        descriptors.append({"file":archive.name,"bytes":archive.stat().st_size,
                            "sha256":sha256(archive.read_bytes()).hexdigest(),
                            "members":members})
    tails={name:int((base/name).stat().st_size)-offset
           for name,offset in offsets.items() if (base/name).is_file()}
    def file_hash(path):
        digest=sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):
                digest.update(block)
        return digest.hexdigest()
    manifest={
        "schema":1,"archive_format":"immutable_gzip_tar_parts",
        "source_files_sha256":{name:file_hash(base/name)
                               for name in offsets if (base/name).is_file()},
        "parts":descriptors, "unarchived_tail_bytes":tails,
        "full_reconstruction_verified":all(v==0 for v in tails.values()),
    }
    import gzip
    folder.mkdir(parents=True,exist_ok=True)
    with gzip.open(folder/"manifest.json.gz","wt",encoding="utf-8",compresslevel=6) as target:
        json.dump(manifest,target,sort_keys=True)
    return manifest

def finalize_local_diagnostics(output):
    """End-of-run offline packaging. Does not communicate with W&B."""
    local=CompressedDiagnosticArchive(output,None)
    local.tick(final=True)
    integrity=verify_archive_parts(output)
    return {"archive":local.stats(),
            "manifest":str(Path(output)/"diagnostic-archives/manifest.json.gz"),
            "reconstruction_verified":integrity["full_reconstruction_verified"],
            "archives":len(integrity["parts"]),
            "unarchived_tail_bytes":integrity["unarchived_tail_bytes"]}
