"""End-of-run compressed journal uploads to avoid duplicating large W&B files."""
from pathlib import Path
import gzip
import shutil


def upload_compressed_journals(tracker):
    """Keep local raw JSONL for investigation; upload verified gzip to W&B Files.

    This never touches training History or model artifacts. W&B may retain its
    separate output.log, now made small by printing compact event summaries.
    """
    for name in ("training-events.jsonl", "diagnostics.jsonl"):
        source = Path(tracker.output) / name
        if not source.is_file():
            continue
        target = source.with_suffix(source.suffix + ".gz")
        try:
            with source.open("rb") as source_stream, gzip.open(target, "wb", compresslevel=6) as packed:
                shutil.copyfileobj(source_stream, packed)
            tracker.run.save(str(target), base_path=str(tracker.output), policy="now")
        except (OSError, ValueError, RuntimeError) as error:
            if tracker.journal is not None:
                tracker.journal._warn(error)
