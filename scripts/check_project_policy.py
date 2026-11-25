#!/usr/bin/env python3
"""Fail-closed governance for new/changed tracked source, with explicit legacy debt.

Changed mode protects PRs without claiming 2026-11 pre-existing 500-line files
already comply. Full mode audits all tracked files and must currently fail.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".txt", ".rst"}
MAX_SOURCE_LINES = 500
MAX_NEW_ARTIFACT_BYTES = 250_000
BLOCKED_COMPONENTS = {
    ".venv", ".pytest_cache", "__pycache__", "outputs", "wandb",
    "node_modules", ".ruff_cache", "build", "dist", ".git",
}
BLOCKED_FILENAMES = {"key.txt", "kaggle.json", ".env", "id_rsa", "id_ed25519"}
TOKEN = re.compile(r"(?:gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_-]{24,}")
RESEARCH_ROOT = "research/zhang_xue/"
LEGACY_ADAPTERS = {
    "uav_v54_7node_env.py",
    "uav_v54_7radio_csmaca_sim.py",
    "uav_v54_ds_and_battery.py",
    "v56_warp_7radio_mac.py",
    "v56_warp_reset_kernels.py",
    "v56_warp_event_kernels.py",
    "v56_warp_injection_kernels.py",
    "v55_state_derived_replay.py",
    "v55_warp_actor_maps.py",
    "v57_warp_phy_kernels.py",
    "zhang_xue_v54_7node_warp.py",
    "train_v55_packet_cpu.py",
}


def git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        text=True, capture_output=True, check=True,
    )
    return [s for s in result.stdout.splitlines() if s]


def sources(scope: str, base: str | None) -> list[str]:
    if scope == "full":
        return sorted(set(git("ls-files")))
    if not base:
        raise ValueError("changed scope requires --base (immutable commit SHA/ref)")
    paths = set(git("diff", "--name-only", "--diff-filter=ACMR", base, "HEAD"))
    paths.update(git("diff", "--name-only", "--diff-filter=ACMR", "HEAD"))
    paths.update(git("ls-files", "--others", "--exclude-standard"))
    return sorted(paths)


def violations(path: str, *, full: bool) -> list[str]:
    problems = []
    file = ROOT / path
    parts = Path(path).parts
    if not file.is_file() or any(p in BLOCKED_COMPONENTS for p in parts):
        if file.is_file() and any(p in BLOCKED_COMPONENTS for p in parts):
            problems.append("generated/downloaded dependency must be gitignored")
        return problems

    name = file.name.lower()
    if name in BLOCKED_FILENAMES or name.startswith("key.") or name.startswith(".env."):
        problems.append("credential or private environment file")
    try:
        payload = file.read_bytes()
    except OSError as error:
        return [f"cannot read: {error}"]
    if TOKEN.search(payload.decode("utf-8", errors="ignore")):
        problems.append("possible access token in source")

    if file.suffix in SOURCE or name in ("readme", "license"):
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            problems.append("non-UTF8 source file")
            return problems
        if len(lines) > MAX_SOURCE_LINES:
            problems.append(f"source exceeds {MAX_SOURCE_LINES} lines ({len(lines)})")
        if file.suffix == ".py":
            try:
                ast.parse("\n".join(lines), filename=path)
            except SyntaxError as error:
                problems.append(f"invalid Python syntax: {error.msg}")
    elif file.suffix.lower() in (".ipynb", ".zip", ".pt", ".pth", ".ckpt", ".npz",
                                  ".parquet", ".h5", ".hdf5", ".csv", ".log"):
        if len(payload) > MAX_NEW_ARTIFACT_BYTES:
            problems.append(f"generated/heavy artifact {len(payload)} bytes; use outputs/ or external store")
        if file.suffix.lower() in (".pt", ".pth", ".ckpt"):
            problems.append("model weights are not source code; exclude from git")

    # Newly introduced research code should have a folder/namespace contract.
    if not full and path.startswith(RESEARCH_ROOT + "src/") and file.suffix == ".py":
        rel = path[len(RESEARCH_ROOT + "src/"):]
        if "/" not in rel and rel not in LEGACY_ADAPTERS:
            problems.append("new source must live in a named package folder; flat files are compatibility-only")
        if "/" in rel and not rel.startswith("zhang_xue/"):
            problems.append("new source must live under src/zhang_xue/")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("changed", "full"), default="changed")
    parser.add_argument("--base", help="Base Git revision for --scope changed")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        files = sources(args.scope, args.base)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    errors = {name: findings for name in files
              if (findings := violations(name, full=args.scope == "full"))}
    report = {
        "scope": args.scope,
        "base": args.base,
        "examined_files": len(files),
        "violation_files": len(errors),
        "violations": errors,
        "result": "PASS" if not errors else "FAIL",
    }
    if args.json_out:
        target = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        if not target.resolve().is_relative_to((ROOT / "outputs").resolve()):
            parser.error("--json-out must be under gitignored outputs/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"PROJECT_POLICY {report['result']} | scope={args.scope} "
          f"files={len(files)} violations={len(errors)}")
    for name, issue in list(errors.items())[:35]:
        print(f"  {name}: {'; '.join(issue)}")
    if len(errors) > 35:
        print(f"  ... {len(errors) - 35} additional files; --json-out for complete report")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
