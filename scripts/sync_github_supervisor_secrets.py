#!/usr/bin/env python3
"""Sync external-supervisor credentials from key.txt into GitHub safely."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

ACCOUNT_RE = re.compile(r"^account_(\d{2})$")


def _gh_command(kind: str, name: str, repo: str | None) -> list[str]:
    command = ["gh", kind, "set", name]
    if repo:
        command += ["--repo", repo]
    return command


def _set_stdin(kind: str, name: str, value: str, repo: str | None, run) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    run(_gh_command(kind, name, repo), input=value, text=True, check=True,
        capture_output=True)


def sync_credentials(key_file: str | Path, *, repo: str | None = None,
                     run=subprocess.run) -> dict[str, list[str]]:
    data = json.loads(Path(key_file).read_text())
    accounts = data.get("kaggle", {}).get("accounts", [])
    if not isinstance(accounts, list) or not accounts:
        raise ValueError("key.txt needs kaggle.accounts")

    secrets: list[tuple[str, str]] = []
    enabled_accounts: list[str] = []
    for item in accounts:
        if not isinstance(item, dict) or item.get("enabled", True) is not True:
            continue
        name, token = item.get("name"), item.get("token")
        match = ACCOUNT_RE.fullmatch(str(name or ""))
        if not match or not isinstance(token, str) or not token:
            raise ValueError("Enabled Kaggle accounts require account_NN names and tokens")
        account_name = f"account_{match.group(1)}"
        enabled_accounts.append(account_name)
        secrets.append((f"KAGGLE_API_TOKEN_ACCOUNT_{match.group(1)}", token))

    wandb = data.get("wandb", {})
    hf = data.get("huggingface", {})
    secrets += [("WANDB_API_KEY", wandb.get("api_key")),
                ("HF_TOKEN", hf.get("token"))]
    variables = [("WANDB_ENTITY", wandb.get("entity")),
                 ("WANDB_PROJECT", wandb.get("project")),
                 ("HF_REPO_ID", hf.get("repo_id")),
                 ("KAGGLE_ENABLED_ACCOUNTS", ",".join(enabled_accounts))]

    for name, value in secrets:
        _set_stdin("secret", name, value, repo, run)
        print(f"set secret {name}")
    for name, value in variables:
        _set_stdin("variable", name, value, repo, run)
        print(f"set variable {name}")
    return {"secrets": sorted(name for name, _ in secrets),
            "variables": sorted(name for name, _ in variables)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sync GitHub external-supervisor credentials")
    parser.add_argument("--key-file", default=str(Path(__file__).resolve().parents[1] / "key.txt"))
    parser.add_argument("--repo", help="OWNER/REPO; defaults to the current gh repository")
    args = parser.parse_args(argv)
    result = sync_credentials(args.key_file, repo=args.repo)
    print(f"synced {len(result['secrets'])} secrets and {len(result['variables'])} variables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
