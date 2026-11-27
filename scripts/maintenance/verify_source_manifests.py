#!/usr/bin/env python3
"""Write/check independent full-tree and scenario SHA256 provenance manifests.

Tracks git-source only; never untracked downloads, caches or credentials.
Run --write after staging all intended source changes; --check is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
try:
    from scripts.maintenance.source_inventory import source_paths
except ModuleNotFoundError:  # Standalone script: scripts/ is the first sys.path entry.
    from source_inventory import source_paths
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOT_MANIFEST = 'SOURCE_SHA256_MANIFEST.json'
SCENARIO_MANIFEST = 'research/zhang_xue/SHA256_MANIFEST.json'
OMIT = {ROOT_MANIFEST, SCENARIO_MANIFEST}


def expected() -> tuple[dict[str, str], dict[str, str]]:
    paths = source_paths(ROOT)
    root_hashes: dict[str, str] = {}
    scenario_hashes: dict[str, str] = {}
    for name in sorted(set(paths) - {'', *OMIT}):
        file = ROOT / name
        if not file.is_file():
            raise RuntimeError(f'Git contains deleted or missing file: {name}')
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        root_hashes[name] = digest
        if name.startswith('research/zhang_xue/'):
            scenario_hashes[name.removeprefix('research/zhang_xue/')] = digest
    return root_hashes, scenario_hashes


def run(*, update: bool) -> int:
    root_hashes, scenario_hashes = expected()
    for name, generated in ((ROOT_MANIFEST, root_hashes),
                            (SCENARIO_MANIFEST, scenario_hashes)):
        output = ROOT / name
        if update:
            output.write_text(json.dumps(generated, indent=2, sort_keys=True) + '\n')
            print(f'WROTE {name}: {len(generated)} checksums')
        else:
            if not output.is_file() or json.loads(output.read_text()) != generated:
                print(f'FAIL {name}: missing, stale or incomplete source manifest')
                return 1
            print(f'PASS {name}: {len(generated)} source checksums')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument('--write', action='store_true')
    flags.add_argument('--check', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(update=args.write))
