#!/usr/bin/env python3
"""Reproducible local scenario generator and optional checksum-pinned downloader.

Does not clone third-party code or include model weights. Writes *oracle*
target locations to outputs only; they must never enter Actor observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SAFE_ID = re.compile(r'^[a-zA-Z0-9_-]{1,72}$')
HEX_SHA256 = re.compile(r'^[a-fA-F0-9]{64}$')


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or set(manifest) != {'sources'}:
        raise ValueError("Source manifest must contain only 'sources'")
    sources = manifest['sources']
    if not isinstance(sources, list) or not sources:
        raise ValueError('At least one declared data source is required')
    seen: set[str] = set()
    for src in sources:
        if not isinstance(src, dict) or src.get('kind') not in ('generated', 'download'):
            raise ValueError('Source kind must be generated/download')
        source_id = src.get('id')
        if not isinstance(source_id, str) or SAFE_ID.fullmatch(source_id) is None or source_id in seen:
            raise ValueError('Each data source needs a unique safe ID')
        seen.add(source_id)
        if src['kind'] == 'generated':
            if set(src) != {'id', 'kind', 'scenario_config'}:
                raise ValueError('Generated source requires scenario_config only')
            config_path = src['scenario_config']
            if not isinstance(config_path, str) or not (ROOT / config_path).resolve().is_relative_to(ROOT):
                raise ValueError('Scenario config must be inside the source repository')
        else:
            if set(src) != {'id', 'kind', 'url', 'sha256'}:
                raise ValueError('Download source must include HTTPS URL and SHA-256')
            url = src['url']; digest = src['sha256']
            if not isinstance(url, str) or not url.startswith('https://'):
                raise ValueError('Only public HTTPS dataset downloads are permitted')
            if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None:
                raise ValueError('A pinned SHA-256 is required for downloads')
    return sources


def _fetch_pinned(src: dict[str, Any], output: Path) -> dict[str, Any]:
    destination = output / f"{src['id']}.bin"
    h = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=output, prefix='.download-', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(src['url'], timeout=30) as resp:
                if not resp.geturl().startswith('https://'):
                    raise ValueError('Redirect to non-HTTPS dataset URL blocked')
                while chunk := resp.read(1 << 20):
                    h.update(chunk)
                    tmp.write(chunk)
            if h.hexdigest().lower() != src['sha256'].lower():
                raise ValueError('Downloaded dataset SHA-256 mismatch; refusing to keep file')
            tmp_path.replace(destination)
        finally:
            tmp_path.unlink(missing_ok=True)
    return {'id':src['id'], 'kind':'download', 'file':destination.name, 'sha256':h.hexdigest()}


def prepare_data(manifest_path: Path, output_dir: Path, *, seed: int, episodes: int) -> dict:
    if not 1 <= episodes <= 128:
        raise ValueError('episodes must be between 1 and 128')
    if not 0 <= seed <= (2**32 - 1) - episodes:
        raise ValueError('seed+episodes outside uint32 range')
    sources = validate_manifest(yaml.safe_load(manifest_path.read_text(encoding='utf-8')))
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {'kind':'generated-synthetic-no-external-dataset', 'seed':seed,
                              'episodes':[], 'downloads':[]}
    for src in sources:
        if src['kind'] == 'download':
            result['downloads'].append(_fetch_pinned(src, output_dir))
            continue
        from zhang_xue.configuration.loader import load_scenario
        from zhang_xue.environment.core import ZhangXue7NodePacketEnv
        from zhang_xue.communication.common import CANONICAL_BUILDINGS_5
        config_path = ROOT / src['scenario_config']
        cfg = load_scenario(config_path)
        env = ZhangXue7NodePacketEnv(cfg, seed=seed)
        for index in range(episodes):
            sample_seed = seed + index
            env.reset(seed=sample_seed)
            target_cells = [int(x) for x in env.true_targets.nonzero()[0].tolist()]
            result['episodes'].append({'source':src['id'], 'seed':sample_seed,
                                       'target_cells':target_cells,
                                       'buildings_3d':CANONICAL_BUILDINGS_5})
    (output_dir / 'metadata.json').write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)+'\n',
        encoding='utf-8',
    )
    return result


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=ROOT/'configs/data/sources.yaml')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs/data/canonical_7radio')
    ap.add_argument('--seed',type=int,default=44)
    ap.add_argument('--episodes',type=int,default=4)
    args=ap.parse_args()
    result=prepare_data(args.manifest,args.output_dir,seed=args.seed,episodes=args.episodes)
    print(json.dumps({'status':'prepared','episodes':len(result['episodes']),
                      'downloads':len(result['downloads']),'path':str(args.output_dir)}))


if __name__=='__main__':
    main()
