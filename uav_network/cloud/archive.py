"""CPU-only checkpoint migration; delete only after verified private backup."""
import argparse
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from ..configuration import Config
from ..experiments.checkpoint_store import (MANIFEST, upload_checkpoint_files,
    download_manifest_file, file_sha256)
from ..experiments.tracking import OnlineTracker


_PATTERN = r'([\w.-]+/[\w.-]+)/(?P<collection>uav-network-(?:masac|matd3)-(?P<run>[0-9a-f]{8})):v(?P<version>[0-9]+)'


def inventory_legacy_artifacts(api, entity, project, protected):
    """Read-only inventory; fail closed unless a legacy version is clearly removable."""
    records = []
    terminal = {'finished', 'failed', 'crashed', 'killed'}
    for run in api.runs(f'{entity}/{project}'):
        for artifact in run.logged_artifacts():
            ref = getattr(artifact, 'qualified_name', '')
            match = re.fullmatch(_PATTERN, ref)
            if match is None:
                continue
            names = set(artifact.manifest.entries)
            candidate = False
            if names == {MANIFEST}:
                status = 'already_external'
            elif not names or not names.issubset({'checkpoint.pt', 'resume.pt'}):
                status = 'out_of_scope'
            elif match['run'] in protected:
                status = 'protected'
            elif run.state != 'finished' or getattr(run, 'id', None) != match['run']:
                status = 'source_not_finished_or_mismatched'
            else:
                try:
                    active = any(consumer.state not in terminal for consumer in artifact.used_by())
                except Exception:
                    active = True
                    status = 'dependency_unknown'
                else:
                    status = 'active_consumer' if active else 'candidate'
                    candidate = not active
            records.append({'ref': ref, 'source_run_id': match['run'], 'status': status,
                            'candidate': candidate, 'files': sorted(names)})
    return records


def archive_artifact(api, tracker, config, ref, directory, protected):
    match = re.fullmatch(_PATTERN, ref)
    if match is None:
        raise ValueError('Only scoped immutable UAV checkpoint versions may be archived')
    run_id = match['run']
    if run_id in protected:
        raise RuntimeError('Source belongs to a protected production run')
    source_run = api.run(match[1] + '/' + run_id)
    if source_run.state != 'finished':
        raise RuntimeError('Source run is still active or has not finished successfully')
    artifact = api.artifact(ref, type='model')
    if any(run.state not in {'finished', 'failed', 'crashed', 'killed'} for run in artifact.used_by()):
        raise RuntimeError('Source artifact has an active consumer')
    names = set(artifact.manifest.entries)
    if names == {MANIFEST}:
        return {'original_ref': ref, 'status': 'already_external'}
    if not names or not names.issubset({'checkpoint.pt', 'resume.pt'}):
        raise ValueError('Artifact includes files outside checkpoint migration scope')
    folder = Path(directory) / 'source'
    downloaded = Path(artifact.download(root=str(folder), skip_cache=True))
    files = {name: downloaded / name for name in names}
    descriptor = upload_checkpoint_files(config, files,
        f"archive/{match['collection']}/v{match['version']}")
    # A different local directory and forced download prove data can be read back.
    for name in files:
        restored = download_manifest_file(descriptor, name, Path(directory) / 'fresh', force_download=True)
        if file_sha256(restored) != descriptor['files'][name]['sha256']:
            raise RuntimeError('Fresh remote backup verification failed; source retained')
    import wandb
    manifest_path = Path(directory) / MANIFEST
    manifest_path.write_text(json.dumps(descriptor, indent=2))
    replacement = wandb.Artifact(match['collection'], type='model',
        metadata={**artifact.metadata, 'archived_from': ref, 'archive_verified': True})
    replacement.add_file(str(manifest_path), name=MANIFEST)
    aliases = [a for a in artifact.aliases if not re.fullmatch(r'v[0-9]+', a)]
    saved = tracker.run.log_artifact(replacement, aliases=[*aliases, 'archive-of-v' + match['version']])
    saved.wait(timeout=config.tracking.artifact_timeout_seconds)
    new_ref = saved.qualified_name
    archives = dict(source_run.summary.get('checkpoint_archives', {}))
    archives[ref] = new_ref
    source_run.summary['checkpoint_archives'] = archives
    if {'latest', 'final', 'best'} & set(aliases):
        source_run.summary['checkpoint_artifact'] = new_ref
    source_run.update()
    # Recheck after slow transfer: a newly started consumer also protects the source.
    if api.run(match[1] + '/' + run_id).state != 'finished' or any(run.state not in {'finished', 'failed', 'crashed', 'killed'}
                                            for run in artifact.used_by()):
        raise RuntimeError('Source acquired an active consumer during backup; source retained')
    record = {'original_ref': ref, 'replacement_ref': new_ref, 'manifest': descriptor,
              'bytes': sum(x['size'] for x in descriptor['files'].values()), 'status': 'verified'}
    (Path(directory) / 'archive-record.json').write_text(json.dumps(record, indent=2))
    artifact.delete(delete_aliases=True)
    return {**record, 'status': 'archived', 'source_deleted': True}


def run_archive(spec, output=Path('/kaggle/working')):
    import torch
    if torch.cuda.is_available():
        raise RuntimeError('Checkpoint maintenance must use a CPU-only session')
    import wandb
    from .client import kernel_status
    from .credentials import Account
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.runtime.device = config.runtime.replay_device = 'cpu'
    config.tracking.enabled = True
    config.tracking.entity, config.tracking.project = spec['entity'], spec['project']
    config.tracking.checkpoint_repo = spec['checkpoint_repo']
    config.tracking.artifact_timeout_seconds = 900
    config.tracking.run_name = spec['runs']['archive']['name']
    os.environ['WANDB_RUN_ID'] = spec['runs']['archive']['id']
    os.environ['UAV_STAGE'] = 'archive'
    tracker = OnlineTracker(config, output)
    tracker.run.summary['stage'] = 'archive'
    api = wandb.Api()
    records = []; failed = True
    try:
        for run_id, cpu_ref in spec.get('completed_cpu_checks', {}).items():
            account = Account(spec['account'], os.environ['KAGGLE_API_TOKEN'])
            if cpu_ref.split('/')[0] != spec['kernel_ref'].split('/')[0] or kernel_status(cpu_ref, account) != 'complete':
                raise RuntimeError('CPU follow-up has not completed; keeping its checkpoints')
        protected = set(spec.get('protected_run_ids', []))
        inventory = inventory_legacy_artifacts(api, spec['entity'], spec['project'], protected)
        inventory_path = output / 'checkpoint-archive-inventory.json'
        inventory_path.write_text(json.dumps(inventory, indent=2))
        tracker.run.save(str(inventory_path), base_path=str(output), policy='now')
        requested = spec.get('archive_candidates', [])
        safe = {item['ref'] for item in inventory if item['candidate']}
        if any(ref not in safe for ref in requested):
            raise RuntimeError('Requested checkpoint is not safe in the fresh archive inventory')
        for index, ref in enumerate(requested):
            tracker.run.log({'global_step': index, 'archive/processed': index,
                            'archive/bytes_verified': sum(r.get('bytes', 0) for r in records)})
            print(json.dumps({'archive_source': ref, 'index': index + 1,
                              'total': len(requested)}), flush=True)
            with TemporaryDirectory(prefix='uav-checkpoint-archive-') as temporary:
                record = archive_artifact(api, tracker, config, ref, Path(temporary), protected)
            records.append(record)
            (output / 'checkpoint-archive-index.json').write_text(json.dumps(records, indent=2))
            tracker.run.save(str(output / 'checkpoint-archive-index.json'), base_path=str(output), policy='now')
        tracker.run.summary.update({'archive/count': len(records),
            'archive/bytes_moved': sum(r.get('bytes', 0) for r in records)})
        failed = False
    finally:
        tracker.close(failed=failed)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    run_archive(json.loads(Path(args.spec).read_text()))


if __name__ == '__main__':
    main()
