"""CPU-only checkpoint migration; delete only after verified private backup."""
import argparse
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time

from ..configuration import Config
from ..experiments.checkpoint_store import (MANIFEST, upload_checkpoint_files,
    download_manifest_file, file_sha256)
from ..experiments.tracking import OnlineTracker


_PATTERN = r'([\w.-]+/[\w.-]+)/(?P<collection>uav-network-(?:masac|matd3)-(?P<run>[0-9a-f]{8})):v(?P<version>[0-9]+)'


def fresh_finished_source(api, path):
    # Public Api.run memoizes Run objects. A second call alone is not a refresh.
    api.flush()
    source = api.run(path)
    if source.state != 'finished':
        raise RuntimeError('Source run is active or has not finished successfully')
    return source


def wait_for_archive_mapping(api, source_path, key, replacement, timeout):
    """The summary write can succeed before it appears on a fresh API read."""
    deadline = time.monotonic() + timeout
    while True:
        refreshed = fresh_finished_source(api, source_path)
        if refreshed.summary.get(key) == replacement:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Archive mapping did not persist; source retained')
        time.sleep(2)


def verify_completed_source(api, spec, run_id, account, verify_outputs=False):
    """Bind a completed GPU/CPU pipeline to this exact checkpoint producer."""
    from .client import kernel_status, run_kaggle
    from .continuation import kernel_ref_for_segment
    proof = spec['completed_sources'][run_id]
    source = fresh_finished_source(api, spec['entity'] + '/' + spec['project'] + '/' + run_id)
    gpu = source.config.get('kaggle_kernel')
    root = (source.config.get('pipeline') or {}).get('root_kernel_ref', gpu)
    owner = spec['kernel_ref'].split('/')[0]
    if (not gpu or proof.get('gpu_kernel_ref') != gpu or proof.get('root_kernel_ref') != root
            or gpu.split('/')[0] != owner or root.split('/')[0] != owner
            or source.summary.get('episode_budget_complete') is not True):
        raise RuntimeError('Completion proof does not match a completed owned source pipeline')
    cpu = kernel_ref_for_segment(root, 0, cpu=True)
    artifact_ref = proof.get('artifact_ref', '')
    matched = re.fullmatch(_PATTERN, artifact_ref)
    if (proof.get('cpu_kernel_ref') != cpu or matched is None or matched['run'] != run_id
            or matched[1] != spec['entity'] + '/' + spec['project']):
        raise RuntimeError('Completion proof has an unrelated CPU session or artifact')
    if any(kernel_status(ref, account) != 'complete' for ref in (gpu, cpu)):
        raise RuntimeError('GPU or CPU follow-up has not completed; keeping its checkpoints')
    if verify_outputs:
        with TemporaryDirectory(prefix='uav-cpu-completion-') as temporary:
            run_kaggle(['kernels', 'output', cpu, '-p', temporary, '-q', '--file-pattern',
                        r'(?:^|/)(job|hardware|visualization-result)\.json$'], account, timeout=120)
            folder = Path(temporary)
            job = json.loads((folder / 'job.json').read_text())
            hardware = json.loads((folder / 'hardware.json').read_text())
            results = json.loads((folder / 'visualization-result.json').read_text())
        algorithm = matched['collection'].split('-')[2]
        result = results.get(algorithm, {})
        if (job.get('stage') != 'visualize' or job.get('parent_kernel_ref') != gpu
                or job.get('root_kernel_ref', gpu) != root
                or job.get('artifacts', {}).get(algorithm) != artifact_ref
                or result.get('checkpoint_artifact') != artifact_ref
                or result.get('device') != 'cpu' or result.get('episodes', 0) < 1
                or hardware.get('cuda_available') is not False or hardware.get('cuda_devices') != []):
            raise RuntimeError('Completed CPU outputs do not prove evaluation of this source checkpoint')


def inventory_legacy_artifacts(api, entity, project, protected, source_ids=None):
    """Read-only inventory; fail closed unless a legacy version is clearly removable."""
    records = []
    terminal = {'finished', 'failed', 'crashed', 'killed'}
    runs = (api.runs(f'{entity}/{project}') if source_ids is None else
            [fresh_finished_source(api, f'{entity}/{project}/{ident}') for ident in sorted(source_ids)])
    for run in runs:
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


def archive_artifact(api, tracker, config, ref, directory, protected, check_source):
    match = re.fullmatch(_PATTERN, ref)
    if match is None:
        raise ValueError('Only scoped immutable UAV checkpoint versions may be archived')
    run_id = match['run']
    if run_id in protected:
        raise RuntimeError('Source belongs to a protected production run')
    source_path = match[1] + '/' + run_id
    source_run = fresh_finished_source(api, source_path)
    check_source()
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
    source_run = fresh_finished_source(api, source_path)
    check_source()
    if any(run.state not in {'finished', 'failed', 'crashed', 'killed'} for run in artifact.used_by()):
        raise RuntimeError('Source acquired an active consumer during backup; source retained')
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
    # W&B flattens nested summary dictionaries. Persist one scalar per version.
    mapping_key = f"checkpoint_archive_{match['collection']}_v{match['version']}"
    summary = {mapping_key: new_ref}
    if {'latest', 'final', 'best'} & set(aliases):
        summary['checkpoint_artifact'] = new_ref
    source_run.summary.update(summary)
    # Recheck after slow transfer: a newly started consumer also protects the source.
    wait_for_archive_mapping(api, source_path, mapping_key, new_ref,
                             min(60, config.tracking.artifact_timeout_seconds))
    check_source()
    if any(run.state not in {'finished', 'failed', 'crashed', 'killed'} for run in artifact.used_by()):
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
    from .client import resolve_username
    from .credentials import Account
    requested = spec.get('archive_candidates', [])
    matches = [re.fullmatch(_PATTERN, ref) for ref in requested]
    proofs = spec.get('completed_sources', {})
    if (not requested or any(m is None or m[1] != spec['entity'] + '/' + spec['project']
                             or not isinstance(proofs.get(m['run']), dict) for m in matches)):
        raise RuntimeError('Every requested checkpoint requires a bound completion proof')
    account = Account(spec['account'], os.environ['KAGGLE_API_TOKEN'])
    if resolve_username(account) != spec['kernel_ref'].split('/')[0]:
        raise RuntimeError('Archive must authenticate as its selected kernel owner')
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
        for run_id in {m['run'] for m in matches}:
            verify_completed_source(api, spec, run_id, account, verify_outputs=True)
        protected = set(spec.get('protected_run_ids', []))
        inventory = inventory_legacy_artifacts(api, spec['entity'], spec['project'], protected,
                                             source_ids={m['run'] for m in matches})
        inventory_path = output / 'checkpoint-archive-inventory.json'
        inventory_path.write_text(json.dumps(inventory, indent=2))
        tracker.run.save(str(inventory_path), base_path=str(output), policy='now')
        safe = {item['ref'] for item in inventory if item['candidate']}
        if any(ref not in safe for ref in requested):
            raise RuntimeError('Requested checkpoint is not safe in the fresh archive inventory')
        for index, ref in enumerate(requested):
            tracker.run.log({'global_step': index, 'archive/processed': index,
                            'archive/bytes_verified': sum(r.get('bytes', 0) for r in records)})
            print(json.dumps({'archive_source': ref, 'index': index + 1,
                              'total': len(requested)}), flush=True)
            with TemporaryDirectory(prefix='uav-checkpoint-archive-') as temporary:
                run_id = re.fullmatch(_PATTERN, ref)['run']
                record = archive_artifact(api, tracker, config, ref, Path(temporary), protected,
                    lambda: verify_completed_source(api, spec, run_id, account))
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
