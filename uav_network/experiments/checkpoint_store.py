"""Small W&B manifests point to verified, immutable, private Hub checkpoints."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


# Native HTTPS/LFS avoids the Xet transport stall reproduced on the host VPN.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

MANIFEST = 'checkpoint.json'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def storage_credentials(config=None):
    data = {}
    path = Path(config.tracking.credentials_file if config is not None else 'key.txt').expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if path.is_file():
        data = json.loads(path.read_text()).get('huggingface', {})
    token = os.environ.get('HF_TOKEN') or data.get('token')
    repo = getattr(getattr(config, 'tracking', None), 'checkpoint_repo', 'auto')
    if repo == 'auto':
        repo = os.environ.get('HF_REPO_ID') or data.get('repo_id')
    if not token:
        raise RuntimeError('Private checkpoint storage requires HF_TOKEN or huggingface.token in key.txt')
    if config is not None and (not isinstance(repo, str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', repo)):
        raise ValueError('Private checkpoint storage needs a valid HF repository ID')
    return token, repo


def ensure_private_store(config):
    from huggingface_hub import HfApi
    token, repo = storage_credentials(config)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type='model', private=True, exist_ok=True)
    if api.repo_info(repo, repo_type='model').private is not True:
        raise RuntimeError('Checkpoint repository must be private; refusing upload')
    return api, repo


def validate_manifest(manifest):
    if (not isinstance(manifest, dict) or manifest.get('schema') != 1
            or manifest.get('backend') != 'huggingface'
            or not isinstance(manifest.get('repo_id'), str)
            or not re.fullmatch(r'[\w.-]+/[\w.-]+', manifest['repo_id'])
            or not re.fullmatch(r'[0-9a-f]{40}', str(manifest.get('revision', '')))
            or not isinstance(manifest.get('files'), dict) or not manifest['files']):
        raise ValueError('Invalid immutable checkpoint manifest')
    for name, item in manifest['files'].items():
        if name not in {'checkpoint.pt', 'resume.pt'} or not isinstance(item, dict):
            raise ValueError('Invalid checkpoint entry')
        path = item.get('path')
        if (not isinstance(path, str) or '\\' in path or not path
                or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
                or PurePosixPath(path).name != name
                or type(item.get('size')) is not int or item['size'] < 1
                or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', '')))):
            raise ValueError('Unsafe checkpoint path, size or checksum')
    return manifest


def upload_checkpoint_files(config, files, prefix):
    from huggingface_hub import CommitOperationAdd
    if (not re.fullmatch(r'[\w./-]+', prefix) or prefix.startswith('/')
            or '..' in PurePosixPath(prefix).parts):
        raise ValueError('Invalid checkpoint storage prefix')
    items = {name: {'path': prefix.rstrip('/') + '/' + name,
                    'size': Path(path).stat().st_size, 'sha256': file_sha256(path)}
             for name, path in files.items()}
    # Validate names and relative paths before any external side effects.
    validate_manifest({'schema': 1, 'backend': 'huggingface', 'repo_id': 'check/check',
                       'revision': '0' * 40, 'files': items})
    api, repo = ensure_private_store(config)
    current = api.repo_info(repo, repo_type='model', files_metadata=True)
    if current.private is not True:
        raise RuntimeError('Checkpoint repository must be private; refusing upload')
    if _remote_files_match(current, items):
        return validate_manifest({'schema': 1, 'backend': 'huggingface', 'repo_id': repo,
                                  'revision': current.sha, 'files': items})
    commit = api.create_commit(repo_id=repo, repo_type='model',
        operations=[CommitOperationAdd(path_in_repo=items[name]['path'], path_or_fileobj=str(path))
                    for name, path in files.items()], commit_message='Save UAV checkpoint ' + prefix)
    manifest = validate_manifest({'schema': 1, 'backend': 'huggingface', 'repo_id': repo,
                                  'revision': commit.oid, 'files': items})
    info = api.repo_info(repo, repo_type='model', revision=commit.oid, files_metadata=True)
    if info.private is not True:
        raise RuntimeError('Uploaded checkpoint repository is not private')
    if not _remote_files_match(info, items):
        raise RuntimeError('Remote checkpoint size/checksum verification failed')
    return manifest


def _remote_files_match(info, items):
    remote = {item.rfilename: item for item in info.siblings}
    for item in items.values():
        actual = remote.get(item['path'])
        lfs = getattr(actual, 'lfs', None)
        checksum = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        if actual is None or actual.size != item['size'] or checksum != item['sha256']:
            return False
    return True


def download_checkpoint_file(artifact, entry, destination):
    destination = Path(destination)
    stored = getattr(artifact, 'manifest', None)
    if stored is None or MANIFEST not in stored.entries:
        return artifact.get_entry(entry).download(root=str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    path = Path(artifact.get_entry(MANIFEST).download(root=str(destination)))
    if path.stat().st_size > 256 * 1024:
        raise ValueError('Checkpoint manifest is too large')
    manifest = validate_manifest(json.loads(path.read_text()))
    return download_manifest_file(manifest, entry, destination)


def download_manifest_file(manifest, entry, destination, force_download=False):
    manifest = validate_manifest(manifest)
    item = manifest['files'].get(entry)
    if item is None:
        raise ValueError('Manifest does not contain requested checkpoint state')
    from huggingface_hub import hf_hub_download
    token, _ = storage_credentials()
    downloaded = Path(hf_hub_download(repo_id=manifest['repo_id'], repo_type='model',
        filename=item['path'], revision=manifest['revision'], token=token,
        local_dir=str(destination), force_download=force_download))
    if downloaded.stat().st_size != item['size'] or file_sha256(downloaded) != item['sha256']:
        raise ValueError('Downloaded checkpoint size/checksum differs from immutable manifest')
    return str(downloaded)


def resolve_checkpoint_artifact(api, ref):
    """Preserve immutable legacy inputs after a verified archive migration."""
    try:
        return api.artifact(ref, type='model')
    except Exception as original:
        match = re.fullmatch(r'([\w.-]+/[\w.-]+)/(?P<c>uav-network-(?:masac|matd3)-(?P<run>[0-9a-f]{8})):v(?P<version>[0-9]+)', ref)
        if match is None:
            raise
        try:
            api.flush()
            summary = api.run(match[1] + '/' + match['run']).summary
            mapped = summary.get(f"checkpoint_archive_{match['c']}_v{match['version']}")
            if mapped is None:
                mapped = summary.get('checkpoint_archives.' + ref)
            if mapped is None:
                mapped = summary.get('checkpoint_archives', {}).get(ref)
        except Exception:
            raise original from None
        if mapped is None:
            raise original from None
        prefix = re.escape(match[1] + '/' + match['c'])
        if not isinstance(mapped, str) or not re.fullmatch(prefix + r':v[0-9]+', mapped):
            raise ValueError('Invalid checkpoint archive mapping') from None
        return api.artifact(mapped, type='model')
