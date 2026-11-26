"""Source path inventory for both Git working trees and git-free release ZIPs.

The archive inventory excludes generated dependencies/artifacts, not credentials:
if someone unpacks a key file into the ZIP, the policy checker must see it.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

SKIP_FOLDERS = {
    '.git', '.venv', '.pytest_cache', '__pycache__', '.ruff_cache',
    'build', 'dist', 'outputs', 'wandb', 'third_party', 'node_modules',
    '.ipynb_checkpoints',
}


def source_paths(root: Path) -> list[str]:
    root = root.resolve()
    if (root / '.git').exists():
        data = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'])
        return sorted({s for s in data.decode('utf-8').split('\0') if s})
    found = []
    for file in root.rglob('*'):
        if not file.is_file():
            continue
        path = file.relative_to(root)
        if any(part in SKIP_FOLDERS or part.endswith('.egg-info') for part in path.parts):
            continue
        if file.suffix in ('.pyc', '.pyo') or file.name == '.DS_Store':
            continue
        found.append(path.as_posix())
    return sorted(found)
