"""Verify ZIP consumers can reproduce policy and checksums without .git metadata."""
from pathlib import Path
import shutil
import subprocess
import sys
from scripts.maintenance.source_inventory import source_paths


def test_portable_source_archive_passes_policy_and_checksums_without_git(tmp_path):
    root = Path(__file__).resolve().parents[1]
    archived = tmp_path / 'unpacked'
    names = source_paths(root)
    for name in names:
        if not name:
            continue
        src = root / name
        target = archived / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    assert not (archived / '.git').exists()
    for script, args in (
        ('scripts/maintenance/verify_source_manifests.py', ['--check']),
        ('scripts/maintenance/check_project_policy.py', ['--scope', 'full']),
    ):
        result = subprocess.run([sys.executable, script, *args], cwd=archived, capture_output=True, text=True)
        assert result.returncode == 0, (result.stdout, result.stderr)
