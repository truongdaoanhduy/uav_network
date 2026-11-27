"""The canonical source package must have reproducible full-project checksums."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CHECKER=ROOT/'scripts/maintenance/verify_source_manifests.py'


def test_source_manifests_cover_entire_git_tree_and_research_subset():
    result=subprocess.run([sys.executable,str(CHECKER),'--check'],cwd=ROOT,
                          capture_output=True,text=True)
    assert result.returncode==0, result.stdout+result.stderr
    root=json.loads((ROOT/'SOURCE_SHA256_MANIFEST.json').read_text())
    research=json.loads((ROOT/'research/zhang_xue/SHA256_MANIFEST.json').read_text())
    assert all(key.startswith('research/zhang_xue/') for key in root if key in {'research/zhang_xue/README.md'})
    assert 'pyproject.toml' in root
    assert 'src/zhang_xue/gpu/kernels/phy.py' in research
    assert all('outputs/' not in name and 'key.txt' not in name for name in root)
