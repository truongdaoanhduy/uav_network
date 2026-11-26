"""Moved legacy script must still be independently runnable from repository root."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def test_moved_cpu_reference_replays_without_pythonpath_env():
    env=os.environ.copy()
    env.pop('PYTHONPATH',None)
    result=subprocess.run(
        [sys.executable,'research/zhang_xue/scripts/legacy/reproduce_80step_reference.py'],
        cwd=ROOT,env=env,text=True,capture_output=True,timeout=30,
    )
    assert result.returncode==0, result.stderr[-900:]
    payload=json.loads(result.stdout)
    assert {'full_comm','zero_comm','seed'}<=payload.keys()
