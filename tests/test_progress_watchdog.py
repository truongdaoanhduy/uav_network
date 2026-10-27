"""Actual native traceback timers diagnose stalled work without terminating it."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def require_watchdog():
    assert importlib.util.find_spec("uav_network.experiments.progress_watchdog"), "No-progress traceback capture is missing"


@pytest.mark.parametrize("case",["stalled","progress","same_step","cancelled","owner"])
def test_real_process_watchdog_timer_progress_cancel_and_ownership(tmp_path,case):
    require_watchdog()
    script = r"""
from pathlib import Path
import sys,time
from uav_network.experiments.progress_watchdog import ProgressWatchdog
out=Path(sys.argv[1]);case=sys.argv[2]
a=ProgressWatchdog(out,timeout=.3)
if case=='progress':
    for step in range(8):
        time.sleep(.06);a.progress(step)
    a.close()
elif case=='cancelled':
    a.close();time.sleep(.4)
elif case=='owner':
    (out/'second').mkdir();b=ProgressWatchdog(out/'second',timeout=.1)
    a.close();time.sleep(.25);b.close()
else:
    if case=='same_step':
        for _ in range(8):time.sleep(.06);a.progress(0)
    else:time.sleep(.45)
    a.close()
print('process survived')
"""
    result=subprocess.run([sys.executable,"-c",script,str(tmp_path),case],capture_output=True,text=True,timeout=8)
    assert result.returncode==0 and "process survived" in result.stdout
    first=(tmp_path/"thread-stacks.log").read_text()
    if case in ("stalled","same_step"):
        assert "Timeout" in first and "File " in first
    else:
        assert "Timeout" not in first
    if case=='owner':
        assert "Timeout" in (tmp_path/"second/thread-stacks.log").read_text()


def test_optional_watchdog_file_failure_never_stops_work(tmp_path,monkeypatch):
    require_watchdog()
    from uav_network.experiments.progress_watchdog import ProgressWatchdog
    monkeypatch.setattr(Path,"open",lambda *a,**k:(_ for _ in ()).throw(OSError("optional output unavailable")))
    watcher=ProgressWatchdog(tmp_path);watcher.progress(4096);watcher.close()
