"""GPU smoke must size and exercise the actual 6-UAV user configuration, not 2-UAV toy."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / 'scripts/gpu/warp_packet_cuda_gate.py').read_text()


def test_cuda_gate_defaults_to_six_uavs_and_twenty_grid():
    assert "parser.add_argument('--uavs', type=int, default=6)" in SRC
    assert "parser.add_argument('--grid', type=int, default=20)" in SRC
    assert 'assert_network_allocation_fits(num_envs, num_uavs, grid_w' in SRC
    assert 'WarpCompressedRadioNetworkV67(num_envs=num_envs, num_uavs=num_uavs, grid_w=grid_w' in SRC


def test_cuda_gate_explicitly_rejects_oversize_grid():
    r = subprocess.run([sys.executable, 'scripts/gpu/warp_packet_cuda_gate.py', '--grid', '65'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert r.returncode == 2, r.stdout+r.stderr
