"""The patch search should scan only configured cells, not a fixed 13x13."""
from pathlib import Path


def test_warp_investigation_scans_just_configured_patch():
    source=Path('uav_network/environment/gpu/bayesian.py').read_text()
    start=source.index('def investigation_potential(')
    end=source.index('@wp.kernel',start)
    fragment=source[start:end]
    assert 'for dy in range(-radius,radius+1):' in fragment
    assert 'for dx in range(-radius,radius+1):' in fragment
    assert 'range(-6,7)' not in fragment
