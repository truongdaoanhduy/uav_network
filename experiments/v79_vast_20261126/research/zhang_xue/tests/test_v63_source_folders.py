"""No historical version-numbered implementation files directly under src/."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_research_implementation_files_are_namespaced():
    assert not list((ROOT/'src').glob('*.py'))
    for relative in ('gpu/kernels/events.py','gpu/kernels/injection.py',
                     'gpu/kernels/reset.py','gpu/kernels/phy.py',
                     'learners/state_derived_replay.py','compat/env.py',
                     'communication/channel.py','environment/core.py'):
        assert (ROOT/'src/zhang_xue'/relative).is_file(), relative
