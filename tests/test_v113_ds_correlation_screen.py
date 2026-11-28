"""V113 DS correlated-evidence synthetic probe mathematical contracts."""
from __future__ import annotations
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest
from zhang_xue.belief.dempster_shafer import counts_to_canonical_ds_stable

PATH=Path(__file__).resolve().parents[1]/'scripts/maintenance/v113_ds_correlation_screen.py'
spec=spec_from_file_location('v113_screen',PATH)
probe=module_from_spec(spec)
spec.loader.exec_module(probe)


def test_ds_infinite_ignorance_without_observations():
    assert counts_to_canonical_ds_stable(0,0,mf=.8,me=.8) == (0.,0.,1.)


def test_zlib_binary_count_discrete_threshold_equivalence():
    for p in range(0,12):
        for n in range(0,12):
            mt, mf, mu=counts_to_canonical_ds_stable(p,n,mf=.8,me=.8)
            assert (mt>=.9 and mu<=.2) == (mt>=.95 and mu<=.2)


def test_synthetic_correlation_worsens_false_positive_rate():
    a=probe.simulate(n_trials=100_000,visits=20,seed=44,rho=0.,probability=.03,thresholds=(.9,.99))
    b=probe.simulate(n_trials=100_000,visits=20,seed=44,rho=.9,probability=.03,thresholds=(.9,.99))
    for threshold in ('0.9','0.99'):
        assert b['results'][threshold]['ever_confirmed_rate']>a['results'][threshold]['ever_confirmed_rate']+0.005


def test_finite_ci_and_reproducible_seed():
    kw=dict(n_trials=1000,visits=10,seed=92,rho=.5,probability=.03,thresholds=(.9,.99))
    one=probe.simulate(**kw)
    two=probe.simulate(**kw)
    assert one==two
    for v in one['results'].values():
        lo,hi=v['wilson_ci95']
        assert 0<=lo<=v['ever_confirmed_rate']<=hi<=1
