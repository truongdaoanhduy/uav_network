"""Ensure only explicitly selected BASE runs and small-window math are plotted."""
import numpy as np
import pytest
from scripts.plot_latest_base_small_reward_windows import (
    RUNS, WINDOWS, moving_average, breakpoints, evaluate_jumps,
)


def test_user_selected_run_ids_exclude_pql_and_kaiming():
    assert RUNS == {'MATD3 BASE (default)': '971f7cc2',
                    'MASAC BASE (mask-aware)': 'f6a196a7'}
    assert WINDOWS == (8, 16, 32, 64)


def test_small_windows_have_no_downsample_and_true_means():
    episodes = np.arange(1, 13)
    rewards = np.arange(1, 13) * 2
    x, y = moving_average(episodes,rewards,4)
    assert len(x) == 9
    np.testing.assert_allclose(x[:3], [2.5,3.5,4.5])
    np.testing.assert_allclose(y[:3], [5,7,9])
    np.testing.assert_allclose(y[-2:], [19,21])


def test_small_window_edge_and_group_means():
    rows = np.array([[ep,float(ep),1.,float(ep*2)] for ep in range(1,13)])
    x,y=breakpoints(rows,4,'block')
    np.testing.assert_allclose(x,[2.5,6.5,10.5])
    np.testing.assert_allclose(y,[2.5,6.5,10.5])
    x2,y2=breakpoints(rows,4,'ma')
    assert len(x2)==9
    assert len(y2)==9
    with pytest.raises(ValueError):moving_average([1],[1],0)
    a,b=moving_average([1,2],[1,2],8)
    assert len(a)==len(b)==0


def test_jump_analysis_does_not_invent_records():
    rows=np.array([[ep,float(ep),float(ep),float(ep)] for ep in range(1,4300)])
    result=evaluate_jumps({'T':rows})
    assert list(result['T'])==['4096']
    assert result['T']['4096']['delta']==64.0
