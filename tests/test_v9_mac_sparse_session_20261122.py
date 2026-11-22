"""Ensure the NoOracle MAC reconciles only its append-only report prefix."""
from pathlib import Path
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv


def test_nooracle_session_limit_must_be_report_prefix_not_total_capacity():
    from uav_network.communication.gpu import streaming
    text=Path(streaming.__file__).read_text()
    target=text[text.index('def clear_pending('):text.index('\n\n@wp.func\ndef fill_session(',text.index('def clear_pending('))]
    assert 's.report_cell[e,target] < 0' in target
    assert 'for source in range(p.n)' in target
    assert 'for target in range(p.report_slots)' in target
    assert 'for dest in range(p.n+1)' in target


def test_nooracle_report_capacity_far_exceeds_allocated_slots_at_reset():
    cfg=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    cfg.runtime.device='cpu';cfg.runtime.num_envs=2
    env=WarpVectorEnv(cfg,device='cpu')
    try:
        env.reset(seed=[44,45]);import warp as wp
        assert env.p.report_slots==140
        assert env.p.session_capacity==env.p.n*env.p.report_slots*(env.p.n+1)
        assert bool((wp.to_torch(env.s.report_cell)<0).all())
        assert bool((wp.to_torch(env.s.session_active)==0).all())
    finally:env.close()
