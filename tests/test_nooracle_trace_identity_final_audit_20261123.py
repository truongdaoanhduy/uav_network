"""NoOracle radio traces carry observed cell IDs, not true target IDs."""
import torch
import warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpMissionEnv
from uav_network.experiments.rendering import _draw_communication


class _FakeAxis:
    def __init__(self):self.labels=[]
    def annotate(self,*args,**kwargs):pass
    def text(self,x,y,label,**kwargs):self.labels.append(label)


def test_nooracle_report_render_label_explicitly_says_cell_not_true_target():
    ax=_FakeAxis()
    events=[dict(event='report_received',time_s=1.,source_xy=[100.,100.],
                 destination_xy=[200.,200.],destination=6,target=820,power_w=.2,
                 report_identity='observed_cell_id')]
    _draw_communication(ax,events,2.,6)
    assert len(ax.labels)==1
    assert ax.labels[0].startswith('Cell 820 '),ax.labels
    historical=dict(events[0],target=2,report_identity='true_target_id')
    ax=_FakeAxis()
    _draw_communication(ax,[historical],2.,6)
    assert ax.labels[0].startswith('T2 '),ax.labels


def test_warp_nooracle_trace_event_exposes_explicit_on_air_identity_type():
    c=load_config('configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml')
    c.runtime.device='cpu';c.runtime.num_envs=1
    e=WarpMissionEnv(c)
    try:
        e.reset(seed=44)
        e.enable_communication_trace()
        s=e.vector.s
        wp.to_torch(s.trace_count)[0]=1
        wp.to_torch(s.trace_kind)[0,0]=2
        wp.to_torch(s.trace_report)[0,0,0]=820.
        wp.to_torch(s.trace_source)[0,0]=0
        wp.to_torch(s.trace_destination)[0,0]=6
        wp.to_torch(s.trace_time)[0,0]=1_000_000.
        e._collect_communication_events()
        event=e.drain_communication_events()[0]
        assert event['target']==820
        assert event['report_identity']=='observed_cell_id'
    finally:e.close()
