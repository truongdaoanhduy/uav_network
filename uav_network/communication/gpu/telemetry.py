"""Passive action and receiver-opportunity accounting for the packet engine."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState
from .reports import active, bytes_used, can_receive
from .phy import xy
from ...environment.gpu.xyz_geometry import range_distance
from .power import link_power_feasible
from . import diagnostics as diag

# Keep the same FP64 operation boundaries as the calling MAC kernel.
wp.set_module_options({"enable_fp_fma": False, "enable_backward": False})


@wp.func
def record_action(s: WorldState, p: WorldParams, e: int, source: int, index: int,
                  continuous: wp.array3d(dtype=wp.float32), now: wp.float64):
    if index == 0:
        s.action_counts[e,0] += wp.int64(1)
    elif index == p.n:
        s.action_counts[e,2] += wp.int64(1)
    else:
        s.action_counts[e,1] += wp.int64(1)
    for j in range(p.continuous_dim-1):
        if wp.abs(continuous[e,source,j]) >= wp.float32(0.99):
            s.action_counts[e,3] += wp.int64(1)
    if p.network_enabled != 0:
        s.diagnostic_counts[e,diag.ACTION_SLOTS] += wp.int64(1)
        if index == 0:
            s.diagnostic_counts[e,diag.ACTION_SILENT] += wp.int64(1)
        is_active = active(s,p,e,source)
        if is_active:
            s.diagnostic_counts[e,diag.ACTION_ACTIVE] += wp.int64(1)
        bearing = is_active and bytes_used(s,p,e,source) > 0
        eligible = bool(False)
        direct_eligible = bool(False)
        closer_peer = bool(False)
        if bearing:
            s.diagnostic_counts[e,diag.ACTION_REPORT_BEARING] += wp.int64(1)
            for receiver in range(p.n+1):
                if receiver != source and active(s,p,e,receiver):
                    reach = p.peer_range
                    if receiver == p.n:
                        reach = p.gcs_range
                    if (range_distance(s,p,e,source,receiver) <= reach
                        and link_power_feasible(s,p,e,source,receiver)):
                        for target in range(p.targets_count):
                            if (s.report_present[e,source,target] != 0
                                and now/wp.float64(1e6)-s.report_created[e,source,target] < p.report_ttl
                                and can_receive(s,p,e,receiver,target)):
                                eligible = True
                                if receiver == p.n:
                                    direct_eligible = True
                                elif (range_distance(s,p,e,receiver,p.n)
                                      < range_distance(s,p,e,source,p.n)-wp.float64(1e-9)):
                                    closer_peer = True
        if eligible:
            s.diagnostic_counts[e,diag.ACTION_RECEIVER_ELIGIBLE] += wp.int64(1)
        else:
            s.diagnostic_counts[e,diag.ACTION_NO_OPPORTUNITY] += wp.int64(1)
        if eligible and index == 0:
            s.diagnostic_counts[e,diag.ACTION_ELIGIBLE_SILENT] += wp.int64(1)
        if closer_peer and not direct_eligible:
            s.diagnostic_counts[e,diag.ACTION_PROGRESS_PEER_OPPORTUNITY] += wp.int64(1)
            if index == 0:
                s.diagnostic_counts[e,diag.ACTION_PROGRESS_PEER_SILENT] += wp.int64(1)
