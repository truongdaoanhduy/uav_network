"""Simulator-only scoring. NEVER call these helpers to generate reports, policy features or masks.

In the new Bayesian cell-report task, a report ID refers to a grid cell, and
true target coordinates are consulted ONLY here to score that hypothesis.
"""
import warp as wp
from .state import WorldParams, WorldState

@wp.func
def true_target_cell(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,t:int)->int:
    xy=s.targets[e,t]
    x=wp.clamp(int(xy[0]/p.cell_width),0,p.grid_n-1)
    y=wp.clamp(int(xy[1]/p.cell_width),0,p.grid_n-1)
    return y*p.grid_n+x

@wp.func
def is_truth_matched_report(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,slot:int)->bool:
    if p.cell_reports==0:
        return slot>=0 and slot<p.targets_count
    cell=s.report_cell[e,slot]
    if cell<0:
        return False
    for t in range(p.targets_count):
        if cell==true_target_cell(s,p,e,t):
            return True
    return False

@wp.func
def truth_found_and_delivered(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int)->wp.vec4d:
    """Counts of unique confirmed true/false cells; only used for rewards/termination."""
    true_found=int(0)
    true_delivered=int(0)
    false_found=int(0)
    false_delivered=int(0)
    for slot in range(p.report_slots):
        found=int(0)
        for i in range(p.n):
            found=wp.max(found,s.known[e,i,slot])
        delivered=s.delivered[e,slot]
        if is_truth_matched_report(s,p,e,slot):
            true_found+=found
            true_delivered+=delivered
        else:
            false_found+=found
            false_delivered+=delivered
    return wp.vec4d(wp.float64(true_found),wp.float64(true_delivered),
                    wp.float64(false_found),wp.float64(false_delivered))
