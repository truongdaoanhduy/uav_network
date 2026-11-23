"""Centralized observed-cell report sketch invariant to report-slot ordering.

Version-6 fixed-width critic columns are spatial buckets, not true-target IDs.
Collisions lose detail, but no report slot is invisible to the critic.
"""
import warp as wp
from .state import WorldState,WorldParams


@wp.func
def write_cell_report_state(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,
                            xy_offset:int,status_offset:int):
    count=p.targets_count
    delivery_offset=status_offset+2*p.n*count
    for bucket in range(count):
        s.state[e,xy_offset+2*bucket]=wp.float32(0)
        s.state[e,xy_offset+2*bucket+1]=wp.float32(0)
        s.state[e,delivery_offset+bucket]=wp.float32(0)
        for i in range(p.n):
            s.state[e,status_offset+i*count+bucket]=wp.float32(0)
            s.state[e,status_offset+p.n*count+i*count+bucket]=wp.float32(0)
    for slot in range(p.report_slots):
        cell=s.report_cell[e,slot]
        if cell>=0:
            # Buckets depend only on transmitted cell identity, not a hidden
            # true target ID or any other UAV's discovery order.
            bucket=cell%count
            x=wp.float32((wp.float64(cell%p.grid_n)+wp.float64(.5))*p.cell_width/p.map_size)
            y=wp.float32((wp.float64(cell//p.grid_n)+wp.float64(.5))*p.cell_width/p.map_size)
            oldx=s.state[e,xy_offset+2*bucket]
            oldy=s.state[e,xy_offset+2*bucket+1]
            if oldy==wp.float32(0) or y<oldy or (y==oldy and x<oldx):
                s.state[e,xy_offset+2*bucket]=x
                s.state[e,xy_offset+2*bucket+1]=y
            s.state[e,delivery_offset+bucket]=wp.max(
                s.state[e,delivery_offset+bucket],wp.float32(s.delivered[e,slot]))
            for i in range(p.n):
                idx=status_offset+i*count+bucket
                age_idx=status_offset+p.n*count+i*count+bucket
                s.state[e,idx]=wp.max(s.state[e,idx],wp.float32(s.known[e,i,slot]))
                if s.report_present[e,i,slot]!=0:
                    age=wp.max(wp.float64(0),wp.float64(1)-
                         (wp.float64(s.steps[e])*p.dt-s.report_created[e,i,slot])/p.report_ttl)
                    s.state[e,age_idx]=wp.max(s.state[e,age_idx],wp.float32(age))
