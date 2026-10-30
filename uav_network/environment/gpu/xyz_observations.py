"""Compact local posterior patches and noisy cell candidates on device."""
import warp as wp
from .state import WorldState,WorldParams
from .xyz_geometry import range_distance
from .bayesian import profile


@wp.func
def patch_value(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,i:int,cell:int):
    xy=s.positions[e,i];half=p.patch_n//2
    gx=wp.min(int(xy[0]/p.cell_width),p.grid_n-1)+cell%p.patch_n-half
    gy=wp.min(int(xy[1]/p.cell_width),p.grid_n-1)+cell//p.patch_n-half
    value=wp.float64(.5)
    if gx>=0 and gx<p.grid_n and gy>=0 and gy<p.grid_n:value=s.belief[e,i,gy*p.grid_n+gx]
    return value


@wp.func
def candidates(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,i:int,offset:int):
    previous=wp.float64(-1);previous_id=int(-1)
    for rank in range(p.target_slots):
        best=wp.float64(1e100);best_id=int(-1)
        for cell in range(p.grid_n*p.grid_n):
            if s.positive[e,i,cell]!=0:
                center=wp.vec2d(wp.min((wp.float64(cell%p.grid_n)+wp.float64(.5))*p.cell_width,p.map_size),wp.min((wp.float64(cell//p.grid_n)+wp.float64(.5))*p.cell_width,p.map_size))
                distance=wp.length(center-s.positions[e,i])
                if (distance>previous or (distance==previous and cell>previous_id)) and distance<best:
                    best=distance;best_id=cell
        if best_id>=0:
            center=wp.vec2d(wp.min((wp.float64(best_id%p.grid_n)+wp.float64(.5))*p.cell_width,p.map_size),wp.min((wp.float64(best_id//p.grid_n)+wp.float64(.5))*p.cell_width,p.map_size))
            rel=(center-s.positions[e,i])/p.map_size
            s.observation[e,i,offset+4*rank]=wp.float32(rel[0])
            s.observation[e,i,offset+4*rank+1]=wp.float32(rel[1])
            s.observation[e,i,offset+4*rank+2]=wp.float32(s.belief[e,i,best_id])
            s.observation[e,i,offset+4*rank+3]=wp.float32(1)
            previous=best;previous_id=best_id


@wp.func
def extras(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,i:int,offset:int):
    s.observation[e,i,offset]=wp.float32(s.altitudes[e,i]/p.altitude_max)
    s.observation[e,i,offset+1]=wp.float32(s.vertical_velocity[e,i]/p.max_speed)
    offset+=2
    radius=profile(p,s.altitudes[e,i])[2];pos=s.positions[e,i]
    active=p.energy_enabled==0 or s.battery[e,i]>wp.float64(0)
    for o in range(p.obstacles_count):
        v=s.obstacles[e,o];d=wp.length(wp.vec2d(v[0],v[1])-pos)
        if active and d<=radius+v[2]:
            rank=int(0)
            for other in range(p.obstacles_count):
                q=s.obstacles[e,other];od=wp.length(wp.vec2d(q[0],q[1])-pos)
                if od<=radius+q[2] and (od<d or (od==d and other<o)):rank+=1
            if rank<p.obstacle_slots:s.observation[e,i,offset+rank]=wp.float32(s.obstacle_heights[e,o]/p.altitude_max)
    offset+=p.obstacle_slots
    slot=int(0)
    for j in range(p.n):
        if j!=i:
            if active and range_distance(s,p,e,i,j)<=p.peer_range:
                s.observation[e,i,offset+2*slot]=wp.float32((s.altitudes[e,j]-s.altitudes[e,i])/p.altitude_max)
                s.observation[e,i,offset+2*slot+1]=wp.float32(s.vertical_velocity[e,j]/p.max_speed)
            slot+=1
    offset+=2*(p.n-1)
    for cell in range(p.patch_n*p.patch_n):s.observation[e,i,offset+cell]=wp.float32(patch_value(s,p,e,i,cell))
    return offset+p.patch_n*p.patch_n


@wp.func
def centralized_extras(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,offset:int):
    for i in range(p.n):
        s.state[e,offset+i]=wp.float32(s.altitudes[e,i]/p.altitude_max)
        s.state[e,offset+p.n+i]=wp.float32(s.vertical_velocity[e,i]/p.max_speed)
    offset+=2*p.n
    for o in range(p.obstacles_count):s.state[e,offset+o]=wp.float32(s.obstacle_heights[e,o]/p.altitude_max)
    offset+=p.obstacles_count
    for i in range(p.n):
        for cell in range(p.patch_n*p.patch_n):s.state[e,offset+i*p.patch_n*p.patch_n+cell]=wp.float32(patch_value(s,p,e,i,cell))
