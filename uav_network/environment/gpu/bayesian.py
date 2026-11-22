"""Reproducible device Bayesian camera; no shared mutable sampling RNG."""
import warp as wp
from .state import WorldState,WorldParams
from .xyz_geometry import position3,camera_clear

wp.set_module_options({"enable_backward":False,"fuse_fp":False})
Cuts=wp.types.vector(length=6,dtype=wp.float64)


@wp.func
def sensor_uniform(seed:wp.int64,tick:int,agent:int,cell:int):
    x=wp.uint32(seed) ^ (wp.uint32(tick)*wp.uint32(0x9e3779b9)) ^ (wp.uint32(agent)*wp.uint32(0x85ebca6b)) ^ (wp.uint32(cell)*wp.uint32(0xc2b2ae35))
    x=(x^(x>>wp.uint32(16)))*wp.uint32(0x7feb352d)
    x=(x^(x>>wp.uint32(15)))*wp.uint32(0x846ca68b)
    return (wp.float64(x^(x>>wp.uint32(16)))+wp.float64(.5))/wp.float64(4294967296.0)


@wp.func
def profile(p:wp.ref[WorldParams],z:wp.float64):
    pd=p.sensor_pd[3];pf=p.sensor_pf[3]
    if z<=p.sensor_anchors[0]:
        pd=p.sensor_pd[0];pf=p.sensor_pf[0]
    else:
        for k in range(3):
            if z>p.sensor_anchors[k] and z<=p.sensor_anchors[k+1]:
                f=(z-p.sensor_anchors[k])/(p.sensor_anchors[k+1]-p.sensor_anchors[k])
                pd=p.sensor_pd[k]+f*(p.sensor_pd[k+1]-p.sensor_pd[k])
                pf=p.sensor_pf[k]+f*(p.sensor_pf[k+1]-p.sensor_pf[k])
    return wp.vec3d(pd,pf,wp.max(wp.float64(0),z)*p.fov_tangent)


@wp.func

def confirmation_allowed(p:wp.ref[WorldParams],altitude:wp.float64,
                         positive_likelihood:wp.float64,false_likelihood:wp.float64):
    if p.confirmation_mode==0:
        return altitude<=p.fine_altitude
    return positive_likelihood>=p.confirmation_min_likelihood_ratio*false_likelihood


@wp.func
def integral(x:wp.float64,r:wp.float64):
    a=wp.clamp(x,-r,r)
    return wp.float64(.5)*(a*wp.sqrt(wp.max(wp.float64(0),r*r-a*a))+r*r*wp.asin(a/r))


@wp.func
def cell_coverage(p:wp.ref[WorldParams],gx:int,gy:int,xy:wp.vec2d,r:wp.float64):
    area=wp.float64(0)
    if r>wp.float64(0):
        x0=wp.float64(gx)*p.cell_width-xy[0];x1=wp.min(wp.float64(gx+1)*p.cell_width,p.map_size)-xy[0]
        y0=wp.float64(gy)*p.cell_width-xy[1];y1=wp.min(wp.float64(gy+1)*p.cell_width,p.map_size)-xy[1]
        dx=wp.max(wp.float64(0),wp.max(x0,-x1))
        dy=wp.max(wp.float64(0),wp.max(y0,-y1))
        if dx*dx+dy*dy>r*r*wp.float64(1.0000000001):return wp.float64(0)
        left=wp.max(x0,-r);right=wp.min(x1,r)
        if right>left and y1>-r and y0<r:
            cuts=Cuts(right);cuts[0]=left;count=int(2)
            for j in range(2):
                y=y0
                if j==1:y=y1
                if wp.abs(y)<r:
                    root=wp.sqrt(wp.max(wp.float64(0),r*r-y*y))
                    for side in range(2):
                        q=-root
                        if side==1:q=root
                        if q>left and q<right:
                            cuts[count]=q;count+=1
            for sort_pass in range(6):
                for sort_j in range(5):
                    if cuts[sort_j]>cuts[sort_j+1]:
                        temp=cuts[sort_j];cuts[sort_j]=cuts[sort_j+1];cuts[sort_j+1]=temp
            for j in range(5):
                a=cuts[j];b=cuts[j+1]
                h=wp.sqrt(wp.max(wp.float64(0),r*r-(a+b)*(a+b)/wp.float64(4)))
                if b>a and wp.min(y1,h)>wp.max(y0,-h):
                    arc=integral(b,r)-integral(a,r)
                    upper=arc;lower=-arc
                    if y1<h:upper=y1*(b-a)
                    if y0>-h:lower=y0*(b-a)
                    area+=upper-lower
            area=wp.clamp(area/((x1-x0)*(y1-y0)),wp.float64(0),wp.float64(1))
    return area


@wp.func
def entropy_bits(value:wp.float64):
    q=wp.clamp(value,wp.float64(1e-12),wp.float64(1)-wp.float64(1e-12))
    return -(q*wp.log(q)+(wp.float64(1)-q)*wp.log(wp.float64(1)-q))/wp.log(wp.float64(2))


@wp.func
def investigation_potential(s:WorldState,p:WorldParams,e:int):
    total=wp.float64(0)
    if p.xyz!=0 and p.reward_investigation!=wp.float64(0):
        radius=p.patch_n//2
        altitude_span=wp.max(p.altitude_max-p.confirmation_shaping_altitude,wp.float64(1e-6))
        for i in range(p.n):
            active=p.energy_enabled==0 or s.battery[e,i]>wp.float64(0)
            if active:
                xy=s.positions[e,i]
                gx=wp.clamp(int(wp.floor(xy[0]/p.cell_width)),0,p.grid_n-1)
                gy=wp.clamp(int(wp.floor(xy[1]/p.cell_width)),0,p.grid_n-1)
                best=wp.float64(-1);best_gx=gx;best_gy=gy
                # For patch=5 only 25 cells are eligible, not the 169
                # iterations of the old fixed-radius=6 implementation.
                for dy in range(-radius,radius+1):
                    for dx in range(-radius,radius+1):
                        xx=gx+dx;yy=gy+dy
                        if xx>=0 and xx<p.grid_n and yy>=0 and yy<p.grid_n:
                            value=s.belief[e,i,yy*p.grid_n+xx]
                            if value>best:
                                best=value;best_gx=xx;best_gy=yy
                evidence=wp.clamp((best-p.investigation_threshold)/wp.max(p.confirmation_threshold-p.investigation_threshold,wp.float64(1e-6)),wp.float64(0),wp.float64(1))
                peak=wp.vec2d((wp.float64(best_gx)+wp.float64(.5))*p.cell_width,(wp.float64(best_gy)+wp.float64(.5))*p.cell_width)
                max_distance=wp.max(wp.sqrt(wp.float64(2))*(wp.float64(radius)+wp.float64(.5))*p.cell_width,wp.float64(1e-6))
                proximity=wp.clamp(wp.float64(1)-wp.length(peak-xy)/max_distance,wp.float64(0),wp.float64(1))
                descent=wp.clamp((p.altitude_max-s.altitudes[e,i])/altitude_span,wp.float64(0),wp.float64(1))
                total+=evidence*wp.float64(.5)*(proximity+descent)
    return total/wp.float64(p.n)


@wp.kernel
def reset_xyz(s:WorldState,p:WorldParams,mask:wp.array(dtype=wp.int32)):
    e,i=wp.tid()
    if mask[e]==0:return
    s.altitudes[e,i]=p.altitude;s.vertical_velocity[e,i]=wp.float64(0)
    for cell in range(p.grid_n*p.grid_n):
        s.belief[e,i,cell]=p.belief_prior;s.visited[e,i,cell]=0;s.positive[e,i,cell]=0
        if i==0:s.coverage_first_seen_step[e,cell]=p.max_steps+1
    prior_entropy=entropy_bits(p.belief_prior)
    for coarse in range(p.coarse_n*p.coarse_n):s.belief_coarse[e,i,coarse]=prior_entropy
    if i==0:
        s.belief_potential[e]=wp.float64(1)-prior_entropy
        s.coverage_auc_weight_sum[e]=wp.int64(0)
        s.sensor_tick[e]=0
        for k in range(6):s.sensor_counts[e,k]=wp.int64(0)
        for o in range(p.obstacles_count):
            s.obstacle_heights[e,o]=p.obstacle_min_height+(p.obstacle_max_height-p.obstacle_min_height)*sensor_uniform(s.seeds[e],0,0,o+1610612736)


@wp.kernel
def build_target_cell_lookup(s:WorldState,p:WorldParams,mask:wp.array(dtype=wp.int32),lookup:wp.array2d(dtype=wp.int32)):
    """One-time NoOracle reverse index after geometry reset, not in every sensor tick.

    NoOracle scene placement certifies unique target cells; the table changes
    only on world reset. No truth is exposed to actors or report generation.
    """
    e=wp.tid()
    if p.cell_reports==0 or mask[e]==0:return
    for cell in range(p.grid_n*p.grid_n):
        lookup[e,cell]=-1
    for t in range(p.targets_count):
        target=s.targets[e,t]
        gx=wp.min(int(target[0]/p.cell_width),p.grid_n-1)
        gy=wp.min(int(target[1]/p.cell_width),p.grid_n-1)
        cell=gy*p.grid_n+gx
        if lookup[e,cell]>=0:
            s.errors[e]=117  # A duplicate cell is not a valid NoOracle scene.
        else:
            lookup[e,cell]=t


# Historical 4-argument public kernel kept for external reference tests.
@wp.kernel
def sense_bayesian(s:WorldState,p:WorldParams,advance:int,mask:wp.array(dtype=wp.int32)):
    e,i=wp.tid()
    if advance==0 and mask[e]==0:return
    s.confirmed_count[e,i]=0
    for t in range(p.report_slots):s.visible[e,i,t]=0
    for cell in range(p.grid_n*p.grid_n):s.positive[e,i,cell]=0
    tick=s.steps[e]+advance
    pos=position3(s,e,i)
    if (p.energy_enabled==0 or s.battery[e,i]>wp.float64(0)) and pos[2]>wp.float64(0):
        v=profile(p,pos[2]);pd=v[0];pf=v[1];r=v[2]
        lowx=wp.max(0,int(wp.floor((pos[0]-r)/p.cell_width)));highx=wp.min(p.grid_n-1,int(wp.floor((pos[0]+r)/p.cell_width)))
        lowy=wp.max(0,int(wp.floor((pos[1]-r)/p.cell_width)));highy=wp.min(p.grid_n-1,int(wp.floor((pos[1]+r)/p.cell_width)))
        for gy in range(lowy,highy+1):
            for gx in range(lowx,highx+1):
                cell=gy*p.grid_n+gx;xy=wp.vec2d(pos[0],pos[1])
                q=cell_coverage(p,gx,gy,xy,r)
                center=wp.vec2d(wp.min((wp.float64(gx)+wp.float64(.5))*p.cell_width,p.map_size),wp.min((wp.float64(gy)+wp.float64(.5))*p.cell_width,p.map_size))
                if q>wp.float64(0) and (p.occlusion==0 or camera_clear(s,p,e,pos,center)):
                    has=bool(False)
                    for t in range(p.targets_count):
                        target=s.targets[e,t]
                        if wp.min(int(target[0]/p.cell_width),p.grid_n-1)==gx and wp.min(int(target[1]/p.cell_width),p.grid_n-1)==gy and wp.length(target-xy)<=r:
                            if p.occlusion==0 or camera_clear(s,p,e,pos,target):has=True
                    ef=wp.float64(1)-wp.pow(wp.float64(1)-pf,q);ed=q*pd+(wp.float64(1)-q)*ef
                    probability=ef
                    if has:probability=pd
                    yes=sensor_uniform(s.seeds[e],tick,i,cell)<probability
                    prior=wp.clamp(s.belief[e,i,cell],wp.float64(1e-8),wp.float64(1)-wp.float64(1e-8))
                    a=ed;b=ef
                    if not yes:a=wp.float64(1)-ed;b=wp.float64(1)-ef
                    posterior=wp.clamp(a*prior/wp.max(a*prior+b*(wp.float64(1)-prior),wp.float64(1e-8)),wp.float64(1e-8),wp.float64(1)-wp.float64(1e-8))
                    final_belief=posterior
                    false_confirmation=yes and confirmation_allowed(p,pos[2],ed,ef) and posterior>=p.confirmation_threshold and not has
                    if p.cell_reports==0 and false_confirmation and p.verified_empty_belief>wp.float64(0):final_belief=p.verified_empty_belief
                    # Original 18d coverage-AUC instrumentation: atomic min
                    # counts the first 2D cell visit by ANY UAV exactly once.
                    # It is logging-only, not belief/reward/policy input.
                    first=wp.atomic_min(s.coverage_first_seen_step,e,cell,tick)
                    if first==p.max_steps+1:
                        wp.atomic_add(s.coverage_auc_weight_sum,e,wp.int64(p.max_steps-tick))
                    s.belief[e,i,cell]=final_belief;s.visited[e,i,cell]=1;s.positive[e,i,cell]=int(yes)
                    if p.cell_reports!=0 and yes and confirmation_allowed(p,pos[2],ed,ef) and posterior>=p.confirmation_threshold:
                        idx=s.confirmed_count[e,i]
                        if idx>=p.confirm_capacity:
                            s.errors[e]=116
                        else:
                            s.confirmed_candidates[e,i,idx]=cell
                            s.confirmed_count[e,i]=idx+1
                    old_entropy=entropy_bits(prior);new_entropy=entropy_bits(final_belief)
                    wp.atomic_add(s.belief_potential,e,(old_entropy-new_entropy)/(wp.float64(p.n)*wp.float64(p.grid_n*p.grid_n)))
                    if p.coarse_n>0:
                        group=p.grid_n//p.coarse_n;coarse=(gy//group)*p.coarse_n+(gx//group)
                        s.belief_coarse[e,i,coarse]=wp.clamp(s.belief_coarse[e,i,coarse]+(new_entropy-old_entropy)/wp.float64(group*group),wp.float64(0),wp.float64(1))
                    wp.atomic_add(s.sensor_counts,e,0,wp.int64(1))
                    wp.atomic_add(s.sensor_counts,e,1,wp.int64(int(yes)))
                    wp.atomic_add(s.sensor_counts,e,2,wp.int64(int(yes and not has)))
                    wp.atomic_add(s.sensor_counts,e,3,wp.int64(int(not yes and has)))
                    if yes and confirmation_allowed(p,pos[2],ed,ef) and posterior>=p.confirmation_threshold:
                        # Truth enters only a diagnostic counter, never the
                        # report/decision path in cell-report mode.
                        wp.atomic_add(s.sensor_counts,e,4,wp.int64(int(not has)))
                    if yes and confirmation_allowed(p,pos[2],ed,ef) and posterior>=p.confirmation_threshold and p.cell_reports==0:
                        for t in range(p.targets_count):
                            target=s.targets[e,t]
                            if wp.min(int(target[0]/p.cell_width),p.grid_n-1)==gx and wp.min(int(target[1]/p.cell_width),p.grid_n-1)==gy and wp.length(target-xy)<=r:
                                if p.occlusion==0 or camera_clear(s,p,e,pos,target):
                                    s.visible[e,i,t]=1
                                    wp.atomic_add(s.sensor_counts,e,5,wp.int64(int(s.known[e,i,t]==0)))
    if i==0:s.sensor_tick[e]=tick+1



# New internally indexed kernel, same sensor RNG/Bayesian update.
@wp.kernel
def sense_bayesian_indexed(s:WorldState,p:WorldParams,advance:int,mask:wp.array(dtype=wp.int32),lookup:wp.array2d(dtype=wp.int32)):
    e,i=wp.tid()
    if advance==0 and mask[e]==0:return
    s.confirmed_count[e,i]=0
    for t in range(p.report_slots):s.visible[e,i,t]=0
    for cell in range(p.grid_n*p.grid_n):s.positive[e,i,cell]=0
    tick=s.steps[e]+advance
    pos=position3(s,e,i)
    if (p.energy_enabled==0 or s.battery[e,i]>wp.float64(0)) and pos[2]>wp.float64(0):
        v=profile(p,pos[2]);pd=v[0];pf=v[1];r=v[2]
        lowx=wp.max(0,int(wp.floor((pos[0]-r)/p.cell_width)));highx=wp.min(p.grid_n-1,int(wp.floor((pos[0]+r)/p.cell_width)))
        lowy=wp.max(0,int(wp.floor((pos[1]-r)/p.cell_width)));highy=wp.min(p.grid_n-1,int(wp.floor((pos[1]+r)/p.cell_width)))
        for gy in range(lowy,highy+1):
            for gx in range(lowx,highx+1):
                cell=gy*p.grid_n+gx;xy=wp.vec2d(pos[0],pos[1])
                q=cell_coverage(p,gx,gy,xy,r)
                center=wp.vec2d(wp.min((wp.float64(gx)+wp.float64(.5))*p.cell_width,p.map_size),wp.min((wp.float64(gy)+wp.float64(.5))*p.cell_width,p.map_size))
                if q>wp.float64(0) and (p.occlusion==0 or camera_clear(s,p,e,pos,center)):
                    has=bool(False)
                    if p.cell_reports!=0:
                        # Same truth condition as the historical 35-target
                        # loop; just avoid scanning unrelated target cells.
                        t=lookup[e,cell]
                        if t>=0:
                            target=s.targets[e,t]
                            if wp.length(target-xy)<=r:
                                if p.occlusion==0 or camera_clear(s,p,e,pos,target):has=True
                    else:
                        for t in range(p.targets_count):
                            target=s.targets[e,t]
                            if wp.min(int(target[0]/p.cell_width),p.grid_n-1)==gx and wp.min(int(target[1]/p.cell_width),p.grid_n-1)==gy and wp.length(target-xy)<=r:
                                if p.occlusion==0 or camera_clear(s,p,e,pos,target):has=True
                    ef=wp.float64(1)-wp.pow(wp.float64(1)-pf,q);ed=q*pd+(wp.float64(1)-q)*ef
                    probability=ef
                    if has:probability=pd
                    yes=sensor_uniform(s.seeds[e],tick,i,cell)<probability
                    prior=wp.clamp(s.belief[e,i,cell],wp.float64(1e-8),wp.float64(1)-wp.float64(1e-8))
                    a=ed;b=ef
                    if not yes:a=wp.float64(1)-ed;b=wp.float64(1)-ef
                    posterior=wp.clamp(a*prior/wp.max(a*prior+b*(wp.float64(1)-prior),wp.float64(1e-8)),wp.float64(1e-8),wp.float64(1)-wp.float64(1e-8))
                    final_belief=posterior
                    false_confirmation=yes and confirmation_allowed(p,pos[2],ed,ef) and posterior>=p.confirmation_threshold and not has
                    if p.cell_reports==0 and false_confirmation and p.verified_empty_belief>wp.float64(0):final_belief=p.verified_empty_belief
                    # Original 18d coverage-AUC instrumentation: atomic min
                    # counts the first 2D cell visit by ANY UAV exactly once.
                    # It is logging-only, not belief/reward/policy input.
                    first=wp.atomic_min(s.coverage_first_seen_step,e,cell,tick)
                    if first==p.max_steps+1:
                        wp.atomic_add(s.coverage_auc_weight_sum,e,wp.int64(p.max_steps-tick))
                    s.belief[e,i,cell]=final_belief;s.visited[e,i,cell]=1;s.positive[e,i,cell]=int(yes)
                    if p.cell_reports!=0 and yes and confirmation_allowed(p,pos[2],ed,ef) and posterior>=p.confirmation_threshold:
                        idx=s.confirmed_count[e,i]
                        if idx>=p.confirm_capacity:
                            s.errors[e]=116
                        else:
                            s.confirmed_candidates[e,i,idx]=cell
                            s.confirmed_count[e,i]=idx+1
                    old_entropy=entropy_bits(prior);new_entropy=entropy_bits(final_belief)
                    wp.atomic_add(s.belief_potential,e,(old_entropy-new_entropy)/(wp.float64(p.n)*wp.float64(p.grid_n*p.grid_n)))
                    if p.coarse_n>0:
                        group=p.grid_n//p.coarse_n;coarse=(gy//group)*p.coarse_n+(gx//group)
                        s.belief_coarse[e,i,coarse]=wp.clamp(s.belief_coarse[e,i,coarse]+(new_entropy-old_entropy)/wp.float64(group*group),wp.float64(0),wp.float64(1))
                    wp.atomic_add(s.sensor_counts,e,0,wp.int64(1))
                    wp.atomic_add(s.sensor_counts,e,1,wp.int64(int(yes)))
                    wp.atomic_add(s.sensor_counts,e,2,wp.int64(int(yes and not has)))
                    wp.atomic_add(s.sensor_counts,e,3,wp.int64(int(not yes and has)))
                    if yes and confirmation_allowed(p,pos[2],ed,ef) and posterior>=p.confirmation_threshold:
                        # Truth enters only a diagnostic counter, never the
                        # report/decision path in cell-report mode.
                        wp.atomic_add(s.sensor_counts,e,4,wp.int64(int(not has)))
                    if yes and confirmation_allowed(p,pos[2],ed,ef) and posterior>=p.confirmation_threshold and p.cell_reports==0:
                        for t in range(p.targets_count):
                            target=s.targets[e,t]
                            if wp.min(int(target[0]/p.cell_width),p.grid_n-1)==gx and wp.min(int(target[1]/p.cell_width),p.grid_n-1)==gy and wp.length(target-xy)<=r:
                                if p.occlusion==0 or camera_clear(s,p,e,pos,target):
                                    s.visible[e,i,t]=1
                                    wp.atomic_add(s.sensor_counts,e,5,wp.int64(int(s.known[e,i,t]==0)))
    if i==0:s.sensor_tick[e]=tick+1


@wp.kernel
def confirm_observed_cells(s:WorldState,p:WorldParams,mask:wp.array(dtype=wp.int32),masked:int):
    """Deterministic, sparse candidate matching across UAVs in a world.

    A REPORT slot is assigned solely from observed cell coordinates and Bayes
    evidence. The TRUE target array never participates in this code path.
    """
    e=wp.tid()
    if p.cell_reports==0 or s.terminated[e]!=0:return
    if masked!=0 and mask[e]==0:return
    for i in range(p.n):
        for idx in range(s.confirmed_count[e,i]):
            cell=s.confirmed_candidates[e,i,idx]
            selected=int(-1)
            free=int(-1)
            # Allocator appends at the first free slot and never deletes an
            # individual cell until a whole-world reset. The first free slot
            # therefore terminates the allocated prefix.
            for slot in range(p.report_slots):
                existing=s.report_cell[e,slot]
                if existing==cell:
                    selected=slot
                    break
                if existing<0:
                    free=slot
                    break
            if selected<0:
                selected=free
                if selected>=0:s.report_cell[e,selected]=cell
            if selected<0:
                # Fail closed rather than discarding legitimate/false hypotheses.
                s.errors[e]=115
            else:
                s.visible[e,i,selected]=1
                if s.known[e,i,selected]==0:
                    s.sensor_counts[e,5]+=wp.int64(1)
