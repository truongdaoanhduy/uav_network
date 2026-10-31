"""One device thread per world: XYZ CBF safety and funded flight."""
import warp as wp
from .state import WorldParams,WorldState
from .xyz_geometry import position3,velocity3,cylinder_hit

wp.set_module_options({"enable_backward":False,"fuse_fp":False})


@wp.func
def cap(v:wp.vec3d,maximum:wp.float64):
    return v*wp.min(wp.float64(1),maximum/wp.max(wp.length(v),wp.float64(1e-12)))


@wp.func
def clip3(v:wp.vec3d,low:wp.vec3d,high:wp.vec3d):
    return wp.vec3d(wp.clamp(v[0],low[0],high[0]),wp.clamp(v[1],low[1],high[1]),wp.clamp(v[2],low[2],high[2]))


@wp.func
def project(desired:wp.vec3d,previous:wp.vec3d,p:wp.ref[WorldParams]):
    step=p.max_accel*p.dt
    lower=previous-wp.vec3d(step);upper=previous+wp.vec3d(step)
    result=clip3(desired,lower,upper)
    if wp.length(result)>p.max_speed:
        if wp.length(clip3(wp.vec3d(0.0),lower,upper))<=p.max_speed:
            low=wp.float64(0);high=wp.float64(1)
            while wp.length(clip3(desired/(wp.float64(1)+high),lower,upper))>p.max_speed:high*=wp.float64(2)
            for k in range(40):
                mid=(low+high)/wp.float64(2)
                if wp.length(clip3(desired/(wp.float64(1)+mid),lower,upper))>p.max_speed:low=mid
                else:high=mid
            result=clip3(desired/(wp.float64(1)+high),lower,upper)
        else:result=cap(result,p.max_speed)
    return result


@wp.func
def distance(a:wp.vec3d,b:wp.vec3d):
    d=b-a;f=wp.clamp(-wp.dot(a,d)/wp.max(wp.dot(d,d),wp.float64(1e-24)),wp.float64(0),wp.float64(1))
    return wp.length(a+f*d)


@wp.func
def cost(prev:wp.vec3d,actual:wp.vec3d,p:wp.ref[WorldParams]):
    a=(actual-prev)/p.dt
    return (p.idle_power+p.hover_power+p.speed_coefficient*wp.dot(actual,actual)+p.accel_coefficient*wp.dot(a,a))*p.dt


@wp.func
def cbf_horizon(closing:wp.float64,decel:wp.float64,p:wp.ref[WorldParams]):
    return wp.max(p.dt,closing/wp.max(decel,wp.float64(1e-12)))


@wp.func
def cbf_bound(gap:wp.float64,closing:wp.float64,horizon:wp.float64,p:wp.ref[WorldParams]):
    coefficient=wp.float64(.5)*horizon*(horizon+p.dt)
    return (gap-closing*horizon)/coefficient


@wp.func
def axis_direction(axis:int,sign:wp.float64):
    result=wp.vec3d(0.0)
    result[axis]=sign
    return result


@wp.func
def clip_acceleration(value:wp.vec3d,p:wp.ref[WorldParams]):
    limit=p.max_accel
    return clip3(value,wp.vec3d(-limit),wp.vec3d(limit))


@wp.func
def project_cbf(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int):
    """Project xyz_work[...,1] onto the same half-spaces as native CBF."""
    tol=wp.float64(1e-9)
    obstacle_count=int(0);boundary_count=int(0)
    for i in range(p.n):s.motion_flags[e,i,1]=0

    # Classify initial nominal violations once for telemetry.
    for i in range(p.n):
        if s.motion_flags[e,i,0]==0:
            pos=position3(s,e,i);vel=velocity3(s,e,i);nominal=s.xyz_work[e,i,0]
            boundary_active=bool(False)
            for axis in range(3):
                low=wp.float64(0.0);high=p.map_size
                if axis==2:low=p.altitude_min;high=p.altitude_max
                closing=wp.max(wp.float64(0),-vel[axis]);H=cbf_horizon(closing,p.max_accel,p)
                b=cbf_bound(pos[axis]-low,closing,H,p)
                if -nominal[axis]-b>tol:boundary_active=True
                closing=wp.max(wp.float64(0),vel[axis]);H=cbf_horizon(closing,p.max_accel,p)
                b=cbf_bound(high-pos[axis],closing,H,p)
                if nominal[axis]-b>tol:boundary_active=True
            if boundary_active:boundary_count+=1

            obstacle_active=bool(False)
            for o in range(p.obstacles_count):
                obs=s.obstacles[e,o];dx=obs[0]-pos[0];dy=obs[1]-pos[1]
                radial=wp.sqrt(dx*dx+dy*dy);direction=wp.vec3d(1.0,0.0,0.0)
                if radial>wp.float64(1e-12):direction=wp.vec3d(dx/radial,dy/radial,0.0)
                radius=obs[2]+p.obstacle_clearance;top=s.obstacle_heights[e,o]+p.obstacle_clearance
                side_closing=wp.max(wp.float64(0),wp.dot(vel,direction));H=cbf_horizon(side_closing,p.max_accel,p)
                side_b=cbf_bound(radial-radius,side_closing,H,p);side_v=wp.dot(nominal,direction)-side_b
                top_closing=wp.max(wp.float64(0),-vel[2]);H=cbf_horizon(top_closing,p.max_accel,p)
                top_b=cbf_bound(pos[2]-top,top_closing,H,p);top_v=-nominal[2]-top_b
                use_side=bool(False);use_top=bool(False)
                if pos[2]<=top and radial>=radius:use_side=True
                elif radial<=radius and pos[2]>=top:use_top=True
                elif pos[2]<=top and radial<=radius:use_side=True;use_top=True
                elif side_v>tol and top_v>tol:
                    if side_v<=top_v:use_side=True
                    else:use_top=True
                if (use_side and side_v>tol) or (use_top and top_v>tol):obstacle_active=True
            if obstacle_active:obstacle_count+=1

    for i in range(p.n):
        for j in range(i):
            pi=position3(s,e,i);pj=position3(s,e,j);delta=pj-pi;dist=wp.length(delta)
            direction=wp.vec3d(1.0,0.0,0.0)
            if dist>wp.float64(1e-12):direction=delta/dist
            closing=wp.max(wp.float64(0),wp.dot(velocity3(s,e,i)-velocity3(s,e,j),direction))
            H=cbf_horizon(closing,wp.float64(2)*p.max_accel,p)
            b=cbf_bound(dist-p.safety_distance,closing,H,p)
            violation=wp.dot(s.xyz_work[e,i,0],direction)-wp.dot(s.xyz_work[e,j,0],direction)-b
            if violation>tol:s.motion_flags[e,i,1]=1;s.motion_flags[e,j,1]=1
    peer_count=int(0)
    for i in range(p.n):peer_count+=s.motion_flags[e,i,1];s.motion_flags[e,i,1]=0

    # Deterministic most-violated-constraint projection, matching native order.
    for iteration in range(32):
        worst=wp.float64(-1e300);kind=int(0);bi=int(-1);bj=int(-1);direction=wp.vec3d(0.0)
        for i in range(p.n):
            if s.motion_flags[e,i,0]==0:
                pos=position3(s,e,i);vel=velocity3(s,e,i);safe=s.xyz_work[e,i,1]
                for axis in range(3):
                    low=wp.float64(0.0);high=p.map_size
                    if axis==2:low=p.altitude_min;high=p.altitude_max
                    closing=wp.max(wp.float64(0),-vel[axis]);H=cbf_horizon(closing,p.max_accel,p)
                    b=cbf_bound(pos[axis]-low,closing,H,p);v=-safe[axis]-b
                    if v>worst:worst=v;kind=1;bi=i;bj=-1;direction=axis_direction(axis,wp.float64(-1))
                    closing=wp.max(wp.float64(0),vel[axis]);H=cbf_horizon(closing,p.max_accel,p)
                    b=cbf_bound(high-pos[axis],closing,H,p);v=safe[axis]-b
                    if v>worst:worst=v;kind=1;bi=i;bj=-1;direction=axis_direction(axis,wp.float64(1))

                nominal=s.xyz_work[e,i,0]
                for o in range(p.obstacles_count):
                    obs=s.obstacles[e,o];dx=obs[0]-pos[0];dy=obs[1]-pos[1]
                    radial=wp.sqrt(dx*dx+dy*dy);side_dir=wp.vec3d(1.0,0.0,0.0)
                    if radial>wp.float64(1e-12):side_dir=wp.vec3d(dx/radial,dy/radial,0.0)
                    radius=obs[2]+p.obstacle_clearance;top=s.obstacle_heights[e,o]+p.obstacle_clearance
                    side_closing=wp.max(wp.float64(0),wp.dot(vel,side_dir));H=cbf_horizon(side_closing,p.max_accel,p)
                    side_b=cbf_bound(radial-radius,side_closing,H,p)
                    nominal_side=wp.dot(nominal,side_dir)-side_b;side_v=wp.dot(safe,side_dir)-side_b
                    top_closing=wp.max(wp.float64(0),-vel[2]);H=cbf_horizon(top_closing,p.max_accel,p)
                    top_b=cbf_bound(pos[2]-top,top_closing,H,p);nominal_top=-nominal[2]-top_b;top_v=-safe[2]-top_b
                    use_side=bool(False);use_top=bool(False)
                    if pos[2]<=top and radial>=radius:use_side=True
                    elif radial<=radius and pos[2]>=top:use_top=True
                    elif pos[2]<=top and radial<=radius:use_side=True;use_top=True
                    elif nominal_side>tol and nominal_top>tol:
                        if nominal_side<=nominal_top:use_side=True
                        else:use_top=True
                    if use_side and side_v>worst:worst=side_v;kind=2;bi=i;bj=-1;direction=side_dir
                    if use_top and top_v>worst:worst=top_v;kind=2;bi=i;bj=-1;direction=wp.vec3d(0.0,0.0,-1.0)

        for i in range(p.n):
            for j in range(i):
                pi=position3(s,e,i);pj=position3(s,e,j);delta=pj-pi;dist=wp.length(delta)
                peer_dir=wp.vec3d(1.0,0.0,0.0)
                if dist>wp.float64(1e-12):peer_dir=delta/dist
                closing=wp.max(wp.float64(0),wp.dot(velocity3(s,e,i)-velocity3(s,e,j),peer_dir))
                H=cbf_horizon(closing,wp.float64(2)*p.max_accel,p)
                b=cbf_bound(dist-p.safety_distance,closing,H,p)
                v=wp.dot(s.xyz_work[e,i,1],peer_dir)-wp.dot(s.xyz_work[e,j,1],peer_dir)-b
                if v>worst:worst=v;kind=3;bi=i;bj=j;direction=peer_dir
        if worst<=tol:break
        if kind==1 or kind==2:
            s.xyz_work[e,bi,1]=clip_acceleration(s.xyz_work[e,bi,1]-worst*direction,p)
        elif kind==3:
            correction=worst/wp.float64(2)
            s.xyz_work[e,bi,1]=clip_acceleration(s.xyz_work[e,bi,1]-correction*direction,p)
            s.xyz_work[e,bj,1]=clip_acceleration(s.xyz_work[e,bj,1]+correction*direction,p)
        for i in range(p.n):
            if s.motion_flags[e,i,0]!=0:s.xyz_work[e,i,1]=wp.vec3d(0.0)

    # One final scan only needs to know whether any constraint still violates.
    max_violation=wp.float64(0)
    for i in range(p.n):
        if s.motion_flags[e,i,0]==0:
            pos=position3(s,e,i);vel=velocity3(s,e,i);safe=s.xyz_work[e,i,1];nominal=s.xyz_work[e,i,0]
            for axis in range(3):
                low=wp.float64(0.0);high=p.map_size
                if axis==2:low=p.altitude_min;high=p.altitude_max
                closing=wp.max(wp.float64(0),-vel[axis]);H=cbf_horizon(closing,p.max_accel,p)
                b=cbf_bound(pos[axis]-low,closing,H,p);max_violation=wp.max(max_violation,-safe[axis]-b)
                closing=wp.max(wp.float64(0),vel[axis]);H=cbf_horizon(closing,p.max_accel,p)
                b=cbf_bound(high-pos[axis],closing,H,p);max_violation=wp.max(max_violation,safe[axis]-b)
            for o in range(p.obstacles_count):
                obs=s.obstacles[e,o];dx=obs[0]-pos[0];dy=obs[1]-pos[1]
                radial=wp.sqrt(dx*dx+dy*dy);side_dir=wp.vec3d(1.0,0.0,0.0)
                if radial>wp.float64(1e-12):side_dir=wp.vec3d(dx/radial,dy/radial,0.0)
                radius=obs[2]+p.obstacle_clearance;top=s.obstacle_heights[e,o]+p.obstacle_clearance
                side_closing=wp.max(wp.float64(0),wp.dot(vel,side_dir));H=cbf_horizon(side_closing,p.max_accel,p)
                side_b=cbf_bound(radial-radius,side_closing,H,p);nominal_side=wp.dot(nominal,side_dir)-side_b;side_v=wp.dot(safe,side_dir)-side_b
                top_closing=wp.max(wp.float64(0),-vel[2]);H=cbf_horizon(top_closing,p.max_accel,p)
                top_b=cbf_bound(pos[2]-top,top_closing,H,p);nominal_top=-nominal[2]-top_b;top_v=-safe[2]-top_b
                if pos[2]<=top and radial>=radius:max_violation=wp.max(max_violation,side_v)
                elif radial<=radius and pos[2]>=top:max_violation=wp.max(max_violation,top_v)
                elif pos[2]<=top and radial<=radius:max_violation=wp.max(max_violation,wp.max(side_v,top_v))
                elif nominal_side>tol and nominal_top>tol:
                    if nominal_side<=nominal_top:max_violation=wp.max(max_violation,side_v)
                    else:max_violation=wp.max(max_violation,top_v)
    for i in range(p.n):
        for j in range(i):
            pi=position3(s,e,i);pj=position3(s,e,j);delta=pj-pi;dist=wp.length(delta)
            direction=wp.vec3d(1.0,0.0,0.0)
            if dist>wp.float64(1e-12):direction=delta/dist
            closing=wp.max(wp.float64(0),wp.dot(velocity3(s,e,i)-velocity3(s,e,j),direction))
            H=cbf_horizon(closing,wp.float64(2)*p.max_accel,p)
            b=cbf_bound(dist-p.safety_distance,closing,H,p)
            max_violation=wp.max(max_violation,wp.dot(s.xyz_work[e,i,1],direction)-wp.dot(s.xyz_work[e,j,1],direction)-b)
    return wp.vec4d(wp.float64(obstacle_count),wp.float64(peer_count),wp.float64(boundary_count),wp.float64(int(max_violation>wp.float64(1e-6))))


@wp.kernel
def motion_xyz(s:WorldState,p:WorldParams,actions:wp.array3d(dtype=wp.float32)):
    e=wp.tid()
    if s.errors[e]!=0 or s.terminated[e]!=0:return
    cbf_counts=wp.vec4d(0.0)
    for i in range(p.n):s.motion_flags[e,i,0]=int(p.energy_enabled!=0 and s.battery[e,i]<=wp.float64(0))
    for retry in range(p.n+1):
        for i in range(p.n):
            command=wp.vec3d(wp.clamp(wp.float64(actions[e,i,0]),wp.float64(-1),wp.float64(1)),
                             wp.clamp(wp.float64(actions[e,i,1]),wp.float64(-1),wp.float64(1)),
                             wp.clamp(wp.float64(actions[e,i,2]),wp.float64(-1),wp.float64(1)))
            accel=command*p.max_accel
            if s.motion_flags[e,i,0]!=0:accel=wp.vec3d(0.0)
            s.xyz_work[e,i,0]=accel;s.xyz_work[e,i,1]=accel
            s.motion_flags[e,i,1]=0;s.motion_flags[e,i,2]=0;s.motion_flags[e,i,3]=0;s.motion_flags[e,i,4]=0
        cbf_counts=wp.vec4d(0.0)
        if p.motion_safety==2:cbf_counts=project_cbf(s,p,e)
        for i in range(p.n):
            nominal_accel=s.xyz_work[e,i,0];safe_accel=s.xyz_work[e,i,1]
            s.motion_flags[e,i,1]=int(wp.length(safe_accel-nominal_accel)>wp.float64(1e-9))
            previous=velocity3(s,e,i)
            nominal=project(previous+nominal_accel*p.dt,previous,p)
            chosen=project(previous+safe_accel*p.dt,previous,p)
            if s.motion_flags[e,i,0]!=0:nominal=wp.vec3d(0.0);chosen=wp.vec3d(0.0)
            s.xyz_work[e,i,0]=nominal;s.xyz_work[e,i,1]=chosen

        for i in range(p.n):
            pos=position3(s,e,i);end=clip3(pos+s.xyz_work[e,i,1]*p.dt,wp.vec3d(0.0,0.0,p.altitude_min),wp.vec3d(p.map_size,p.map_size,p.altitude_max))
            if s.motion_flags[e,i,0]!=0:end=pos
            blocked=bool(False)
            for o in range(p.obstacles_count):
                v=s.obstacles[e,o]
                if cylinder_hit(pos,end,wp.vec4d(v[0],v[1],v[2],s.obstacle_heights[e,o]),p.obstacle_clearance):blocked=True
            s.motion_flags[e,i,2]=int(blocked);s.motion_flags[e,i,3]=int(blocked)
            s.motion_flags[e,i,4]=int(wp.length(end-(pos+s.xyz_work[e,i,1]*p.dt))>wp.float64(0))
            s.xyz_work[e,i,2]=end
        for repeat in range(p.n+1):
            for i in range(p.n):
                if s.motion_flags[e,i,2]!=0:s.xyz_work[e,i,2]=position3(s,e,i)
            changed=bool(False)
            for i in range(p.n):
                for j in range(i):
                    if distance(position3(s,e,i)-position3(s,e,j),s.xyz_work[e,i,2]-s.xyz_work[e,j,2])<p.safety_distance-wp.float64(1e-9):
                        if s.motion_flags[e,i,2]==0 or s.motion_flags[e,j,2]==0:changed=True
                        s.motion_flags[e,i,2]=1;s.motion_flags[e,j,2]=1
            if not changed:break
        unaffordable=bool(False)
        for i in range(p.n):
            if s.motion_flags[e,i,2]!=0:s.xyz_work[e,i,2]=position3(s,e,i)
            actual=(s.xyz_work[e,i,2]-position3(s,e,i))/p.dt
            if p.energy_enabled!=0 and s.battery[e,i]>wp.float64(0) and s.motion_flags[e,i,0]==0 and cost(velocity3(s,e,i),actual,p)>s.battery[e,i]:
                s.motion_flags[e,i,0]=1;unaffordable=True
        if not unaffordable:break
    correction=wp.float64(0);active_count=int(0)
    for i in range(p.n):
        if s.motion_flags[e,i,0]==0:
            correction+=wp.length(s.xyz_work[e,i,1]-s.xyz_work[e,i,0])/p.max_speed;active_count+=1
    if active_count>0:correction/=wp.float64(active_count)
    s.shield_correction_step[e]=correction;s.shield_correction[e]+=correction
    s.motion_counts[e,4]+=int(cbf_counts[0]);s.motion_counts[e,5]+=int(cbf_counts[1]);s.motion_counts[e,6]+=int(cbf_counts[2]);s.motion_counts[e,7]+=int(cbf_counts[3])
    for i in range(p.n):
        pos=position3(s,e,i);end=s.xyz_work[e,i,2];actual=(end-pos)/p.dt
        energy=cost(velocity3(s,e,i),actual,p)
        if p.energy_enabled!=0:
            if s.battery[e,i]<=wp.float64(0):energy=wp.float64(0)
            elif s.motion_flags[e,i,0]!=0:energy=s.battery[e,i]
            s.battery[e,i]=wp.max(wp.float64(0),s.battery[e,i]-energy)
        s.flight_energy[e,i]+=energy
        s.positions[e,i]=wp.vec2d(end[0],end[1]);s.altitudes[e,i]=end[2]
        s.velocities[e,i]=wp.vec2d(actual[0],actual[1]);s.vertical_velocity[e,i]=actual[2]
        if s.motion_flags[e,i,0]==0:
            s.blocked[e]+=s.motion_flags[e,i,2];s.clipped[e]+=s.motion_flags[e,i,4]
            s.motion_counts[e,0]+=s.motion_flags[e,i,3]
            s.motion_counts[e,1]+=int(s.motion_flags[e,i,2]!=0 and s.motion_flags[e,i,3]==0)
            cmd=wp.vec3d(wp.float64(actions[e,i,0]),wp.float64(actions[e,i,1]),wp.float64(actions[e,i,2]))
            s.motion_counts[e,2]+=int(wp.length(cmd)>wp.float64(.1) and wp.length(actual)<wp.float64(1e-8))
            s.motion_counts[e,3]+=s.motion_flags[e,i,1]
