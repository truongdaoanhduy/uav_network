"""One device thread per world: XYZ candidate/braking safety and funded flight."""
import warp as wp
from .state import WorldParams,WorldState
from .xyz_geometry import position3,velocity3,cylinder_hit

wp.set_module_options({"enable_backward":False,"fuse_fp":False})


@wp.func
def cap(v:wp.vec3d,maximum:wp.float64):
    return v*wp.min(wp.float64(1),maximum/wp.max(wp.length(v),wp.float64(1e-12)))


@wp.func
def brake(v:wp.vec3d,step:wp.float64):
    return v*wp.max(wp.float64(0),wp.float64(1)-step/wp.max(wp.length(v),wp.float64(1e-12)))


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
def static_path(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,pos:wp.vec3d,velocity:wp.vec3d):
    lo=wp.vec3d(0.0,0.0,p.altitude_min);hi=wp.vec3d(p.map_size,p.map_size,p.altitude_max)
    point=pos;speed=velocity;valid=bool(True)
    for k in range(int(wp.ceil(p.max_speed/(p.max_accel*p.dt)))+1):
        end=point+speed*p.dt
        for a in range(3):
            if point[a]<lo[a] or point[a]>hi[a] or end[a]<lo[a] or end[a]>hi[a]:valid=False
        for o in range(p.obstacles_count):
            v=s.obstacles[e,o]
            if cylinder_hit(point,end,wp.vec4d(v[0],v[1],v[2],s.obstacle_heights[e,o]),p.obstacle_clearance):valid=False
        point=end;speed=brake(speed,p.max_accel*p.dt)
    return valid


@wp.func
def peer_path(a:wp.vec3d,av:wp.vec3d,b:wp.vec3d,bv:wp.vec3d,p:wp.ref[WorldParams]):
    pos=a;other=b;v=av;ov=bv;valid=bool(True)
    for k in range(int(wp.ceil(p.max_speed/(p.max_accel*p.dt)))+1):
        end=pos+v*p.dt;oe=other+ov*p.dt
        if distance(pos-other,end-oe)<p.safety_distance-wp.float64(1e-9):valid=False
        pos=end;other=oe;v=brake(v,p.max_accel*p.dt);ov=brake(ov,p.max_accel*p.dt)
    return valid


@wp.func
def cost(prev:wp.vec3d,actual:wp.vec3d,p:wp.ref[WorldParams]):
    a=(actual-prev)/p.dt
    return (p.idle_power+p.hover_power+p.speed_coefficient*wp.dot(actual,actual)+p.accel_coefficient*wp.dot(a,a))*p.dt


@wp.kernel
def motion_xyz(s:WorldState,p:WorldParams,actions:wp.array3d(dtype=wp.float32)):
    e=wp.tid()
    if s.errors[e]!=0 or s.terminated[e]!=0:return
    step=p.max_accel*p.dt
    for i in range(p.n):s.motion_flags[e,i,0]=int(p.energy_enabled!=0 and s.battery[e,i]<=wp.float64(0))
    for retry in range(p.n+1):
        for i in range(p.n):
            previous=velocity3(s,e,i)
            command=wp.vec3d(wp.float64(actions[e,i,0]),wp.float64(actions[e,i,1]),wp.float64(actions[e,i,2]))
            nominal=project(previous+command*step,previous,p)
            chosen=brake(previous,step)
            if s.motion_flags[e,i,0]!=0:nominal=wp.vec3d(0.0);chosen=wp.vec3d(0.0)
            s.xyz_work[e,i,0]=nominal;s.xyz_work[e,i,1]=chosen
            s.motion_flags[e,i,1]=0
        joint_safe=bool(True)
        if p.motion_safety!=0:
            for i in range(p.n):
                if s.motion_flags[e,i,0]==0 and not static_path(s,p,e,position3(s,e,i),s.xyz_work[e,i,0]):
                    joint_safe=False
                for j in range(i):
                    if not peer_path(position3(s,e,i),s.xyz_work[e,i,0],
                                     position3(s,e,j),s.xyz_work[e,j,0],p):
                        joint_safe=False
        if p.motion_safety!=0 and joint_safe:
            for i in range(p.n):
                s.xyz_work[e,i,1]=s.xyz_work[e,i,0]
                s.motion_flags[e,i,1]=0
        else:
            for i in range(p.n):
                if s.motion_flags[e,i,0]==0:
                    previous=velocity3(s,e,i);pos=position3(s,e,i);nominal=s.xyz_work[e,i,0]
                    best=wp.float64(1e100);selected=bool(False)
                    if p.motion_safety!=0:
                        for candidate in range(29):
                            v=nominal
                            if candidate==1:v=s.xyz_work[e,i,1]
                            elif candidate>=2:
                                q=candidate-2
                                v=project(previous+step*wp.vec3d(wp.float64(q//9-1),wp.float64((q//3)%3-1),wp.float64(q%3-1)),previous,p)
                            score=wp.dot(v-nominal,v-nominal)
                            if score<best and static_path(s,p,e,pos,v):
                                safe=bool(True)
                                for j in range(p.n):
                                    if j!=i and not peer_path(pos,v,position3(s,e,j),s.xyz_work[e,j,1],p):safe=False
                                if safe:best=score;s.xyz_work[e,i,1]=v;selected=True
                        s.motion_flags[e,i,1]=int(not selected)
                    else:s.xyz_work[e,i,1]=nominal
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
            s.motion_counts[e,3]+=int(wp.length(s.xyz_work[e,i,1]-s.xyz_work[e,i,0])>wp.float64(1e-9))
            s.motion_counts[e,7]+=s.motion_flags[e,i,1]
