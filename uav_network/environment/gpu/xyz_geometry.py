"""Device XYZ geometry; legacy XY packet coordinates remain supported."""
import warp as wp
from .state import WorldState,WorldParams

wp.set_module_options({"enable_backward":False,"fuse_fp":False})


@wp.func
def position3(s:WorldState,e:int,i:int):
    a=s.positions[e,i]
    return wp.vec3d(a[0],a[1],s.altitudes[e,i])


@wp.func
def velocity3(s:WorldState,e:int,i:int):
    a=s.velocities[e,i]
    return wp.vec3d(a[0],a[1],s.vertical_velocity[e,i])


@wp.func
def node_z(s:WorldState,p:WorldParams,e:int,i:int):
    z=wp.float64(0)
    if i<p.n:
        z=p.altitude
        if p.xyz!=0:z=s.altitudes[e,i]
    return z


@wp.func
def node3(s:WorldState,p:WorldParams,e:int,i:int):
    a=s.gcs[e]
    if i<p.n:a=s.positions[e,i]
    return wp.vec3d(a[0],a[1],node_z(s,p,e,i))


@wp.func
def range_distance(s:WorldState,p:WorldParams,e:int,i:int,j:int):
    a,b=node3(s,p,e,i),node3(s,p,e,j)
    delta=a-b
    if p.xyz==0:delta[2]=wp.float64(0)
    return wp.length(delta)


@wp.func
def cylinder_hit(a:wp.vec3d,b:wp.vec3d,o:wp.vec4d,margin:wp.float64):
    d=b-a;low=wp.float64(0);high=wp.float64(1);valid=bool(True)
    if wp.abs(d[2])<wp.float64(1e-15):
        if a[2]<wp.float64(0) or a[2]>o[3]+margin:valid=False
    else:
        u=(wp.float64(0)-a[2])/d[2];v=(o[3]+margin-a[2])/d[2]
        low=wp.max(low,wp.min(u,v));high=wp.min(high,wp.max(u,v))
    q=wp.vec2d(a[0]-o[0],a[1]-o[1]);xy=wp.vec2d(d[0],d[1])
    aa=wp.dot(xy,xy);cc=wp.dot(q,q)-(o[2]+margin)*(o[2]+margin)
    if aa<wp.float64(1e-24):
        if cc>wp.float64(0):valid=False
    else:
        bb=wp.float64(2)*wp.dot(q,xy);disc=bb*bb-wp.float64(4)*aa*cc
        if disc<wp.float64(0):valid=False
        else:
            root=wp.sqrt(wp.max(wp.float64(0),disc))
            low=wp.max(low,(-bb-root)/(wp.float64(2)*aa))
            high=wp.min(high,(-bb+root)/(wp.float64(2)*aa))
    return valid and low<=high


@wp.func
def camera_clear(s:WorldState,p:WorldParams,e:int,a:wp.vec3d,xy:wp.vec2d):
    b=wp.vec3d(xy[0],xy[1],0.0);clear=bool(True)
    for o in range(p.obstacles_count):
        v=s.obstacles[e,o]
        if cylinder_hit(a,b,wp.vec4d(v[0],v[1],v[2],s.obstacle_heights[e,o]),wp.float64(0)):clear=False
    return clear
