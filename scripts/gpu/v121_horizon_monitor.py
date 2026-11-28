"""On-device persistent monitors independent of the shield's success flag."""
import warp as wp

wp.set_module_options({'enable_backward':False, 'max_unroll':2})


@wp.kernel
def monitor(pos:wp.array3d(dtype=wp.float32), vel:wp.array3d(dtype=wp.float32),
            battery:wp.array2d(dtype=wp.float32), reward:wp.array(dtype=wp.float32),
            components:wp.array2d(dtype=wp.float32), failed:wp.array(dtype=wp.uint8),
            overflow:wp.array(dtype=wp.uint8), done:wp.array(dtype=wp.uint8),
            prev_pos:wp.array3d(dtype=wp.float32), prev_bat:wp.array2d(dtype=wp.float32),
            buildings:wp.array2d(dtype=wp.float32), nbuild:int,
            sticky:wp.array2d(dtype=wp.int32), errors:wp.array2d(dtype=wp.float32),
            step:int, vmax:float, safe_d:float):
    e=wp.tid()
    if failed[e]!=wp.uint8(0): sticky[e,0]=1
    if overflow[e]!=wp.uint8(0): sticky[e,1]=1
    if done[e]!=wp.uint8(0) and sticky[e,4]==0: sticky[e,4]=step
    if not wp.isfinite(reward[e]): sticky[e,2]=1
    total=float(0.0)
    for k in range(8):
        if not wp.isfinite(components[e,k]): sticky[e,2]=1
        total=total+components[e,k]
    errors[e,0]=wp.max(errors[e,0],wp.abs(total-reward[e]))
    for u in range(7):
        for axis in range(3):
            if not wp.isfinite(pos[e,u,axis]): sticky[e,2]=1
    for u in range(6):
        if not wp.isfinite(battery[e,u]): sticky[e,2]=1
        if battery[e,u]>prev_bat[e,u]+0.001: sticky[e,3]=1
        vx=vel[e,u,0]; vy=vel[e,u,1]
        if not wp.isfinite(vx) or not wp.isfinite(vy): sticky[e,2]=1
        errors[e,1]=wp.max(errors[e,1],wp.sqrt(vx*vx+vy*vy)-vmax)
        if prev_bat[e,u]>0.0:
            x=pos[e,u,0]; y=pos[e,u,1]
            px=prev_pos[e,u,0]; py=prev_pos[e,u,1]
            if x<9.9998 or x>990.0002 or y<9.9998 or y>990.0002: sticky[e,5]=1
            dx=x-px; dy=y-py
            denom=wp.max(1e-12,dx*dx+dy*dy)
            for k in range(nbuild):
                tau=wp.clamp(((buildings[k,0]-px)*dx+(buildings[k,1]-py)*dy)/denom,0.0,1.0)
                bx=px+tau*dx-buildings[k,0]; by=py+tau*dy-buildings[k,1]
                radius=buildings[k,2]+2.0-0.0002
                if bx*bx+by*by<radius*radius: sticky[e,5]=1
        for j in range(u+1,6):
            if prev_bat[e,u]>0.0 or prev_bat[e,j]>0.0:
                rx=prev_pos[e,u,0]-prev_pos[e,j,0]
                ry=prev_pos[e,u,1]-prev_pos[e,j,1]
                dx=(pos[e,u,0]-prev_pos[e,u,0])-(pos[e,j,0]-prev_pos[e,j,0])
                dy=(pos[e,u,1]-prev_pos[e,u,1])-(pos[e,j,1]-prev_pos[e,j,1])
                tau=wp.clamp(-(rx*dx+ry*dy)/wp.max(1e-12,dx*dx+dy*dy),0.0,1.0)
                sx=rx+tau*dx; sy=ry+tau*dy
                if sx*sx+sy*sy<(safe_d-0.0002)*(safe_d-0.0002): sticky[e,5]=1
    for u in range(7):
        for axis in range(3): prev_pos[e,u,axis]=pos[e,u,axis]
    for u in range(6): prev_bat[e,u]=battery[e,u]
