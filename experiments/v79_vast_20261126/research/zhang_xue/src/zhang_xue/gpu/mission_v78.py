"""V78 experimental *real independent* Warp UAV worlds coupled to packet MAC.

NOT scientifically interchangeable with ZhangXue7NodePacketEnv until exact
CPU-Warp observation, trajectory safety, reward and event parity gates pass.
Supports fixed-altitude XY, acceleration-bounded movement, NFZ cylinder
footprint safety, Bernoulli multi-cell sensing with 3D building occlusion,
UQ8.8-local DS evidence, packet MAC/PHY peer/GCS and genuine per-world resets.
No ground-truth target map is given to policy observations.

One world = one independent random target set, robot state, radio state,
packet events, belief ledger, reward and termination. Zero CPU transition
fixtures or repeated replay records are used by this runtime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
from zhang_xue.gpu.state_v67 import HAS_WARP, wp, NetworkBuffersV67, SimDimsV67
from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
from zhang_xue.communication.common import scale_canonical_geometry

# The V78 experimental GPU mission contract fixes six actors / 20x20 cells.
N_UAV=6
GRID=20
NUM_CELLS=400
N_TARGETS=15

if HAS_WARP:
    @wp.func
    def _ds_v78(p: float,n: float,logmf: float,logme: float)->wp.vec3:
        if p+n<=0.0:
            return wp.vec3(0.0,0.0,1.0)
        a=p*logmf
        b=n*logme
        shift=wp.max(a,b)
        pa=wp.exp(a-shift)
        pb=wp.exp(b-shift)
        pu=wp.exp(wp.min(a,b))
        mt=pb*(1.0-wp.exp(a))
        me=pa*(1.0-wp.exp(b))
        norm=wp.max(1.0e-20,mt+me+pu)
        return wp.vec3(mt/norm,me/norm,pu/norm)

    @wp.func
    def _segment_dist2_v78(x0:float,y0:float,x1:float,y1:float,cx:float,cy:float)->float:
        dx=x1-x0;dy=y1-y0
        den=wp.max(1.0e-12,dx*dx+dy*dy)
        t=wp.clamp(((cx-x0)*dx+(cy-y0)*dy)/den,0.0,1.0)
        ax=x0+t*dx-cx;ay=y0+t*dy-cy
        return ax*ax+ay*ay

    @wp.func
    def _safe_xy_v78(x:float,y:float,vx:float,vy:float,
                      b:wp.array2d(dtype=wp.float32),n_b:int)->int:
        nx=x+vx;ny=y+vy
        brkx=vx*vx/10.0
        brky=vy*vy/10.0
        if nx-wp.max(0.0,-vx)*wp.max(0.0,-vx)/10.0<10.0 or nx+wp.max(0.0,vx)*wp.max(0.0,vx)/10.0>990.0:
            return 0
        if ny-wp.max(0.0,-vy)*wp.max(0.0,-vy)/10.0<10.0 or ny+wp.max(0.0,vy)*wp.max(0.0,vy)/10.0>990.0:
            return 0
        for k in range(n_b):
            radius=b[k,2]+2.0
            if _segment_dist2_v78(x,y,nx,ny,b[k,0],b[k,1]) <=radius*radius:
                return 0
        return 1

    @wp.kernel
    def _reset_world_v78(
       pos:wp.array3d(dtype=wp.float32), vel:wp.array3d(dtype=wp.float32),
       energy:wp.array2d(dtype=wp.float32),truth:wp.array2d(dtype=wp.uint8),
       u_prev:wp.array2d(dtype=wp.float32),gcs_tp:wp.array2d(dtype=wp.uint8),
       gcs_fp:wp.array2d(dtype=wp.uint8), ep_step:wp.array(dtype=wp.int32),
       ep_id:wp.array(dtype=wp.int32), done:wp.array(dtype=wp.uint8),
       failed:wp.array(dtype=wp.uint8), reward:wp.array(dtype=wp.float32),
       pos_xy:wp.array3d(dtype=wp.float32),
       legal:wp.array(dtype=wp.int32),legal_size:int,n_targets:int,
       battery_max:float,seed:int,
    ):
       e=wp.tid()
       if done[e]==wp.uint8(0):return
       ep_step[e]=0;ep_id[e]=ep_id[e]+1;failed[e]=wp.uint8(0);reward[e]=0.0
       for c in range(NUM_CELLS):
           truth[e,c]=wp.uint8(0)
           u_prev[e,c]=1.0
           gcs_tp[e,c]=wp.uint8(0)
           gcs_fp[e,c]=wp.uint8(0)
       for u in range(N_UAV):
           pos[e,u,0]=90.0+65.0*float(u)
           pos[e,u,1]=110.0+30.0*float(u%2)
           pos[e,u,2]=50.0
           pos_xy[e,u,0]=pos[e,u,0]
           pos_xy[e,u,1]=pos[e,u,1]
           vel[e,u,0]=0.0;vel[e,u,1]=0.0
           energy[e,u]=battery_max
       pos[e,N_UAV,0]=50.0;pos[e,N_UAV,1]=50.0;pos[e,N_UAV,2]=15.0
       rng=wp.rand_init(seed+e*4093+ep_id[e]*11939)
       count=int(0)
       # Exact N distinct targets sampled from legal cells by rejection sampling.
       for _ in range(300):
           if count>=n_targets:break
           cid=legal[wp.randi(rng,0,legal_size)]
           if truth[e,cid]==wp.uint8(0):
               truth[e,cid]=wp.uint8(1)
               count+=1
       if count!=n_targets: failed[e]=wp.uint8(1)

    @wp.kernel
    def _candidate_v78(act:wp.array3d(dtype=wp.float32),
                       vel:wp.array3d(dtype=wp.float32),cand:wp.array3d(dtype=wp.float32)):
       e,u=wp.tid()
       x=wp.clamp(act[e,u,0],-1.0,1.0)*15.0
       y=wp.clamp(act[e,u,1],-1.0,1.0)*15.0
       m=wp.sqrt(x*x+y*y)
       if m>15.0:x=x*15.0/m;y=y*15.0/m
       dx=x-vel[e,u,0];dy=y-vel[e,u,1]
       a=wp.sqrt(dx*dx+dy*dy)
       if a>5.0:dx=dx*5.0/a;dy=dy*5.0/a
       vx=vel[e,u,0]+dx;vy=vel[e,u,1]+dy
       m2=wp.sqrt(vx*vx+vy*vy)
       if m2>15.0:vx=vx*15.0/m2;vy=vy*15.0/m2
       cand[e,u,0]=vx;cand[e,u,1]=vy

    @wp.kernel
    def _move_v78(
      pos:wp.array3d(dtype=wp.float32),vel:wp.array3d(dtype=wp.float32),
      cand:wp.array3d(dtype=wp.float32), pos_xy:wp.array3d(dtype=wp.float32),
      battery:wp.array2d(dtype=wp.float32),b:wp.array2d(dtype=wp.float32),n_b:int,
      net:NetworkBuffersV67, failed:wp.array(dtype=wp.uint8),
      intervention:wp.array(dtype=wp.float32),battery_max:float,
    ):
      e=wp.tid()
      interventions=float(0.0)
      for u in range(N_UAV):
          if battery[e,u]<=0.0:
              vel[e,u,0]=0.0;vel[e,u,1]=0.0
              net.radio_alive[e,u]=wp.uint8(0)
              continue
          px=pos[e,u,0];py=pos[e,u,1]
          vx=cand[e,u,0];vy=cand[e,u,1]
          safe=_safe_xy_v78(px,py,vx,vy,b,n_b)
          # Pairwise swept segment check; later peers use candidates, earlier use accepted velocities.
          for peer in range(N_UAV):
              if peer==u or battery[e,peer]<=0.0:continue
              ux=vel[e,peer,0] if peer<u else cand[e,peer,0]
              uy=vel[e,peer,1] if peer<u else cand[e,peer,1]
              rx=px-pos[e,peer,0];ry=py-pos[e,peer,1]
              dx=vx-ux;dy=vy-uy
              denom=wp.max(1.0e-9,dx*dx+dy*dy)
              alpha=wp.clamp(-(rx*dx+ry*dy)/denom,0.0,1.0)
              rx=rx+alpha*dx;ry=ry+alpha*dy
              if rx*rx+ry*ry < 2500.0:safe=0
          if safe==0:
              # Braking is acceleration-constrained, NOT instantaneous zero.
              ox=vel[e,u,0];oy=vel[e,u,1]
              vv=wp.sqrt(ox*ox+oy*oy)
              ratio=wp.max(0.0,vv-5.0)/wp.max(1e-10,vv)
              vx=ox*ratio;vy=oy*ratio
              interventions+=1.0
              if _safe_xy_v78(px,py,vx,vy,b,n_b)==0:failed[e]=wp.uint8(1)
          vel[e,u,0]=vx;vel[e,u,1]=vy
          pos[e,u,0]=px+vx;pos[e,u,1]=py+vy
          pos_xy[e,u,0]=pos[e,u,0];pos_xy[e,u,1]=pos[e,u,1]
          v=wp.sqrt(vx*vx+vy*vy)
          x=v*v/(2.0*4.03*4.03)
          p_blade=79.86*(1.0+3.0*v*v/14400.0)
          p_ind=88.63*wp.sqrt(1.0/(wp.sqrt(1.0+x*x)+x))
          p_par=0.5*0.6*1.225*0.05*0.503*v*v*v
          air_s=wp.float32(net.tx_airtime[e,u])/1.0e6
          battery[e,u]=wp.max(0.0,battery[e,u]-(p_blade+p_ind+p_par+15.0)-1.0*air_s)
          if battery[e,u]<=0.0:net.radio_alive[e,u]=wp.uint8(0)
      intervention[e]=interventions

    @wp.kernel
    def _sense_visible_v78(
       net:NetworkBuffersV67, pos:wp.array3d(dtype=wp.float32),
       truth:wp.array2d(dtype=wp.uint8), ep_step:wp.array(dtype=wp.int32),
       battery:wp.array2d(dtype=wp.float32),
       b:wp.array2d(dtype=wp.float32),n_b:int,
       seed:int,pd:float,pf:float,eta:float,
    ):
       e,u=wp.tid()
       if battery[e,u]<=0.0:return
       x=pos[e,u,0];y=pos[e,u,1]
       cx=int(wp.floor(x/50.0));cy=int(wp.floor(y/50.0))
       t=int(ep_step[e])+1
       rng=wp.rand_init(seed+e*4093+u*131+t*1013)
       for i in range(3):
          for j in range(3):
             ix=cx+i-1;iy=cy+j-1
             if ix<0 or iy<0 or ix>=GRID or iy>=GRID:continue
             cell_x=(float(ix)+0.5)*50.0
             cell_y=(float(iy)+0.5)*50.0
             ddx=cell_x-x;ddy=cell_y-y
             if ddx*ddx+ddy*ddy>2500.0:continue
             blocked=int(0)
             for k in range(n_b):
                if b[k,3]>=50.0 and _segment_dist2_v78(x,y,cell_x,cell_y,b[k,0],b[k,1])<=b[k,2]*b[k,2]:
                   blocked=1;break
             if blocked!=0:continue
             cid=iy*GRID+ix
             q=wp.float32(pd) if truth[e,cid]!=wp.uint8(0) else wp.float32(pf)
             detected=wp.randf(rng)<q
             old_t=int(net.ledger_gen[e,u,u,cid])
             p=float(net.ledger_pos[e,u,u,cid])/256.0
             n=float(net.ledger_neg[e,u,u,cid])/256.0
             if old_t>=0 and t>old_t:
                 decay=wp.exp(-eta*float(t-old_t))
                 p=p*decay;n=n*decay
             if detected:p=p+1.0
             else:n=n+1.0
             net.ledger_pos[e,u,u,cid]=wp.uint16(wp.clamp(int(wp.round(p*256.0)),0,65535))
             net.ledger_neg[e,u,u,cid]=wp.uint16(wp.clamp(int(wp.round(n*256.0)),0,65535))
             net.ledger_gen[e,u,u,cid]=wp.int16(t)

    @wp.func
    def _fuse_uq88_v78(net:NetworkBuffersV67,e:int,receiver:int,cid:int,t:int,
                       logmf:float,logme:float,eta:float)->wp.vec3:
        p=float(0.0);n=float(0.0)
        for org in range(N_UAV):
           if net.ledger_gen[e,receiver,org,cid]<0:continue
           age=wp.max(0.0,float(t-int(net.ledger_gen[e,receiver,org,cid])))
           weight=wp.exp(-eta*age)*(1.0/256.0)
           p+=float(net.ledger_pos[e,receiver,org,cid])*weight
           n+=float(net.ledger_neg[e,receiver,org,cid])*weight
        return _ds_v78(p,n,logmf,logme)

    @wp.kernel
    def _reward_v78(
       net:NetworkBuffersV67, truth:wp.array2d(dtype=wp.uint8),
       prev_u:wp.array2d(dtype=wp.float32),tp:wp.array2d(dtype=wp.uint8),
       fp:wp.array2d(dtype=wp.uint8), battery:wp.array2d(dtype=wp.float32),
       ep_step:wp.array(dtype=wp.int32), failed:wp.array(dtype=wp.uint8),
       intervention:wp.array(dtype=wp.float32),
       reward:wp.array(dtype=wp.float32),done:wp.array(dtype=wp.uint8),
       logmf:float,logme:float,eta:float,confirm:float,uncert_ceiling:float,
       r_tp:float,r_fp:float,w_aoi:float,w_loc:float,w_block:float,
       lambda_b:float,aoi_penalty:float,ref_age:float,horizon:int,
    ):
       e=wp.tid();t=int(ep_step[e])+1
       loc=float(0.0); delivered=float(0.0);fp_pen=float(0.0)
       aoi=float(0.0);seen=int(0);confirmed=int(0)
       for cid in range(NUM_CELLS):
           p=float(0.0);n=float(0.0)
           for u in range(N_UAV):
               if net.ledger_gen[e,u,u,cid]<0:continue
               age=wp.max(0.0,float(t-int(net.ledger_gen[e,u,u,cid])))
               discount=wp.exp(-eta*age)*(1.0/256.0)
               p+=float(net.ledger_pos[e,u,u,cid])*discount
               n+=float(net.ledger_neg[e,u,u,cid])*discount
           if p+n>0.0:
               ds=_ds_v78(p,n,logmf,logme)
               uncertainty=ds[2]
               if uncertainty<prev_u[e,cid]:
                   loc+=w_loc*(prev_u[e,cid]-uncertainty)
                   prev_u[e,cid]=uncertainty
           m=_fuse_uq88_v78(net,e,N_UAV,cid,t,logmf,logme,eta)
           if m[0]>=confirm and m[2]<=uncert_ceiling:
               if truth[e,cid]!=wp.uint8(0) and tp[e,cid]==wp.uint8(0):
                   tp[e,cid]=wp.uint8(1)
                   rx_t=float(net.gcs_last_rx_us[e,cid])/1e6
                   gen_t=float(net.gcs_latest[e,cid])
                   delay=wp.max(0.0,rx_t-gen_t)
                   delivered+=r_tp+w_aoi*wp.exp(-delay/ref_age)
               if truth[e,cid]==wp.uint8(0) and fp[e,cid]==wp.uint8(0):
                   fp[e,cid]=wp.uint8(1);fp_pen+=r_fp
           if truth[e,cid]!=wp.uint8(0) and tp[e,cid]!=wp.uint8(0):confirmed+=1
           last=int(net.gcs_latest[e,cid])
           aoi+=wp.max(0.0,float(t-last)) if last>=0 else float(t)
       energy_alive=int(0)
       air=float(0.0)
       for u in range(N_UAV):
           if battery[e,u]>0.0:energy_alive+=1
           air+=float(net.tx_airtime[e,u])/1e6
       reward[e]=loc+delivered-fp_pen-w_block*intervention[e]-lambda_b*air-aoi_penalty*aoi/(float(NUM_CELLS)*ref_age)
       ep_step[e]=t
       if confirmed>=N_TARGETS or energy_alive==0 or failed[e]!=wp.uint8(0) or t>=horizon:
           done[e]=wp.uint8(1)
       else:
           done[e]=wp.uint8(0)

    @wp.kernel
    def _observe_core_v78(net:NetworkBuffersV67,
       pos:wp.array3d(dtype=wp.float32),vel:wp.array3d(dtype=wp.float32),
       batt:wp.array2d(dtype=wp.float32),obs:wp.array3d(dtype=wp.float32),
       mask:wp.array3d(dtype=wp.uint8),battery_max:float,qcap:float):
       e,u=wp.tid()
       x=pos[e,u,0];y=pos[e,u,1]
       gx=pos[e,N_UAV,0];gy=pos[e,N_UAV,1]
       dx=(gx-x)/1000.0;dy=(gy-y)/1000.0
       soc=wp.clamp(batt[e,u]/battery_max,0.0,1.0)
       obs[e,u,0]=x/1000.0;obs[e,u,1]=y/1000.0
       obs[e,u,2]=vel[e,u,0]/15.0;obs[e,u,3]=vel[e,u,1]/15.0
       obs[e,u,4]=soc
       obs[e,u,5]=float(net.radio_alive[e,u])
       obs[e,u,6]=float(net.q_count[e,u])/qcap
       obs[e,u,7]=dx;obs[e,u,8]=dy;obs[e,u,9]=wp.sqrt(dx*dx+dy*dy)
       valid=net.route_hop[e,u,N_UAV]>=0 and net.route_expiry[e,u,N_UAV]>net.now[e]
       obs[e,u,10]=1.0 if valid else 0.0
       obs[e,u,11]=float(net.dirty_dest[e,u,N_UAV])
       # ACK watermark fraction in V77 consumes too many reads for a 4096-world
       # step: reserved as missing observable until an exact reduction is added.
       obs[e,u,12]=0.0
       count=int(0)
       for peer in range(N_UAV+1):
           if net.route_hop[e,u,peer]>=0 and net.route_expiry[e,u,peer]>net.now[e]:count+=1
       obs[e,u,13]=float(count)/7.0
       mask[e,u,0]=wp.uint8(1)
       mask[e,u,1]=wp.uint8(1) if soc>0.0 and net.dirty_dest[e,u,N_UAV]!=wp.uint8(0) else wp.uint8(0)
       allowed=wp.uint8(0)
       for peer in range(N_UAV):
           if peer==u:continue
           if net.route_hop[e,u,peer]>=0 and net.route_expiry[e,u,peer]>net.now[e] and net.dirty_dest[e,u,peer]!=wp.uint8(0):allowed=wp.uint8(1)
       mask[e,u,2]=allowed if soc>0.0 else wp.uint8(0)

    @wp.kernel
    def _observe_map_v78(net:NetworkBuffersV67,
       pos:wp.array3d(dtype=wp.float32),ep_step:wp.array(dtype=wp.int32),
       obs:wp.array3d(dtype=wp.float32),logmf:float,logme:float,eta:float,ref_age:float):
       e,u,bid=wp.tid()
       cy=bid//10;cx=bid%10
       t_max=float(0.0);f_min=float(1.0);u_avg=float(0.0);age_max=float(0.0)
       now=int(ep_step[e])
       for yy in range(2):
           for xx in range(2):
               cid=(2*cy+yy)*20+2*cx+xx
               ds=_fuse_uq88_v78(net,e,u,cid,now,logmf,logme,eta)
               t_max=wp.max(t_max,ds[0]);f_min=wp.min(f_min,ds[1]);u_avg+=0.25*ds[2]
               latest=int(-1)
               for org in range(N_UAV):
                   if int(net.ledger_pos[e,u,org,cid])+int(net.ledger_neg[e,u,org,cid])>0:
                       latest=wp.max(latest,int(net.ledger_gen[e,u,org,cid]))
               age=1.0 if latest<0 else wp.clamp(float(now-latest)/ref_age,0.0,1.0)
               age_max=wp.max(age_max,age)
       obs[e,u,14+bid]=t_max
       obs[e,u,114+bid]=f_min
       obs[e,u,214+bid]=u_avg
       obs[e,u,314+bid]=age_max
       px=int(wp.clamp(wp.floor(pos[e,u,0]/50.0),0.0,19.0))//2
       py=int(wp.clamp(wp.floor(pos[e,u,1]/50.0),0.0,19.0))//2
       obs[e,u,414+bid]=1.0 if (px==cx and py==cy) else 0.0


@dataclass
class GPUWorldV78Params:
    envs:int=4
    device:str='cpu'
    seed:int=44
    queue_capacity:int=64
    obstacle_model:str='multi_obstacle'
    per_building_loss_db:float=20.0
    noise_power_w:float=4.38957709293155e-13
    tx_power_w:float=0.1
    horizon:int=900
    max_events:int=4096


class WarpParallelMissionV78:
    """A genuine independent E-world GPU *experimental* environment.

    Outputs raw (E,6,514) local observations and (E,6,3) legal masks.
    Caller can use `wp.to_torch` and the policy adapter on same/different GPU.
    Timing is deterministic at the 1-second packet step level; V77 CPU parity
    is not asserted and the gate defaults to research-ineligible.
    """
    def __init__(self,params:GPUWorldV78Params):
        if not HAS_WARP:raise RuntimeError('Requires NVIDIA Warp 1.18')
        if not(1<=params.envs<=8192 and params.horizon<=32767 and params.horizon>=1):
            raise ValueError('invalid env count/horizon')
        if params.obstacle_model not in ('binary_nlos','multi_obstacle','rf_opaque'):
            raise ValueError('invalid obstacle model')
        self.params=params
        self.device=params.device
        self.envs=params.envs
        self.network=WarpCompressedRadioNetworkV67(params.envs,num_uavs=6,grid_w=20,
                      device=params.device,seed=params.seed,q_cap=params.queue_capacity,
                      enable_peer=True,enable_data_mule=True)
        self._raw={}
        specs={'positions':((params.envs,7,3),wp.float32),
          'velocities':((params.envs,6,2),wp.float32),
          'candidates':((params.envs,6,2),wp.float32),
          'pos_xy':((params.envs,6,2),wp.float32),
          'battery':((params.envs,6),wp.float32),
          'truth':((params.envs,400),wp.uint8),
          'team_u_prev':((params.envs,400),wp.float32),
          'gcs_tp':((params.envs,400),wp.uint8),
          'gcs_fp':((params.envs,400),wp.uint8),
          'episode_step':((params.envs,),wp.int32),
          'episode_id':((params.envs,),wp.int32),
          'done':((params.envs,),wp.uint8),
          'failed':((params.envs,),wp.uint8),
          'reward':((params.envs,),wp.float32),
          'intervention':((params.envs,),wp.float32),
          'obs':((params.envs,6,514),wp.float32),
          'mask':((params.envs,6,3),wp.uint8)}
        for name,(shape,dtype) in specs.items():self._raw[name]=wp.zeros(shape,dtype=dtype,device=params.device)
        self.buffers=self._raw
        buildings,blackouts,blackout_r=scale_canonical_geometry(1000.,20)
        self.buildings=buildings;self.blackouts=blackouts;self.blackout_r=blackout_r
        self.build_wp=wp.array(np.asarray(buildings,dtype=np.float32),dtype=wp.float32,device=params.device)
        self.blackout_wp=wp.array(np.asarray(blackouts,dtype=np.int32),dtype=wp.int32,device=params.device)
        legal=[]
        for cid in range(400):
            x=(cid%20+0.5)*50.;y=(cid//20+0.5)*50.
            if all((x-b[0])**2+(y-b[1])**2>b[2]**2 for b in buildings):legal.append(cid)
        self.legal=wp.array(np.asarray(legal,dtype=np.int32),dtype=wp.int32,device=params.device)
        self.obstacle_model_id=('binary_nlos','multi_obstacle','rf_opaque').index(params.obstacle_model)
        self.logmf=math.log1p(-.8);self.logme=math.log1p(-.8)
        self.reset()

    def _launch(self,kernel,dim,inputs):
        wp.launch(kernel,dim=dim,inputs=inputs,device=self.device)

    def reset(self,done:Optional[np.ndarray]=None):
        a=np.ones(self.envs,dtype=np.uint8) if done is None else np.asarray(done,dtype=np.uint8)
        if a.shape!=(self.envs,):raise ValueError('done mask shape mismatch')
        w=self.buffers
        wp.copy(w['done'],wp.array(a,dtype=wp.uint8,device=self.device))
        self.network.reset(a)
        self._launch(_reset_world_v78,self.envs,[w[k] for k in ('positions','velocities','battery','truth','team_u_prev','gcs_tp','gcs_fp','episode_step','episode_id','done','failed','reward','pos_xy')]
             +[self.legal,int(len(self.legal)),15,180000.0,int(self.params.seed)])
        self._update_phy()
        self.network.step(duration_us=600000,noise_w=self.params.noise_power_w,tx_power_w=self.params.tx_power_w,
                          max_events=self.params.max_events,check_host_guards=False)
        self._observe()
        return w['obs'],w['mask']

    def _update_phy(self):
        from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67
        n=self.network
        self._launch(compute_channel_gain_v67,(self.envs,7),[
             self.buffers['positions'],n.data['radio_alive'],self.build_wp,self.blackout_wp,
             n.data['gain'],7,len(self.buildings),len(self.blackouts),self.blackout_r,
             20,50.0,self.obstacle_model_id,float(self.params.per_building_loss_db)])

    def _observe(self):
        from zhang_xue.gpu.kernels.inject_v67 import compute_dirty_dest_v67
        n=self.network;w=self.buffers
        self._launch(compute_dirty_dest_v67,(self.envs,6,7),[n.state,n.dims])
        self._launch(_observe_core_v78,(self.envs,6),[
            n.state,w['positions'],w['velocities'],w['battery'],w['obs'],w['mask'],180000.0,float(self.params.queue_capacity)])
        self._launch(_observe_map_v78,(self.envs,6,100),[
            n.state,w['positions'],w['episode_step'],w['obs'],self.logmf,self.logme,.02,60.0])

    def step(self,continuous,discrete):
        """Accept Warp arrays on the mission device; all stages remain on device."""
        from zhang_xue.gpu.kernels.inject_v67 import compute_dirty_dest_v67,resolve_compact3_v67,inject_v67
        w=self.buffers;n=self.network
        if tuple(continuous.shape)!=(self.envs,6,2) or tuple(discrete.shape)!=(self.envs,6):
            raise ValueError('act shape should be (E,6,2) and (E,6)')
        self._launch(compute_dirty_dest_v67,(self.envs,6,7),[n.state,n.dims])
        self._launch(resolve_compact3_v67,(self.envs,6),[n.state,n.dims,discrete,w['pos_xy'],n._dest_buf])
        self._launch(inject_v67,(self.envs,6),[n.state,n.dims,n._dest_buf])
        n.step(duration_us=1_000_000,max_events=self.params.max_events,
               noise_w=self.params.noise_power_w,tx_power_w=self.params.tx_power_w,check_host_guards=False)
        self._launch(_candidate_v78,(self.envs,6),[continuous,w['velocities'],w['candidates']])
        self._launch(_move_v78,self.envs,[w['positions'],w['velocities'],w['candidates'],w['pos_xy'],
                 w['battery'],self.build_wp,len(self.buildings),n.state,w['failed'],w['intervention'],180000.0])
        self._update_phy()
        self._launch(_sense_visible_v78,(self.envs,6),[n.state,w['positions'],w['truth'],w['episode_step'],
                 w['battery'],self.build_wp,len(self.buildings),self.params.seed,.85,.03,.02])
        self._launch(_reward_v78,self.envs,[n.state,w['truth'],w['team_u_prev'],w['gcs_tp'],w['gcs_fp'],
             w['battery'],w['episode_step'],w['failed'],w['intervention'],w['reward'],w['done'],
             self.logmf,self.logme,.02,.9,.2,25.,12.5,6.,.5,.2,0.,0.,60.,self.params.horizon])
        self._observe()
        return w['obs'],w['mask'],w['reward'],w['done']

    def guard(self):
        """Synchronizing audit. Safe to call between rollout batches, never silently accept overflow."""
        errors=self.network.data['overflow'].numpy()
        failed=self.buffers['failed'].numpy()
        if np.any(errors):raise RuntimeError(f'Warp packet event/gen overflow in {int(np.count_nonzero(errors))} worlds')
        if np.any(failed):raise RuntimeError(f'unsafe flight or sampling error in {int(np.count_nonzero(failed))} worlds')
        if not np.isfinite(self.buffers['obs'].numpy()).all():raise FloatingPointError('nonfinite observations')
        return {'environments':self.envs,'episode_steps':int(self.buffers['episode_step'].numpy().max()),
                'active_truth_per_world_min':int(self.buffers['truth'].numpy().sum(axis=1).min()),
                'active_truth_per_world_max':int(self.buffers['truth'].numpy().sum(axis=1).max()),
                'distinct_target_layouts':int(np.unique(self.buffers['truth'].numpy(),axis=0).shape[0])}
