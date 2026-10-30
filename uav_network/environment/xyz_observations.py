"""Local Bayesian evidence features; never query hidden targets for actor slots."""
import numpy as np
from .bayesian import belief_patch,sensor_profile


def xyz_features(env):
    w,s,n=env.config.world,env.config.sensor,env.config.network
    from .observations import destination_mask,report_status,link_state_features
    masks=destination_mask(env);status=report_status(env);rows=[]
    for i,p in enumerate(env.positions):
        own=np.r_[p[:2]/w.map_size_m,env.velocities[i,:2]/w.max_speed_mps,
                  (env.gcs[:2]-p[:2])/w.map_size_m,env.steps/w.max_steps,
                  env.reports.bytes_used(i)/n.buffer_bytes,env.battery_fraction[i],env.active[i]]
        targets=np.zeros((s.visible_target_slots,4))
        cells=np.argwhere(env.positive[i])
        cells=sorted(cells,key=lambda a:np.linalg.norm((a[::-1]+.5)*s.grid_cell_m-p[:2]))
        for slot,(gy,gx) in enumerate(cells[:s.visible_target_slots]):
            center=np.minimum((np.array([gx,gy])+.5)*s.grid_cell_m,w.map_size_m)
            targets[slot]=np.r_[(center-p[:2])/w.map_size_m,env.belief[i,gy,gx],1]
        obs=np.zeros((s.visible_obstacle_slots,4));heights=np.zeros(s.visible_obstacle_slots)
        radius=sensor_profile(s,p[2])[2]
        visible=sorted([j for j,o in enumerate(env.obstacles)
                        if env.active[i] and np.linalg.norm(o[:2]-p[:2])<=radius+o[2]],
                       key=lambda j:np.linalg.norm(env.obstacles[j,:2]-p[:2]))
        for slot,j in enumerate(visible[:s.visible_obstacle_slots]):
            o=env.obstacles[j];obs[slot]=np.r_[(o[:2]-p[:2])/w.map_size_m,o[2]/w.map_size_m,1]
            heights[slot]=env.obstacle_heights[j]/w.altitude_max_m
        peers=np.zeros((w.num_uavs-1,6));vertical=np.zeros((w.num_uavs-1,2))
        for slot,j in enumerate(k for k in range(w.num_uavs) if k!=i):
            if env.active[i] and np.linalg.norm(env.positions[j]-p)<=n.peer_range_m:
                peers[slot]=np.r_[(env.positions[j,:2]-p[:2])/w.map_size_m,
                                 env.velocities[j,:2]/w.max_speed_mps,env.battery_fraction[j],env.active[j]]
                vertical[slot]=[(env.positions[j,2]-p[2])/w.altitude_max_m,env.velocities[j,2]/w.max_speed_mps]
        link=link_state_features(env,i) if n.link_state_observation else []
        extra=np.r_[p[2]/w.altitude_max_m,env.velocities[i,2]/w.max_speed_mps,
                    heights,vertical.ravel(),belief_patch(env,i)]
        rows.append(np.r_[own,env.reports.known[i],status[i],targets.ravel(),obs.ravel(),
                          peers.ravel(),link,env.coverage[i],extra,masks[i]])
    return np.asarray(rows,dtype=np.float32)
