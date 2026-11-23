"""Scripted camera demonstration, not a learned policy or convergence result."""
import argparse
import json
from pathlib import Path
import numpy as np

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.experiments.rendering import animation_frame, render_episode, render_animation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",default="outputs/xyz-camera-demo")
    args=parser.parse_args()
    out=Path(args.output_dir)
    if out.exists():
        raise FileExistsError("Choose a new demo output directory")
    out.mkdir(parents=True)
    c=Config()
    c.world.motion_dimensions=3;c.sensor.model="bayesian"
    c.world.num_uavs,c.world.num_targets,c.world.num_obstacles=2,2,2
    c.world.max_steps=60
    c.network.tx_power_min_w=.001;c.network.tx_power_mapping="log"
    c.network.power_limited_admission=True;c.network.link_state_observation=True
    c.network.propagation_delay_enabled=True
    env=SwarmSearchEnv(c)
    try:
        env.enable_communication_trace()
        env.reset(seed=44,options=dict(positions=[[500,350,100],[550,200,100]],
            targets=[[512.5,362.5],[762.5,762.5]],
            obstacles=[[650,400,30,80],[300,650,50,110]]))
        paths=[env.positions.copy()];frames=[animation_frame(env)]
        high=render_episode(env,paths,out/"high.png")
        for step in range(60):
            action=np.zeros((2,4),np.float32)
            # Descend, inspect, then ascend. UAV1 stays high for a FOV comparison.
            desired=25 if step<40 else 100
            action[0,2]=np.clip(.05*(desired-env.positions[0,2])-.6*env.velocities[0,2],-1,1)
            action[:,-1]=1
            _,_,done,truncated,_=env.step(Action(action,np.array([2,2],np.int64)))
            paths.append(env.positions.copy())
            if step%3==0 or done or truncated:frames.append(animation_frame(env))
            if step==35:render_episode(env,paths,out/"low.png")
            if done or truncated:break
        image=render_episode(env,paths,out/"final.png")
        video=render_animation(env,paths,frames,out/"camera.gif",video_format="gif",fps=6)
        result=dict(controller="Scripted altitude controller, not a trained policy",
                    purpose="Camera/FOV/Bayes/report pipeline demonstration",
                    initial_image=high,final_image=image,animation=video,
                    targets_found=env.info()["targets_found"],targets_delivered=env.info()["targets_delivered"],
                    sensing=env.info()["sensing"],steps=env.steps)
        (out/"demo.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result))
    finally:env.close()


if __name__=="__main__":
    main()
