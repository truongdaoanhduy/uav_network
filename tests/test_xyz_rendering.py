"""Visualization stays 2D while FOV/labels/belief use the captured XYZ frame."""
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from test_xyz_sensing import xyz_config,world
from uav_network.configuration import load_config
from uav_network.environment.env import Action
from uav_network.experiments.rendering import _draw,animation_frame,render_episode,render_animation


@pytest.mark.parametrize("name",[
    "comparison-20k-hybrid75-1kb",
    "comparison-20k-hybrid75-1kb-shared-actor-critic",
    "comparison-20k-fullgpu-1kb-shared-actor-critic"])
def test_xyz_profiles_preserve_controlled_budget(name):
    c=load_config("configs/"+name+"-xyz-sensing.yaml")
    old=load_config("configs/"+name+".yaml")
    assert c.motion_dim==3 and c.continuous_dim==4 and old.motion_dim==2
    assert (c.world.num_uavs,c.world.map_size_m,c.world.num_targets,c.world.num_obstacles,c.world.max_steps)==(6,1000,20,15,1200)
    assert (c.runtime.num_envs,c.masac.batch_size,c.masac.replay_capacity)==(4096,8192,1000000)
    assert c.runtime.amp_mode=="off" and c.network.power_control=="learned"
    assert c.masac.target_continuous_entropy==-4
    assert c.runtime.replay_cpu_fraction==old.runtime.replay_cpu_fraction


def test_draw_uses_captured_altitude_and_stays_2d():
    env=world(xyz_config(),z=25)
    frame=animation_frame(env)
    assert frame["belief"].shape==(1,40,40)
    before=env.belief.copy()
    env.positions[:,2]=150
    fig,ax=plt.subplots()
    try:
        _draw(ax,env,[frame["positions"]],frame["positions"],frame["battery"],frame["found"],frame["delivered"],"frame")
        circles=[p for p in ax.patches if isinstance(p,Circle)]
        assert circles[-1].radius==pytest.approx(25)
        assert ax.name=="rectilinear"
        assert any("z=25" in text.get_text() for text in ax.texts)
    finally:plt.close(fig)
    np.testing.assert_array_equal(env.belief,before)


def test_xyz_real_png_and_gif_render(tmp_path):
    env=world(xyz_config(),z=25,target=(950,950))
    frames=[animation_frame(env)];paths=[env.positions.copy()]
    for _ in range(2):
        env.step(Action(np.array([[0,0,1,0]],np.float32),np.array([0])))
        paths.append(env.positions.copy());frames.append(animation_frame(env))
    png=render_episode(env,paths,tmp_path/"preview.png")
    gif=render_animation(env,paths,frames,tmp_path/"preview.gif",video_format="gif")
    assert Path(png).stat().st_size>1000 and Path(gif).stat().st_size>1000
