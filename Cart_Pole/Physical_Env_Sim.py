import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import Cart_Pole_Env

#returns the desired position of the cart at time t. Currently returns a constant position of 0.
def trajectory(t):
    T= 100
    A = 0.1
    return 0
    #return A* np.sin(2 * np.pi / T * t)

#gets the current errors for balance and position from the sensors.
def observations(data):
    angle = data.sensor("pole_angle").data[0]
    omega = data.sensor("pole_angular_vel").data[0]
    pos = data.sensor("cart_pos").data[0] + np.sin(angle)*0.2
    vel = data.sensor("cart_vel").data[0] + np.cos(angle)*0.2*omega
    return np.array([angle, omega, pos, vel], dtype=np.float32)

#to enable input from the keyboard to push the pole or disable the controller
def key_callback(keycode):
    global push, disable

    if chr(keycode) == 'P':
        push = 1
    if chr(keycode) == 'L' and disable == 0:
        disable = 1
        print("disable")
    elif chr(keycode) == 'L' and disable == 1:
        disable = 0
        print("enable")

xml = """
<mujoco model="Cart_Pole">
  <option integrator="RK4"/>
  <option gravity="0 0 -0.1"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>
<worldbody>
    <geom name= "floor" size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="cart" pos="0 0 .02">
        <joint name="cart_slide"
            type="slide"
            axis="1 0 0"
            limited="true"
            range="-0.15 0.15"/>
        <geom name="cart"
            type="box"
            size=".05 .05 .02"
            mass=".1"/>
        <body name="pole">
            <inertial
                pos="0 0 0.2"
                mass="1"
                diaginertia="4e-6 4e-6 4e-6"/>
            <joint name="pole_hinge"
                type="hinge"
                axis="0 1 0"/>
            <geom name="stem"
                type="capsule"
                size=".01"
                fromto="0 0 0 0 0 0.2"
                mass="0"/>
        </body>
    </body>
</worldbody>
<actuator>
    <motor name="cart_motor"
        joint="cart_slide"
        gear="1"
        forcerange="-.2 .2"
        forcelimited="true"/>
</actuator>
<sensor>
    <jointpos name="cart_pos"
        joint="cart_slide"/>
    <jointpos name="pole_angle"
        joint="pole_hinge"/>
    <jointvel name="cart_vel"
        joint="cart_slide"/>
    <jointvel name="pole_angular_vel"
        joint="pole_hinge"/>
</sensor>    
</mujoco>
"""
#initialise the MuJoCo model and data.
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

#initialise variables.
push = 0
pole_dof = m.joint("pole_hinge").dofadr[0]

#load the normalizer for the observations.
dummy_vec_env = make_vec_env("CartPoleCustom-v0",
    n_envs=1,
    env_kwargs={"ep_len": 20,
                "max_init_angle": 15,
                "max_init_velocity": 0.2,
                "termination_angle": 20,
                }
    )
normalizer = VecNormalize.load(
    "vec_normalize.pkl",
    dummy_vec_env
)

normalizer.training = False
normalizer.norm_reward = False

#load the trained policy
policy = PPO.load("ppo_cartpole")

mujoco.mj_resetData(m, d)

#set initial state
d.qpos[0] = 0.0
d.qpos[1] = np.deg2rad(5)
d.qvel[:] = 0

mujoco.mj_forward(m, d)


with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    while viewer.is_running():
        #get the current errors for balance and position from the sensors.
        obs = observations(d)

        #VecNormalize expects a batch dimension.
        obs_batch = obs.reshape(1, -1)
        normalised_obs = normalizer.normalize_obs(obs_batch)

        #get the control signals from the trained policy.
        action, _ = policy.predict(normalised_obs[0], deterministic=True)

        #gives the cart-pole system a push if the 'P' key is pressed.
        if push:
            print("PUSH!")
            d.qvel[pole_dof] += 0.4
            push = 0

        #apply the control signal to the cart-pole system.
        d.ctrl[0] = action[0] * 0.2

        #step the simulation forward by one time step.
        mujoco.mj_step(m, d)

        viewer.sync()