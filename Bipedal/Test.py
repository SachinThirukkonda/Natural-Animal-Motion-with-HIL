import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import time

from gymnasium.utils.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from Helper import test_env, key_callback, Camera
import Bipedal_Env

def key_callback(keycode):
    global push

    if chr(keycode) == 'P':
        push = 1

trained_model = "_uprightv3"
deterministic = True
env_kwargs = {
    "ep_len": 100,
    "terminal_height": 0.2,
    "variation": [0., 0.],
    "impulse_magnitude": 1,
}
init_camera_settings = {
    "distance": 3.0,
    "azimuth": 90,
    "elevation": -15,
    "offset": np.array([0, 0, -0.4]),
    "lookat": True
}


trained_model_path = "Bipedal/Training/Saved_Models/ppo" + trained_model
vec_normalize_path = "Bipedal/Training/Saved_Models/vec_norm" + trained_model + ".pkl"

test_env(env_kwargs)

#create the environment
vec_env = make_vec_env("Bipedal-v0",
                       n_envs=1,
                       env_kwargs=env_kwargs
                        )

env = VecNormalize(vec_env,
                   norm_obs=True,
                   norm_reward=True,
                   clip_obs=10.0
                   )

#load the normalization statistics
env = VecNormalize.load(vec_normalize_path,
                        vec_env
                        )

#set to evaluating, not training
env.training = False
env.norm_reward = False


#load trained PPO
model = PPO.load(trained_model_path,
                 env=env
                 )


#get the underlying MuJoCo environment
base_env = env.venv.envs[0].unwrapped

m = base_env.model
d = base_env.data

obs = env.reset()

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    cam = Camera(viewer, m, d, **init_camera_settings)
    push = 0
    while viewer.is_running():
        step_start = time.time()
        # Ask the trained policy what action to take
        action, _ = model.predict(obs, deterministic=deterministic)

        if push:
            base_env.Cassie.apply_impulse(base_env.impulse_magnitude)
            push = 0

        # Apply action and advance environment
        obs, reward, done, info = env.step(action)

        cam.update()

        remaining = m.opt.timestep * 40 - (time.time() - step_start)

        if remaining > 0:
            time.sleep(remaining)

        if done[0]:
            obs = env.reset()