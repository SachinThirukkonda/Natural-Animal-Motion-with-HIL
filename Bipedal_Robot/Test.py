import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from gymnasium.utils.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import Bipedal_Env

dummy_env = gym.make("Bipedal-v0",
                     ep_len = 5
                     )

try:
    check_env(dummy_env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

#create the environment
vec_env = make_vec_env("Bipedal-v0",
                       n_envs=1,
                       env_kwargs={"ep_len": 20
                                   }
                        )

env = VecNormalize(vec_env,
                   norm_obs=True,
                   norm_reward=True,
                   clip_obs=10.0
                   )

'''
#load the normalization statistics
env = VecNormalize.load("vec_normalize.pkl",
                        vec_env
                        )

#set to evaluating, not training
env.training = False
env.norm_reward = False


#load trained PPO
model = PPO.load("ppo_bipedal",
                 env=env
                 )
'''

#get the underlying MuJoCo environment
base_env = env.venv.envs[0].unwrapped

m = base_env.model
d = base_env.data


obs = env.reset()

with mujoco.viewer.launch_passive(m, d) as viewer:

    while viewer.is_running():

        # Ask the trained policy what action to take
        action = np.zeros((1, 10), dtype=np.float32)

        # Apply action and advance environment
        obs, reward, done, info = env.step(action)

        viewer.sync()

        if done[0]:
            obs = env.reset()