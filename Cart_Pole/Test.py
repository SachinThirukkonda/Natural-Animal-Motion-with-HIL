import gymnasium as gym
import mujoco
import mujoco.viewer

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import Cart_Pole_Env


#create the environment
vec_env = make_vec_env("CartPoleCustom-v0",
                       n_envs=1,
                       env_kwargs={"ep_len": 20,
                                   "max_init_angle": 15,
                                    "max_init_velocity": 0.2,
                                   "termination_angle": 20,
                                   }
                        )

#load the normalization statistics
env = VecNormalize.load("vec_normalize.pkl",
                        vec_env
                        )

#set to evaluating, not training
env.training = False
env.norm_reward = False

#load trained PPO
model = PPO.load("ppo_cartpole",
                 env=env
                 )


#get the underlying MuJoCo environment
base_env = env.venv.envs[0].unwrapped

m = base_env.model
d = base_env.data


obs = env.reset()

with mujoco.viewer.launch_passive(m, d) as viewer:

    while viewer.is_running():

        # Ask the trained policy what action to take
        action, _ = model.predict(obs,
                                  deterministic=True
                                  )

        # Apply action and advance environment
        obs, reward, done, info = env.step(action)

        viewer.sync()

        if done[0]:
            obs = env.reset()