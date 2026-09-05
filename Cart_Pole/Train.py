import mujoco
import numpy as np
import gymnasium as gym
from typing import Optional
import Cart_Pole_Env
from gymnasium.utils.env_checker import check_env
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

dummy_env = gym.make("CartPoleCustom-v0",
                     ep_len = 5,
                     max_init_angle = 15,
                     max_init_velocity = 0.2,
                     termination_angle = 20
                     )

try:
    check_env(dummy_env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")


vec_env = make_vec_env("CartPoleCustom-v0",
                       n_envs=1, 
                       env_kwargs={"ep_len": 200,
                                   "max_init_angle": 15,
                                   "max_init_velocity": 0.1,
                                   "termination_angle": 20,
                                   }
                        )

env = VecNormalize(vec_env,
                   norm_obs=True,
                   norm_reward=True,
                   clip_obs=10.0
                   )

model = PPO("MlpPolicy", 
            env, 
            learning_rate=1e-4,

            n_steps=1024,
            batch_size=64,
            n_epochs=10,

            gamma=0.99,
            gae_lambda=0.98,

            clip_range=0.2,

            ent_coef=0.0,
            vf_coef=0.5,

            verbose=1,
            tensorboard_log="./tensorboard/")

model.learn(total_timesteps=100000)
model.save("ppo_cartpole")
env.save("vec_normalize.pkl")

