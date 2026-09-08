import Bipedal_Env
import numpy as np
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from Helper import AsyncPolicyViewer, SaveVecNormalizeCallback, clear_file, test_env
from pathlib import Path


train_on_existing_model = True
view_training = True
trained_model = ""
env_kwargs = {
    "ep_len": 6,
    "terminal_height": 0.7,
    "variation": [0.1, 0.2]
}
ppo_parameters = {

    "learning_rate": 1e-4,
    "n_steps": 4096,
    "batch_size": 64,
    "n_epochs": 5,

    "gamma": 0.99,
    "gae_lambda": 0.98,

    "clip_range": 0.2,

    "ent_coef": 0.001,
    "vf_coef": 0.15,
    "policy_kwargs": {
        "log_std_init": np.log(0.5)
    }
}
async_viewer_settings = {
    "distance": 3.0,
    "azimuth": 90,
    "elevation": -15,
    "offset": np.array([0, 0, -0.4])
}


trained_model_path = "Bipedal/Training/Saved_Models/ppo" + trained_model
vec_normalize_path = "Bipedal/Training/Saved_Models/vec_norm" + trained_model + ".pkl"
    

# Delete everything previously generated
training_dir = Path(__file__).parent / "Training/Temp_Checkpoint_Models"
clear_file(training_dir)

test_env(env_kwargs)

vec_env = make_vec_env("Bipedal-v0",
                       n_envs=1, 
                       env_kwargs=env_kwargs
                        )

if train_on_existing_model:
    #load the normalization statistics
    env = VecNormalize.load(vec_normalize_path,
                            vec_env
                            )

    #load trained PPO
    model = PPO.load(trained_model_path,
                     env=env,
                     **ppo_parameters,
                     verbose=1,
                     tensorboard_log="./Bipedal/Training/tensorboard/"
                     )

else:
    env = VecNormalize(vec_env,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0
                    )

    model = PPO("MlpPolicy", 
                env, 
                **ppo_parameters,
                verbose=1,
                tensorboard_log="./Bipedal/Training/tensorboard/"
                )


checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=str(training_dir), name_prefix="ppo")
normalize_callback = SaveVecNormalizeCallback(save_freq=100_000, save_path=training_dir)
callbacks = [checkpoint_callback, normalize_callback]
if view_training:
    viewer_callback = AsyncPolicyViewer(env_kwargs=env_kwargs, update_freq=5_000, init_camera_settings=async_viewer_settings)
    callbacks.append(viewer_callback)

callback = CallbackList(callbacks)

try:
    model.learn(total_timesteps=2_000_000,
                callback=callback,
                tb_log_name="PPO"
                )

finally:
    print("Saving model...")
    model.save("Bipedal/Training/Saved_Models/ppo")
    env.save("Bipedal/Training/Saved_Models/vec_norm.pkl")
    print("Model and normalization statistics saved.")

