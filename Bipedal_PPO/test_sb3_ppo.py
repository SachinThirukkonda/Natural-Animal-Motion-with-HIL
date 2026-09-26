import gymnasium as gym
from stable_baselines3 import PPO
from Helper import clear_file, file_name, PPOCallback
from pathlib import Path


train_on_existing_model = False
training_len = 1_000_000
trained_model = ""
save_as = "CartPole-v1_SB3_"
env = gym.make('CartPole-v1')

ppo_parameters = {

    "learning_rate": 0.005,
    "n_steps": 4800,
    "batch_size": 64,
    "n_epochs": 5,

    "gamma": 0.95,
    "gae_lambda": 0.98,

    "clip_range": 0.2,

    "ent_coef": 0.001,
    "vf_coef": 0.15,
}

xml_path = str(Path(__file__).parent)

# Delete everything previously generated
training_dir = Path(__file__).parent / "Training/Temp_Checkpoint_Models"
clear_file(training_dir)



if train_on_existing_model:
    #load trained PPO
    trained_model_path = xml_path + "/Training/Saved_Models/" + trained_model + "actor.pth"
    model = PPO.load(trained_model_path,
                     env=env,
                     **ppo_parameters,
                     verbose=1,
                     tensorboard_log=xml_path + "/Training/tensorboard/"
                     )

else:
    model = PPO("MlpPolicy", 
                env, 
                **ppo_parameters,
                verbose=1,
                tensorboard_log=xml_path + "/Training/tensorboard/"
                )

try:
    model.learn(total_timesteps=training_len,
                tb_log_name="sb3_ppo_tensorboard"
                )
    

finally:
    print("Saving model...")
    file = file_name(f"{xml_path}/Training/Saved_Models/{save_as}model")
    Path(file).mkdir()
    model.save(file)
    #env.save("Bipedal/Training/Saved_Models/vec_norm" + save_as + ".pkl")
    print(f"Model and normalization statistics saved as {save_as}model")