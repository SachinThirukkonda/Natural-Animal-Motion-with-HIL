import gymnasium as gym
from pathlib import Path
from ppo import PPO
from Helper import clear_file, file_name, PPOCallback



train_on_existing_model = False
training_len = 1_000_000
view_training = True
trained_model = "_"
save_as = "custom_ppo_"
env = gym.make('Pendulum-v1')

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

model = PPO(env, 
                hyperparameters=ppo_parameters,
                verbose=1
                )

if train_on_existing_model:
    try:
        trained_actor_path = xml_path + "/Training/Saved_Models/" + trained_model + "/actor.pth"
        trained_critic_path = xml_path + "/Training/Saved_Models/" + trained_model + "/critic.pth"
        #load trained PPO
        model.load(trained_actor_path, trained_critic_path)
    except:
        print(f"Error: {trained_model} not found")
        raise




callback = PPOCallback(
    parent_dir=xml_path,
    save_freq=10,
    save_as=save_as
    )


try:
    model.learn(total_timesteps=training_len,
                callback=callback
                )

finally:
    print("Saving model...")
    file = file_name(f"{xml_path}/Training/Saved_Models/{save_as}model")
    Path(file).mkdir()
    model.save(file)
    #env.save("Bipedal/Training/Saved_Models/vec_norm" + save_as + ".pkl")
    print(f"Model and normalization statistics saved as {save_as}model")