import gymnasium as gym
import torch
from networks.network import FeedForwardNN
from pathlib import Path


env = gym.make("Pendulum-v1", render_mode="human")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

actor = FeedForwardNN(obs_dim, act_dim, 64)

trained_model = "custom_ppo_model"
trained_model_path = str(Path(__file__).parent / "Training" / "Saved_models" / trained_model / "actor.pth")
actor.load_state_dict(
    torch.load(trained_model_path)
)

actor.eval()


obs, _ = env.reset()

for _ in range(1000):

    obs_tensor = torch.from_numpy(obs).float()

    with torch.no_grad():
        action = actor(obs_tensor)

    obs, reward, terminated, truncated, _ = env.step(
        action.numpy()
    )

    if terminated or truncated:
        print("Episode finished")
        obs, _ = env.reset()

env.close()