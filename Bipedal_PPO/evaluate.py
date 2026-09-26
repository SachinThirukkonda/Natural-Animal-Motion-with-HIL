import gymnasium as gym
import torch
from networks.network import FeedForwardNN
from pathlib import Path


env = gym.make("CartPole-v1", render_mode="human")

if isinstance(env.observation_space, gym.spaces.Box):
    obs_dim = env.observation_space.shape[0]
else:    
    raise NotImplementedError(f"Unsupported action space: {env.action_space}")

if isinstance(env.action_space, gym.spaces.Box):
    act_dim = env.action_space.shape[0]
    discrete = False

elif isinstance(env.action_space, gym.spaces.Discrete):
    act_dim = env.action_space.n
    discrete = True

else:
    raise NotImplementedError(f"Unsupported action space: {env.action_space}")

actor = FeedForwardNN(obs_dim, act_dim, 64)

trained_model = "CartPole-v1_Custom_model"
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

    if discrete:
        action = torch.argmax(action).item()
    else:
        action = action.numpy()
    
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        print("Episode finished")
        obs, _ = env.reset()

env.close()