import mujoco
import mujoco.viewer
import time
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import stat
import threading
import numpy as np
import copy
from pathlib import Path
import torch

class Camera():
    def __init__(self, viewer, model, data, distance, azimuth, elevation, offset, lookat):
        self.viewer = viewer
        # Initial camera settings
        viewer.cam.distance = distance
        viewer.cam.azimuth = azimuth
        viewer.cam.elevation = elevation
        self.camera_offset = offset
        self.model = model
        self.data = data
        self.lookat = lookat

    def update(self):
        if self.lookat:
            # Move the camera's orbit centre with Cassie
            self.viewer.cam.lookat[:] = self.data.xpos[self.model.body("cassie-pelvis").id] + self.camera_offset
        else:
            self.viewer.cam.lookat[:] = [0, 0, 0.5]
        self.viewer.sync()

class AsyncPolicyViewer(BaseCallback):

    def __init__(self, env_kwargs, init_camera_settings, update_freq=10_000):
        super().__init__()

        self.env_kwargs = env_kwargs
        self.update_freq = update_freq

        self.running = True

        # Latest frozen policy
        self.latest_policy = None

        # Latest VecNormalize statistics
        self.obs_mean = None
        self.obs_var = None
        self.clip_obs = 10.0

        # Protects policy/statistics while being replaced
        self.lock = threading.Lock()

        self.viewer_thread = None

        self.init_camera_settings = init_camera_settings

    def _on_training_start(self):

        print("Starting asynchronous policy viewer...")

        # Take an initial snapshot immediately
        self._update_policy_snapshot()

        # Start completely separate visualisation thread
        self.viewer_thread = threading.Thread(
            target=self._viewer_loop,
            daemon=True
        )

        self.viewer_thread.start()

    def _on_step(self):

        # Every update_freq training steps, copy the current
        # policy and normalisation statistics.
        if self.num_timesteps % self.update_freq == 0:

            self._update_policy_snapshot()

        return True

    # --------------------------------------------------------
    # COPY CURRENT PPO POLICY
    # --------------------------------------------------------

    def _update_policy_snapshot(self):

        print(
            f"\n[Viewer] Creating policy snapshot at "
            f"{self.num_timesteps:,} steps"
        )


        policy_copy = copy.deepcopy(self.model.policy)

        # Viewer only needs inference
        policy_copy.set_training_mode(False)

        # Put the copy on CPU so it doesn't interfere with
        # whatever device the training policy is using.
        policy_copy = policy_copy.to("cpu")

        # ----------------------------------------------------
        # Copy VecNormalize statistics
        # ----------------------------------------------------

        obs_rms = self.training_env.obs_rms

        obs_mean = obs_rms.mean.copy()
        obs_var = obs_rms.var.copy()

        clip_obs = self.training_env.clip_obs

        # ----------------------------------------------------
        # Atomically replace viewer snapshot
        # ----------------------------------------------------

        with self.lock:

            self.latest_policy = policy_copy
            self.obs_mean = obs_mean
            self.obs_var = obs_var
            self.clip_obs = clip_obs

        print(
            f"[Viewer] Snapshot ready at "
            f"{self.num_timesteps:,} steps"
        )

    # --------------------------------------------------------
    # NORMALISE OBSERVATION
    # --------------------------------------------------------

    def _normalize_obs(self, obs):

        normalized = (
            obs - self.obs_mean
        ) / np.sqrt(self.obs_var + 1e-8)

        normalized = np.clip(
            normalized,
            -self.clip_obs,
            self.clip_obs
        )

        return normalized.astype(np.float32)

    # --------------------------------------------------------
    # VIEWER THREAD
    # --------------------------------------------------------

    def _viewer_loop(self):

        # --------------------------------------------------------
        # Create completely independent visualisation environment
        # --------------------------------------------------------

        env = gym.make("Bipedal-v0", **self.env_kwargs)

        mujoco_env = env.unwrapped

        print("[Viewer] Visualisation environment created.")

        # --------------------------------------------------------
        # Reset once
        # --------------------------------------------------------

        obs, info = env.reset()

        # --------------------------------------------------------
        # Create viewer ONCE
        # --------------------------------------------------------

        with mujoco.viewer.launch_passive(mujoco_env.model, mujoco_env.data) as viewer:
            cam = Camera(viewer, mujoco_env.model, mujoco_env.data, **self.init_camera_settings)
            while self.running and viewer.is_running():

                # ------------------------------------------------
                # Get latest policy snapshot
                # ------------------------------------------------

                with self.lock:

                    if self.latest_policy is None:
                        time.sleep(0.01)
                        continue

                    policy = self.latest_policy
                    obs_mean = self.obs_mean.copy()
                    obs_var = self.obs_var.copy()
                    clip_obs = self.clip_obs

                # ------------------------------------------------
                # Normalise observation
                # ------------------------------------------------

                normalized_obs = (
                    obs - obs_mean
                ) / np.sqrt(obs_var + 1e-8)

                normalized_obs = np.clip(
                    normalized_obs,
                    -clip_obs,
                    clip_obs
                )

                normalized_obs = normalized_obs.astype(
                    np.float32
                )

                # ------------------------------------------------
                # Policy inference
                # ------------------------------------------------

                action, _ = policy.predict(
                    normalized_obs,
                    deterministic=True
                )

                # ------------------------------------------------
                # Advance visualisation environment
                # ------------------------------------------------

                obs, reward, terminated, truncated, info = env.step(
                    action
                )

                # ------------------------------------------------
                # Update viewer
                # ------------------------------------------------

                cam.update()

                # ------------------------------------------------
                # Reset episode WITHOUT closing viewer
                # ------------------------------------------------

                if terminated or truncated:

                    obs, info = env.reset()

                # ------------------------------------------------
                # Viewer refresh rate
                # ------------------------------------------------

                time.sleep(1 / 60)

        # --------------------------------------------------------
        # Viewer was closed by user
        # --------------------------------------------------------

        env.close()

        print("[Viewer] Thread stopped.")

    # --------------------------------------------------------
    # TRAINING END
    # --------------------------------------------------------

    def _on_training_end(self):

        print("[Viewer] Stopping viewer...")

        self.running = False

        if self.viewer_thread is not None:

            self.viewer_thread.join(
                timeout=2
            )

        print("[Viewer] Viewer stopped.")

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq, save_path):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = self.save_path / f"vec_norm_{self.num_timesteps}_steps.pkl"
            self.training_env.save(str(path))
            print(f"Saved VecNormalize: {path}")

        return True

class RewardComponentsCallback(BaseCallback):
    def __init__(self, log_every=100, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every

        self.rewards = []
        self.episode_count = 0

    def _on_step(self):
        for info in self.locals["infos"]:

            if "reward_components" in info:

                self.episode_count += 1
                # Store this episode's component means
                self.rewards.append(info["reward_components"])

                # Log mean across the last N episodes
                if self.episode_count % self.log_every == 0:

                    mean_components = {
                        name: sum(
                            episode[name] for episode in self.rewards
                        ) / len(self.rewards)
                        for name in self.rewards[0]
                    }

                    for name, value in mean_components.items():
                        self.logger.record(
                            f"reward_components/{name}",
                            value
                        )

                    self.logger.dump(self.num_timesteps)

                    # Start collecting the next N episodes
                    self.rewards = []

        return True

class PPOCallback():
    def __init__(self, parent_dir, save_as="", save_freq=10, render=False, render_freq=10):
        self.log_dir = file_name(f"{parent_dir}/Training/tensorboard/{save_as}tensorboard")
        self.checkoint_save_dir = parent_dir + "/Training/Temp_Checkpoint_Models"
        self.final_save_dir = parent_dir + "/Training/Saved_Models"

        self.writer = SummaryWriter(self.log_dir)

        os.makedirs(self.checkoint_save_dir, exist_ok=True)

        self.save_freq = save_freq

        self.render = render
        self.render_freq = render_freq

        self.iteration = 0
        self.timesteps = 0

    def on_training_start(self, model):
        self.model = model

    def on_rollout_end(self, batch_lens, batch_rews):
        """Log rollout statistics."""

        episode_returns = [
            np.sum(rews) for rews in batch_rews
        ]

        mean_return = np.mean(episode_returns)
        mean_length = np.mean(batch_lens)

        self.writer.add_scalar(
            "rollout/ep_rew_mean",
            mean_return,
            self.timesteps
        )

        self.writer.add_scalar(
            "rollout/ep_len_mean",
            mean_length,
            self.timesteps
        )

    def on_iteration_end(self, actor_loss, critic_loss):

        self.writer.add_scalar(
            "train/actor_loss",
            actor_loss,
            self.timesteps
        )

        self.writer.add_scalar(
            "train/critic_loss",
            critic_loss,
            self.timesteps
        )

        self.iteration += 1

        if self.iteration % self.save_freq == 0:
            self.save()
            

        self.writer.flush()

    def save(self):

        torch.save(
            self.model.actor.state_dict(),
            f"{self.checkoint_save_dir}/actor_{self.timesteps}.pth"
        )

        torch.save(
            self.model.critic.state_dict(),
            f"{self.checkoint_save_dir}/critic_{self.timesteps}.pth"
        )

    def on_training_end(self):
        self.writer.close()

def key_callback(keycode):
    global push

    if chr(keycode) == 'P':
        push = 1

def test_env(env_kwargs):
    dummy_env = gym.make("Bipedal-v0", **env_kwargs)

    try:
        check_env(dummy_env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

def remove_readonly(func, path, exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clear_file(dir):
    dir.mkdir(exist_ok=True)


    for item in dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item, onexc=remove_readonly)
        else:
            item.unlink()

def file_name(str):
    counter = 1
    file = Path(str)
    if (file).exists():
        while (Path(f"{str}_{counter}")).exists():
            counter += 1
        return f"{str}_{counter}"
    else:
        return str