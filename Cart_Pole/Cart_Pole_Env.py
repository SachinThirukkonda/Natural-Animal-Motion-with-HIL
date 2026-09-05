import mujoco
import numpy as np
import gymnasium as gym
from typing import Optional
from pathlib import Path

class cart_pole(gym.Env):
    def __init__(self, ep_len, max_init_angle, max_init_velocity,termination_angle):
        xml_path = Path(__file__).parent / "Model.xml" 
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.observation_space = gym.spaces.Box(low=np.array([-np.pi / 2,
                                                              -np.inf,
                                                              -0.15,
                                                              -np.inf
                                                              ], dtype=np.float32),
                                                high=np.array([np.pi / 2,
                                                               np.inf,
                                                               0.15,
                                                               np.inf
                                                               ], dtype=np.float32),
                                                dtype=np.float32
                                                )

        self.action_space = gym.spaces.Box(low=np.array([-1], dtype=np.float32),
                                           high=np.array([1], dtype=np.float32),
                                           dtype=np.float32
                                           )
        
        dt = self.model.opt.timestep
        self.max_steps = int(ep_len / dt)
        self.step_count = 0
        self.max_init_angle = max_init_angle
        self.max_init_velocity = max_init_velocity
        self.termination_angle = termination_angle
        #self.push = 0


    def _get_obs(self):
        angle = self.data.sensor("pole_angle").data[0]
        omega = self.data.sensor("pole_angular_vel").data[0]
        cart_pos = self.data.sensor("cart_pos").data[0]
        cart_vel = self.data.sensor("cart_vel").data[0]
        pos = cart_pos + np.sin(angle)*0.2
        vel = cart_vel + np.cos(angle)*omega*0.2
        return np.array([
            angle,
            omega,
            pos,
            vel
        ], dtype=np.float32)
    
    def _get_info(self):
        angle = self.data.sensor("pole_angle").data[0]
        omega = self.data.sensor("pole_angular_vel").data[0]
        cart_pos = self.data.sensor("cart_pos").data[0]
        cart_vel = self.data.sensor("cart_vel").data[0]
        pos = cart_pos + np.sin(angle)*0.2
        vel = cart_vel + np.cos(angle)*omega*0.2
        return {
            "angle": angle,
            "omega": omega,
            "pos": pos,
            "vel": vel,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        self.data.qpos[0] = self.np_random.uniform(-0.05, 0.05)
        pos_diff = 0.2*np.sin(np.deg2rad(self.max_init_angle))
        self.data.qpos[1] = self.np_random.uniform(self.data.qpos[0] - pos_diff, self.data.qpos[0] + pos_diff)

        self.data.qvel[:] = 0
        self.data.qvel[1] = self.np_random.uniform(-self.max_init_velocity, self.max_init_velocity)

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def termination_check(self, observation):
        if abs(observation[0]) > np.deg2rad(self.termination_angle):
            return True
        return False

    def reward_function(self, observation, terminated):
        angle, omega, pos, vel = observation.tolist()

        if terminated:
            return -1000.0

        #reward being upright and stationary at the origin.
        reward = 1.0 - 10*abs(pos)
        return float(reward)


    def step(self, action):
        self.data.ctrl[0] = action[0] * 0.2  #scale the action to the control range of the motor

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        '''    
        if self.data.time >= 5 and self.push == 0:
            self.data.qvel[1] += self.np_random.uniform(-self.max_init_velocity, self.max_init_velocity)
            self.push = 1
        '''    
        self.step_count += 1
        observation = self._get_obs()
        terminated = self.termination_check(observation)
        truncated = self.step_count >= self.max_steps
        reward = self.reward_function(observation, terminated)
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="CartPoleCustom-v0",
    entry_point="Cart_Pole_Env:cart_pole",   
)
