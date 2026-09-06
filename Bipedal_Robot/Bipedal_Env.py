import mujoco
import numpy as np
import gymnasium as gym
from typing import Optional
from pathlib import Path

class bipedal(gym.Env):
    def __init__(self, ep_len):
        xml_path = Path(__file__).parent / "agility_cassie" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(42,),
                                                dtype=np.float32
                                                )

        self.action_space = gym.spaces.Box(low=np.array([-4.5,   # left-hip-roll
                                                         -4.5,   # left-hip-yaw
                                                         -12.2,  # left-hip-pitch
                                                         -12.2,  # left-knee
                                                         -0.9,   # left-foot
                                                         -4.5,   # right-hip-roll
                                                         -4.5,   # right-hip-yaw
                                                         -12.2,  # right-hip-pitch
                                                         -12.2,  # right-knee
                                                         -0.9    # right-foot
                                                         ], dtype=np.float32),
                                          high=np.array([4.5,    # left-hip-roll
                                                         4.5,    # left-hip-yaw
                                                         12.2,   # left-hip-pitch
                                                         12.2,   # left-knee
                                                         0.9,    # left-foot
                                                         4.5,    # right-hip-roll
                                                         4.5,    # right-hip-yaw
                                                         12.2,   # right-hip-pitch
                                                         12.2,   # right-knee
                                                         0.9     # right-foot
                                                         ], dtype=np.float32),
                                          dtype=np.float32
                                          )
                                
        
        dt = self.model.opt.timestep
        self.max_steps = int(ep_len / dt)
        self.step_count = 0

    def _get_obs(self):
        left_hip_roll_angle = self.data.sensor("left-hip-roll-input").data[0]
        left_hip_yaw_angle = self.data.sensor("left-hip-yaw-input").data[0]
        left_hip_pitch_angle = self.data.sensor("left-hip-pitch-input").data[0]
        left_knee_angle = self.data.sensor("left-knee-input").data[0]
        left_foot_angle = self.data.sensor("left-foot-input").data[0]

        left_hip_roll_vel = self.data.sensor("left-hip-roll-input-vel").data[0]
        left_hip_yaw_vel = self.data.sensor("left-hip-yaw-input-vel").data[0]
        left_hip_pitch_vel = self.data.sensor("left-hip-pitch-input-vel").data[0]
        left_knee_vel = self.data.sensor("left-knee-input-vel").data[0]
        left_foot_vel = self.data.sensor("left-foot-input-vel").data[0]

        left_shin_angle = self.data.sensor("left-shin-output").data[0]
        left_tarsus_angle = self.data.sensor("left-tarsus-output").data[0]
        left_foot_angle = self.data.sensor("left-foot-output").data[0]

        left_shin_vel = self.data.sensor("left-shin-output-vel").data[0]
        left_tarsus_vel = self.data.sensor("left-tarsus-output-vel").data[0]
        left_foot_vel = self.data.sensor("left-foot-output-vel").data[0]

        right_hip_roll_angle = self.data.sensor("right-hip-roll-input").data[0]
        right_hip_yaw_angle = self.data.sensor("right-hip-yaw-input").data[0]
        right_hip_pitch_angle = self.data.sensor("right-hip-pitch-input").data[0]
        right_knee_angle = self.data.sensor("right-knee-input").data[0]
        right_foot_angle = self.data.sensor("right-foot-input").data[0]

        right_hip_roll_vel = self.data.sensor("right-hip-roll-input-vel").data[0]
        right_hip_yaw_vel = self.data.sensor("right-hip-yaw-input-vel").data[0]
        right_hip_pitch_vel = self.data.sensor("right-hip-pitch-input-vel").data[0]
        right_knee_vel = self.data.sensor("right-knee-input-vel").data[0]
        right_foot_vel = self.data.sensor("right-foot-input-vel").data[0]

        right_shin_angle = self.data.sensor("right-shin-output").data[0]
        right_tarsus_angle = self.data.sensor("right-tarsus-output").data[0]
        right_foot_angle = self.data.sensor("right-foot-output").data[0]

        right_shin_vel = self.data.sensor("right-shin-output-vel").data[0]
        right_tarsus_vel = self.data.sensor("right-tarsus-output-vel").data[0]
        right_foot_vel = self.data.sensor("right-foot-output-vel").data[0]

        pelvis_orientation_w = self.data.sensor("pelvis-orientation").data[0]
        pelvis_orientation_x = self.data.sensor("pelvis-orientation").data[1]
        pelvis_orientation_y = self.data.sensor("pelvis-orientation").data[2]
        pelvis_orientation_z = self.data.sensor("pelvis-orientation").data[3]

        pelvis_angular_velocity_x = self.data.sensor("pelvis-angular-velocity").data[0]
        pelvis_angular_velocity_y = self.data.sensor("pelvis-angular-velocity").data[1]
        pelvis_angular_velocity_z = self.data.sensor("pelvis-angular-velocity").data[2]

        pelvis_linear_acceleration_x = self.data.sensor("pelvis-linear-acceleration").data[0]
        pelvis_linear_acceleration_y = self.data.sensor("pelvis-linear-acceleration").data[1]
        pelvis_linear_acceleration_z = self.data.sensor("pelvis-linear-acceleration").data[2]

        return np.array([left_hip_roll_angle,
                        left_hip_yaw_angle,
                        left_hip_pitch_angle,
                        left_knee_angle,
                        left_foot_angle,
                        left_hip_roll_vel,
                        left_hip_yaw_vel,
                        left_hip_pitch_vel,
                        left_knee_vel,
                        left_foot_vel,
                        left_shin_angle,
                        left_tarsus_angle,
                        left_foot_angle,
                        left_shin_vel,
                        left_tarsus_vel,
                        left_foot_vel,
            
                        right_hip_roll_angle,
                        right_hip_yaw_angle,
                        right_hip_pitch_angle,
                        right_knee_angle,
                        right_foot_angle,
                        right_hip_roll_vel,
                        right_hip_yaw_vel,
                        right_hip_pitch_vel,
                        right_knee_vel,
                        right_foot_vel,
                        right_shin_angle,
                        right_tarsus_angle,
                        right_foot_angle,
                        right_shin_vel,
                        right_tarsus_vel,
                        right_foot_vel,
            
                        pelvis_orientation_w, 
                        pelvis_orientation_x, 
                        pelvis_orientation_y, 
                        pelvis_orientation_z,
            
                        pelvis_angular_velocity_x,
                        pelvis_angular_velocity_y,
                        pelvis_angular_velocity_z,
                        
                        pelvis_linear_acceleration_x,
                        pelvis_linear_acceleration_y,
                        pelvis_linear_acceleration_z
                        ], dtype=np.float32
                        )
            
    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        self.data.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def termination_check(self, observation):
        body_id = self.model.body("cassie-pelvis").id

        height = self.data.xpos[body_id][2]  # z-coordinate of the pelvis
        if height < 0.7:
            return True
        return False

    def reward_function(self, observation, terminated):
        return 0

    def step(self, action):
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        observation = self._get_obs()
        terminated = self.termination_check(observation)
        truncated = self.step_count >= self.max_steps
        reward = self.reward_function(observation, terminated)
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="Bipedal-v0",
    entry_point="Bipedal_Env:bipedal",   
)
