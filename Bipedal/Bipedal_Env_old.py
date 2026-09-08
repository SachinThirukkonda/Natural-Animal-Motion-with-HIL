import mujoco
import numpy as np
import gymnasium as gym
from typing import Optional
from pathlib import Path
import random

class bipedal(gym.Env):
    def __init__(self, ep_len, terminal_height, variation):
        xml_path = Path(__file__).parent / "agility_cassie" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.initial_pos = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.initial_pos)

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
        self.action_ranges = np.array([4.5,     # left-hip-roll
                                       4.5,     # left-hip-yaw
                                       12.2,    # left-hip-pitch
                                       12.2,    # left-knee
                                       0.9,     # left-foot
                                       4.5,     # right-hip-roll
                                       4.5,     # right-hip-yaw
                                       12.2,    # right-hip-pitch
                                       12.2,    # right-knee
                                       0.9      # right-foot
                                       ], dtype=np.float32
                                       )                        

        self.frame_skip = 40
        dt = self.model.opt.timestep
        self.max_steps = int(ep_len / (dt * self.frame_skip))
        self.step_count = 0
        self.terminal_height = terminal_height
        self.init_var = variation

    def _get_obs(self):
        left_hip_roll_angle = self.data.sensor("left-hip-roll-input").data[0]
        left_hip_yaw_angle = self.data.sensor("left-hip-yaw-input").data[0]
        left_hip_pitch_angle = self.data.sensor("left-hip-pitch-input").data[0]
        left_knee_angle = self.data.sensor("left-knee-input").data[0]
        left_foot_input_angle = self.data.sensor("left-foot-input").data[0]

        left_hip_roll_vel = self.data.sensor("left-hip-roll-input-vel").data[0]
        left_hip_yaw_vel = self.data.sensor("left-hip-yaw-input-vel").data[0]
        left_hip_pitch_vel = self.data.sensor("left-hip-pitch-input-vel").data[0]
        left_knee_vel = self.data.sensor("left-knee-input-vel").data[0]
        left_foot_input_vel = self.data.sensor("left-foot-input-vel").data[0]

        left_shin_angle = self.data.sensor("left-shin-output").data[0]
        left_tarsus_angle = self.data.sensor("left-tarsus-output").data[0]
        left_foot_output_angle = self.data.sensor("left-foot-output").data[0]

        left_shin_vel = self.data.sensor("left-shin-output-vel").data[0]
        left_tarsus_vel = self.data.sensor("left-tarsus-output-vel").data[0]
        left_foot_output_vel = self.data.sensor("left-foot-output-vel").data[0]

        right_hip_roll_angle = self.data.sensor("right-hip-roll-input").data[0]
        right_hip_yaw_angle = self.data.sensor("right-hip-yaw-input").data[0]
        right_hip_pitch_angle = self.data.sensor("right-hip-pitch-input").data[0]
        right_knee_angle = self.data.sensor("right-knee-input").data[0]
        right_foot_input_angle = self.data.sensor("right-foot-input").data[0]

        right_hip_roll_vel = self.data.sensor("right-hip-roll-input-vel").data[0]
        right_hip_yaw_vel = self.data.sensor("right-hip-yaw-input-vel").data[0]
        right_hip_pitch_vel = self.data.sensor("right-hip-pitch-input-vel").data[0]
        right_knee_vel = self.data.sensor("right-knee-input-vel").data[0]
        right_foot_input_vel = self.data.sensor("right-foot-input-vel").data[0]

        right_shin_angle = self.data.sensor("right-shin-output").data[0]
        right_tarsus_angle = self.data.sensor("right-tarsus-output").data[0]
        right_foot_output_angle = self.data.sensor("right-foot-output").data[0]

        right_shin_vel = self.data.sensor("right-shin-output-vel").data[0]
        right_tarsus_vel = self.data.sensor("right-tarsus-output-vel").data[0]
        right_foot_output_vel = self.data.sensor("right-foot-output-vel").data[0]

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
                        left_foot_input_angle,
                        left_hip_roll_vel,
                        left_hip_yaw_vel,
                        left_hip_pitch_vel,
                        left_knee_vel,
                        left_foot_input_vel,
                        left_shin_angle,
                        left_tarsus_angle,
                        left_foot_output_angle,
                        left_shin_vel,
                        left_tarsus_vel,
                        left_foot_output_vel,
            
                        right_hip_roll_angle,
                        right_hip_yaw_angle,
                        right_hip_pitch_angle,
                        right_knee_angle,
                        right_foot_input_angle,
                        right_hip_roll_vel,
                        right_hip_yaw_vel,
                        right_hip_pitch_vel,
                        right_knee_vel,
                        right_foot_input_vel,
                        right_shin_angle,
                        right_tarsus_angle,
                        right_foot_output_angle,
                        right_shin_vel,
                        right_tarsus_vel,
                        right_foot_output_vel,
            
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
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.initial_pos)


        random_initial = self.np_random.uniform(-self.init_var, self.init_var, size=36) + 1
        self.data.xpos = np.multiply(self.data.xpos, random_initial[0:26])
        self.data.qvel += np.multiply(random_initial[26:36], self.action_ranges)
        self.step_count = 0

        self.data.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def termination_check(self, observation):
        body_id = self.model.body("cassie-pelvis").id

        height = self.data.xpos[body_id][2]  # z-coordinate of the pelvis
        if height < self.terminal_height:
            return True
        return False

    def reward_function(self, observation, terminated, action): 
        if terminated:
            return -100.0   # Large negative reward for falling

        angular_velocity = observation[36:39]
        acceleration = observation[39:]
        downward_acceleration = observation[-1]  # z-component of the acceleration
        body_id = self.model.body("cassie-pelvis").id
        position = self.data.xpos[body_id][0:2]
        height = self.data.xpos[body_id][2]  # z-coordinate of the pelvis
        quaternion = observation[32:36]  # w, x, y, z
        reward = 2    # Reward for staying upright

        reward -= 1 * np.sum(position**2)  # Penalize deviation from the desired position
        reward -= 0.005 * np.sum(angular_velocity**2) # Penalize high angular velocity
        #reward -= 0.1 * np.sum(quaternion**2) # Penalize high quaternion values
        #reward -= 0.1 * downward_acceleration**2 # Penalize high acceleration

        # Additional penalty for being too far from the anchor point. Imagine a virtual rope that keeps the robot upright.
        anchor = np.array([0.0, 0.0, 2.0])
        pelvis = self.data.xpos[body_id]
        distance = np.linalg.norm(anchor - pelvis)
        tension = max(0.0, distance - 1.0)  # Penalize if the pelvis is too far from the anchor point
        #reward -= 0.1 * tension**2

        normalized_action = action / self.action_ranges
        reward -= 0.1 * np.sum(normalized_action**2)  # Penalize large actions to encourage smoother control

        return float(reward)

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        observation = self._get_obs()
        terminated = self.termination_check(observation)
        truncated = self.step_count >= self.max_steps
        reward = self.reward_function(observation, terminated, action)
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="Bipedal-v0",
    entry_point="Bipedal_Env:bipedal",   
)
