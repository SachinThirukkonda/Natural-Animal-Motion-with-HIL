import mujoco
import numpy as np
import gymnasium as gym
from typing import Optional
from pathlib import Path
from CassieModel import CassieModel

class bipedal(gym.Env):
    def __init__(self, ep_len, terminal_height, variation):
        xml_path = Path(__file__).parent / "agility_cassie" / "scene.xml"

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.Cassie = CassieModel(self.model, self.data)

        self.initial_pos_id = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.initial_pos_id)
        self.initial_hinge_jnt_pos = self.Cassie.get_joint_qpos(self.Cassie.hinge_joints)

        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(42,),
                                                dtype=np.float32
                                                )

        self.action_space = gym.spaces.Box(low=-1.0,
                                                high=1.0,
                                                shape=(10,),
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
        sensor_readings = self.Cassie.get_all_sensor_readings()
        return sensor_readings
            
    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.initial_pos_id)

        jntpos_noise = self.np_random.uniform(-self.init_var[0], self.init_var[0], size=len(self.Cassie.hinge_joints))
        jntvel_noise = self.np_random.uniform(-self.init_var[1], self.init_var[1], size=len(self.Cassie.hinge_joints))
        self.Cassie.set_joints_qpos(self.Cassie.hinge_joints, jntpos_noise)
        self.data.qvel[:] = 0
        self.Cassie.set_joints_qvel(self.Cassie.hinge_joints, jntvel_noise)
        
        self.step_count = 0

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def termination_check(self, observation):
        height = self.Cassie.get_body_xpos(["cassie-pelvis"])[0][2]  # z-coordinate of the pelvis
        if height < self.terminal_height:
            return True
        return False

    def reward_function(self, observation, terminated, action): 
        if terminated:
            return -100.0   # Large negative reward for falling

        joint_weights = np.ones((len(self.Cassie.hinge_joints)))
        joint_weights[(self.Cassie.hinge_joints == "left-foot") | (self.Cassie.hinge_joints == "right-foot")] = 5.0
        joint_error = self.initial_hinge_jnt_pos - self.Cassie.get_joint_qpos(self.Cassie.hinge_joints)
        pose_penalty = 0.01 * np.sum((np.multiply(joint_error, joint_weights))**2)

        joint_vel = self.Cassie.get_joint_qvel(self.Cassie.hinge_joints)
        vel_penalty = 0.001 * np.sum(joint_vel**2)

        normalized_action = action
        act_penalty = 0.001 * np.sum(normalized_action**2)

        reward = 1.0 - pose_penalty - vel_penalty - act_penalty
        return float(reward)
    def get_pose_manifold(self):

        pass

    def step(self, action):
        physical_action = np.multiply(action, self.action_ranges)
        self.data.ctrl[:] = physical_action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        observation = self._get_obs()
        terminated = self.termination_check(observation)
        truncated = self.step_count >= self.max_steps
        reward = self.reward_function(observation, terminated, physical_action)
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="Bipedal-v0",
    entry_point="Bipedal_Env:bipedal",   
)
