import mujoco
import numpy as np
import gymnasium as gym
from typing import Optional
from pathlib import Path
from CassieModel import CassieModel

class bipedal(gym.Env):
    def __init__(self, ep_len, terminal_height, variation, impulse_magnitude):
        xml_path = Path(__file__).parent / "agility_cassie" / "scene.xml"

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.Cassie = CassieModel(self.model, self.data, self.np_random)

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
        self.impulse_magnitude = impulse_magnitude
        self.prev_action = np.zeros_like(self.action_ranges)
        self.reward_components = {
            "survival_reward": 0,
            "separation_reward": 0,
            "uprightness_reward": 0,
            "pelvis_offset_penalty": 0,
            "velocity_penalty": 0,
            "act_penalty": 0,
            "pos_penalty": 0
            }

    def _get_obs(self):
        sensor_readings = self.Cassie.get_all_sensor_readings()
        return sensor_readings
            
    def _get_info(self, terminated, truncated):
        if terminated or truncated:
            self.reward_components = {name: total / self.step_count for name, total in self.reward_components.items()}
            self.reward_components["survival_reward"] = self.step_count
            return {"reward_components": self.reward_components}
        else:
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
        self.reward_components = {
                    "survival_reward": 0,
                    "separation_reward": 0,
                    "uprightness_reward": 0,
                    "pelvis_offset_penalty": 0,
                    "velocity_penalty": 0,
                    "act_penalty": 0,
                    "pos_penalty": 0
                    }

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info(False, False)
        return observation, info

    def termination_check(self):
        height = self.Cassie.get_body_xpos(["cassie-pelvis"])[0][2]  # z-coordinate of the pelvis
        if height < self.terminal_height:
            return True
        return False

    def reward_function(self, terminated, action): 
        if terminated:
            return -100.0  # Large negative reward for falling


        '''
        Attempted to use a known stable pose as a target standing position
        joint_weights = np.ones((len(self.Cassie.hinge_joints)))
        joint_weights[(self.Cassie.hinge_joints == "left-foot") | (self.Cassie.hinge_joints == "right-foot")] = 5.0
        joint_error = self.initial_hinge_jnt_pos - self.Cassie.get_joint_qpos(self.Cassie.hinge_joints)
        pose_penalty = 0.01 * np.sum((np.multiply(joint_error, joint_weights))**2)
        '''

        pose_manifold = self.Cassie.get_pose_manifold()

        
        stance_width = pose_manifold["stance_width"]
        separation_reward = 0.01 * stance_width

        pelvis_xmat = self.Cassie.get_body_xmat(["cassie-pelvis"])[0]
        pelvis_xmat = pelvis_xmat.reshape(3, 3)
        uprightness = np.dot(pelvis_xmat[:, 2], np.array([0.0, 0.0, 1.0]))
        uprightness_reward = 0.5 * uprightness**2

        offset_weights = np.array([3, 1])
        com_offset = pose_manifold["com_offset"]
        uncentered_com_penalty = 0.5 * np.linalg.norm(np.multiply(offset_weights, com_offset))**2

        joint_vel = self.Cassie.get_joint_qvel(self.Cassie.hinge_joints)
        vel_penalty = 0.001 * np.sum(joint_vel**2)

        delta_action = action - self.prev_action
        act_penalty = 0.1 * np.sum(delta_action**2)

        pos = self.Cassie.get_body_xpos(["cassie-pelvis"])[0][:2]
        pos_penalty = 0.25 * np.linalg.norm(pos)**2


        reward = 1.0 - vel_penalty - uncentered_com_penalty - act_penalty - pos_penalty
        self.reward_components["separation_reward"] += separation_reward
        self.reward_components["uprightness_reward"] += uprightness_reward
        self.reward_components["pelvis_offset_penalty"] += uncentered_com_penalty
        self.reward_components["velocity_penalty"] += vel_penalty
        self.reward_components["act_penalty"] += act_penalty
        self.reward_components["pos_penalty"] += pos_penalty
        
    
        return float(reward)
    
    def step(self, action):
        physical_action = np.multiply(action, self.action_ranges)
        self.data.ctrl[:] = physical_action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        if self.step_count == int(self.max_steps/3):
            self.Cassie.apply_impulse(self.impulse_magnitude)
        observation = self._get_obs()
        terminated = self.termination_check()
        truncated = self.step_count >= self.max_steps
        reward = self.reward_function(terminated, action)
        info = self._get_info(terminated, truncated)

        self.prev_action = action

        return observation, reward, terminated, truncated, info




gym.register(
    id="Bipedal-v0",
    entry_point="Bipedal_Env:bipedal",   
)
