import mujoco
import numpy as np
from typing import Optional

class CassieModel:

    def __init__(self, model, data, rng):
        self.model = model
        self.data = data
        self.np_random = rng

        self.hinge_joints = np.array([
            self.model.joint(j).name
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ])

    def get_body_xpos(self, items: list[str]):
        """
        Get world-frame position of the specified bodies.

        Returns:
            np.ndarray with shape (n, 3)
        """
        return np.array([
            self.data.xpos[self.model.body(item).id].copy()
            for item in items
        ])

    def get_body_xquat(self, items: list[str]):
        """
        Get world-frame orientation quaternion of the specified bodies.

        Returns:
            np.ndarray with shape (n, 4)
        """
        return np.array([
            self.data.xquat[self.model.body(item).id].copy()
            for item in items
        ])

    def get_body_xmat(self, items: list[str]):
        """
        Get world-frame orientation matrix of the specified bodies.

        Returns:
            np.ndarray with shape (n, 9)
        """
        return np.array([
            self.data.xmat[self.model.body(item).id].copy()
            for item in items
        ])

    def get_body_xipos(self, items: list[str]):
        """
        Get world-frame position of the body's center of mass.

        Returns:
            np.ndarray with shape (n, 3)
        """
        return np.array([
            self.data.xipos[self.model.body(item).id].copy()
            for item in items
        ])

    def get_body_ximat(self, items: list[str]):
        """
        Get world-frame orientation of the body's inertia.

        Returns:
            np.ndarray with shape (n, 9)
        """
        return np.array([
            self.data.ximat[self.model.body(item).id].copy()
            for item in items
        ])

    def get_body_cvel(self, items: list[str]):
        """
        Get 6D Cartesian velocity of the specified bodies.

        MuJoCo's cvel contains:
            [angular_velocity, linear_velocity]

        Returns:
            np.ndarray with shape (n, 6)
        """
        return np.array([
            self.data.cvel[self.model.body(item).id].copy()
            for item in items
        ])

    def set_body_xvel(self, items: list[str], values: list[str]):
        for item, value in zip(items, values):
            body = self.model.body(item)
            self.data.xvel[body.dofadr[0]] += value

    # =========================
    # Joint data
    # =========================

    def get_joint_qpos(self, items: list[str]):
        """
        Get joint positions.

        For single-DoF joints, returns one scalar per joint.

        Returns:
            np.ndarray with shape (n,)
        """
        return np.array([
            self.data.qpos[self.model.joint(item).qposadr[0]].copy()
            for item in items
        ])

    def get_joint_qvel(self, items: list[str]):
        """
        Get joint velocities.

        For single-DoF joints, returns one scalar per joint.

        Returns:
            np.ndarray with shape (n,)
        """
        return np.array([
            self.data.qvel[self.model.joint(item).dofadr[0]].copy()
            for item in items
        ])

    def set_joints_qpos(self, items: list[str], values):
        for item, value in zip(items, values):
            joint = self.model.joint(item)
            self.data.qpos[joint.qposadr[0]] += value


    def set_joints_qvel(self, items: list[str], values):
        for item, value in zip(items, values):
            joint = self.model.joint(item)
            self.data.qvel[joint.dofadr[0]] += value

    def get_com(self):
        return self.data.subtree_com[0].copy()

    def get_all_sensor_readings(self):
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

    def get_pose_manifold(self):
            manifold_items = ["left-foot","right-foot","cassie-pelvis"]
            left_foot, right_foot, pelvis = self.get_body_xpos(manifold_items)
            com = self.get_com()
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, self.get_body_xquat(["cassie-pelvis"])[0])
            direction = R.reshape(3, 3)[:2, 0]
            direction /= np.linalg.norm(direction)
            feet_separation = (right_foot - left_foot)[:2]
            stance_width = abs(feet_separation[0] * direction[1] - feet_separation[1] * direction[0])
            feet_midpoint = (left_foot + right_foot)/2
            stance = com - feet_midpoint
    
            com_horizontal_offset = stance[:2]
            com_offset_along_stance = abs(np.dot(pelvis - left_foot, right_foot - left_foot) / np.linalg.norm(right_foot - left_foot)**2 - 0.5)
            com_offset_across_stance = abs(com_horizontal_offset @ direction)
            com_height = stance[2]
            pelvis_height = pelvis[2] - feet_midpoint[2]
    
            return {"direction": direction,
                    "stance_width": stance_width,
                    "feet_midpoint": feet_midpoint,
                    "com_offset": np.array([com_offset_across_stance, com_offset_along_stance]),
                    "com_height": com_height,
                    "pelvis_height": pelvis_height
            }

    def apply_impulse(self, impulse_magnitude):
        #magnitude = np.random.uniform(0, impulse_magnitude)
        magnitude = impulse_magnitude
        impulse_direction = self.np_random.normal(size=2)
        impulse_direction = np.array([1, 0])
        impulse_direction = self.np_random.choice((-1.0, 0.0, 1.0), size= 2,)
        impulse_direction = np.array([self.np_random.choice((-1.0, 1.0)), 0])
        if np.linalg.norm(impulse_direction) > 1:
            impulse_direction /= np.linalg.norm(impulse_direction)
        impulse = magnitude * impulse_direction

        pelvis = self.model.body("cassie-pelvis")

        # Free joint's 6 DOFs:
        # [vx, vy, vz, wx, wy, wz]
        dof = pelvis.dofadr[0]

        self.data.qvel[dof:dof + 2] += impulse