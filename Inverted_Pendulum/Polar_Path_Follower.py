import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

def follower_control(pos0, R, max_vel):
    pos = np.array([0, 0, 1])
    norm = lambda x: x/np.linalg.norm(x)
    # Orientation error
    d_pos = R.T @ pos0
    
    e_p = np.cross(pos, d_pos)

    return norm(e_p) * max_vel

def trajectory(t):
   T = 100
   return np.sqrt(2)/2 * np.array([np.sin(2*np.pi/T * t), np.cos(2*np.pi/T * t), 1]), np.pi/4, 2*np.pi/T * t


def pos_error(pos0, R):
  pos = np.array([0, 0, 1])

  # Orientation error
  d_pos = R.T @ pos0
  e_p = np.arcsin(np.linalg.norm(np.cross(pos, d_pos))/np.linalg.norm(d_pos))
  return e_p

def angle(R):
   vertical = np.array([0, 0, 1])
   pos = R @ vertical
   return np.arccos(np.dot(vertical, pos)) * 180 / (np.pi)


xml = """
<mujoco model="Inverse_Pendulum">
  <option integrator="RK4"/>
  <option gravity="0 0 0"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>
<worldbody>
    <geom name= "floor" size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="base">
      <joint name="pen_base" type="ball"/>
      <inertial
        pos="0 0 0.05"
        mass="0.1"
        diaginertia="4e-6 4e-6 4e-6"/>
      <geom type="capsule" name= "stem" size=".001" fromto="0 0 0 0 0 0.05" mass= "0"/>
      <geom name="ball" type="sphere" pos="0 0 0.05" size=".01" mass= "0"/>
      <site name="IMU" pos="0 0 0.5"/>
      <site name="x_axis"
      type="capsule"
      fromto="0 0 0.05 .02 0 0.05"
      size=".0005"
      rgba="1 0 0 1"/>
      <site name="y_axis"
      type="capsule"
      fromto="0 0 0.05 0 .02 0.05"
      size=".0005"
      rgba="0 1 0 1"/>
      <site name="z_axis"
      type="capsule"
      fromto="0 0 0.05 0 0 0.07"
      size=".0005"
      rgba="0 0 1 1"/>
    </body>

  </worldbody>
  <actuator>
    <motor name="motor_x" joint="pen_base" gear="1 0 0"/>
    <motor name="motor_y" joint="pen_base" gear="0 1 0"/>
    <motor name="motor_z" joint="pen_base" gear="0 0 1"/>
  </actuator>
  <sensor>
    <accelerometer name="accelerometer" site="IMU"/>
  </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
max_vel = 0.0000005

mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        pos0 = trajectory(data.time)[0]
        #pos0 = np.array([[np.cos(polar), -np.sin(polar), 0],[np.sin(polar), np.cos(polar), 0],[0,0,1]]) @ np.array([[np.cos(polar), 0, np.sin(polar)],[0,1,0],[-np.sin(polar), 0, np.cos(polar)]]) @ np.array([1,0,0])
        #print(pos0)
        R = np.reshape(data.xmat[1], (3,3))
        ctrl = follower_control(pos0, R, max_vel)
        #print(ctrl)
        data.ctrl = ctrl
        
        mujoco.mj_step(model, data)
        viewer.sync()