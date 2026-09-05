import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

#PID control function for the cart-pole system. Takes in the proportional, integral, and derivative errors, as well as the PID gains, and returns the control signal and updated integral error.
#for g = -0.001, ctrl1: k_p=0.006, k_i=6.3e-8, k_d=1.6e-2, ctrl2: k_p=0.006, k_i=6.3e-8, k_d=1.6e-2
def pid_control(e_p, e_i, e_d, k_p, k_i, k_d):
    #Integral error
    e_i += e_p
    return e_p * k_p + e_d * k_d + e_i * k_i, e_i

#returns the desired position of the cart at time t. Currently returns a constant position of 0.
def trajectory(t):
    T= 100
    A = 0.1
    return 0
    #return A* np.sin(2 * np.pi / T * t)

#to enable input from the keyboard to push the pole or disable the controller
def key_callback(keycode):
    global push, disable

    if chr(keycode) == 'P':
        push = 1
    if chr(keycode) == 'L' and disable == 0:
        disable = 1
        print("disable")
    elif chr(keycode) == 'L' and disable == 1:
        disable = 0
        print("enable")

#gets the current errors for balance and position from the sensors.
def observations(data):
    balance_e_p = data.sensor("pole_angle").data[0]
    balance_e_d = data.sensor("pole_angular_vel").data[0]
    pos = data.sensor("cart_pos").data[0]
    vel = data.sensor("cart_vel").data[0]
    pos_e_p = pos + np.sin(balance_e_p)*0.2
    pos_e_d = vel + np.cos(balance_e_p)*0.2*balance_e_d
    return balance_e_p, balance_e_d, pos_e_p, pos_e_d

xml = """
<mujoco model="Cart_Pole">
  <option integrator="RK4"/>
  <option gravity="0 0 -0.1"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>
<worldbody>
    <geom name= "floor" size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="cart" pos="0 0 .02">
        <joint name="cart_slide"
            type="slide"
            axis="1 0 0"
            limited="true"
            range="-0.15 0.15"/>
        <geom name="cart"
            type="box"
            size=".05 .05 .02"
            mass=".1"/>
        <body name="pole">
            <inertial
                pos="0 0 0.2"
                mass="1"
                diaginertia="4e-6 4e-6 4e-6"/>
            <joint name="pole_hinge"
                type="hinge"
                axis="0 1 0"/>
            <geom name="stem"
                type="capsule"
                size=".01"
                fromto="0 0 0 0 0 0.2"
                mass="0"/>
        </body>
    </body>
</worldbody>
<actuator>
    <motor name="cart_motor"
        joint="cart_slide"
        gear="1"
        forcerange="-.2 .2"
        forcelimited="true"/>
</actuator>
<sensor>
    <jointpos name="cart_pos"
        joint="cart_slide"/>
    <jointpos name="pole_angle"
        joint="pole_hinge"/>
    <jointvel name="cart_vel"
        joint="cart_slide"/>
    <jointvel name="pole_angular_vel"
        joint="pole_hinge"/>
</sensor>    
</mujoco>
"""

#initialise the MuJoCo model and data.
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

#initialise variables.
pole_dof = model.joint("pole_hinge").dofadr[0]
push = 0
disable = 0
balance_e_i = 0
pos_e_i = 0
testing = 1

if testing:
    k_p = np.power(10, np.linspace(0.3, -0.5, 10))
    k_i = np.power(10, np.linspace(-4, -12, 8))
    k_d = np.power(10, np.linspace(0.5, -1, 1))

   # prepare plotting axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    duration = 50 # (seconds)

    for k in k_d:
        mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0
        timevals = []
        pole_angle = []
        cart_pos = []
        balance_e_i = 0
        pos_e_i = 0
        data.qvel[pole_dof] = .4

        while data.time < duration:
            #get the current errors for balance and position from the sensors.
            balance_e_p, balance_e_d, pos_e_p, pos_e_d = observations(data)

            #calculate the control signals for balance and position using the PID controller.
            balance, balance_e_i = pid_control(balance_e_p, balance_e_i, balance_e_d, k_p=1, k_i=1e-8, k_d=0.2) #k_p=0.25, k_i=1e-6, k_d=3.6)
            position, pos_e_i = pid_control(pos_e_p, pos_e_i, pos_e_d, k_p=1.3, k_i=1e-12, k_d=3.2)  # k_p=0.00056, k_i=1e-4, k_d=40)

            #apply the control signals to the cart-pole system.
            data.ctrl[0] = balance + position

            #step the simulation forward by one time step.
            mujoco.mj_step(model, data)

            #record the time, pole angle, and cart position for plotting.
            timevals.append(data.time)
            pole_angle.append(balance_e_p)
            cart_pos.append(pos_e_p)

        ax[0].plot(timevals, pole_angle, label='k = {:2.2g}'.format(k))
        ax[1].plot(timevals, cart_pos, label='k = {:2.2g}'.format(k))

    ax[0].set_title('Coefficient Tuning')
    ax[0].set_ylabel('angle')
    ax[0].set_xlabel('second')
    ax[0].legend(frameon=True);
    ax[1].set_title('Coefficient Tuning')
    ax[1].set_ylabel('position')
    ax[1].set_xlabel('second')
    ax[1].legend(frameon=True);
    plt.tight_layout()
    plt.show()
else:
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            #gives the cart-pole system a push if the 'P' key is pressed.
            if push:
                data.qvel[pole_dof] += .6
                push = 0

            #get the current errors for balance and position from the sensors.
            balance_e_p, balance_e_d, pos_e_p, pos_e_d = observations(data)

            #calculate the control signals for balance and position using the PID controller.
            balance, balance_e_i = pid_control(balance_e_p, balance_e_i, balance_e_d, k_p=1, k_i=1e-8, k_d=0.2) #k_p=0.25, k_i=1e-6, k_d=3.6)
            position, pos_e_i = pid_control(pos_e_p, pos_e_i, pos_e_d, k_p=1.3, k_i=1e-12, k_d=3.2)  # k_p=0.00056, k_i=1e-4, k_d=40)

            #apply the control signals to the cart-pole system if the pole is within a certain angle range.
            if abs(balance_e_p) < np.pi / 4 and  not disable:
                data.ctrl[0] = balance + position

            #step the simulation forward by one time step.
            mujoco.mj_step(model, data)
            viewer.sync()
