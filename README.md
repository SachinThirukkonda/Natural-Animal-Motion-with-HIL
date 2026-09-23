# Natural-Animal-Motion-with-HIL

The core investigation is whether **Hybrid Imitation Learning (HIL)** can produce natural looking and robust quadrupedal locomotion. The long-term objective is to implement and adapt the the work of Jiashun Wang et al. in their paper titled *"HIL: Hybrid Imitation Learning for Dynamic Athletic Control"*, adapted for simulated animal motion. I aim to take their work from simulation to a physical quadruped and and investigate the challenges involved with sim-to-real transfer. The target motion is dog-like locomotion, using motion capture data from the **3D Multi-modal Dog Dataset (3DDogs)** - https://cvssp.org/data/3DDogs/. For the initial simulation, the quadruped model will be taken from the **MuJoCo Menagerie** - https://github.com/google-deepmind/mujoco_menagerie. Once the components have been selected and mechanical design of the physical quadruped finalised, the agent will be retrained using a model matching the physical robot, before investigating sim-to-real transfer. The HIL model will be taken from the official code release of the paper (https://github.com/jiashunwang/Hybrid-Motion-Imitation).

# Knowledge Gained

This is a working list of concepts and technologies developed throughout the project.

### Simulation

- MuJoCo XML modelling
- Joint, body and geometry data
- Actuators and control inputs
- Sensor-based observations
- External forces and impulses
- Physics simulation and simulation timestep

### Reinforcement Learning

- Markov Decision Processes (MDPs)
- Reward function design/ Reward shaping
- Policy gradients
- Deep Q-Networks (DQN)
- Proximal Policy Optimisation (PPO)
- Generalised Advantage Estimation (GAE)
- Intrinsic Curiosity Module (ICM)

### Machine Learning

- Neural network training
- Actor-critic architectures
- Policy/value networks
- Model checkpointing
- Hyperparameter selection
- Training stability
- Observation normalisation
- Experiment tracking

### Software

- NumPy
- PyTorch
- Gymnasium
- Stable-Baselines3
- TensorBoard
- GitHub

---


Educational Projects
In order to learn the skills required to carry out the investigation, I built up the required knowledge through a series of progressively more complex projects. Stable-Baselines3 was initially used to avoid implementing the entire RL algorithm from scratch. This allowed me to concentrate on understanding the environment, reward design and behaviour of the trained agent. I made use of additional functionality provided by Stable-Baselines3, including:

- TensorBoard logging
- callbacks
- checkpointing
- environment wrappers
- model evaluation

The eventual intention is to replace Stable-Baselines3 with a custom PyTorch implementation of PPO.

# Inverted Pendulum

This project provided my initial introduction to MuJoCo.

I learnt how to:

- create a MuJoCo model from an XML string
- define bodies, joints and actuators
- access the dynamic state directly
- work with quantities such as `xmat`, `qvel` and `xpos`
- update motor actuation through `data.ctrl`
- interact with the MuJoCo simulation state
- control motion using a PID controller

This established the basic understanding of MuJoCo required for the later reinforcement-learning environments.

---


# Cart Pole

The cart-pole project was used to gain familiarity with Gymnasium and reinforcement learning.

I learnt how to construct a custom Gymnasium environment, including initialisation, creating reset, reward, termination check, step, info, and observation functions. I specifically aimed to make the environment **sensor-based**, rather than simply taking data from MuJoCo's dynamic state. This simulates the sensor data gathered by the physical robot, enabling the controller to work on an identical physical setup. I also tained a neural network using stable-baselines3's implementation of PPO. I made of some additional functionality such as tensorboard logging and  environment wrappers. 

## Reward shaping

The cart-pole project was my first  experience with reward shaping for reinforcement learning.

An important lesson was that a simpler reward function is often easier to train than a highly complicated one.

I also found that progressively increasing environment complexity produced more stable training.

For example, training initially involved:

1. keeping the pole upright initialising with a small initial angle
2. increasing the initial angle
3. later introducing a second objective to keep the cart near the origin

This approach was more reliable than introducing all objectives and difficulties simultaneously.

I also created a separate physical-environment simulation in which the trained policy interacted directly with the MuJoCo model rather than through the Gymnasium environment. This established the distinction between the robot, controller, and physical environment, recreating how the trained model would be implemented in real life.

# Bipedal Stable-Baselines3

This project was a step up in complexity from the cart pole and inverted pendulum projects. 
I used the Agility Cassie model from the MuJoCo Menagerie and developed an environment for training the robot on standing, dynamic recovery and eventually walking.

## Observation Space Design
The standing environment initially used 42 observations describing the robot's physical state, taking information from the joints (position and velocities) as well as a gyroscope and accelerometer. The observation space was expanded with additional directional/velocity information for the walking implementation. Potential future additions include foot contact and observation history.
A key learning was in shaping the reward function and the significance of the reward coefficients. With the increase in training times, I sought the need for saving checkpoint models in case the program crashed, or some instability midway through training required me to rollback training, and the use of try and finally structure to pause training when needed. The need for more complex reward function shaping also made it necessary to view the individual reward contributions in the tensorboard. I also added some more quality of life functionalities such as asynchronous live viewing of training (by extracting a snapshot of the policy as the model is training and simulating that asynchronously). I also reorganised the code and moved the mujoco model related functionality calls from the gym environment to a separate CassieModel file to improve readability.

Current progress:
Bipedal_SB3
├── agility_cassie/
│   - Imported from Mujoco Menagerie. Contains model assets, cassie model xml and scene xml
│
├── Training/
│   ├── Saved_Models
│   │   - Contains trained models for standing and walking
│   ├── Temp_Checkpoint_Models
│   │   - Used for storing checkpoint models when training. Automatically cleared before every training session
│   └── tensorboard
│       - Stored tensorboard logs for tensorboard functionality
│
├── Bipedal_Crumple_Test.py
│   -  Used for initial environment set up and testing. Used for extracting relevant information about body/geometry ids and initial positions, etc.
│
├── Bipedal_Env_Standing.py
│   - The environment for the standing implementation. A distinction was required since training required different observations and a different reward function. Due to the difference in size of observations, trained models are only compatible with the environment it was trained in.
│
├── Bipedal_Env.py
│   - In addition to the 42 observations for the standing implementation, it has another 2 for its current velocity.
│
├── CassieModel.py
│   - A quality of life wrapper that provides convenient access to joint/body/geom positions and velocity data, sensor readings and application of external forces.
│
├── Helper.py
│   - Contains all the helper functions, including functions for testing the environment (using check_env), and callbacks. Creating a separate file declutters the rest of the code.
│
├── Test.py
│   - 
│
└── Train.py
