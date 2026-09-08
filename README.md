# Natural-Animal-Motion-with-HIL

The core investigation is whether Hybrid Imitation Learning (HIL) can produce natural looking and robust quadrupedal locomotion. The aim is to implement the results of the work of Jiashun Wang et al. in their paper titled "HIL: Hybrid Imitation Learning for Dynamic Athletic Control", adapted for simulated animal motion. I aim to take their work from simulation to reality in the future. The motion is supposed to imitate that of a dog and will use motion capture data from the 3D Multi-modal Dog Dataset, 3DDogs (https://cvssp.org/data/3DDogs/). For training, the dog model will be taken from the mujoco_menagerie github (https://github.com/google-deepmind/mujoco_menagerie). Once the components are selected and the CAD for my quadruped is designed, I will retrain the the agent using my CAD models for deployment. The HIL model will be taken from the official code release of the paper (https://github.com/jiashunwang/Hybrid-Motion-Imitation).

Knowledge Gained (working list):
Mujoco
Use of Gymnasium: creating my own training environment and training the model
Stable-Baselines-3 for implementation of premade networks
Reward function shaping
Understanding of concepts such as Reinforcement Learning, Markov Decision Processes (MDP), Deep Q-Networks (DQN), Policy Gradients, Proximal Policy Optimisation (PPO), Intrinsic Curiousity Module (ICM)

Educational Projects
In order to learn the skills required to carry out the investigation, I 
