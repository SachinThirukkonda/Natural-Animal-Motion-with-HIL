import torch
import torch.nn as nn
import numpy as np
import time
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from networks.network import FeedForwardNN




class PPO:
    def __init__(self, env, hyperparameters, verbose=0, h_dim=64):
        #Initialise hyperparameters
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]   

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim, h_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1, h_dim)

        #Initialise optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        # Create the covariance matrix for get_action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.verbose = verbose
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }

        

    def learn(self, total_timesteps, callback=None):
            print(f"Learning... Running {self.ep_len} timesteps per episode, ", end='')
            print(f"{self.n_steps} timesteps per batch for a total of {total_timesteps} timesteps")

            if callback is not None:
                callback.on_training_start(self)

            t_so_far = 0 # Timesteps simulated so far
            i_so_far = 0 # Iterations ran so far
            while t_so_far < total_timesteps:              # ALG STEP 2
                # From the pseudocode, we need to collect the observations and actions per timestep, action probabilities, rewards-to-go and episode length
                # ALG STEP 3
                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews, batch_values, batch_masks = self.rollout()

                print("ROLLOUT FINISHED")

                # Calculate how many timesteps we collected this batch
                t_so_far += np.sum(batch_lens)

                if callback is not None:
                    callback.timesteps = t_so_far
                    callback.on_rollout_end(batch_lens, batch_rews)

                # Increment the number of iterations
                i_so_far += 1

                # Logging timesteps so far and iterations so far
                self.logger['t_so_far'] = t_so_far
                self.logger['i_so_far'] = i_so_far

                # ALG STEP 5
                # Calculate advantage
                #A_k = batch_rtgs - V.detach()
                A_k_GAE = self.calculate_gae(batch_rews, batch_values, batch_masks)

                old_V = torch.tensor([v for ep in batch_values for v in ep[:-1]])
                returns = A_k_GAE + old_V

                # Normalize advantages
                #A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
                A_k_GAE = (A_k_GAE - A_k_GAE.mean()) / (A_k_GAE.std() + 1e-10)

                
                
                # ALG STEP 6 and 7
                for _ in range(self.n_epochs):
                    # Calculate V_phi and pi_theta(a_t | s_t)    
                    V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                    # Calculate ratios
                    ratios = torch.exp(curr_log_probs - batch_log_probs)

                    # Calculate surrogate losses
                    surr1 = ratios * A_k_GAE
                    surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * A_k_GAE

                    # Caluclate actor and critic loss
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, returns)

                    # Calculate gradients and perform backward propagation for actor 
                    # network
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network    
                    self.critic_optim.zero_grad()    
                    critic_loss.backward()    
                    self.critic_optim.step()

                    # Log actor loss
                    self.logger['actor_losses'].append(actor_loss.detach())

                if callback is not None:
                    callback.on_iteration_end(
                        actor_loss.item(),
                        critic_loss.item()
                    )

                # Print a summary of our training so far
                self._log_summary()

    def evaluate(self, batch_obs, batch_acts):
            # Query critic network for a value V for each obs in batch_obs.
            V = self.critic(batch_obs).squeeze()
    
            # Calculate the log probabilities of batch actions using most 
            # recent actor network.
            # This segment of code is similar to that in get_action()
            mean = self.actor(batch_obs)
            dist = MultivariateNormal(mean, self.cov_mat)
            log_probs = dist.log_prob(batch_acts)
    
            # Return predicted values V and log probs log_probs
            return V, log_probs

    def calculate_gae(self, rewards, values, masks):
        batch_advantages = []
        for ep_rews, ep_vals, ep_masks in zip(rewards, values, masks):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                delta = ep_rews[t] + self.gamma * ep_vals[t+1] * ep_masks[t] - ep_vals[t]
                advantage = delta + self.gamma * self.gae_lambda * ep_masks[t] * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)
    def rollout(self):
            # Batch data
            batch_obs = []              # batch observations
            batch_acts = []             # batch actions
            batch_log_probs = []        # log probs of each action
            batch_rews = []             # batch rewards
            batch_values = []           # batch values
            batch_rtgs = []             # batch rewards-to-go
            batch_lens = []             # episodic lengths in batch
            batch_masks = []

            '''
            observations: (number of timesteps per batch, dimension of observation)
            actions: (number of timesteps per batch, dimension of action)
            log probabilities: (number of timesteps per batch)
            rewards: (number of episodes, number of timesteps per episode)
            reward-to-gos: (number of timesteps per batch)
            batch lengths: (number of episodes)
            '''

            # Number of timesteps run so far this batch
            t = 0 
            while t < self.n_steps:
                # Rewards this episode
                ep_rews = []
                ep_vals = []
                ep_masks = []
                obs, _ = self.env.reset()
                for ep_t in range(self.ep_len):
                    # If render is specified, render the environment
                    if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                        self.env.render()

                    # Increment timesteps ran this batch so far
                    t += 1 
                    
                    # Collect observation
                    batch_obs.append(obs)
                    obs_tensor = torch.from_numpy(obs).float()
                    value = self.critic(obs_tensor).detach().item()
                    ep_vals.append(value)
                    action, log_prob = self.get_action(obs)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                
                    # Collect reward, action, and log prob
                    ep_rews.append(reward)
                    batch_acts.append(action)
                    batch_log_probs.append(log_prob)

                    if terminated:
                        # No bootstrap
                        ep_masks.append(0)

                        # Add dummy value for V_{t+1}
                        ep_vals.append(0.0)
                        break
                    if truncated:
                        # Bootstrap from next observation
                        ep_masks.append(1)

                        next_obs_tensor = torch.from_numpy(obs).float()
                        next_value = self.critic(next_obs_tensor).detach().item()
                        ep_vals.append(next_value)

                        break
                        
                    ep_masks.append(1)

                # Collect episodic length and rewards
                batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
                batch_rews.append(ep_rews)
                batch_values.append(ep_vals)
                batch_masks.append(ep_masks)

            # Reshape data as tensors in the shape specified before returning
            batch_obs = torch.from_numpy(np.array(batch_obs)).float()
            batch_acts = torch.from_numpy(np.array(batch_acts)).float()
            batch_log_probs = torch.from_numpy(np.array(batch_log_probs)).float()
            # ALG STEP #4
            batch_rtgs = self.compute_rtgs(batch_rews)

            # Log the episodic returns and episodic lengths in this batch.
            self.logger['batch_rews'] = batch_rews
            self.logger['batch_lens'] = batch_lens

            # Return the batch data
            return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews, batch_values, batch_masks

    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)

        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return the sampled action and the log prob of that action
        # Note that detach() is called since the action and log_prob  
        # are tensors with computation graphs. detach() gets rid
        # of the graph and numpy() converts the action to numpy array.
        return action.detach().numpy(), log_prob.detach()
    
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.from_numpy(np.array(batch_rtgs)).float()

        return batch_rtgs

    def _init_hyperparameters(self, hyperparameters):
        # Default values for hyperparameters, will need to change later.
        self.n_steps = 4800             # timesteps per batch
        self.ep_len = 1600       # timesteps per episode
        self.gamma = 0.95                           # discount factor
        self.n_epochs = 5            # number of epochs per iteration
        self.clip_range = 0.2                             # Clip threshold (set at 0.2 as recommended by the paper)
        self.learning_rate = 0.005                             # learning rate of optimisers
        self.gae_lambda = 0.98

        # Miscellaneous parameters
        self.render = False                              # If we should render during rollout
        self.render_every_i = 10000                        # Only render every n iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        if self.verbose == 1:
            # Print logging statements
            print(flush=True)
            print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
            print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
            print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
            print(f"Average Loss: {avg_actor_loss}", flush=True)
            print(f"Timesteps So Far: {t_so_far}", flush=True)
            print(f"Iteration took: {delta_t} secs", flush=True)
            print(f"------------------------------------------------------", flush=True)
            print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def save(self, save_dir):
        torch.save(self.actor.state_dict(), save_dir + "/actor.pth")
        torch.save(self.critic.state_dict(), save_dir + "/critic.pth")

    def load(self, trained_actor_path, trained_critic_path):
        self.actor.load_state_dict(torch.load(trained_actor_path))
        self.critic.load_state_dict(torch.load(trained_critic_path))

