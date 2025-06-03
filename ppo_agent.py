import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt
import os
import time

gym.envs.registration.register(
    id='MalmoZombie-v0',
    entry_point='stable_baseline_env:MalmoZombieEnv'
)

# Improved Hyperparameters
GAMMA = 0.99
LR = 5e-4  # Increased for faster initial learning
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
UPDATE_EVERY = 512  # Smaller buffer for more frequent updates
MAX_EPISODES = 2000
SAVE_PATH = "ppo_malmo_model_best.pth"
CHECKPOINT_DIR = "checkpoints"  # Directory for saving checkpoints
ENTROPY_COEF = 0.05  # Increased to encourage more exploration
VALUE_COEF = 0.5
REWARD_SCALING = 0.1  # Scale rewards to a reasonable range

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        
        # Improved network architecture with more capacity for ray observations
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),  # Increased from 128 to 256
            nn.ReLU(),
            nn.Linear(256, 256),      # Increased from 128 to 256
            nn.ReLU(),
            nn.Linear(256, 128),      # Increased from 64 to 128
            nn.ReLU()
        )
        
        # Actor network - outputs mean of action distribution
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()  # Bounded actions
        )
        
        # Actor network - outputs log std of action distribution
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        
        # Critic network with additional layer
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        shared = self.shared(x)
        action_mean = self.actor_mean(shared)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        value = self.critic(shared)
        return action_mean, action_std, value
    
    def get_action_and_value(self, x, action=None):
        action_mean, action_std, value = self.forward(x)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

class PPO:
    def __init__(self, obs_dim, act_dim):
        self.agent = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=LR, eps=1e-5)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        # Running statistics for observation normalization
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        
        # Track total steps for annealing exploration
        self.total_steps = 0
        
    def normalize_obs(self, obs):
        """Normalize observations using running statistics"""
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10.0, 10.0)
    
    def update_obs_stats(self, obs):
        """Update observation normalization statistics"""
        self.obs_rms.update(obs)
    
    def preprocess_obs(self, obs):
        """Emphasize important features early in training"""
        processed = obs.copy()
        
        # Early in training, emphasize zombie detection and facing
        if self.total_steps < 200000:
            if processed[3] > 0:  # If zombie present
                # Emphasize ray hit and distance information
                processed[4] *= 2.0  # Emphasize ray hit
                
                # Make distance more salient
                if processed[5] < 3.0:  # If close to optimal distance
                    processed[5] = 2.0  # Make it more distinct
        
        return processed
        
    def select_action(self, state):
        # Preprocess and normalize state
        state_processed = self.preprocess_obs(state)
        state_norm = self.normalize_obs(state_processed)
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0)
        
        # Anneal exploration over time
        exploration_factor = max(0.5, 1.0 - self.total_steps / 500000)
        
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(state_tensor)
            
            # Add extra exploration noise early in training
            if self.total_steps < 100000:
                action = action.squeeze(0).numpy()
                # Add more noise to turning action
                action[0] += np.random.normal(0, 0.3 * exploration_factor)
                # Clip to valid range
                action = np.clip(action, -1.0, 1.0)
            else:
                action = action.squeeze(0).numpy()
        
        self.total_steps += 1
        return action
    
    def compute_gae(self, next_obs, rewards, dones, values, gamma=GAMMA, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        next_obs_processed = self.preprocess_obs(next_obs)
        next_obs_norm = self.normalize_obs(next_obs_processed)
        with torch.no_grad():
            next_value = self.agent.forward(torch.FloatTensor(next_obs_norm).unsqueeze(0))[2].squeeze()

        # Create a copy of values to avoid modifying the original list
        values_extended = values.copy()
        values_extended.append(next_value.item())  # Append as scalar, not tensor

        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values_extended[step + 1] * (1 - dones[step]) - values_extended[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values_extended[step])

        return returns

    def update(self, rollout_buffer):
        """Update policy using collected rollout buffer"""
        # Preprocess and normalize observations
        processed_obs = [self.preprocess_obs(obs) for obs in rollout_buffer['observations']]
        norm_obs = [self.normalize_obs(obs) for obs in processed_obs]
        
        # Convert to tensors
        b_obs = torch.FloatTensor(np.array(norm_obs))
        b_actions = torch.FloatTensor(np.array(rollout_buffer['actions']))
        b_logprobs = torch.FloatTensor(np.array(rollout_buffer['logprobs']))
        b_rewards = rollout_buffer['rewards']
        b_dones = rollout_buffer['dones']
        b_values = rollout_buffer['values']

        # Compute returns and advantages using GAE
        b_returns = self.compute_gae(
            rollout_buffer['next_obs'], b_rewards, b_dones, b_values
        )
        b_returns = torch.FloatTensor(b_returns)

        # Make sure values is a tensor of the same length as returns
        b_values_tensor = torch.FloatTensor(b_values)

        # Calculate advantages
        b_advantages = b_returns - b_values_tensor

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Flatten batch dimensions
        b_inds = np.arange(len(b_obs))

        # Training metrics
        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'approx_kl': 0
        }

        # Training loop
        for epoch in range(K_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl for early stopping
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]

                # Policy loss with clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with clipping
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values_tensor[mb_inds] + torch.clamp(
                    newvalue - b_values_tensor[mb_inds], -EPS_CLIP, EPS_CLIP
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENTROPY_COEF * entropy_loss + v_loss * VALUE_COEF

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()

                # Update metrics
                metrics['policy_loss'] += pg_loss.item()
                metrics['value_loss'] += v_loss.item()
                metrics['entropy'] += entropy_loss.item()
                metrics['approx_kl'] += approx_kl.item()

        # Step learning rate scheduler
        self.scheduler.step()

        # Average metrics over iterations
        num_iterations = K_EPOCHS * ((len(b_obs) + BATCH_SIZE - 1) // BATCH_SIZE)
        for k in metrics:
            metrics[k] /= num_iterations

        return metrics

# Training loop with improved memory management and monitoring
def train():
    env = gym.make('MalmoZombie-v0')
    ppo = PPO(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0])
    
    # Try to load existing model
    try:
        ppo.agent.load_state_dict(torch.load(SAVE_PATH))
        print("Loaded existing model")
    except:
        print("Starting with new model")
    
    # Rollout buffer - more memory efficient
    rollout_buffer = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'values': [],
        'logprobs': [],
        'next_obs': None
    }
    
    # Tracking metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_kills = deque(maxlen=100)
    facing_percentages = deque(maxlen=100)
    avg_distances = deque(maxlen=100)
    best_reward = -float('inf')
    total_steps = 0
    episode_count = 0
    
    # Training metrics for plotting
    reward_history = []
    length_history = []
    kill_history = []
    facing_history = []
    distance_history = []
    loss_history = []
    
    print("Starting training...")
    
    try:
        for episode in range(MAX_EPISODES):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            
            # Track facing and distance metrics
            episode_facing_frames = 0
            episode_distance_sum = 0
            episode_distance_frames = 0
            
            # Update observation statistics with initial state
            ppo.update_obs_stats(np.array([state]))
            
            # Check if zombie is present in initial state
            if state[3] == 0:
                print("Warning: Episode starting without zombie in view")
            
            # Wait for a short time at the beginning of each episode to let the environment stabilize
            time.sleep(0.5)
            
            while not done and episode_steps < 500:  # Max episode length
                # Get action
                action = ppo.select_action(state)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Track facing and distance metrics
                if next_state[3] > 0:  # If zombie present
                    episode_distance_frames += 1
                    episode_distance_sum += next_state[5]
                    if next_state[4] > 0:  # If facing zombie
                        episode_facing_frames += 1
                
                # Scale reward
                scaled_reward = reward * REWARD_SCALING
                
                # Update observation statistics
                ppo.update_obs_stats(np.array([next_state]))
                
                # Get value estimate
                with torch.no_grad():
                    state_processed = ppo.preprocess_obs(state)
                    norm_state = ppo.normalize_obs(state_processed)
                    _, logprob, _, value = ppo.agent.get_action_and_value(
                        torch.FloatTensor(norm_state).unsqueeze(0),
                        torch.FloatTensor(action).unsqueeze(0)
                    )
                
                # Store transition
                rollout_buffer['observations'].append(state)
                rollout_buffer['actions'].append(action)
                rollout_buffer['rewards'].append(scaled_reward)
                rollout_buffer['dones'].append(done)
                rollout_buffer['values'].append(value.item())
                rollout_buffer['logprobs'].append(logprob.item())
                
                state = next_state
                episode_reward += reward  # Track unscaled reward for reporting
                episode_steps += 1
                total_steps += 1
                
                # Update policy when buffer is full
                if total_steps % UPDATE_EVERY == 0 and len(rollout_buffer['observations']) > 0:
                    rollout_buffer['next_obs'] = next_state
                    update_info = ppo.update(rollout_buffer)
                    loss_history.append(update_info)
                    
                    # Clear buffer
                    rollout_buffer = {
                        'observations': [],
                        'actions': [],
                        'rewards': [],
                        'dones': [],
                        'values': [],
                        'logprobs': [],
                        'next_obs': None
                    }
                    
                    print("Update at step {}: Policy Loss: {:.4f}, Value Loss: {:.4f}, Entropy: {:.4f}, KL: {:.4f}".format(
                        total_steps, update_info['policy_loss'], update_info['value_loss'], 
                        update_info['entropy'], update_info['approx_kl']))
            
            # Calculate episode metrics
            facing_percentage = 0
            avg_distance = 0
            if episode_distance_frames > 0:
                avg_distance = episode_distance_sum / episode_distance_frames
                facing_percentage = episode_facing_frames / episode_distance_frames * 100
            
            # Track episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            zombie_killed = info.get('zombie_killed', False)
            episode_kills.append(1 if zombie_killed else 0)
            facing_percentages.append(facing_percentage)
            avg_distances.append(avg_distance)
            
            reward_history.append(episode_reward)
            length_history.append(episode_steps)
            kill_history.append(1 if zombie_killed else 0)
            facing_history.append(facing_percentage)
            distance_history.append(avg_distance)
            episode_count += 1
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(ppo.agent.state_dict(), SAVE_PATH)
                print("New best model saved! Reward: {:.2f}".format(best_reward))
            
            # Save periodic checkpoints
            if episode_count % 50 == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_ep{}.pth".format(episode_count))
                torch.save(ppo.agent.state_dict(), checkpoint_path)
                print("Checkpoint saved at episode {}".format(episode_count))
            
            # Logging
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_kills = np.mean(episode_kills) * 100  # Convert to percentage
            avg_facing = np.mean(facing_percentages)
            avg_dist = np.mean(avg_distances)
            
            print("Episode {}: Reward: {:.2f}, Avg(100): {:.2f}, Steps: {}, Zombie Killed: {}, Kill Rate: {:.1f}%".format(
                episode_count, episode_reward, avg_reward, episode_steps, zombie_killed, avg_kills))
            print("Facing: {:.1f}%, Avg Distance: {:.2f}".format(facing_percentage, avg_distance))
            
            # Plot progress every 50 episodes
            if episode_count % 50 == 0:
                plt.figure(figsize=(15, 10))
                
                plt.subplot(2, 3, 1)
                plt.plot(reward_history)
                plt.title('Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                
                plt.subplot(2, 3, 2)
                plt.plot([np.mean(reward_history[max(0, i-50):i+1]) for i in range(len(reward_history))])
                plt.title('Moving Average Reward (50 episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                
                plt.subplot(2, 3, 3)
                plt.plot(kill_history)
                plt.title('Zombie Kill Success')
                plt.xlabel('Episode')
                plt.ylabel('Kill (1=Yes, 0=No)')
                
                plt.subplot(2, 3, 4)
                window_size = 50
                kill_rate = [np.mean(kill_history[max(0, i-window_size):i+1])*100 for i in range(len(kill_history))]
                plt.plot(kill_rate)
                plt.title('Kill Success Rate (Moving Avg {} episodes)'.format(window_size))
                plt.xlabel('Episode')
                plt.ylabel('Kill Rate (%)')
                
                plt.subplot(2, 3, 5)
                plt.plot(facing_history)
                plt.title('Facing Percentage')
                plt.xlabel('Episode')
                plt.ylabel('% Time Facing Zombie')
                
                plt.subplot(2, 3, 6)
                plt.plot(distance_history)
                plt.title('Average Distance to Zombie')
                plt.xlabel('Episode')
                plt.ylabel('Distance (blocks)')
                
                plt.tight_layout()
                plt.savefig('training_progress_{}.png'.format(episode_count))
                plt.close()
                
                # Save detailed metrics
                np.save('metrics_ep{}.npy'.format(episode_count), {
                    'rewards': reward_history,
                    'lengths': length_history,
                    'kills': kill_history,
                    'facing': facing_history,
                    'distances': distance_history,
                    'losses': loss_history
                })
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save final model on interrupt
        torch.save(ppo.agent.state_dict(), "ppo_malmo_model_interrupted.pth")
    
    finally:
        env.close()
        print("Training completed!")

if __name__ == "__main__":
    train()