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

# Hyperparameters
GAMMA = 0.99
LR = 5e-4
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 64
UPDATE_EVERY = 512
MAX_EPISODES = 2000
ENTROPY_COEF = 0.05
VALUE_COEF = 0.5
REWARD_SCALING = 0.1

SAVE_PATH = "ppo_malmo_model_best.pth"
CHECKPOINT_DIR = "checkpoints"

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
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
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
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.total_steps = 0
        
    def normalize_obs(self, obs):
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10.0, 10.0)
    
    def update_obs_stats(self, obs):
        self.obs_rms.update(obs)
    
    def preprocess_obs(self, obs):
        processed = obs.copy()
        
        if self.total_steps < 200000:
            if processed[3] > 0:
                processed[4] *= 2.0
                
                if processed[5] < 3.0:
                    processed[5] = 2.0
        
        return processed
        
    def select_action(self, state):
        state_processed = self.preprocess_obs(state)
        state_norm = self.normalize_obs(state_processed)
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0)
        
        exploration_factor = max(0.5, 1.0 - self.total_steps / 500000)
        
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(state_tensor)
            
            if self.total_steps < 100000:
                action = action.squeeze(0).numpy()
                action[0] += np.random.normal(0, 0.3 * exploration_factor)
                action = np.clip(action, -1.0, 1.0)
            else:
                action = action.squeeze(0).numpy()
        
        self.total_steps += 1
        return action
    
    def compute_gae(self, next_obs, rewards, dones, values, gamma=GAMMA, lam=0.95):
        next_obs_processed = self.preprocess_obs(next_obs)
        next_obs_norm = self.normalize_obs(next_obs_processed)
        with torch.no_grad():
            next_value = self.agent.forward(torch.FloatTensor(next_obs_norm).unsqueeze(0))[2].squeeze()

        values_extended = values.copy()
        values_extended.append(next_value.item())

        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values_extended[step + 1] * (1 - dones[step]) - values_extended[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values_extended[step])

        return returns

    def update(self, rollout_buffer):
        states = torch.FloatTensor(np.array(rollout_buffer['states']))
        actions = torch.FloatTensor(np.array(rollout_buffer['actions']))
        old_log_probs = torch.FloatTensor(np.array(rollout_buffer['log_probs']))
        returns = torch.FloatTensor(np.array(rollout_buffer['returns']))
        advantages = torch.FloatTensor(np.array(rollout_buffer['advantages']))
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(K_EPOCHS):
            batch_size = states.size(0)
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                _, log_probs, entropy, values = self.agent.get_action_and_value(batch_states, batch_actions)
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
        
        self.scheduler.step()
    
    def save_model(self, path):
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
            'obs_rms_count': self.obs_rms.count,
            'total_steps': self.total_steps
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.obs_rms.mean = checkpoint['obs_rms_mean']
        self.obs_rms.var = checkpoint['obs_rms_var']
        self.obs_rms.count = checkpoint['obs_rms_count']
        self.total_steps = checkpoint['total_steps']

def train():
    env = gym.make('MalmoZombie-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    agent = PPO(obs_dim, act_dim)
    
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    
    rollout_buffer = {
        'states': [],
        'actions': [],
        'log_probs': [],
        'rewards': [],
        'values': [],
        'dones': []
    }
    
    print("Starting PPO training...")
    
    for episode in range(MAX_EPISODES):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(UPDATE_EVERY):
            action = agent.select_action(obs)
            
            next_obs, reward, done, _ = env.step(action)
            
            rollout_buffer['states'].append(obs)
            rollout_buffer['actions'].append(action)
            rollout_buffer['rewards'].append(reward * REWARD_SCALING)
            rollout_buffer['dones'].append(done)
            
            obs_processed = agent.preprocess_obs(obs)
            obs_norm = agent.normalize_obs(obs_processed)
            with torch.no_grad():
                value = agent.agent.forward(torch.FloatTensor(obs_norm).unsqueeze(0))[2].squeeze().item()
            rollout_buffer['values'].append(value)
            
            obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            with torch.no_grad():
                _, log_prob, _, _ = agent.agent.get_action_and_value(obs_tensor, action_tensor)
            rollout_buffer['log_probs'].append(log_prob.item())
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        agent.update_obs_stats(np.array(rollout_buffer['states']))
        
        returns = agent.compute_gae(obs, rollout_buffer['rewards'], rollout_buffer['dones'], rollout_buffer['values'])
        advantages = [r - v for r, v in zip(returns, rollout_buffer['values'])]
        rollout_buffer['returns'] = returns
        rollout_buffer['advantages'] = advantages
        
        agent.update(rollout_buffer)
        
        rollout_buffer = {key: [] for key in rollout_buffer}
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(SAVE_PATH)
            print(f"New best model saved with reward: {best_reward:.2f}")
        
        if (episode + 1) % 100 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ppo_checkpoint_episode_{episode + 1}.pth")
            agent.save_model(checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}")
    
    env.close()
    print("Training completed!")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    train()