import gym
import numpy as np
from stable_baselines import PPO2, SAC
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback
from stable_baseline_env import MalmoZombieEnv

# Register your environment
gym.envs.registration.register(
    id='MalmoZombie-v0',
    entry_point='stable_baseline_env:MalmoZombieEnv'
)

class ZombieRewardWrapper(gym.Wrapper):
    """Wrapper that adds meaningful rewards to the zombie environment"""
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_zombie_distance = None
        self.steps_taken = 0
        self.zombie_was_present = False  # Track if zombie was previously visible
        self.zombie_killed = False       # Track if zombie has been killed
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.steps_taken += 1
        
        # Calculate our custom reward
        custom_reward = self._calculate_reward(obs, action)
        
        return obs, custom_reward, done, info
    
    def _calculate_reward(self, obs, action):
        reward = 0.0
        
        if obs[5] == 1:  # Zombie present (assuming yaw at obs[1])
            self.zombie_was_present = True  # Mark that we've seen the zombie
            
            # Extract data - now with yaw at obs[1]
            agent_yaw = obs[1]  # Agent's yaw angle
            zombie_pos = obs[2:5]  # Zombie x, y, z coordinates
            distance = np.linalg.norm(zombie_pos)
            
            # Reward for approaching zombie
            if self.prev_zombie_distance is not None:
                if distance < self.prev_zombie_distance:
                    reward += 0.3  # Approaching zombie
                elif distance > self.prev_zombie_distance:
                    reward -= 0.1  # Moving away from zombie
            
            self.prev_zombie_distance = distance
            
            # Distance-based rewards
            if distance < 2.0:
                reward += 3.0    # Very close - excellent
            elif distance < 4.0:
                reward += 1.5    # Close - good
            elif distance < 6.0:
                reward += 0.5    # Moderate distance - okay
            
            # Attack rewards
            if action[2] > 0.5:  # Attacking
                if distance < 2.5:
                    reward += 5.0    # Attack at close range - excellent
                elif distance < 4.0:
                    reward += 1.0    # Attack at medium range - okay
                else:
                    reward -= 1.0    # Attack at long range - bad
            
            # Bonus for facing zombie (using yaw data)
            zombie_x, zombie_z = zombie_pos[0], zombie_pos[2]  # Ignore Y for angle calc
            
            # Calculate angle to zombie
            angle_to_zombie = np.degrees(np.arctan2(-zombie_x, zombie_z))  # Minecraft coordinate system
            
            # Normalize angles to [0, 360)
            angle_to_zombie = (angle_to_zombie + 360) % 360
            agent_yaw_normalized = (agent_yaw + 360) % 360
            
            # Calculate angular difference
            angle_diff = abs(angle_to_zombie - agent_yaw_normalized)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Reward for facing zombie (smaller angle difference = better)
            if angle_diff < 30:      # Facing directly at zombie
                reward += 1.0
            elif angle_diff < 60:    # Roughly facing zombie
                reward += 0.5
            elif angle_diff < 90:    # Somewhat facing zombie
                reward += 0.2
            else:                    # Not facing zombie
                reward -= 0.1
            
        else:
            # No zombie visible
            if self.zombie_was_present and not self.zombie_killed:
                # Zombie disappeared after being present - likely killed!
                print("ZOMBIE KILLED! Maximum reward!")
                reward += 100.0  # HUGE reward for killing zombie
                self.zombie_killed = True
            elif not self.zombie_was_present:
                # Haven't found zombie yet
                reward -= 0.5  # Penalty for not finding zombie
            elif self.zombie_killed:
                # Zombie already killed, maintain high reward
                reward += 10.0  # Continue rewarding success
            
            self.prev_zombie_distance = None
        
        # Smaller time penalty since we want to reward success more
        reward -= 0.01
        
        return reward
    
    def reset(self):
        self.prev_zombie_distance = None
        self.steps_taken = 0
        self.zombie_was_present = False
        self.zombie_killed = False
        return self.env.reset()

# Custom callback for logging (SB2 style)
class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            # Log episode info
            if 'episode_rewards' in self.locals:
                episode_reward = self.locals['episode_rewards'][0]
                episode_length = self.locals['episode_lengths'][0]
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                print("Episode reward: {:.2f}, Length: {}".format(episode_reward, episode_length))
        
        return True

def create_wrapped_env():
    """Create the wrapped environment"""
    base_env = gym.make('MalmoZombie-v0')
    return ZombieRewardWrapper(base_env)

def train_ppo2_agent():
    """Train a PPO2 agent using Stable-Baselines 2"""
    print("Training PPO2 agent with Stable-Baselines 2...")
    
    # Create vectorized environment
    env = DummyVecEnv([create_wrapped_env])
    
    # Create the PPO2 model
    model = PPO2(
        MlpPolicy,              # Policy type
        env,
        verbose=1,              # Print training progress
        learning_rate=3e-4,     # Learning rate
        n_steps=128,            # Steps per update (smaller for SB2)
        nminibatches=4,         # Number of minibatches
        noptepochs=4,          # Number of optimization epochs
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,       # GAE parameter
        cliprange=0.2,         # PPO clipping parameter
        tensorboard_log="./ppo2_malmo_tensorboard/"
    )
    
    # Create callback
    callback = LoggingCallback()
    
    # Train the agent
    model.learn(
        total_timesteps=50000,
        callback=callback
    )
    
    # Save the trained model
    model.save("ppo2_malmo_zombie")
    print("PPO2 model saved as 'ppo2_malmo_zombie'")
    
    return model

def train_sac_agent():
    """Train a SAC agent using Stable-Baselines 2"""
    print("Training SAC agent with Stable-Baselines 2...")
    
    env = DummyVecEnv([create_wrapped_env])
    
    model = SAC(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,      # Replay buffer size (smaller for SB2)
        batch_size=64,          # Batch size
        tau=0.005,             # Soft update coefficient
        gamma=0.99,            # Discount factor
        tensorboard_log="./sac_malmo_tensorboard/"
    )
    
    callback = LoggingCallback()
    
    model.learn(
        total_timesteps=50000,
        callback=callback
    )
    
    model.save("sac_malmo_zombie")
    print("SAC model saved as 'sac_malmo_zombie'")
    
    return model

def test_trained_agent(model_path, algorithm="PPO2", num_episodes=5):
    """Test a trained agent"""
    print("Testing {} agent...".format(algorithm))
    
    # Load the trained model
    if algorithm == "PPO2":
        model = PPO2.load(model_path)
    elif algorithm == "SAC":
        model = SAC.load(model_path)
    else:
        raise ValueError("Unsupported algorithm")
    
    # Create environment for testing
    env = create_wrapped_env()
    
    for episode in range(num_episodes):
        print("Test Episode {}".format(episode + 1))
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < 200:  # Max steps per episode
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            print("Step {}: Action = {}, Reward = {:.2f}".format(step_count, action, reward))
            print("Observation: {}".format(obs))
        
        print("Episode {} finished. Total reward: {:.2f}, Steps: {}\n".format(
            episode + 1, total_reward, step_count))

# Main training script
if __name__ == "__main__":
    # Choose algorithm
    algorithm = "PPO2"  # or "SAC"
    
    if algorithm == "PPO2":
        # Train PPO2 agent
        trained_model = train_ppo2_agent()
        
        # Test the trained agent
        test_trained_agent("ppo2_malmo_zombie", "PPO2", num_episodes=3)
        
    elif algorithm == "SAC":
        # Train SAC agent
        trained_model = train_sac_agent()
        
        # Test the trained agent
        test_trained_agent("sac_malmo_zombie", "SAC", num_episodes=3)
    
    print("Training and testing completed!")