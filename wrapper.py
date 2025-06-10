"""
Reinforcement Learning Wrapper for Malmo Zombie Environment

This module provides wrappers and training utilities for the Malmo zombie fighting environment.
It includes reward shaping, logging callbacks, and training functions for PPO2 and SAC agents.
"""

import gym
import numpy as np
from stable_baselines import PPO2, SAC
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback
from stable_baseline_env import MalmoZombieEnv

# Constants
APPROACH_BONUS = 0.3
RETREAT_PENALTY = -0.1
CLOSE_DISTANCE_BONUS = 3.0
MEDIUM_DISTANCE_BONUS = 1.5
FAR_DISTANCE_BONUS = 0.5
CLOSE_ATTACK_BONUS = 5.0
MEDIUM_ATTACK_BONUS = 1.0
FAR_ATTACK_PENALTY = -1.0
FACE_DIRECT_BONUS = 1.0
FACE_ROUGH_BONUS = 0.5
FACE_SOME_BONUS = 0.2
NOT_FACE_PENALTY = -0.1
ZOMBIE_KILL_BONUS = 100.0
CONTINUE_SUCCESS_BONUS = 10.0
NO_ZOMBIE_PENALTY = -0.5
TIME_PENALTY = -0.01

VERY_CLOSE = 2.0
CLOSE = 4.0
MODERATE = 6.0
CLOSE_ATTACK = 2.5
MEDIUM_ATTACK = 4.0

DIRECT_ANGLE = 30
ROUGH_ANGLE = 60
SOME_ANGLE = 90

# Register environment
gym.envs.registration.register(
    id='MalmoZombie-v0',
    entry_point='stable_baseline_env:MalmoZombieEnv'
)

class ZombieRewardWrapper(gym.Wrapper):
    """
    Wrapper that adds meaningful rewards to the zombie environment.
    
    This wrapper implements a sophisticated reward function that encourages
    the agent to approach zombies, face them, and attack at appropriate distances.
    """
    
    def __init__(self, env):
        """Initialize the reward wrapper."""
        super().__init__(env)
        self.prev_zombie_distance = None
        self.steps_taken = 0
        self.zombie_was_present = False
        self.zombie_killed = False
        
    def step(self, action):
        """Execute one step in the environment with custom reward calculation."""
        obs, reward, done, info = self.env.step(action)
        self.steps_taken += 1
        
        # Calculate our custom reward
        custom_reward = self._calculate_reward(obs, action)
        
        return obs, custom_reward, done, info
    
    def _calculate_reward(self, obs, action):
        reward = 0.0
        
        if obs[5] == 1:  # Zombie present
            self.zombie_was_present = True
            
            # Extract data
            agent_yaw = obs[1]
            zombie_pos = obs[2:5]
            distance = np.linalg.norm(zombie_pos)
            
            # Reward for approaching zombie
            reward += self._calculate_distance_reward(distance)
            
            # Attack rewards
            reward += self._calculate_attack_reward(action, distance)
            
            # Bonus for facing zombie
            reward += self._calculate_facing_reward(agent_yaw, zombie_pos)
            
        else:
            # No zombie visible
            reward += self._calculate_no_zombie_reward()
            
        # Time penalty
        reward += TIME_PENALTY
        
        return reward
    
    def _calculate_distance_reward(self, distance):
        """Calculate reward based on distance to zombie."""
        reward = 0.0
        
        if self.prev_zombie_distance is not None:
            if distance < self.prev_zombie_distance:
                reward += APPROACH_BONUS
            elif distance > self.prev_zombie_distance:
                reward += RETREAT_PENALTY
        
        self.prev_zombie_distance = distance
        
        # Distance-based rewards
        if distance < VERY_CLOSE:
            reward += CLOSE_DISTANCE_BONUS
        elif distance < CLOSE:
            reward += MEDIUM_DISTANCE_BONUS
        elif distance < MODERATE:
            reward += FAR_DISTANCE_BONUS
        
        return reward
    
    def _calculate_attack_reward(self, action, distance):
        """Calculate reward for attack actions."""
        reward = 0.0
        
        if action[2] > 0.5:  # Attacking
            if distance < CLOSE_ATTACK:
                reward += CLOSE_ATTACK_BONUS
            elif distance < MEDIUM_ATTACK:
                reward += MEDIUM_ATTACK_BONUS
            else:
                reward += FAR_ATTACK_PENALTY
        
        return reward
    
    def _calculate_facing_reward(self, agent_yaw, zombie_pos):
        """Calculate reward for facing the zombie."""
        zombie_x, zombie_z = zombie_pos[0], zombie_pos[2]
        
        # Calculate angle to zombie
        angle_to_zombie = np.degrees(np.arctan2(-zombie_x, zombie_z))
        
        # Normalize angles to [0, 360)
        angle_to_zombie = (angle_to_zombie + 360) % 360
        agent_yaw_normalized = (agent_yaw + 360) % 360
        
        # Calculate angular difference
        angle_diff = abs(angle_to_zombie - agent_yaw_normalized)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Reward for facing zombie
        if angle_diff < DIRECT_ANGLE:
            return FACE_DIRECT_BONUS
        elif angle_diff < ROUGH_ANGLE:
            return FACE_ROUGH_BONUS
        elif angle_diff < SOME_ANGLE:
            return FACE_SOME_BONUS
        else:
            return NOT_FACE_PENALTY
    
    def _calculate_no_zombie_reward(self):
        """Calculate reward when no zombie is visible."""
        if self.zombie_was_present and not self.zombie_killed:
            # Zombie disappeared after being present - likely killed!
            print("ZOMBIE KILLED! Maximum reward!")
            self.zombie_killed = True
            return ZOMBIE_KILL_BONUS
        elif not self.zombie_was_present:
            # Haven't found zombie yet
            return NO_ZOMBIE_PENALTY
        elif self.zombie_killed:
            # Zombie already killed, maintain high reward
            return CONTINUE_SUCCESS_BONUS
        
        return 0.0
    
    def reset(self):
        """Reset the wrapper state."""
        self.prev_zombie_distance = None
        self.steps_taken = 0
        self.zombie_was_present = False
        self.zombie_killed = False
        return self.env.reset()


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        """Log episode information when episode ends."""
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
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        nminibatches=4,
        noptepochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        cliprange=0.2,
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
    """Train a SAC agent using Stable-Baselines 2."""
    print("Training SAC agent with Stable-Baselines 2...")
    
    env = DummyVecEnv([create_wrapped_env])
    
    model = SAC(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
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
    """
    Test a trained agent.
    
    Args:
        model_path: Path to the saved model
        algorithm: Algorithm type ("PPO2" or "SAC")
        num_episodes: Number of episodes to test
    """
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


def main():
    """Main training script."""
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


if __name__ == "__main__":
    main()