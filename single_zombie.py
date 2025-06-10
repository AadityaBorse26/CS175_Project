import gym
from stable_baseline_env import MalmoZombieEnv

gym.envs.registration.register(
    id='MalmoZombie-v0',
    entry_point='stable_baseline_env:MalmoZombieEnv'
)

def test_env(num_episodes=2):
    env = gym.make('MalmoZombie-v0')
    
    for episode in range(num_episodes):
        print("Starting episode " + str(episode + 1))
        obs = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        
        if episode == num_episodes - 1:
            print("Episode " + str(episode + 1) + " finished. Done.")
            break
        else:
            print("Episode " + str(episode + 1) + " finished. Restarting mission...")

if __name__ == "__main__":
    test_env()