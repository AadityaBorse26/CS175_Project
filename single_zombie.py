import gym
from stable_baseline_env import MalmoZombieEnv

gym.envs.registration.register(
    id='MalmoZombie-v0',
    entry_point='stable_baseline_env:MalmoZombieEnv'
)

# now create an env instance
env = gym.make('MalmoZombie-v0')

num_episodes = 3  # or any number you want
for episode in range(num_episodes):
    print("Starting episode episode " + str(episode + 1) )
    obs = env.reset()  # starts the mission and gets initial observation
    done = False
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    
    if episode == num_episodes - 1:
        print("Episode " + str(episode + 1) + " finished. Done.")
    else:
        print("Episode " + str(episode + 1) + " finished. Restarting mission...")