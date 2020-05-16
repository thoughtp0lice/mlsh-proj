import test_envs
import gym
import numpy as np

env = gym.make("MovementBandits-v0")
total_reward = 0
for i in range(5):
    goals = env.env.goals
    env.reset()
    #env.env.env.goals = goals
    for k in range(50):
        env.render()
        obs, reward, done, _ = env.step(np.random.choice(5, 1))
        total_reward += reward
        print(env.env.obs())
        
print(total_reward)
env.close()