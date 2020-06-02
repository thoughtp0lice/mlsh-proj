import test_envs
import gym
import numpy as np
import mlsh_util

env = gym.make("MovementBandits4-v0")
total_reward = 0
for i in range(100):
    goals = env.env.goals
    obs = env.reset()
    for k in range(50):
        env.render()
        obs, reward, done, _ = env.step(np.random.choice(5, 1))
        total_reward += reward
    if i % 10 == 0:
        print(obs)

        
print(total_reward)
env.close()