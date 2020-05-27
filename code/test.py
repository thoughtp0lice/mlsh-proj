import test_envs
import gym
import numpy as np
import mlsh_util

env = gym.make("MovementBandits-v0")
total_reward = 0
rms = mlsh_util.RunningMeanStd(6)
for i in range(100):
    goals = env.env.goals
    obs = env.reset()
    obs = rms.filter(obs)
    for k in range(50):
        obs, reward, done, _ = env.step(np.random.choice(5, 1))
        total_reward += reward
        obs = rms.filter(obs)
    if i % 10 == 0:
        print(obs)
        print(rms.mean, rms.var)

        
print(total_reward)
env.close()