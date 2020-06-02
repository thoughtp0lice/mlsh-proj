import test_envs
import gym
import numpy as np
import mlsh_util

env = gym.make("MovementBandits4-v0")
total_reward = 0
for i in range(100):
    goals = env.env.goals
    env.env.randomizeCorrect()
    obs = env.reset()
    for k in range(50):
        obs, reward, done, _ = env.step(np.random.choice(5, 1))
        total_reward += reward
    print(env.env.realgoal)

        
print(total_reward)
env.close()