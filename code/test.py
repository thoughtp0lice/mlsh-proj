import test_envs
import gym
import numpy as np
import mlsh_util
import policy
import torch

env = gym.make("InvertedPendulum-v1")
env.reset()
for i in range(1):
    env.reset()
    for _ in range(5000):
        env.render()
        _, _, done, _ = env.step(env.action_space.sample())  # take a random action
        if done:
            break
        
env.close()
