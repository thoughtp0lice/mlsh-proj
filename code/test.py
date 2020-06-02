import test_envs
import gym
import numpy as np
import mlsh_util
import policy
import torch


agent = torch.load("./agent.pt")
env = gym.make("AntBandits-v1")
print(env.observation_space.shape[0])
total_reward = 0
print(agent.rollout_render(env, 50, 10))
env.close()
