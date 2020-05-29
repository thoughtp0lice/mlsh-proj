import test_envs
import gym
import numpy as np
import mlsh_util
import policy
import pickle

agent = pickle.load(open("agent.p","rb"))
env = gym.make("MovementBandits-v0")
print(env.render(mode='rgb_array'))
total_reward = 0
print(agent.rollout_render(env, 50, 10))
env.close()