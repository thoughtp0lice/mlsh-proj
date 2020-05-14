"""rollout memory for PPO continous"""
import torch
import numpy as np


class RolloutMemory:
    def __init__(self, capacity, state_size, action_size):
        self.curr = 0
        self.advantage_memory = torch.zeros(capacity)
        self.prob_memory = torch.zeros(capacity)
        self.prev_state_memory = torch.zeros(capacity, state_size)
        self.action_memory = torch.zeros(capacity, action_size)
        self.reward_memory = torch.zeros(capacity)
        self.post_state_memory = torch.zeros(capacity, state_size)
        self.done_memory = torch.zeros(capacity)
        self.v_targ = torch.zeros(capacity)
        self.capacity = capacity
        self.state_size = state_size
        self.action_size = action_size

    def clear(self):
        self.__init__(self.capacity, self.state_size, self.action_size)

    def put_batch(
        self, prev_state, action, prob, reward, post_state, advantages, v_targ, done
    ):
        size = len(prev_state)
        self.prob_memory[self.curr : self.curr + size] = prob
        self.action_memory[self.curr : self.curr + size][:] = action
        self.prev_state_memory[self.curr : self.curr + size][:] = prev_state
        self.post_state_memory[self.curr : self.curr + size][:] = post_state
        self.reward_memory[self.curr : self.curr + size] = reward
        self.advantage_memory[self.curr : self.curr + size] = advantages
        self.v_targ[self.curr : self.curr + size] = v_targ
        self.done_memory[self.curr : self.curr + size] = done
        self.curr += size

    def get_batch(self, size):
        if size > self.curr:
            size = self.curr
        out = np.random.choice(self.curr, size, replace=False)
        return (
            self.prev_state_memory[out],
            self.action_memory[out],
            self.reward_memory[out],
            self.post_state_memory[out],
            self.prob_memory[out],
            self.advantage_memory[out],
            self.v_targ[out],
            self.done_memory[out],
        )