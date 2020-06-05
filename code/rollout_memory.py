"""rollout memory for PPO continous"""
import torch
import numpy as np


class RolloutMemory:
    def __init__(self, capacity, state_size, action_size):
        self.curr = 0
        self.iter_curr = 0
        self.advantage_memory = torch.zeros(capacity)
        self.prob_memory = torch.zeros(capacity)
        self.prev_state_memory = torch.zeros(capacity, state_size)
        self.action_memory = torch.zeros(capacity, action_size)
        self.reward_memory = torch.zeros(capacity)
        self.post_state_memory = torch.zeros(capacity, state_size)
        self.done_memory = torch.zeros(capacity)
        self.v_targ = torch.zeros(capacity)
        self.v_old = torch.zeros(capacity)
        self.capacity = capacity
        self.state_size = state_size
        self.action_size = action_size

    def clear(self):
        self.__init__(self.capacity, self.state_size, self.action_size)

    def put_batch(
        self,
        prev_state,
        action,
        prob,
        reward,
        post_state,
        advantages,
        v_targ,
        v_old,
        done,
    ):
        size = len(prev_state)
        self.prob_memory[self.curr : self.curr + size] = prob
        self.action_memory[self.curr : self.curr + size][:] = action
        self.prev_state_memory[self.curr : self.curr + size][:] = prev_state
        self.post_state_memory[self.curr : self.curr + size][:] = post_state
        self.reward_memory[self.curr : self.curr + size] = reward
        self.advantage_memory[self.curr : self.curr + size] = advantages
        self.v_targ[self.curr : self.curr + size] = v_targ
        self.v_old[self.curr : self.curr + size] = v_old
        self.done_memory[self.curr : self.curr + size] = done
        self.curr += size

    def iterate(self, size):
        """
        generater through the dataset with a batch of size [size] in each return
        """
        self.iter_curr = 0
        shuffle = np.random.permutation(self.curr)

        self.advantage_memory[: self.curr] = self.advantage_memory[shuffle]
        self.prob_memory[: self.curr] = self.prob_memory[shuffle]
        self.prev_state_memory[: self.curr] = self.prev_state_memory[shuffle]
        self.action_memory[: self.curr] = self.action_memory[shuffle]
        self.reward_memory[: self.curr] = self.reward_memory[shuffle]
        self.post_state_memory[: self.curr] = self.post_state_memory[shuffle]
        self.done_memory[: self.curr] = self.done_memory[shuffle]
        self.v_targ[: self.curr] = self.v_targ[shuffle]
        self.v_old[: self.curr] = self.v_old[shuffle]

        while (self.curr - size) >= self.iter_curr:
            yield (
                self.prev_state_memory[self.iter_curr : self.iter_curr + size],
                self.action_memory[self.iter_curr : self.iter_curr + size],
                self.reward_memory[self.iter_curr : self.iter_curr + size],
                self.post_state_memory[self.iter_curr : self.iter_curr + size],
                self.prob_memory[self.iter_curr : self.iter_curr + size],
                self.advantage_memory[self.iter_curr : self.iter_curr + size],
                self.v_targ[self.iter_curr : self.iter_curr + size],
                self.v_old[self.iter_curr : self.iter_curr + size],
                self.done_memory[self.iter_curr : self.iter_curr + size],
            )
            self.iter_curr += size

    def normalize_adv(self):
        """
        call to normalize the advantages stroed in the data set
        """
        advs = self.advantage_memory[: self.curr]
        self.advantage_memory[: self.curr] = (advs - advs.mean()) / max(
            advs.std(), 0.000001
        )
