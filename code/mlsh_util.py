"""utility functions"""
import torch
from torch.distributions import normal
from torch.distributions import Categorical


def entropy_cont(y, d):
    return normal.Normal(y, d).entropy()


def entropy_disc(distribution):
    return Categorical(probs=distribution).entropy()


def get_v_targ(r_batch, gamma):
    out = torch.zeros(len(r_batch))
    for i in range(len(r_batch)):
        curr = len(r_batch) - i - 1
        if i == 0:
            out[curr] = r_batch[curr]
        else:
            out[curr] = r_batch[curr] + gamma * out[curr + 1]
    return out


def get_disc_prob(prob, a_batch):
    return prob.gather(1, a_batch.long()).view(-1)


def get_cont_prob(y, d, raw_a_batch, scale):
    a_batch = scale * torch.tanh(raw_a_batch)
    out = (
        torch.log((1 / scale))
        - 2 * torch.log(1 - (a_batch / scale) ** 2 + 1e-6)
        + normal.Normal(y, d).log_prob(raw_a_batch)
    )
    return out


def advantage(t, deltas, gamma, lam):
    curr_t = len(deltas) - 1
    out = 0.0
    while curr_t >= t:
        out = lam * gamma * out
        out += deltas[curr_t]
        curr_t -= 1
    return out.item()