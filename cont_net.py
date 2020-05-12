"""Actor and Critic network of PPO continous"""
import torch
import torch.nn as nn
from torch.distributions import normal
import mlsh_util

# out put mean and standard deviation
class Actor(nn.Module):
    def __init__(self, input_size, output_size, action_scale):
        super(Actor, self).__init__()
        # mean
        self.mean_fc1 = nn.Linear(input_size, 64)
        self.mean_fc2 = nn.Linear(64, 64)
        self.mean_fc3 = nn.Linear(64, 32)
        self.mean_fc4 = nn.Linear(32, 32)
        self.mean_fc5 = nn.Linear(32, output_size)

        # std
        self.std_fc1 = nn.Linear(input_size, 16)
        self.std_fc2 = nn.Linear(16, 16)
        self.std_fc3 = nn.Linear(16, 16)
        self.std_fc4 = nn.Linear(16, output_size)

        nn.init.orthogonal_(self.mean_fc1.weight)
        nn.init.orthogonal_(self.mean_fc2.weight)
        nn.init.orthogonal_(self.mean_fc3.weight)
        nn.init.orthogonal_(self.mean_fc4.weight)
        nn.init.orthogonal_(self.mean_fc5.weight)
        nn.init.orthogonal_(self.std_fc1.weight)
        nn.init.orthogonal_(self.std_fc2.weight)
        nn.init.orthogonal_(self.std_fc3.weight)
        nn.init.orthogonal_(self.std_fc4.weight)

        self.s = torch.tensor([action_scale]).float()

    def forward(self, x):
        mean = torch.relu(self.mean_fc1(x))
        mean = torch.relu(self.mean_fc2(mean))
        mean = torch.relu(self.mean_fc3(mean))
        mean = torch.relu(self.mean_fc4(mean))
        mean = self.mean_fc5(mean)

        std = torch.relu(self.std_fc1(x))
        std = torch.relu(self.std_fc2(std))
        std = torch.relu(self.std_fc3(std))
        std = torch.clamp(torch.exp(self.std_fc4(std)), 1e-9, 1e10)
        return mean, std

    # select a action
    # return action and log probability
    def action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
        # mean and standard deviation of distribution
        mean, std = self.forward(state)
        dist = normal.Normal(mean, std)
        raw_a = dist.sample()
        a = self.s * torch.tanh(raw_a)
        log_p_a = mlsh_util.get_cont_prob(mean, std, raw_a, self.s).sum()
        return a.tolist(), log_p_a.detach(), raw_a.tolist()

    # tensor for all the mean and tensor for all the standard deviation
    def policy_out(self, state):
        mean, std = self.forward(state)
        return mean, std


# value function
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.orthogonal_(self.fc4.weight)
        nn.init.orthogonal_(self.fc5.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def delta(self, s1, s2, r, gamma):
        return r + gamma * self.forward(s2).view(-1) - self.forward(s1).view(-1)
