import torch
import torch.nn as nn
from torch.distributions import normal
from torch.distributions import Categorical
import mlsh_util

# out put mean and standard deviation
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        # mean
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        if output_size == 2:
            self.output_size = 1
        else:
            self.output_size = output_size
        self.fc4 = nn.Linear(64, self.output_size)

        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.orthogonal_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.output_size)
        if self.output_size == 1:
            x = torch.sigmoid(x)
            x = torch.cat([x, 1 - x], dim=1)
        else:
            x = torch.softmax(x, dim=1)
        return x

    # select a action
    # return action and probability
    def action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
        probs = self.forward(state).view(-1)
        dist = Categorical(probs=probs)
        a = dist.sample()
        p_a = probs[a]
        return a.item(), p_a.detach(), a.item()


# value function
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.orthogonal_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def delta(self, s1, s2, r, done, gamma):
        nonterminal = 1 - done.float()
        return r + gamma * self.forward(s2).view(-1) * nonterminal - self.forward(s1).view(-1)
