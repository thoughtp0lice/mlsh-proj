import torch
import torch.nn as nn
import numpy as np
import wandb
import rollout_memory
import mlsh_util


class DiscPolicy:
    """
    PPO that gives discrete output
    """

    def __init__(self, input_size, output_size, memory_capacity, lr):
        self.actor = DiscNet.Actor(input_size, output_size)
        self.critic = DiscNet.Critic(input_size)
        self.memory = rollout_memory.RolloutMemory(memory_capacity, input_size, 1)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def optim_epi(self, epsilon, batch_size, c1, c2, log="", vclip=False):
        """
        optimize epi for an episode that goes through all the data in memory
        log changes the name of hte log in wandb
        """
        if self.memory.curr == 0 or batch_size == 0:
            return 0

        if batch_size > self.memory.curr:
            batch_size = self.memory.curr

        losses = []

        for data in self.memory.iterate(batch_size):
            (
                prev_s_batch,
                a_batch,
                r_batch,
                post_s_batch,
                prob_batch,
                advantage_batch,
                v_targ,
                v_old,
                done_batch,
            ) = data

            probs = self.actor(prev_s_batch)
            new_prob = mlsh_util.get_disc_prob(probs, a_batch)
            ratio = torch.exp(new_prob - prob_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage_batch
            surr_loss = torch.mean(torch.min(surr1, surr2))

            v_curr = self.critic(prev_s_batch).view(-1)
            if vclip:
                v_targ = v_targ.view(-1).detach()
                v_old = v_old.view(-1).detach()
                v_loss1 = torch.pow(v_curr - v_targ, 2)
                clipped_v = v_old + torch.clamp(v_curr - v_old, -epsilon, epsilon)
                v_loss2 = torch.pow(clipped_v - v_targ, 2)
                v_loss = torch.mean(torch.min(v_loss1, v_loss2))
            else:
                v_targ = v_targ.detach()
                v_loss = torch.mean(torch.pow(v_curr.view(-1) - v_targ.view(-1), 2))

            ent_loss = torch.mean(mlsh_util.entropy_disc(probs))

            self.optimizer.zero_grad()
            loss = -surr_loss + c1 * v_loss - c2 * ent_loss
            loss.backward()
            losses.append(loss.item())

            grad_size = 0
            for param in list(self.actor.parameters()) + list(self.critic.parameters()):
                grad_size += torch.sum(param.grad.data ** 2).item()
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            grad_size = grad_size ** 0.5

            wandb.log(
                {
                    log + "surr_loss": surr_loss,
                    log + "v_loss": v_loss,
                    log + "ent_loss": ent_loss,
                    log + "loss": loss,
                    log + "advantage": torch.mean(advantage_batch),
                    log + "ratio": torch.mean(abs(1 - ratio)),
                    log + "grad_size": grad_size,
                }
            )

        return np.mean(losses)


class DiscNet:
    '''
    Actor and Critics used in DiscPolicy
    '''

    class Actor(nn.Module):
        def __init__(self, input_size, output_size, hidden_size=64, hidden_num=2):
            super().__init__()
            self.output_size = output_size
            self.linears = nn.ModuleList([nn.Linear(input_size, hidden_size)])
            for i in range(hidden_num):
                self.linears.append(nn.Linear(hidden_size, hidden_size))
            self.linears.append(nn.Linear(hidden_size, output_size))

            for l in self.linears:
                nn.init.orthogonal_(l.weight)

        def forward(self, x):
            for i in range(len(self.linears) - 1):
                x = torch.relu(self.linears[i](x))
            x = self.linears[-1](x)
            x = x.view(-1, self.output_size)
            x = torch.softmax(x, dim=1)
            return x

        def action(self, state):
            """
            select a action return action and probability and action as raw_a
            """
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state).float()
            probs = self.forward(state).view(-1)
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()
            p_a = probs[a]
            return a.item(), p_a.detach(), a.item()

    class Critic(nn.Module):
        def __init__(self, input_size, hidden_size=64, hidden_num=2):
            super().__init__()
            self.linears = nn.ModuleList([nn.Linear(input_size, hidden_size)])
            for i in range(hidden_num):
                self.linears.append(nn.Linear(hidden_size, hidden_size))
            self.linears.append(nn.Linear(hidden_size, 1))

            for l in self.linears:
                nn.init.orthogonal_(l.weight)

        def forward(self, x):
            for i in range(len(self.linears) - 1):
                x = torch.relu(self.linears[i](x))
            x = self.linears[-1](x)
            return x

        def delta(self, s1, s2, r, done, gamma):
            nonterminal = 1 - done.float()
            return (
                r
                + gamma * self.forward(s2).view(-1) * nonterminal
                - self.forward(s1).view(-1)
            )


class ContPolicy:
    '''
    PPO that gives continous output
    '''

    def __init__(self, input_size, output_size, action_scale, memory_capacity, lr):
        self.actor = ContNet.Actor(input_size, output_size, action_scale)
        self.critic = ContNet.Critic(input_size)
        self.memory = rollout_memory.RolloutMemory(
            memory_capacity, input_size, output_size
        )
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def optim_epi(
        self, epsilon, batch_size, c1, c2, log="", vclip=False,
    ):
        """
        optimize epi for an episode that goes through all the data in memory
        """
        if self.memory.curr == 0 or batch_size == 0:
            return 0

        losses = []
        for data in self.memory.iterate(batch_size):
            (
                prev_s_batch,
                a_batch,
                r_batch,
                post_s_batch,
                prob_batch,
                advantage_batch,
                v_targ,
                v_old,
                done_batch,
            ) = data

            y, d = self.actor.policy_out(prev_s_batch)
            new_prob = mlsh_util.get_cont_prob(y, d, a_batch, self.actor.s).sum(axis=1)
            ratio = torch.exp(new_prob - prob_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage_batch
            surr_loss = torch.mean(torch.min(surr1, surr2))
            v_curr = self.critic(prev_s_batch).view(-1)
            if vclip:
                v_targ = v_targ.view(-1).detach()
                v_old = v_old.view(-1).detach()
                v_loss1 = torch.pow(v_curr - v_targ, 2)
                clipped_v = v_old + torch.clamp(v_curr - v_old, -epsilon, epsilon)
                v_loss2 = torch.pow(clipped_v - v_targ, 2)
                v_loss = torch.mean(torch.min(v_loss1, v_loss2))
            else:
                v_targ = v_targ.detach()
                v_loss = torch.mean(torch.pow(v_curr.view(-1) - v_targ.view(-1), 2))

            ent_loss = torch.mean(mlsh_util.entropy_cont(y, d))

            self.optimizer.zero_grad()
            loss = -surr_loss + c1 * v_loss - c2 * ent_loss
            loss.backward()
            losses.append(loss.item())

            grad_size = 0
            for param in list(self.actor.parameters()) + list(self.critic.parameters()):
                grad_size += torch.sum(param.grad.data ** 2).item()
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            grad_size = grad_size ** 0.5

            wandb.log(
                {
                    log + "surr_loss": surr_loss,
                    log + "v_loss": v_loss,
                    log + "ent_loss": ent_loss,
                    log + "loss": loss,
                    log + "advantage": torch.mean(advantage_batch),
                    log + "ratio": torch.mean(abs(1 - ratio)),
                    log + "grad_size": grad_size,
                }
            )

            return np.mean(losses)


class ContNet:
    '''
    Actor and critic used in ContPolicy
    '''
    class Actor(nn.Module):
        def __init__(
            self, input_size, output_size, action_scale, hidden_num=3, hidden_size=64
        ):
            super().__init__()
            # mean
            self.output_size = output_size
            self.mean_linears = nn.ModuleList([nn.Linear(input_size, hidden_size)])
            for i in range(hidden_num):
                self.mean_linears.append(nn.Linear(hidden_size, hidden_size))
            self.mean_linears.append(nn.Linear(hidden_size, output_size))

            # std
            self.output_size = output_size
            self.std_linears = nn.ModuleList([nn.Linear(input_size, hidden_size)])
            for i in range(hidden_num):
                self.std_linears.append(nn.Linear(hidden_size, hidden_size))
            self.std_linears.append(nn.Linear(hidden_size, output_size))

            for l in self.mean_linears:
                nn.init.orthogonal_(l.weight)
            for l in self.std_linears:
                nn.init.orthogonal_(l.weight)

            self.s = torch.tensor([action_scale]).float()

        def forward(self, x):
            std = mean = x
            for i in range(len(self.mean_linears) - 1):
                mean = torch.relu(self.mean_linears[i](mean))
            mean = self.mean_linears[i](mean)

            for i in range(len(self.std_linears) - 1):
                std = torch.relu(self.std_linears[i](std))
            std = torch.clamp(torch.exp(self.std_linears[-1](std)), 1e-9, 1e10)
            return mean, std

        def action(self, state):
            """
            select a action return clipped action and log probability
            and raw action output by the network
            """
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state).float()
            # mean and standard deviation of distribution
            mean, std = self.forward(state)
            dist = torch.distributions.normal.Normal(mean, std)
            raw_a = dist.sample()
            a = self.s * torch.tanh(raw_a)
            log_p_a = mlsh_util.get_cont_prob(mean, std, raw_a, self.s).sum()
            return a.tolist(), log_p_a.detach(), raw_a.tolist()

        def policy_out(self, state):
            """
            tensor for all the mean and tensor for all the standard deviation
            """
            mean, std = self.forward(state)
            return mean, std

    # value function
    class Critic(nn.Module):
        def __init__(self, input_size, hidden_num=3, hidden_size=64):
            super().__init__()
            self.linears = nn.ModuleList([nn.Linear(input_size, hidden_size)])
            for i in range(hidden_num):
                self.linears.append(nn.Linear(hidden_size, hidden_size))
            self.linears.append(nn.Linear(hidden_size, 1))

        def forward(self, x):
            for i in range(len(self.linears) - 1):
                x = torch.relu(self.linears[i](x))
            x = self.linears[-1](x)
            return x

        def delta(self, s1, s2, r, done, gamma):
            nonterminal = 1 - done.float()
            return (
                r
                + gamma * self.forward(s2).view(-1) * nonterminal
                - self.forward(s1).view(-1)
            )
