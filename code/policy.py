import torch
import numpy as np
import wandb
import cont_net
import disc_net
import rollout_memory
import mlsh_util


class HierPolicy:
    def __init__(
        self,
        input_size,
        output_size,
        memory_capacity,
        num_low,
        lr,
        disc=True,
        action_scale=1.0,
    ):
        self.high = DiscPolicy(input_size, num_low, memory_capacity, lr)
        self.low = []
        self.input_size = input_size
        self.num_low = num_low
        self.memory_capacity = memory_capacity
        self.lr = lr
        for i in range(num_low):
            if disc:
                self.low.append(
                    DiscPolicy(input_size, output_size, memory_capacity, lr)
                )
            else:
                self.low.append(
                    ContPolicy(
                        input_size, output_size, action_scale, memory_capacity, lr
                    )
                )

    def save(self):
        torch.save(self.high.actor.state_dict(), "../policy/high/model_actor_cont.pt")
        torch.save(self.low.critic.state_dict(), "../policy/high/model_critic_cont.pt")
        for i, low_p in enumerate(self.low):
            torch.save(
                low_p.actor.state_dict(), "../policy/low/model_actor_cont%r.pt" % i
            )
            torch.save(
                low_p.critic.state_dict(), "../policy/low/model_critic_cont%r.pt" % i
            )

    def forget(self):
        self.high.memory.clear()
        for low_p in self.low:
            low_p.memory.clear()

    def high_init(self):
        self.high = DiscPolicy(
            self.input_size, self.num_low, self.memory_capacity, self.lr
        )

    def warmup_optim_step(self, epsilon, gamma, batch_size, c1, c2):
        self.high.optim_step(epsilon, gamma, batch_size, c1, c2, log=True)

    def joint_optim_step(self, epsilon, gamma, batch_size, c1, c2):
        self.high.optim_step(epsilon, gamma, batch_size, c1, c2, log=True)
        for low_p in self.low:
            low_p.optim_step(epsilon, gamma, batch_size, c1, c2)

    def high_rollout(self, env, T, high_len, gamma, lam, render=False, record=False):
        total_reward = 0
        advantages = []
        probs = []
        prev_states = []
        actions = []
        rewards = []
        post_states = []
        dones = []

        curr_steps = 0
        if record:
            post_state = env.env.env.obs()
        else:
            post_state = env.env.obs()
        while curr_steps < T:
            prev_state = post_state
            action, prob, raw_a = self.high.actor.action(prev_state)
            post_state, r, done = self.low_rollout(
                env, action, high_len, gamma, lam, render=render, record=record
            )
            probs.append(prob)
            prev_states.append(prev_state)
            post_states.append(post_state)
            actions.append(raw_a)
            rewards.append(r)
            dones.append(done)
            total_reward += r
            curr_steps += high_len
            if done:
                break

        probs = torch.Tensor(probs)
        prev_states = torch.Tensor(prev_states)
        actions = torch.Tensor(actions).reshape(-1, self.high.memory.action_size)
        rewards = torch.Tensor(rewards)
        post_states = torch.Tensor(post_states)
        dones = torch.Tensor(dones)

        deltas = self.high.critic.delta(prev_states, post_states, rewards, gamma)
        for t in range(len(deltas)):
            advantages.append(mlsh_util.advantage(t, deltas, gamma, lam))
        advantages = torch.Tensor(advantages)

        v_targ = mlsh_util.get_v_targ(rewards, gamma)

        self.high.memory.put_batch(
            prev_states, actions, probs, rewards, post_states, advantages, v_targ, dones
        )

        return total_reward

    def low_rollout(
        self, env, action, high_len, gamma, lam, render=False, record=False
    ):
        low_policy = self.low[action]

        total_reward = 0
        advantages = []
        probs = []
        prev_states = []
        actions = []
        rewards = []
        post_states = []
        dones = []

        done = False
        if record:
            post_state = env.env.env.obs()
        else:
            post_state = env.env.obs()
        for i in range(high_len):
            prev_state = post_state
            action, prob, raw_a = low_policy.actor.action(prev_state)
            if render:
                env.render()
            post_state, r, done, _ = env.step(action)
            probs.append(prob)
            prev_states.append(prev_state)
            post_states.append(post_state)
            actions.append(raw_a)
            rewards.append(r)
            dones.append(done)
            total_reward += r
            if done:
                break

        probs = torch.Tensor(probs)
        prev_states = torch.Tensor(prev_states)
        actions = torch.Tensor(actions).reshape(-1, low_policy.memory.action_size)
        rewards = torch.Tensor(rewards)
        post_states = torch.Tensor(post_states)
        dones = torch.Tensor(dones)

        deltas = low_policy.critic.delta(prev_states, post_states, rewards, gamma)
        for t in range(len(deltas)):
            advantages.append(mlsh_util.advantage(t, deltas, gamma, lam))
        advantages = torch.Tensor(advantages)

        v_targ = mlsh_util.get_v_targ(rewards, gamma)
        low_policy.memory.put_batch(
            prev_states, actions, probs, rewards, post_states, advantages, v_targ, dones
        )

        return post_state, total_reward, done


class DiscPolicy:
    def __init__(self, input_size, output_size, memory_capacity, lr):
        self.actor = disc_net.Actor(input_size, output_size)
        self.critic = disc_net.Critic(input_size)
        self.memory = rollout_memory.RolloutMemory(memory_capacity, input_size, 1)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def optim_step(self, epsilon, gamma, batch_size, c1, c2, log=False):
        if self.memory.curr == 0:
            return 0

        (
            prev_s_batch,
            a_batch,
            r_batch,
            post_s_batch,
            prob_batch,
            advantage_batch,
            v_targ,
            done_batch,
        ) = self.memory.get_batch(batch_size)

        probs = self.actor(prev_s_batch)
        new_prob = mlsh_util.get_disc_prob(probs, a_batch)
        ratio = torch.exp(new_prob - prob_batch)
        surr1 = ratio * advantage_batch
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage_batch
        surr_loss = -torch.mean(torch.min(surr1, surr2))

        v_curr = self.critic(prev_s_batch).view(-1)
        v_targ = r_batch + gamma*self.critic(prev_s_batch).view(-1)
        v_loss = c1 * torch.mean(torch.pow(v_curr.view(-1) - v_targ.view(-1), 2))

        ent_loss = -c2 * torch.mean(mlsh_util.entropy_disc(probs))

        self.optimizer.zero_grad()
        loss = surr_loss + v_loss + ent_loss
        loss.backward()

        for param in list(self.actor.parameters()) + list(self.critic.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if log:
            wandb.log(
                {
                    "surr_loss": surr_loss,
                    "v_loss": v_loss / c1,
                    "ent_loss": ent_loss / c2,
                    "loss": loss,
                    "advantage": torch.mean(advantage_batch),
                    "ratio": torch.mean(abs(1 - ratio)),
                }
            )

        return loss


class ContPolicy:
    def __init__(self, input_size, output_size, action_scale, memory_capacity, lr):
        self.actor = cont_net.Actor(input_size, output_size, action_scale)
        self.critic = cont_net.Critic(input_size)
        self.memory = rollout_memory.RolloutMemory(
            memory_capacity, input_size, output_size
        )
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def optim_step(
        self, optimizer, memory, epsilon, gamma, batch_size, c1, c2, log=False
    ):
        if self.memory.curr == 0:
            return 0

        (
            prev_s_batch,
            a_batch,
            r_batch,
            post_s_batch,
            prob_batch,
            advantage_batch,
            v_targ,
            done_batch,
        ) = self.memory.get_batch(batch_size)

        y, d = self.actor.policy_out(prev_s_batch)
        new_prob = mlsh_util.get_cont_prob(y, d, a_batch, self.actor.s).sum(axis=1)
        ratio = torch.exp(new_prob - prob_batch)
        surr1 = ratio * advantage_batch
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage_batch
        surr_loss = -torch.mean(torch.min(surr1, surr2))

        v_curr = self.critic(prev_s_batch).view(-1)
        v_loss = c1 * torch.mean(torch.pow(v_curr.view(-1) - v_targ.view(-1), 2))

        ent_loss = -c2 * torch.mean(mlsh_util.entropy_cont(y, d))

        self.optimizer.zero_grad()
        loss = surr_loss + v_loss + ent_loss
        loss.backward()

        for param in list(self.actor.parameters()) + list(self.critic.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if log:
            wandb.log(
                {
                    "surr_loss": surr_loss,
                    "v_loss": v_loss / c1,
                    "ent_loss": ent_loss / c2,
                    "loss": loss,
                    "advantage": torch.mean(advantage_batch),
                    "ratio": torch.mean(abs(1 - ratio)),
                }
            )

        return loss
