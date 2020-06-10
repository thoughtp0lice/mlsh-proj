import torch
import torch.nn as nn
import numpy as np
import wandb
import rollout_memory
import mlsh_util


class HierPolicy:
    def __init__(
        self,
        input_size,
        output_size,
        memory_capacity,
        num_low,
        llr,
        hlr,
        disc=True,
        action_scale=1.0,
    ):
        """
        set up a hierchical policy
        use input_size, output_size to set the size of output
        use num_low to set num of low level policy
        llr set the learning rate of lowlevel policies
        hlr set the learning rate of highlevel policies
        set disc to false to enable continous control and true for discrete control
        the range of action in lowlevel policy is (-action_scale, action_scale)
        """
        self.high = DiscPolicy(input_size, num_low, memory_capacity, hlr)
        self.low = []
        self.input_size = input_size
        self.num_low = num_low
        self.memory_capacity = memory_capacity
        self.llr = llr
        self.hlr = hlr
        self.rms = mlsh_util.RunningMeanStd(input_size)
        for i in range(num_low):
            if disc:
                self.low.append(
                    DiscPolicy(input_size, output_size, memory_capacity, llr)
                )
            else:
                self.low.append(
                    ContPolicy(
                        input_size, output_size, action_scale, memory_capacity, llr
                    )
                )

    def forget(self):
        """
        use to clear all the replay buffer
        """
        self.high.memory.clear()
        for low_p in self.low:
            low_p.memory.clear()

    def high_init(self):
        """
        use to clear high level policy and initialize again
        """
        self.high = DiscPolicy(
            self.input_size, self.num_low, self.memory_capacity, self.hlr
        )

    def warmup_optim_epi(self, epsilon, batch_size, c1, c2, vclip=False):
        """
        update only high level policy for one epoch
        set vclip to True to clip v value while optimizing
        """
        self.high.optim_epi(epsilon, batch_size, c1, c2, log="high_", vclip=vclip)

    def normalize_adv(self):
        """normalize stored advantage in all the replay buffers"""
        self.high.memory.normalize_adv()
        for low_p in self.low:
            low_p.memory.normalize_adv()

    def joint_optim_epi(
        self, epsilon, batch_size, c1, c2, c2_low, num_batch=15, vclip=False
    ):
        """
        update all the policies for one epoch
        num_batch set the number of batchs to seperate the memories into
        set vclip to True to clip v value while optimizing
        """
        self.high.optim_epi(epsilon, batch_size, c1, c2, log="high_", vclip=vclip)
        for i, low_p in enumerate(self.low):
            if low_p.memory.curr < num_batch:
                continue
            size = int(low_p.memory.curr / num_batch)
            low_p.optim_epi(epsilon, size, c1, c2_low, log=str(i) + "low_", vclip=vclip)

    def rollout_render(self, env, T, high_len):
        """
        rollot and render agent on env for T time steps
        return a np series of rendered images
        dimensions are time, channels, width, height
        this does not store information in memory
        """
        video = []
        obs = env.reset()
        obs = self.rms.filter(obs)
        video.append(env.render(mode="rgb_array"))
        t = 0
        done = False
        while t < T and not done:
            high_action, _, _ = self.high.actor.action(obs)
            for _ in range(high_len):
                low_action, _, _ = self.low[high_action].actor.action(obs)
                obs, _, done, _ = env.step(low_action)
                env.render()
                obs = self.rms.filter(obs)
                video.append(env.render(mode="rgb_array"))
                t += 1
                if done:
                    break
        return np.transpose(np.array(video), (0, 3, 1, 2))

    def high_rollout(self, env, T, high_len, gamma, lam):
        """
        rollout agent but not render agent for T time step on env
        store all the necessary data in memory
        return total reward and sum of all high level actions
        """
        total_reward = 0
        advantages = []
        probs = []
        prev_states = []
        actions = []
        rewards = []
        post_states = []
        dones = []
        low_roll_lens = []

        # dictionary to store in low rollout data
        low_roll = {
            "advantages": [],
            "probs": [],
            "prev_states": [],
            "actions": [],
            "rewards": [],
            "post_states": [],
            "dones": [],
            "deltas": torch.tensor([]),
            "v_targ": [],
            "vpred": torch.tensor([]),
        }

        curr_steps = 0

        post_state = env.reset()
        post_state = self.rms.filter(post_state)

        while curr_steps < T:
            prev_state = post_state
            action, prob, raw_a = self.high.actor.action(prev_state)
            if np.random.random() < 0.0:
                state = torch.from_numpy(prev_state).float()
                action = np.random.choice(self.num_low)
                prob = self.high.actor(state).view(-1)[action].item()
                raw_a = action

            post_state, r, done, roll_len = self.low_rollout(
                env, prev_state, action, high_len, gamma, low_roll,
            )
            probs.append(prob)
            prev_states.append(prev_state)
            post_states.append(post_state)
            actions.append(raw_a)
            rewards.append(r)
            dones.append(done)
            low_roll_lens.append(roll_len)
            total_reward += r
            curr_steps += high_len
            if done:
                break

        # process and store high data
        probs = torch.Tensor(probs)
        prev_states = torch.Tensor(prev_states)
        actions = torch.Tensor(actions).reshape(-1, self.high.memory.action_size)
        rewards = torch.Tensor(rewards)
        post_states = torch.Tensor(post_states)
        dones = torch.Tensor(dones)
        vpred = self.high.critic(prev_states).view(-1).detach()

        deltas = self.high.critic.delta(prev_states, post_states, rewards, dones, gamma)
        for t in range(len(deltas)):
            advantages.append(mlsh_util.advantage(t, deltas, gamma, lam))
        advantages = torch.Tensor(advantages)

        v_targ = advantages + vpred

        self.high.memory.put_batch(
            prev_states,
            actions,
            probs,
            rewards,
            post_states,
            advantages,
            v_targ,
            vpred,
            dones,
        )

        # process low data
        low_roll["probs"] = torch.Tensor(low_roll["probs"])
        low_roll["prev_states"] = torch.Tensor(low_roll["prev_states"])
        low_roll["actions"] = torch.Tensor(low_roll["actions"]).reshape(
            -1, self.low[0].memory.action_size
        )
        low_roll["rewards"] = torch.Tensor(low_roll["rewards"])
        low_roll["post_states"] = torch.Tensor(low_roll["post_states"])
        low_roll["dones"] = torch.Tensor(low_roll["dones"])

        for t in range(len(low_roll["deltas"])):
            low_roll["advantages"].append(
                mlsh_util.advantage(t, low_roll["deltas"], gamma, lam)
            )
        low_roll["advantages"] = torch.Tensor(low_roll["advantages"])

        low_roll["v_targ"] = (low_roll["advantages"] + low_roll["vpred"]).detach()

        # break up low data and store in each low memories
        curr_t = 0
        for roll_len, high_action in zip(low_roll_lens, actions):
            low_policy = self.low[int(high_action.item())]
            roll_range = slice(curr_t, curr_t + roll_len)
            curr_t += roll_len
            low_policy.memory.put_batch(
                low_roll["prev_states"][roll_range],
                low_roll["actions"][roll_range],
                low_roll["probs"][roll_range],
                low_roll["rewards"][roll_range],
                low_roll["post_states"][roll_range],
                low_roll["advantages"][roll_range],
                low_roll["v_targ"][roll_range],
                low_roll["vpred"][roll_range],
                low_roll["dones"][roll_range],
            )

        return total_reward, np.sum(list(actions))

    def low_rollout(self, env, init_state, action, high_len, gamma, low_roll):
        """
        rollout high_len time steps using low policy selected by action
        store all the data in low_roll
        first three return value same as the env return
        last one returns the number of time steps of this rollout
        """
        low_policy = self.low[action]

        total_reward = 0
        probs = low_roll["probs"]
        prev_states = low_roll["prev_states"]
        actions = low_roll["actions"]
        rewards = low_roll["rewards"]
        post_states = low_roll["post_states"]
        dones = low_roll["dones"]

        rollout_len = 0

        done = False
        post_state = init_state
        for _ in range(high_len):
            prev_state = post_state
            action, prob, raw_a = low_policy.actor.action(prev_state)

            post_state, r, done, _ = env.step(action)
            post_state = self.rms.filter(post_state)

            probs.append(prob)
            prev_states.append(prev_state)
            post_states.append(post_state)
            actions.append(raw_a)
            rewards.append(r)
            dones.append(done)
            total_reward += r
            rollout_len += 1
            if done:
                break

        prev_d = torch.tensor(prev_states[-rollout_len:]).float()
        post_d = torch.tensor(post_states[-rollout_len:]).float()
        rewards_d = torch.tensor(rewards[-rollout_len:]).float()
        dones_d = torch.tensor(dones[-rollout_len:]).float()

        deltas = low_policy.critic.delta(
            prev_d, post_d, rewards_d, dones_d, gamma
        ).view(-1)
        vpred = low_policy.critic(prev_d).view(-1).detach()
        low_roll["deltas"] = torch.cat((low_roll["deltas"], deltas), 0)
        low_roll["vpred"] = torch.cat((low_roll["vpred"], vpred), 0)

        return post_state, total_reward, done, rollout_len


class DiscPolicy:
    def __init__(self, input_size, output_size, memory_capacity, lr):
        """
        implements ppo that gives discrete out put
        """
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


class ContPolicy:
    def __init__(self, input_size, output_size, action_scale, memory_capacity, lr):
        """
        optimize epi for an episode that goes through all the data in memory
        action is in range (-action_scale, action_scale)
        """
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
    class Actor(nn.Module):
        def __init__(self, input_size, output_size, action_scale):
            super().__init__()
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
        def __init__(self, input_size):
            super().__init__()
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

        def delta(self, s1, s2, r, done, gamma):
            nonterminal = 1 - done.float()
            return (
                r
                + gamma * self.forward(s2).view(-1) * nonterminal
                - self.forward(s1).view(-1)
            )


class DiscNet:
    class Actor(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
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

    # value function
    class Critic(nn.Module):
        def __init__(self, input_size):
            super().__init__()
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
            return (
                r
                + gamma * self.forward(s2).view(-1) * nonterminal
                - self.forward(s1).view(-1)
            )
