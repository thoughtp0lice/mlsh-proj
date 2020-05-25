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
        llr,
        hlr,
        disc=True,
        action_scale=1.0,
    ):
        self.high = DiscPolicy(input_size, num_low, memory_capacity, hlr)
        self.low = []
        self.input_size = input_size
        self.num_low = num_low
        self.memory_capacity = memory_capacity
        self.llr = llr
        self.hlr = hlr
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
            self.input_size, self.num_low, self.memory_capacity, self.hlr
        )

    def warmup_optim_epi(self, epsilon, gamma, batch_size, c1, c2, bootstrap = False):
        self.high.optim_epi(
            epsilon, gamma, batch_size, c1, c2, log="high_", bootstrap = bootstrap
        )

    def joint_optim_epi(self, epsilon, gamma, batch_size, c1, c2, c2_low, num_batch=15, bootstrap = False):
        self.high.optim_epi(
            epsilon, gamma, batch_size, c1, c2, log="high_", bootstrap= bootstrap
        )
        for i, low_p in enumerate(self.low):
            if low_p.memory.curr < num_batch:
                continue
            size = int(low_p.memory.curr / num_batch)
            low_p.optim_epi(
                epsilon, gamma, size, c1, c2_low, log=str(i) + "low_", bootstrap = bootstrap
            )

    def high_rollout(self, env, T, high_len, gamma, lam, render=False, record=False):
        total_reward = 0
        advantages = []
        probs = []
        prev_states = []
        actions = []
        rewards = []
        post_states = []
        dones = []
        low_roll_lens = []

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
            "vpred": torch.tensor([])
        }

        curr_steps = 0

        if record:
            post_state = env.env.env.obs()
        else:
            post_state = env.env.obs()

        while curr_steps < T:
            prev_state = post_state

            action, prob, raw_a = self.high.actor.action(prev_state)
            if np.random.random() < 0.1:
                state = torch.from_numpy(prev_state).float()
                action = np.random.choice(self.num_low)
                prob = self.high.actor(state).view(-1)[action].item()
                raw_a = action

            post_state, r, done, roll_len = self.low_rollout(
                env,
                action,
                high_len,
                gamma,
                lam,
                low_roll,
                render=render,
                record=record,
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
            prev_states, actions, probs, rewards, post_states, advantages, v_targ, vpred, dones
        )

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

    def low_rollout(
        self, env, action, high_len, gamma, lam, low_roll, render=False, record=False
    ):
        low_policy = self.low[action]

        total_reward = 0
        advantages = low_roll["advantages"]
        probs = low_roll["probs"]
        prev_states = low_roll["prev_states"]
        actions = low_roll["actions"]
        rewards = low_roll["rewards"]
        post_states = low_roll["post_states"]
        dones = low_roll["dones"]

        rollout_len = 0

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
            rollout_len += 1
            if done:
                break

        prev_d = torch.tensor(prev_states[-rollout_len:]).float()
        post_d = torch.tensor(post_states[-rollout_len:]).float()
        rewards_d = torch.tensor(rewards[-rollout_len:]).float()
        dones_d = torch.tensor(dones[-rollout_len:]).float()

        deltas = low_policy.critic.delta(prev_d, post_d, rewards_d, dones_d, gamma).view(-1)
        vpred = low_policy.critic(prev_d).view(-1).detach()
        low_roll["deltas"] = torch.cat((low_roll["deltas"], deltas), 0)
        low_roll["vpred"] = torch.cat((low_roll["vpred"], vpred), 0)

        return post_state, total_reward, done, rollout_len


class DiscPolicy:
    def __init__(self, input_size, output_size, memory_capacity, lr):
        self.actor = disc_net.Actor(input_size, output_size)
        self.critic = disc_net.Critic(input_size)
        self.memory = rollout_memory.RolloutMemory(memory_capacity, input_size, 1)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def optim_epi(self, epsilon, gamma, batch_size, c1, c2, log="", bootstrap=False):
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

            advantage_batch = (advantage_batch-torch.mean(advantage_batch))/max(torch.std(advantage_batch), 0.000001)
            probs = self.actor(prev_s_batch)
            new_prob = mlsh_util.get_disc_prob(probs, a_batch)
            ratio = torch.exp(new_prob - prob_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage_batch
            surr_loss = torch.mean(torch.min(surr1, surr2))

            v_curr = self.critic(prev_s_batch).view(-1)
            v_targ = v_targ.view(-1).detach()
            v_old = v_old.view(-1).detach()
            v_loss1 = torch.pow(v_curr - v_targ, 2)
            clipped_v = v_old + torch.clamp(v_curr - v_old, -epsilon, epsilon)
            v_loss2 = torch.pow(clipped_v - v_targ, 2)
            v_loss = torch.mean(torch.min(v_loss1, v_loss2))

            ent_loss = torch.mean(mlsh_util.entropy_disc(probs))

            self.optimizer.zero_grad()
            loss = - surr_loss + c1 * v_loss - c2 * ent_loss
            loss.backward()
            losses.append(loss.item())

            for param in list(self.actor.parameters()) + list(self.critic.parameters()):
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            wandb.log(
                {
                    log + "surr_loss": surr_loss,
                    log + "v_loss": v_loss,
                    log + "ent_loss": ent_loss,
                    log + "loss": loss,
                    log + "advantage": torch.mean(advantage_batch),
                    log + "ratio": torch.mean(abs(1 - ratio)),
                }
            )

        return np.mean(losses)


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

    def optim_epi(
        self,
        optimizer,
        memory,
        epsilon,
        gamma,
        batch_size,
        c1,
        c2,
        log=False,
        bootstrap=False,
    ):
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

            advantage_batch = (advantage_batch-torch.mean(advantage_batch))/max(torch.std(advantage_batch), 0.000001)
            y, d = self.actor.policy_out(prev_s_batch)
            new_prob = mlsh_util.get_cont_prob(y, d, a_batch, self.actor.s).sum(axis=1)
            ratio = torch.exp(new_prob - prob_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage_batch
            surr_loss = torch.mean(torch.min(surr1, surr2))

            v_curr = self.critic(prev_s_batch).view(-1)
            v_targ = v_targ.view(-1).detach()
            v_old = v_old.view(-1).detach()
            v_loss1 = torch.pow(v_curr - v_targ, 2)
            clipped_v = v_old + torch.clamp(v_curr - v_old, -epsilon, epsilon)
            v_loss2 = torch.pow(clipped_v - v_targ, 2)
            v_loss = torch.mean(torch.min(v_loss1, v_loss2))

            ent_loss = torch.mean(mlsh_util.entropy_cont(y, d))

            self.optimizer.zero_grad()
            loss = - surr_loss + c1 * v_loss - c2 * ent_loss
            loss.backward()
            losses.append(loss.item())

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

            return np.mean(losses)
