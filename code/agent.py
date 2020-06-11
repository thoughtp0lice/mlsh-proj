import torch
import torch.nn as nn
import numpy as np
import wandb
import rollout_memory
import mlsh_util
import policy


class mlshAgent:
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
        self.high = policy.DiscPolicy(input_size, num_low, memory_capacity, hlr)
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
                    policy.DiscPolicy(input_size, output_size, memory_capacity, llr)
                )
            else:
                self.low.append(
                    policy.ContPolicy(
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
        self.high = policy.DiscPolicy(
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
