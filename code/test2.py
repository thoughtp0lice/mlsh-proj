import policy
import mlsh_util
import torch
import time
import wandb
import test_envs
import gym
from gym import wrappers
from pyvirtualdisplay import Display
import numpy as np

def rollout(N, env, T, p, gamma, lam):
    total_rewards = []
    for i in range(N):
        total_reward = 0
        advantages = []
        probs = []
        prev_states = []
        actions = []
        rewards = []
        post_states = []
        dones = []

        done = False
        env.reset()
        env.env.realgoal = 0
        post_state = env.env.obs()
        for i in range(T):
            prev_state = post_state
            action, prob, raw_a = p.actor.action(prev_state)
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
        actions = torch.Tensor(actions).reshape(-1, p.memory.action_size)
        rewards = torch.Tensor(rewards)
        post_states = torch.Tensor(post_states)
        dones = torch.Tensor(dones)

        deltas = p.critic.delta(prev_states, post_states, rewards, gamma)
        for t in range(len(deltas)):
            advantages.append(mlsh_util.advantage(t, deltas, gamma, lam))
        advantages = torch.Tensor(advantages)

        v_targ = mlsh_util.get_v_targ(rewards, gamma)
        p.memory.put_batch(
            prev_states, actions, probs, rewards, post_states, advantages, v_targ, dones
        )
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

if __name__ == "__main__":
    time_stamp = str(int(time.time()))

    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()

    # num of actors
    N = 100
    # number of episodes
    iterations = 10000
    # number of optimization epochs
    K = 50
    # Horizon
    T = 50
    # batch size
    batch_size = 64
    # learning rate
    lr = 1e-4
    # decay
    gamma = 0.99
    # GAE prameters
    lam = 0.95
    # clipping
    epsilon = 0.2
    # parameters in loss fuction
    c1 = 1
    c2 = 1
    # display step
    display = 10

    wandb.init(
        config={
            "num of actors": N,
            "iterations": iterations,
            "epochs": K,
            "Horizon": T,
            "batch size": batch_size,
            "decay": gamma,
            "GAE prameters": lam,
            "clipping": epsilon,
            "c1": c1,
            "c2": c2,
        },
        name="PPO-" + time_stamp,
    )

    env = gym.make("MovementBandits-v0")
    env.reset()
    p = policy.DiscPolicy(6, 5, N*T, lr)
    optimizer = torch.optim.Adam(
        list(p.actor.parameters()) + list(p.critic.parameters()), lr=lr
    )

    for iter in range(iterations):
        env.reset()
        p.memory.clear()
        reward = rollout(N, env, T, p, gamma, lam)
        wandb.log({"reward": reward})
        for i in range(K):
            loss = p.optim_step(
                epsilon, gamma, batch_size, c1, c2, bootstrap=True
            )
        if iter % display == 0 and iter != 0:
            print(
                "Iteration %d Loss = %.3f reward = %.3f" % (iter, loss, reward)
            )
        if iter % 200 == 0:
            p.memory.clear()
            record_env = wrappers.Monitor(env, './ppo_videos/run-%s/epi-%d'%(time_stamp,iter))
            record_env.reset()
            observation = record_env.env.env.obs()
            done = False
            x = 0
            while not done:
                x += 1
                action, prob, raw_a = p.actor.action(observation)
                observation, reward, done, info = record_env.step(action)
            record_env.close()
