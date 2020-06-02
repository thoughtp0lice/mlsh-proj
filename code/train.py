import time
import pickle
import argparse
import random
import atexit
import test_envs
import gym
from gym import wrappers
import torch
import numpy as np
import wandb
from pyvirtualdisplay import Display
import policy
import mlsh_util


def rollout(env, agent, N, T, high_len, gamma, lam, test=False):
    agent.forget()
    reward = 0
    action = 0
    for i in range(N):
        env.reset()
        r, a = agent.high_rollout(env, T, high_len, gamma, lam)
        reward += r
        action += a
    agent.normalize_adv()
    if not test:
        wandb.log(
            {
                "reward": reward / N,
                "action": (action * high_len) / (T * N),
                "current_task": env.env.realgoal,
            }
        )
    return reward / N, (action * high_len) / (T * N)


def save_files():
    wandb.save("train.py")
    wandb.save("rollout_memory.py")
    wandb.save("cont_net.py")
    wandb.save("disc_net.py")
    wandb.save("mlsh_util.py")
    wandb.save("policy.py")
    wandb.save("agent.pt")


def save_agent(agent):
    torch.save(agent, "agent.pt")


def load_agent(file_name="agent.pt"):
    return torch.load(file_name)


if __name__ == "__main__":
    time_stamp = str(int(time.time()))

    parser = argparse.ArgumentParser()
    # num of envs
    parser.add_argument("-N", default=40, type=int)
    # warm up length
    parser.add_argument("-W", default=60, type=int)
    # joint training
    parser.add_argument("-U", default=1, type=int)
    # number of tasks
    parser.add_argument("--tasks", default=5000, type=int)
    # number of optimization epochs
    parser.add_argument("-K", default=10, type=int)
    # Horizon
    parser.add_argument("-T", default=50, type=int)
    # master policy last for
    parser.add_argument("--high_len", default=10, type=int)
    # batch size
    parser.add_argument("--bs", default=64, type=int)
    # learning rate low
    parser.add_argument("--llr", default=3e-4, type=float)
    # learning rate high
    parser.add_argument("--hlr", default=1e-2, type=float)
    # decay
    parser.add_argument("--gamma", default=0.99, type=float)
    # GAE prameters
    parser.add_argument("--lam", default=0.95, type=float)
    # clipping
    parser.add_argument("--epsilon", default=0.2, type=float)
    # parameters in loss fuction
    parser.add_argument("--c1", default=0.5, type=float)
    parser.add_argument("--c2", default=0, type=float)
    parser.add_argument("--c2_low", default=0, type=float)
    # record step
    parser.add_argument("--record", default=1, type=int)
    # random seed
    parser.add_argument("--seed", default=626, type=int)
    # number of low level policies
    parser.add_argument("--num_low", default=2, type=int)
    # name of the environment
    parser.add_argument("--env", default="MovementBandits-v0", type=str)
    # continue training
    parser.add_argument("-c", action="store_true")
    # virutal display setting
    parser.add_argument("--virdis", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = gym.make(args.env)
    env.seed(args.seed)

    if args.virdis:
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    wandb.init(
        config={
            "num of actors": args.N,
            "tasks": args.tasks,
            "W": args.W,
            "U": args.U,
            "high_len": args.high_len,
            "epochs": args.K,
            "Horizon": args.T,
            "batch size": args.bs,
            "decay": args.gamma,
            "GAE prameters": args.lam,
            "clipping": args.epsilon,
            "c1": args.c1,
            "c2": args.c2,
            "c2_low": args.c2_low,
            "llr": args.llr,
            "hlr": args.hlr,
            "seed": args.seed,
            "env": args.env,
        },
        name="mlsh-" + time_stamp,
    )

    save_files()

    from gym import spaces

    action_size = 0
    disc = True
    if isinstance(env.action_space, spaces.Box):
        action_size = env.action_space.shape[0]
        disc = False
    elif isinstance(env.action_space, spaces.Discrete):
        action_size = env.action_space.n

    if args.c:
        agent = load_agent()
    else:
        agent = policy.HierPolicy(
            env.observation_space.shape[0],
            action_size,
            400 * args.T,
            args.num_low,
            args.llr,
            args.hlr,
            disc=disc,
        )

    atexit.register(save_agent, agent)

    # main training loop
    for i in range(args.tasks):
        print("Current task num:", i)
        env.reset()
        # randomize task
        env.env.randomizeCorrect()
        print("Current goal:", env.env.realgoal)
        agent.high_init()

        # log video
        if i % args.record == 0:
            video = agent.rollout_render(env, args.T, args.high_len)
            wandb.log(
                {
                    "pretrain-video-%d"
                    % (env.env.realgoal): wandb.Video(video, fps=24, format="gif")
                }
            )

        # warm up
        for w in range(args.W):
            rollout(env, agent, args.N, args.T, args.high_len, args.gamma, args.lam)
            for _ in range(args.K):
                agent.warmup_optim_epi(
                    args.epsilon, args.gamma, args.bs, args.c1, args.c2
                )

        # log video
        if i % args.record == 0:
            video = agent.rollout_render(env, args.T, args.high_len)
            wandb.log(
                {
                    "after-warmup-video-%d"
                    % (env.env.realgoal): wandb.Video(video, fps=24, format="gif")
                }
            )

        # log reward
        trained_reward, train_action = rollout(
            env, agent, 400, args.T, args.high_len, args.gamma, args.lam, test=True
        )
        wandb.log(
            {
                "trained_reward": trained_reward,
                "trained_action": train_action,
                str(env.env.realgoal) + "_trained_reward": trained_reward,
            }
        )
        print("Trained reward:", trained_reward)

        # joint update
        for _ in range(args.U):
            rollout(env, agent, args.N, args.T, args.high_len, args.gamma, args.lam)
            for _ in range(args.K):
                agent.joint_optim_epi(
                    args.epsilon, args.gamma, args.bs, args.c1, args.c2, args.c2_low
                )

        # log video
        if i % args.record == 0:
            video = agent.rollout_render(env, args.T, args.high_len)
            wandb.log(
                {
                    "after-joint-video-%d"
                    % (env.env.realgoal): wandb.Video(video, fps=24, format="gif")
                }
            )
