import time
import argparse
import random
import atexit
import torch
import numpy as np
import wandb
from pyvirtualdisplay import Display
import test_envs
import gym
import agent


def rollout(env, agent, N, T, high_len, gamma, lam, test=False):
    """
    rollout on env for N episodes that last for T time steps
    """
    agent.forget()
    reward = 0
    action = 0
    for _ in range(N):
        r, a = agent.high_rollout(env, T, high_len, gamma, lam)
        reward += r
        action += a
    agent.normalize_adv()
    if not test:
        wandb.log({"reward": reward / N, "action": (action * high_len) / (T * N)})
    return reward / N, (action * high_len) / (T * N)


def save_agent(agent):
    """
    save current agent
    """
    torch.save(agent, "agent.pt")


def load_agent(file_name="agent.pt"):
    """
    load saved agent
    """
    return torch.load(file_name)


if __name__ == "__main__":
    time_stamp = str(int(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N", default=50, type=int, help="num of episodes for each rollout"
    )
    parser.add_argument(
        "-W", default=20, type=int, help="arm up length"
    )
    parser.add_argument(
        "-U", default=40, type=int, help="joint training length"
    )
    parser.add_argument(
        "--tasks", default=2000, type=int, help="number of tasks"
    )
    parser.add_argument(
        "-K", default=10, type=int, help="number of optimization epochs"
    )
    parser.add_argument(
        "-T", default=300, type=int, help="horizon"
    )
    parser.add_argument(
        "--high_len", default=60, type=int, help="master action length"
    )
    parser.add_argument(
        "--bs", default=64, type=int, help="batch size"
    )
    parser.add_argument(
        "--llr", default=3e-4, type=float, help="low-level policy learning rate"
    )
    parser.add_argument(
        "--hlr", default=1e-2, type=float, help="high-level policy learning rate"
    )
    parser.add_argument(
        "--gamma", default=0.99, type=float, help="decay factor"
    )
    parser.add_argument(
        "--lam", default=0.95, type=float, help="GAE prameter"
    )
    parser.add_argument(
        "--epsilon", default=0.2, type=float, help="clipping parameter"
    )
    parser.add_argument(
        "--c1", default=0.5, type=float, help="critic loss parameter"
    )
    parser.add_argument(
        "--c2", default=0, type=float, help="entropy loss parameter for high-level policy",
    )
    parser.add_argument(
        "--c2_low", default=0, type=float, help="entropy loss for low level policy"
    )
    parser.add_argument(
        "--record", default=1, type=int, help="num of tasks between each record"
    )
    parser.add_argument(
        "--seed", default=626, type=int, help="random seed"
    )
    parser.add_argument(
        "--num_low", default=2, type=int, help="number of low level policies"
    )
    parser.add_argument(
        "--env", default="AntBandits-v1", type=str, help="name of the environment"
    )
    parser.add_argument(
        "-c", action="store_true", help="continue training"
    )
    parser.add_argument(
        "--virdis", action="store_true", help="set virutal display"
    )

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
        config=args, name="mlsh-" + time_stamp,
    )

    # find action space shape
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
        agent = agent.mlshAgent(
            env.observation_space.shape[0],
            action_size,
            args.N * args.T,
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
        # randomly initialize high-level policy
        agent.high_init()

        # log video when high-level policy is just initialized
        if i % args.record == 0:
            video = agent.rollout_render(env, args.T, args.high_len)
            wandb.log(
                {
                    "pretrain-video-%s"
                    % (str(env.env.realgoal)): wandb.Video(video, fps=60, format="gif")
                }
            )

        # warm up
        for w in range(args.W):
            # rollout for N episodes all memories are stored in agent.memory
            rollout(env, agent, args.N, args.T, args.high_len, args.gamma, args.lam)
            for _ in range(args.K):
                # update high-level policy only
                agent.warmup_optim_epi(args.epsilon, args.bs, args.c1, args.c2)

        # log video when high-level policy is updated
        if i % args.record == 0:
            video = agent.rollout_render(env, args.T, args.high_len)
            wandb.log(
                {
                    "warmup-video-%s"
                    % (str(env.env.realgoal)): wandb.Video(video, fps=60, format="gif")
                }
            )

        # log reward when high-level policy is updated
        trained_reward, train_action = rollout(
            env, agent, args.N, args.T, args.high_len, args.gamma, args.lam, test=True
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
                # update both high and low-level policy
                agent.joint_optim_epi(
                    args.epsilon, args.bs, args.c1, args.c2, args.c2_low
                )

        # log video after low-level policy is trained
        if i % args.record == 0:
            video = agent.rollout_render(env, args.T, args.high_len)
            wandb.log(
                {
                    "afterjoint-video-%s"
                    % (str(env.env.realgoal)): wandb.Video(video, fps=60, format="gif")
                }
            )
