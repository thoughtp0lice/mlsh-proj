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
        # reset env while keep the same task
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


def save_agent(agent):
    with open("agent.p", "wb") as outfile:
        pickle.dump(agent, outfile)


def load_agent(file_name="agent.p"):
    with open(file_name, "rb") as infile:
        return pickle.load(infile)


if __name__ == "__main__":
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()

    time_stamp = str(int(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", default=40, type=int)
    parser.add_argument("-W", default=60, type=int)
    parser.add_argument("-U", default=1, type=int)
    parser.add_argument("--tasks", default=5000, type=int)
    parser.add_argument("-K", default=10, type=int)
    parser.add_argument("--K2", default=10, type=int)
    parser.add_argument("-T", default=50, type=int)
    parser.add_argument("--high_len", default=10, type=int)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--llr", default=3e-4, type=float)
    parser.add_argument("--hlr", default=1e-2, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--lam", default=0.95, type=float)
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--c1", default=0.5, type=float)
    parser.add_argument("--c2", default=0, type=float)
    parser.add_argument("--c2_low", default=0, type=float)
    parser.add_argument("--display", default=10, type=int)
    parser.add_argument("--record", default=1, type=int)
    parser.add_argument("--seed", default=626, type=int)
    parser.add_argument("-c", action="store_true")  # continue training

    args = parser.parse_args()

    # num of envs
    N = args.N
    # number of tasks
    num_tasks = args.tasks
    # warm up length
    W = args.W
    # joint training
    U = args.U
    # number of optimization epochs
    # warm up
    K = args.K
    # joint
    K2 = args.K2
    # Horizon
    T = args.T
    # master policy last for
    high_len = args.high_len
    # batch size
    batch_size = args.bs
    # learning rate
    llr = args.llr
    hlr = args.hlr
    # decay
    gamma = args.gamma
    # GAE prameters
    lam = args.lam
    # clipping
    epsilon = args.epsilon
    # parameters in loss fuction
    c1 = args.c1
    c2 = args.c2
    c2_low = args.c2_low
    # display step
    display = args.display
    # record step
    record = args.record
    # random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make("MovementBandits-v0")
    env.seed(seed)

    wandb.init(
        config={
            "num of actors": N,
            "tasks": num_tasks,
            "W": W,
            "U": U,
            "high_len": high_len,
            "epochs": K,
            "Horizon": T,
            "batch size": batch_size,
            "decay": gamma,
            "GAE prameters": lam,
            "clipping": epsilon,
            "c1": c1,
            "c2": c2,
            "c2_low": c2_low,
            "llr": llr,
            "hlr": hlr,
            "seed": seed,
        },
        name="mlsh-" + time_stamp,
    )

    save_files()
    agent = policy.HierPolicy(6, 5, 400 * T, 2, llr, hlr)
    if args.c:
        agent = load_agent()
    atexit.register(save_agent, agent)

    for i in range(num_tasks):
        print("Current task num:", i)
        env.reset()
        env.env.randomizeCorrect()
        # env.env.realgoal = i % 2
        print("Current goal:", env.env.realgoal)
        agent.high_init()

        # log video
        if i % record == 0:
            video = agent.rollout_render(env, T, high_len)
            wandb.log(
                {
                    "pretrain-video-%d"
                    % (env.env.realgoal): wandb.Video(video, fps=24, format="gif")
                }
            )

        # warm up
        for w in range(W):
            rollout(env, agent, N, T, high_len, gamma, lam)
            for _ in range(K):
                agent.warmup_optim_epi(epsilon, gamma, batch_size, c1, c2)

        # log video
        if i % record == 0:
            video = agent.rollout_render(env, T, high_len)
            wandb.log(
                {
                    "after-warmup-video-%d"
                    % (env.env.realgoal): wandb.Video(video, fps=24, format="gif")
                }
            )

        # log reward
        trained_reward, train_action = rollout(
            env, agent, 400, T, high_len, gamma, lam, test=True
        )
        wandb.log({"trained_reward": trained_reward, "trained_action": train_action})
        if env.env.realgoal == 0:
            wandb.log({"0_trained_reward": trained_reward})
        else:
            wandb.log({"1_trained_reward": trained_reward})
        print("Trained reward:", trained_reward)

        # joint update
        for _ in range(U):
            rollout(env, agent, N, T, high_len, gamma, lam)
            for _ in range(K2):
                agent.joint_optim_epi(epsilon, gamma, batch_size, c1, c2, c2_low)

        # log video
        if i % record == 0:
            video = agent.rollout_render(env, T, high_len)
            wandb.log(
                {
                    "after-joint-video-%d"
                    % (env.env.realgoal): wandb.Video(video, fps=24, format="gif")
                }
            )
