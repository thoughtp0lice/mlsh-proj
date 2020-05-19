import time
import argparse
import random
import test_envs
import gym
from gym import wrappers
import torch
import numpy as np
import wandb
from pyvirtualdisplay import Display
import policy
import mlsh_util


def rollout(env, agent, N, T, high_len, gamma, lam):
    agent.forget()
    reward = 0
    for i in range(N):
        # reset env while keep the same task
        realgoal = env.env.realgoal
        env.reset()
        env.env.realgoal = realgoal
        reward += agent.high_rollout(env, T, high_len, gamma, lam)
    wandb.log({"reward": reward / N})


def save_files(agent):
    agent.save()
    wandb.save("../policy")
    wandb.save("train.py")
    wandb.save("rollout_memory.py")


if __name__ == "__main__":
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()     
    
    time_stamp = str(int(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", default=200, type=int)
    parser.add_argument("-W", default=60, type=int)
    parser.add_argument("-U", default=1, type=int)
    parser.add_argument("--tasks", default=5000, type=int)
    parser.add_argument("-K", default=50, type=int)
    parser.add_argument("-T", default=50, type=int)
    parser.add_argument("--high_len", default=10, type=int)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--lam", default=0.95, type=float)
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--c1", default=0.5, type=float)
    parser.add_argument("--c2", default=1e-3, type=float)
    parser.add_argument("--c2_low", default=2, type=float)
    parser.add_argument("--display", default=10, type=int)
    parser.add_argument("--record", default=1, type=int)
    parser.add_argument("--seed", default=12345, type=int)
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
    K = args.K
    # Horizon
    T = args.T
    # master policy last for
    high_len = args.high_len
    # batch size
    batch_size = args.bs
    # learning rate
    lr = args.lr
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
            "lr": lr,
            "seed": seed,
        },
        name="mlsh-" + time_stamp,
    )

    agent = policy.HierPolicy(6, 5, N * T, 2, lr)
    for i in range(num_tasks):
        print("Current task num:", i)
        env.reset()
        env.env.randomizeCorrect()
        realgoal = env.env.realgoal
        agent.high_init()
        if i % record == 0:
            record_env = wrappers.Monitor(
                env, "../mlsh_videos/run-%s/task-%d" % (time_stamp, i)
            )
            agent.forget()
            record_env.reset()
            record_env.env.env.realgoal = realgoal
            agent.high_rollout(record_env, T, high_len, gamma, lam, record=True)

        for _ in range(W):
            rollout(env, agent, N, T, high_len, gamma, lam)
            for _ in range(K):
                agent.warmup_optim_step(epsilon, gamma, batch_size, c1, c2)

        if i % record == 0:
            record_env = wrappers.Monitor(
                env, "../mlsh_videos/run-%s/task-%d" % (time_stamp, i)
            )
            agent.forget()
            record_env.reset()
            record_env.env.env.realgoal = realgoal
            agent.high_rollout(record_env, T, high_len, gamma, lam, record=True)

        for _ in range(U):
            rollout(env, agent, N, T, high_len, gamma, lam)
            for _ in range(K):
                agent.joint_optim_step(epsilon, gamma, batch_size, c1, c2, c2_low)

        if i % record == 0:
            agent.forget()
            record_env.reset()
            record_env.env.env.realgoal = realgoal
            agent.high_rollout(record_env, T, high_len, gamma, lam, record=True)
