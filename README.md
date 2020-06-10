# Meta Learning Shared Hierarchies in PyTorch

## Description

This repo is an implementation of [Meta Learning Shared Hierarchies](https://arxiv.org/abs/1710.09767) using pytorch.

MLSH is a hierchical model that solves the problem of finding a set of low-level motor primitives that enable the high-level master policy to be learned quickly for each task sampled from a task distribution

The architecture of MLSH is illustrated in the graph below. It consits of a high-level policy which is parameterized by $\theta$ and needs to be learned from strach per-task and a set of low-level policies parameterized by $\{\phi_i\}$ that are shared between all tasks and held fixed at test time. During rollout, high-level policy selects a low-level policy to activate, then for the next $N$ time step, actions are taken accroding to the output of the activated low-level policy.

![](assets/mlsh-arch.png)

MLSH trains the agent by reinitializing the high-level policy whenever a new task is sampled from the given distribution. It then goes through a warmup period in which updates only high-level policy then a joint update period where both high-level and low-level policies are updated. It repeats the above process untill convergence.

The full algorithm is as follows:
![](assets/mlsh_algo.png)



## Installation
1. Clone the [original repo](https://github.com/openai/mlsh) and copy gym from it then run `pip install -e .` inside that directory
1. Install `test_envs` following the instructions from [original repo](https://github.com/openai/mlsh)

## Dependices

- pytorch
- wandb
- mujoco-py
- Python3.6
- pyglet 1.3.1

## Running experiments

cd into `code` and run
```
python train.py -W 60 -U 1 --env MovementBandits-v0
```
Change `-W` to adjust warm-up period length and `-U` to adjust joint update period. Use `--env` to specify the environment to run experiment on, `AntBandits-v1` and `MovementBandits-v0` are supported. More options and usages can be find `train.py`  

## Results
- MovementBandit <br>
  - In this environment, an agent is placed in a world and shown the positions of two randomly placed points. The agent may take discrete actions to move in the four cardinal directions, or opt to stay still. One of the two points is marked as correct, although the agent does not receive information on which one it is. The agent receives a reward of 1 if it is within a certain distance of the correct point, and a reward of 0 otherwise.
  - Learning curves
    |Task 1|Task 0|Both Task|
    |------|------|---------|
    |![](assets/1_trained_reward_mb.png)|![](assets/0_trained_reward_mb.png)|![](assets/trained_reward_mb.png)|
  - Videos (after training for 500 episodes) <br>
    **high-level policy initialized**: Video generaged when high-level policy is randomly initialized<br/>
    **high-level policy trained**: Video generated after warm-up period when high-level policy is trained to convergence.
    | Task| high-level policy initialized| high-level policy trained| 
    |-----|------------------------------|--------------------------|
    |Yellow|<img src="assets/pretrain-video-0.gif" width="175">|<img src="assets/after-warmup-video-0.gif" width="175">|
    |Purple|<img src="assets/pretrain-video-1.gif" width="175">|<img src="assets/after-warmup-video-1.gif" width="175">|
- AntBandits <br>
  - In this environment, ant must maneuver towards red goal point, either towards the top or towards the right.
  - Learning curves
    |Task [5 0]|Task [0 5]|Both Task|
    |----------|----------|---------|
    |![](assets/50_trained_reward.png)|![](assets/05_trained_reward.png)|![](assets/trained_reward_ab.png)|
  - Videos (after training for 50 episodes)<br>
    **high-level policy initialized**: Video generaged when high-level policy is randomly initialized<br/>
    **high-level policy trained**: Video generated after warm-up period when high-level policy is trained to convergence.
    | Task| high-level policy initialized| high-level policy trained| 
    |-----|------------------------------|--------------------------|
    |Right|<img src="assets/pretrain-video-50.gif" width="175">|<img src="assets/warmup-video-50.gif" width="175">|
    |Up   |<img src="assets/pretrain-video-05.gif" width="175">|<img src="assets/warmup-video-05.gif" width="175">|


## References
- [Meta Learning Shared Hierarchies](https://arxiv.org/abs/1710.09767)
- [Original repo of MLSH](https://github.com/openai/mlsh)