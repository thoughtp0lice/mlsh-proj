# Meta Learning Shared Hierarchies in PyTorch

## Description

This repo is an implementation of [Meta Learning Shared Hierarchies](https://arxiv.org/abs/1710.09767) using pytorch.

## Installation

1. Copy gym from the [original repo](https://github.com/openai/mlsh) of MLSH and run `pip install -e .` inside that directory
2. Install `test_envs` following the instructions from [original repo](https://github.com/openai/mlsh)

## Dependices

- pytorch
- wandb
- mujoco-py
- Python3.6

## Running experiments

cd into `code` and run
```
python train.py -W 60 -U 1 --env MovementBandit-v1
```
Change `-W` to adjust warm-up period length and `-U` to adjust joint update period. Use `--env` to specify the environment to run experiment on. More options and usages can be find `train.py` 

# Results
- MovementBandit
  - Learning curve
    ![curve_on_MB.png](./assets/curve_on_MB.png)
  - Videos
    | Task| Pretrain| After Warm-up| 
    |-----|---------|------------|
    |    0|![](assets/pretrain-video-0.gif)|![](assets/after-warmup-video-0.gif)|
    |    1|![](assets/pretrain-video-1.gif)|![](assets/after-warmup-video-1.gif)|


## References
- [Meta Learning Shared Hierarchies](https://arxiv.org/abs/1710.09767)