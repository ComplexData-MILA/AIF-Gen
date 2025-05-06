# Benchmarks

We adapt several methods from TRL for continual learning, and include the commands to run the algorithms end-to-end using AIF-Gen data. We integrate with *WANDB* and *DeepSpeed* for multi-GPU support.

Scripts are available [here](../benchmarks/)

## Basic Usage

First, you'll need to sync additional dependencies for running the benchmarks. Assuming you did the `uv` install, you can simply issue:

```sh
uv sync --group benchmarks
```

Each algorithm (aside from DPO) involves first training a reward model, then training an RL agent with respect to the learned model.

### Reward Model Training

Under construction.

### Agent Training

Under construction.

### Evaluation

Under construction.

## Algorithms

### PPO

Under construction.

Reference: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

### DPO

Under construction.

Reference: [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

### COPR

Under construction.

Reference: [https://arxiv.org/abs/2402.14228](https://arxiv.org/abs/2402.14228)

### CPPO

Under construction.

Reference: [https://openreview.net/forum?id=86zAUE80pP](https://openreview.net/forum?id=86zAUE80pP)

### EWC

Under construction.

Reference: [https://arxiv.org/abs/1612.00796](https://arxiv.org/abs/1612.00796)
