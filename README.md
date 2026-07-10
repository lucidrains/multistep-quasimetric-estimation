<img src="./mqe.png" width="400px"></img>

## Multistep Quasimetric Estimation - (wip)

Exploration and eventually practical implementation for the [Multistep Quasimetric Estimation](https://arxiv.org/abs/2511.07730) proposed by Zheng et al. of Berkeley.

This paper is a coming together of a few ideas: quasimetric distance spaces and successor representations, along with an action invariance loss and other loss designs

## Install

```bash
pip install MQE
```

## Usage

```python
import torch
from torch import nn

from MQE import MQE, MRN, Policy, ContinuousAction
from x_mlps_pytorch import MLP

state_dim, action_dim = 16, 4

mrn = MRN(
    sym_network = MLP(32, 64),
    asym_network = MLP(32, 64),
    distance_groups = 8
)

mqe = MQE(
    state_encoder = MLP(state_dim, 32),
    state_action_encoder = MLP(state_dim + action_dim, 32),
    metric_residual_network = mrn
)

policy = Policy(
    action_dim = action_dim,
    dim = 32,
    state_encoder = MLP(state_dim, 32),
    goal_encoder = MLP(state_dim, 32),
    action_dist = ContinuousAction()
)

states = torch.randn(4, 10, state_dim)
actions = torch.randn(4, 10, action_dim)
goals = torch.randn(4, 10, state_dim)

# train critic from offline trajectories

critic_loss, _ = mqe(states, actions, goals)

critic_loss.backward()

# train actor using critic

policy_loss, _ = mqe.extract_policy(
    policy,
    states,
    actions,
    goals,
    bc_loss_weight = 0.1
)

policy_loss.backward()

# inference

action = policy(states[:, 0], goals[:, 0]).sample() # (4, 4)
```

## Citations

```bibtex
@misc{zheng2026multistepquasimetriclearningscalable,
    title   = {Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning},
    author  = {Bill Chunyuan Zheng and Vivek Myers and Benjamin Eysenbach and Sergey Levine},
    year    = {2026},
    eprint  = {2511.07730},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2511.07730},
}
```

```bibtex
@misc{liu2023metricresidualnetworkssample,
    title   = {Metric Residual Networks for Sample Efficient Goal-Conditioned Reinforcement Learning},
    author  = {Bo Liu and Yihao Feng and Qiang Liu and Peter Stone},
    year    = {2023},
    eprint  = {2208.08133},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2208.08133},
}
```
