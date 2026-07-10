import pytest
param = pytest.mark.parametrize

import torch
from torch import nn
import torch.nn.functional as F

def exists(v):
    return v is not None

@param('custom_fn', (False, True))
def test_quasimetric_distance(
    custom_fn
):
    from MQE.MQE import quasimetric_distance

    x = torch.randn(32)
    y = torch.randn(32)

    dist_kwargs = dict()

    if custom_fn:
        dist_kwargs = dict(
            sym_fn = lambda x, y: (x - y).abs().sum(dim = -1),
            asym_fn = lambda x, y: F.softplus(x - y).logsumexp(dim = -1)
        )

    dist = quasimetric_distance(x, y, **dist_kwargs)

    assert dist.numel() == 1

def test_critic():
    from x_mlps_pytorch import MLP
    from MQE import MQE, MRN

    dim_state = 10
    dim_action = 2
    dim_goal = 10

    mrn = MRN(
        sym_network = MLP(16, 32),
        asym_network = MLP(16, 32)
    )

    mqe = MQE(
        state_encoder = MLP(dim_goal, 32, 16),
        state_action_encoder = MLP(dim_state + dim_action, 32, 16),
        metric_residual_network = mrn
    )

    states = torch.randn(4, 10, dim_state)
    actions = torch.rand(4, 10, dim_action)
    goals = torch.randn(4, 10, dim_goal)

    loss, loss_breakdown = mqe(states, actions, goals)
    loss.backward()

def test_policy_discrete():
    from MQE import Policy
    from MQE.MQE import DiscreteAction

    policy = Policy(
        action_dim = 10,
        action_dist = DiscreteAction()
    )

    state = torch.randn(2, 3, 224, 224)
    goal = torch.randn(2, 3, 224, 224)

    dist = policy(state, goal)
    action = dist.sample()

    assert action.shape == (2,)

def test_policy_continuous():
    from MQE import Policy
    from MQE.MQE import ContinuousAction

    policy = Policy(
        action_dim = 4,
        action_dist = ContinuousAction()
    )

    state = torch.randn(2, 3, 224, 224)
    goal = torch.randn(2, 3, 224, 224)

    dist = policy(state, goal)
    action = dist.sample()

    assert action.shape == (2, 4)

def test_policy_extraction():
    from x_mlps_pytorch import MLP
    from MQE import MQE, MRN, ContinuousAction

    dim_state, dim_action = 16, 4

    mrn = MRN(
        sym_network = MLP(16, 32),
        asym_network = MLP(16, 32)
    )

    mqe = MQE(
        state_encoder = MLP(dim_state, 32, 16),
        state_action_encoder = MLP(dim_state + dim_action, 32, 16),
        metric_residual_network = mrn
    )

    class DummyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = MLP(dim_state * 2, dim_action * 2)

        def forward(self, state, goal):
            mean, log_std = self.net(torch.cat((state, goal), dim = -1)).chunk(2, dim = -1)
            return ContinuousAction()(torch.cat((mean, log_std), dim = -1))

    policy = DummyPolicy()

    states, goals = torch.randn(2, 4, dim_state)
    actions = torch.randn(4, dim_action)

    total_loss, _ = mqe.extract_policy(policy, states, actions, goals)

    total_loss.backward()
    assert any(exists(p.grad) for p in policy.parameters())
