import pytest
param = pytest.mark.parametrize

import torch
import torch.nn.functional as F

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

def test_mrn():
    from x_mlps_pytorch import MLP
    from MQE import MetricResidualNetwork

    dim_state = 10
    dim_action = 2
    dim_goal = 10

    mrn = MetricResidualNetwork(
        encoders = [MLP(dim_state + dim_action, 32, 16), MLP(dim_goal, 32, 16)],
        sym_network = MLP(16, 32),
        asym_network = MLP(16, 32)
    )

    state = torch.randn(10)
    actions = torch.rand(2)
    goal = torch.randn(10)

    distance = mrn((state, actions), goal)
    distance.backward()

def test_mqe():
    from x_mlps_pytorch import MLP
    from MQE import MQE, MRN

    dim_state = 10
    dim_action = 2
    dim_goal = 10

    mrn = MRN(
        encoders = [MLP(dim_state + dim_action, 32, 16), MLP(dim_goal, 32, 16)],
        sym_network = MLP(16, 32),
        asym_network = MLP(16, 32)
    )

    mqe = MQE(mrn)

    states = torch.randn(4, 10)
    actions = torch.rand(4, 2)
    goals = torch.randn(4, 10)

    distance, action_inv_loss = mqe(states, actions, goals, return_action_invariance_loss=True)

    assert distance.shape == (4,)
    assert action_inv_loss.numel() == 1

    loss = distance.mean() + action_inv_loss
    loss.backward()
