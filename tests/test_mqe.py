import pytest
param = pytest.mark.parametrize

import torch

def test_quasimetric_distance():
    from MQE.MQE import quasimetric_distance

    x = torch.randn(32)
    y = torch.randn(32)

    dist = quasimetric_distance(x, y)
    assert dist.numel() == 1

def test_mrn():
    from x_mlps_pytorch import MLP
    from MQE import MetricResidualNetwork

    dim_state = 10
    dim_action = 2
    dim_goal = 10

    mrn = MetricResidualNetwork(
        encoders = [MLP(dim_state + dim_action, 32, 16), MLP(dim_state + dim_goal, 32, 16)],
        sym_network = MLP(16, 32),
        asym_network = MLP(16, 32)
    )

    state = torch.randn(10)
    actions = torch.rand(2)
    goal = torch.randn(10)

    distance = mrn((state, actions), (state, goal))
    distance.backward()

def test_mqe():
    assert True
