import pytest
param = pytest.mark.parametrize

import torch

def test_quasimetric_distance():
    from MQE.MQE import quasimetric_distance

    x = torch.randn(32)
    y = torch.randn(32)

    dist = quasimetric_distance(x, y)
    assert dist.numel() == 1

def test_mqe():
    assert True
