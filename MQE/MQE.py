from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, reduce

from x_mlps_pytorch import create_mlp

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# quasimetric distance

def default_sym_fn(x, y):
    return (x - y).norm(p = 2, dim = -1)

def default_sym_fn(x, y):
    return (x - y).relu().amax(dim = -1)

def quasimetric_distance(
    x, y,
    asym_x = None,
    asym_y = None,
    *,
    sym_fn = default_sym_fn,
    asym_fn = default_sym_fn,
    groups = 8 # the paper splits the representation into N equally sized components (Table 2 uses 8)
):
    dim_embed = x.shape[-1]

    asym_x, asym_y = default(asym_x, x), default(asym_y, y)

    assert x.shape == y.shape == asym_x.shape == asym_y.shape
    assert divisible_by(dim_embed, groups)

    # separate out groups

    x, y, asym_x, asym_y = (rearrange(t, '... (g d) -> ... g d', g = groups) for t in (x, y, asym_x, asym_y))

    # symmetric

    sym = sym_fn(x, y)

    # asymmetric

    asym = asym_fn(x, y)

    # eq (4)

    distance = sym + asym

    # average

    return reduce(distance, '... g -> ...', 'mean')

# metric residual network
# https://arxiv.org/abs/2208.08133

class MetricResidualNetwork(Module):
    def __init__(
        self,
        *,
        encoders: list[Module],
        sym_network: Module,
        asym_network: Module,
        distance_groups = 8
    ):
        super().__init__()

        # encoders - would be state-action and state-goal for MEQ

        self.encoders = ModuleList(encoders)

        # the two network backbones, producing inputs for symmetric and asymmetric half of quasimetric distance

        self.sym_network = sym_network
        self.asym_network = asym_network

        # distance related

        self.distance_groups = distance_groups

    def forward(
        self,
        *encoder_inputs
    ):
        encoded = [fn(inputs) for fn, inputs in zip(self.encoders, encoder_inputs)]

        sym_x, sym_y = [self.sym_network(t) for t in encoded]
        asym_x, asym_y = [self.asym_network(t) for t in encoded]

        dist = quasimetric_distance(sym_x, sym_y, asym_x, asym_y, groups = self.distance_groups)

        return -dist

# classes

class MQE(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self,
        state
    ):
        return state
