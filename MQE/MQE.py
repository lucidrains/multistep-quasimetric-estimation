from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, reduce

from x_mlps_pytorch import create_mlp

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

# quasimetric distance

def default_sym_fn(x, y):
    return (x - y).norm(p = 2, dim = -1)

def default_asym_fn(x, y):
    return (x - y).relu().amax(dim = -1)

def quasimetric_distance(
    x, y,
    asym_x = None,
    asym_y = None,
    *,
    sym_fn = default_sym_fn,
    asym_fn = default_asym_fn,
    groups = 8, # the paper splits the representation into N equally sized components (Table 2 uses 8)
    reduce_groups = True
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

    asym = asym_fn(asym_x, asym_y)

    # eq (4)

    distance = sym + asym

    # maybe not reduce, for action invariance loss

    if not reduce_groups:
        return distance

    # average

    return reduce(distance, '... g -> ...', 'mean')

# metric residual network
# https://arxiv.org/abs/2208.08133

class MetricResidualNetwork(Module):
    def __init__(
        self,
        *,
        encoders: tuple[Module, Module],
        sym_network: Module,
        asym_network: Module,
        distance_groups = 8
    ):
        super().__init__()

        # encoders - would be state-action and state-goal for MEQ

        self.encoders = ModuleList(list(encoders))

        # the two network backbones, producing inputs for symmetric and asymmetric half of quasimetric distance

        self.sym_network = sym_network
        self.asym_network = asym_network

        # distance related

        self.distance_groups = distance_groups

    def forward(
        self,
        *encoder_inputs,
        calc_reverse_distance = False,
        reduce_groups = True
    ):
        maybe_reverse = reversed if calc_reverse_distance else identity
        encoded = [fn(inputs) for fn, inputs in maybe_reverse(list(zip(self.encoders, encoder_inputs)))]

        sym_x, sym_y = [self.sym_network(t) for t in encoded]
        asym_x, asym_y = [self.asym_network(t) for t in encoded]

        return quasimetric_distance(sym_x, sym_y, asym_x, asym_y, groups = self.distance_groups, reduce_groups = reduce_groups)

# classes

class MultistepQuasimetricEstimation(Module):
    def __init__(
        self,
        metric_residual_network: MetricResidualNetwork,
    ):
        super().__init__()
        self.mrn = metric_residual_network

    def calc_action_invariance_loss(
        self,
        states,
        actions
    ):
        dist_action_invariance = self.mrn((states, actions), states, calc_reverse_distance = True, reduce_groups = False) # (... g)

        # section 4.2

        return F.mse_loss(dist_action_invariance.neg().exp(), torch.ones_like(dist_action_invariance))

    def forward(
        self,
        states,
        actions,
        goals,
        return_action_invariance_loss = False
    ):

        distance = self.mrn((states, actions), goals)

        if not return_action_invariance_loss:
            return distance

        loss_action_invariance = self.calc_action_invariance_loss(states, actions)

        return distance, loss_action_invariance

# shorthand

MRN = MetricResidualNetwork
MQE = MultistepQuasimetricEstimation
