from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, reduce

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# quasimetric distance

def quasimetric_distance(
    x, y,
    groups = 8 # the paper splits the representation into N equally sized components (Table 2 uses 8)
):
    dim_embed = x.shape[-1]

    assert x.shape == y.shape
    assert divisible_by(dim_embed, groups)

    # separate out groups

    x, y = (rearrange(t, '... (g d) -> ... g d', g = groups) for t in (x, y))

    # diff

    diff = x - y

    # symmetric

    sym = diff.norm(p = 2, dim = -1)

    # asymmetric

    asym = diff.relu().amax(dim = -1)

    # eq (4)

    distance = sym + asym

    # average

    return reduce(distance, '... g -> ...', 'mean')

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
