from __future__ import annotations
from math import log
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Module, Linear

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# policy related

import torchvision.models as models
from torch.distributions import Categorical, Normal, Beta
from x_mlps_pytorch import FiLMableMLP

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# constants

LinearNoBias = partial(Linear, bias = False)

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

    assert x.shape[-1] == y.shape[-1] == asym_x.shape[-1] == asym_y.shape[-1]
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
        sym_network: Module,
        asym_network: Module,
        distance_groups = 8
    ):
        super().__init__()

        # the two network backbones, producing inputs for symmetric and asymmetric half of quasimetric distance

        self.sym_network = sym_network
        self.asym_network = asym_network

        # distance related

        self.distance_groups = distance_groups

    def forward(
        self,
        encoded_left,
        encoded_right,
        reduce_groups = True
    ):
        encoded = [encoded_left, encoded_right]

        sym_x, sym_y = [self.sym_network(t) for t in encoded]
        asym_x, asym_y = [self.asym_network(t) for t in encoded]

        return quasimetric_distance(sym_x, sym_y, asym_x, asym_y, groups = self.distance_groups, reduce_groups = reduce_groups)

# critic

class Critic(Module):
    def __init__(
        self,
        state_encoder: Module,
        state_action_encoder: Module,
        metric_residual_network: MetricResidualNetwork,
        discount_factor = 0.95,
        action_invariance_loss_weight = 1.,
        paired_loss_weight = 0.5
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.state_action_encoder = state_action_encoder
        self.metric_residual_network = metric_residual_network

        # hyperparameters

        self.discount_factor = discount_factor

        # loss related

        self.action_invariance_loss_weight = action_invariance_loss_weight
        self.paired_loss_weight = paired_loss_weight
        self.has_paired_loss_weight = paired_loss_weight > 0

    def forward(
        self,
        states,
        actions,
        goals,
        waypoints,
        waypoint_dist, # int(b)
    ):
        γ, batch = self.discount_factor, states.shape[0]
        state_encoder, state_action_encoder, metric_residual_network = self.state_encoder, self.state_action_encoder, self.metric_residual_network

        encoded_states = state_encoder(states)
        encoded_state_actions = state_action_encoder((states, actions))
        encoded_waypoints = state_encoder(waypoints)
        encoded_goals = state_encoder(goals)

        # eq (10)

        def linex_loss(d, d_target):
            return (d - d_target).exp() - d

        # eq (11) - cross-batch goals

        encoded_state_actions_i = rearrange(encoded_state_actions, 'i d -> i 1 d')
        encoded_waypoints_i = rearrange(encoded_waypoints, 'i d -> i 1 d')
        encoded_goals_j = rearrange(encoded_goals, 'j d -> 1 j d')

        dist_q_to_goal = metric_residual_network(
            encoded_state_actions_i,
            encoded_goals_j
        )

        dist_waypoint_to_goal = metric_residual_network(
            encoded_waypoints_i,
            encoded_goals_j
        )

        waypoint_dist = rearrange(waypoint_dist, 'i -> i 1')

        # handle loss

        loss_matrix = linex_loss(dist_q_to_goal, dist_waypoint_to_goal.detach() - waypoint_dist * log(γ))

        loss = loss_matrix.mean()

        if self.has_paired_loss_weight:
            loss = torch.lerp(loss, loss_matrix.diag().mean(), self.paired_loss_weight)

        # section 4.2 - action invariance

        dist_action_invariance = metric_residual_network(
            encoded_states,
            encoded_state_actions,
            reduce_groups = False
        )

        loss_action_invariance = F.mse_loss(dist_action_invariance.neg().exp(), torch.ones_like(dist_action_invariance))

        total_loss = loss + loss_action_invariance * self.action_invariance_loss_weight

        return total_loss, (loss, loss_action_invariance)

    def actor_loss(
        self,
        states,
        actions,
        goals
    ):
        encoded_state_actions = self.state_action_encoder((states, actions))
        encoded_goals = self.state_encoder(goals)

        dist_q_to_goal = self.metric_residual_network(
            encoded_state_actions,
            encoded_goals
        )

        return dist_q_to_goal.mean()

# classes
class MultistepQuasimetricEstimation(Module):
    def __init__(
        self,
        state_encoder: Module,
        state_action_encoder: Module,
        metric_residual_network: MetricResidualNetwork,
        discount_factor = 0.95,
        waypoint_discount = 0.95,
        max_waypoint_dist = 10,
        next_timestep_prob = 0.2,
        action_invariance_loss_weight = 1.,
        paired_loss_weight = 0.5
    ):
        super().__init__()

        self.critic = Critic(
            state_encoder = state_encoder,
            state_action_encoder = state_action_encoder,
            metric_residual_network = metric_residual_network,
            discount_factor = discount_factor,
            action_invariance_loss_weight = action_invariance_loss_weight,
            paired_loss_weight = paired_loss_weight
        )

        self.max_waypoint_dist = max_waypoint_dist
        self.waypoint_discount = waypoint_discount
        self.discount_factor = discount_factor
        self.next_timestep_prob = next_timestep_prob

    def forward(
        self,
        states,
        actions,
        goals
    ):
        batch, timesteps, device = *states.shape[:2], states.device

        is_next_timestep = torch.full((batch,), self.next_timestep_prob, device = device).bernoulli() == 1
        max_waypoint = min(self.max_waypoint_dist, timesteps - 1)
        waypoint_dist = torch.empty((batch,), device = device).geometric_(1. - self.waypoint_discount).clamp(1, max_waypoint)

        waypoint_dist = torch.where(is_next_timestep, 1, waypoint_dist).long()

        waypoint_indices = rearrange(waypoint_dist, 'b -> b 1')
        batch_arange = rearrange(torch.arange(batch, device = device), 'b -> b 1')

        waypoints = states[batch_arange, waypoint_indices]
        waypoints = rearrange(waypoints, 'b 1 ... -> b ...')

        return self.critic(states[:, 0], actions[:, 0], goals[:, -1], waypoints, waypoint_dist)

# shorthand

MRN = MetricResidualNetwork
MQE = MultistepQuasimetricEstimation

# policy

class ResNet34Encoder(Module):
    def __init__(
        self,
        *,
        pretrained = False,
        pool = True
    ):
        super().__init__()
        resnet = models.resnet34(pretrained = pretrained)

        modules = list(resnet.children())[:-1]

        if not pool:
            modules = modules[:-1]
            modules.append(Rearrange('b c h w -> b (h w) c'))
        else:
            modules.append(Rearrange('b c 1 1 -> b c'))

        self.encoder = nn.Sequential(*modules)

    @property
    def output_dim(self):
        return 512

    def forward(self, x):
        return self.encoder(x)

# action distributions

class ActionDistribution(Module):
    @property
    def expansion_factor(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

class DiscreteAction(ActionDistribution):
    @property
    def expansion_factor(self):
        return 1

    def forward(self, x):
        return Categorical(logits = x)

class ContinuousAction(ActionDistribution):
    @property
    def expansion_factor(self):
        return 2

    def forward(self, x):
        mean, log_std = x.chunk(2, dim = -1)
        return Normal(mean, log_std.exp())

class BetaAction(ActionDistribution):
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps

    @property
    def expansion_factor(self):
        return 2

    def forward(self, x):
        alpha, beta = x.chunk(2, dim = -1)
        alpha, beta = [F.softplus(t) + 1. + self.eps for t in (alpha, beta)]
        return Beta(alpha, beta)

# policy

class Policy(Module):
    def __init__(
        self,
        *,
        action_dim,
        action_dist = None,
        pretrained = False,
        mlp_depth = 3,
        mlp_hidden_dim = 256
    ):
        super().__init__()
        self.action_dist = default(action_dist, ContinuousAction())

        self.state_encoder = ResNet34Encoder(pretrained = pretrained, pool = False)
        self.goal_encoder = ResNet34Encoder(pretrained = pretrained, pool = True)

        dim = self.state_encoder.output_dim
        dim_out = action_dim * self.action_dist.expansion_factor

        self.mlp = FiLMableMLP(
            dim,
            *((mlp_hidden_dim,) * mlp_depth),
            dim_out,
            cond_dim = dim
        )

    def forward(self, state, goal):
        state_tokens = self.state_encoder(state)
        goal_embed = self.goal_encoder(goal)

        embed = reduce(state_tokens, 'b n d -> b d', 'mean')

        out = self.mlp(embed, goal_embed)

        return self.action_dist(out)
