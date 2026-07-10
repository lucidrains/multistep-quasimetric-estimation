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
from x_mlps_pytorch import create_filmable_mlp

# helpers

from torch_einops_utils import batched_index_select

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
    asym_fn = default_asym_fn
):
    asym_x, asym_y = default(asym_x, x), default(asym_y, y)

    assert x.shape[-1] == y.shape[-1] == asym_x.shape[-1] == asym_y.shape[-1]

    # symmetric

    sym = sym_fn(x, y)

    # asymmetric

    asym = asym_fn(asym_x, asym_y)

    # eq (4)

    distance = sym + asym

    return distance

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

        dim_embed = sym_x.shape[-1]
        assert divisible_by(dim_embed, self.distance_groups)

        sym_x, sym_y, asym_x, asym_y = (rearrange(t, '... (g d) -> ... g d', g = self.distance_groups) for t in (sym_x, sym_y, asym_x, asym_y))

        distance = quasimetric_distance(sym_x, sym_y, asym_x, asym_y)

        if not reduce_groups:
            return distance

        return reduce(distance, '... g -> ...', 'mean')

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

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def extract_policy(
        self,
        policy: Module,
        states,
        actions,
        goals,
        bc_loss_weight = 0.1,
        is_image = False
    ):
        # extracts goal-conditioned policy using behavior-regularization

        batch, device = states.shape[0], states.device

        is_seq = states.ndim == (5 if is_image else 3)

        if is_seq:
            states = states[:, 0]
            actions = actions[:, 0]
            goals = goals[:, -1]

        # behavior cloning loss

        action_dist = policy(states, goals)

        bc_loss = self.zero

        if bc_loss_weight > 0.:
            log_prob = action_dist.log_prob(actions)
            log_prob = reduce(log_prob, 'b ... -> b', 'sum')
            bc_loss = -log_prob.mean()

        # minimize distance is the same as maximizing Q

        rand_indices = torch.randperm(batch, device = device)
        goals_j = goals[rand_indices]

        action_dist_j = policy(states, goals_j)
        pred_actions_j = action_dist_j.rsample()

        q_loss = self.actor_loss(states, pred_actions_j, goals_j)

        total_loss = q_loss + bc_loss_weight * bc_loss

        return total_loss, (q_loss, bc_loss)

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

# main classe

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

    def extract_policy(
        self,
        *args,
        **kwargs
    ):
        return self.critic.extract_policy(*args, **kwargs)

    def forward(
        self,
        states,
        actions,
        goals
    ):
        batch, timesteps, device = *states.shape[:2], states.device

        # section 4.1 - multistep returns with quasimetric metric residual network

        is_next_timestep = torch.full((batch,), self.next_timestep_prob, device = device).bernoulli() == 1
        max_waypoint = min(self.max_waypoint_dist, timesteps - 1)
        waypoint_dist = torch.empty((batch,), device = device).geometric_(1. - self.waypoint_discount).clamp(1, max_waypoint)
        waypoint_dist = torch.where(is_next_timestep, 1, waypoint_dist).long()

        # waypoints selected, then waypoints and their sampled timesteps from starting state is used to calculate the loss

        waypoints = batched_index_select(states, waypoint_dist)

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
        dim = None,
        action_dist = None,
        state_encoder = None,
        goal_encoder = None,
        pretrained = False,
        mlp_depth = 3,
        mlp_hidden_dim = 256
    ):
        super().__init__()
        self.action_dist = default(action_dist, ContinuousAction())

        self.state_encoder = default(state_encoder, ResNet34Encoder(pretrained = pretrained, pool = False))
        self.goal_encoder = default(goal_encoder, ResNet34Encoder(pretrained = pretrained, pool = True))

        if not exists(dim):
            assert hasattr(self.state_encoder, 'output_dim'), 'dim must be given if using custom state_encoder'
            dim = self.state_encoder.output_dim

        dim_out = action_dim * self.action_dist.expansion_factor

        self.mlp = create_filmable_mlp(
            mlp_hidden_dim,
            mlp_depth,
            dim_in = dim,
            dim_out = dim_out,
            cond_dim = dim
        )

    def forward(self, state, goal):
        state_tokens = self.state_encoder(state)
        goal_tokens = self.goal_encoder(goal)

        embed = reduce(state_tokens, 'b n d -> b d', 'mean') if state_tokens.ndim == 3 else state_tokens
        cond = reduce(goal_tokens, 'b n d -> b d', 'mean') if goal_tokens.ndim == 3 else goal_tokens

        out = self.mlp(embed, cond)

        return self.action_dist(out)
