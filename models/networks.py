from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal




def mlp(in_dim: int, hidden_dim: int, out_dim: int, num_hidden_layers: int = 1) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class GaussianHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, min_std: float = 1e-3):
        super().__init__()
        self.mean_net = mlp(in_dim, hidden_dim, out_dim, num_hidden_layers=2)
        self.std_net = mlp(in_dim, hidden_dim, out_dim, num_hidden_layers=2)
        self.min_std = min_std
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mean_net(x)
        sigma = F.softplus(self.std_net(x)) + self.min_std
        return mu, sigma

    def dist(self, x: torch.Tensor) -> Normal:
        mu, sigma = self.forward(x)
        return Normal(mu, sigma)


class SkillPosterior(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, z_dim: int,
                 hidden_dim: int = 256, gru_hidden_dim: int = 128,
                 bidirectional: bool = True, min_std: float = 1e-3):
        super().__init__()
        self.state_embed = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        gru_in_dim = hidden_dim + action_dim
        self.gru = nn.GRU(
            input_size=gru_in_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        gru_out_dim = gru_hidden_dim * (2 if bidirectional else 1)
        self.head = GaussianHead(gru_out_dim, hidden_dim, z_dim, min_std=min_std)

    def forward(self, states_seq: torch.Tensor, actions_seq: torch.Tensor):
        s_emb = self.state_embed(states_seq)                
        gru_in = torch.cat([s_emb, actions_seq], dim=-1)     
        gru_out, _ = self.gru(gru_in)                       
        pooled = gru_out.mean(dim=1)                        
        return self.head(pooled)

    def dist(self, states_seq: torch.Tensor, actions_seq: torch.Tensor) -> Normal:
        mu, sigma = self.forward(states_seq, actions_seq)
        return Normal(mu, sigma)


class SkillPrior(nn.Module):

    def __init__(self, state_dim: int, z_dim: int, hidden_dim: int = 256, min_std: float = 1e-3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.head = GaussianHead(hidden_dim, hidden_dim, z_dim, min_std=min_std)

    def forward(self, s0: torch.Tensor):
        h = self.trunk(s0)
        return self.head(h)


class LidarEncoder(nn.Module):
    def __init__(self, lidar_dim: int, embed_dim: int, hidden_dim: int = 256,
                 max_range: float = 30.0):
        super().__init__()
        self.max_range = max_range
        self.net = mlp(lidar_dim, hidden_dim, embed_dim, num_hidden_layers=2)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        o_norm = torch.clamp(o, min=0.0, max=self.max_range) / self.max_range
        return F.relu(self.net(o_norm))


class LowLevelPolicy(nn.Module):
    def __init__(self, obs_embed_dim: int, z_dim: int, action_dim: int,
                 hidden_dim: int = 256, min_std: float = 1e-3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_embed_dim + z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = GaussianHead(hidden_dim, hidden_dim, action_dim, min_std=min_std)

    def forward(self, o_embed: torch.Tensor, z: torch.Tensor):
        h = self.trunk(torch.cat([o_embed, z], dim=-1))
        return self.head(h)

    def dist(self, o_embed: torch.Tensor, z: torch.Tensor) -> Normal:
        mu, sigma = self.forward(o_embed, z)
        return Normal(mu, sigma)


class TemporallyAbstractWorldModel(nn.Module):
    def __init__(self, state_dim: int, z_dim: int, hidden_dim: int = 256,
                 min_std: float = 1e-3, predict_delta: bool = False):
        super().__init__()
        self.predict_delta = predict_delta
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = GaussianHead(hidden_dim, hidden_dim, state_dim, min_std=min_std)

    def forward(self, s0: torch.Tensor, z: torch.Tensor):
        h = self.trunk(torch.cat([s0, z], dim=-1))
        mu, sigma = self.head(h)
        if self.predict_delta:
            mu = s0 + mu
        return mu, sigma

    def dist(self, s0: torch.Tensor, z: torch.Tensor) -> Normal:
        mu, sigma = self.forward(s0, z)
        return Normal(mu, sigma)
