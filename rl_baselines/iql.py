"""Small continuous-control IQL implementation for offline F1TENTH logs.

Algorithm: Kostrikov, Nair, Levine (2021), Offline Reinforcement Learning
with Implicit Q-Learning, https://arxiv.org/abs/2110.06169.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim))


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, obs: torch.Tensor) -> TransformedDistribution:
        features = self.trunk(obs)
        mean = self.mean(features)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return TransformedDistribution(Normal(mean, std), [TanhTransform(cache_size=1)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean(self.trunk(obs)))


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, action), dim=-1))


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = _mlp(obs_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class IQLConfig:
    obs_dim: int
    action_dim: int = 2
    gamma: float = 0.99
    expectile: float = 0.7
    temperature: float = 3.0
    max_advantage_weight: float = 100.0
    learning_rate: float = 3e-4
    target_tau: float = 0.005


class IQL:
    """IQL learner and deterministic policy loader for vector observations."""
    def __init__(self, config: IQLConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.actor = GaussianActor(config.obs_dim, config.action_dim).to(self.device)
        self.q1 = QNetwork(config.obs_dim, config.action_dim).to(self.device)
        self.q2 = QNetwork(config.obs_dim, config.action_dim).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.value = ValueNetwork(config.obs_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.learning_rate)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=config.learning_rate)

    def update(self, obs, actions, rewards, next_obs, terminals) -> dict[str, float]:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1, 1)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(terminals, dtype=torch.float32, device=self.device).reshape(-1, 1)

        with torch.no_grad():
            q_data = torch.minimum(self.target_q1(obs, actions), self.target_q2(obs, actions))
        value_error = q_data - self.value(obs)
        expectile_weight = torch.where(value_error > 0, self.config.expectile,
                                       1.0 - self.config.expectile)
        value_loss = (expectile_weight * value_error.square()).mean()
        self.value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_opt.step()

        with torch.no_grad():
            q_target = rewards + self.config.gamma * (1.0 - terminals) * self.value(next_obs)
        q1_loss = (self.q1(obs, actions) - q_target).square().mean()
        q2_loss = (self.q2(obs, actions) - q_target).square().mean()
        critic_loss = q1_loss + q2_loss
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        with torch.no_grad():
            for target, source in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
                for target_param, source_param in zip(target.parameters(), source.parameters()):
                    target_param.lerp_(source_param, self.config.target_tau)

        with torch.no_grad():
            advantage = torch.minimum(self.q1(obs, actions), self.q2(obs, actions)) - self.value(obs)
            weights = torch.exp(self.config.temperature * advantage).clamp(max=self.config.max_advantage_weight)
        safe_actions = actions.clamp(-0.999, 0.999)
        actor_loss = -(weights * self.actor.distribution(obs).log_prob(safe_actions).sum(-1, keepdim=True)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        return {"value_loss": float(value_loss.detach()), "critic_loss": float(critic_loss.detach()),
                "actor_loss": float(actor_loss.detach())}

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        dist = self.actor.distribution(tensor)
        action = self.actor(tensor) if deterministic else dist.sample()
        result = action.cpu().numpy()
        return result[0] if np.asarray(obs).ndim == 1 else result

    def save(self, path: str, metadata: dict | None = None) -> None:
        torch.save({"config": self.config.__dict__, "actor": self.actor.state_dict(),
                    "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> tuple["IQL", dict]:
        payload = torch.load(path, map_location=device, weights_only=False)
        learner = cls(IQLConfig(**payload["config"]), device=device)
        learner.actor.load_state_dict(payload["actor"])
        learner.actor.eval()
        return learner, payload.get("metadata", {})


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Expectile regression loss, exposed for focused numerical checks."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.square()).mean()

