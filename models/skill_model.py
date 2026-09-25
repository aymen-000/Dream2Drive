from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.distributions import kl_divergence

from models.networks import (
    GaussianHead,
    LidarEncoder,
    SkillPosterior,
    SkillPrior,
    LowLevelPolicy,
    TemporallyAbstractWorldModel,
)

class OPOSMSkillModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, z_dim: int, lidar_dim: int,
                 hidden_dim: int = 256, gru_hidden_dim: int = 128,
                 gru_bidirectional: bool = True, min_std: float = 1e-3,
                 obs_embed_dim: int = 128, lidar_max_range: float = 30.0,
                 tawm_predict_delta: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.lidar_dim = lidar_dim
        self.obs_embed_dim = obs_embed_dim
        self.tawm_predict_delta = tawm_predict_delta

        self.skill_posterior = SkillPosterior(
            state_dim, action_dim, z_dim, hidden_dim, gru_hidden_dim,
            gru_bidirectional, min_std=min_std,
        )
        self.skill_prior = SkillPrior(state_dim, z_dim, hidden_dim, min_std=min_std)
        self.lidar_encoder = LidarEncoder(lidar_dim, obs_embed_dim, hidden_dim, max_range=lidar_max_range)
        self.low_level_policy = LowLevelPolicy(obs_embed_dim, z_dim, action_dim, hidden_dim, min_std=min_std)
        self.tawm = TemporallyAbstractWorldModel(
            state_dim, z_dim, hidden_dim, min_std=min_std,
            predict_delta=tawm_predict_delta,
        )
        self.lidar_max_range = lidar_max_range
    def _sample_z(self, s0, states_seq, actions_seq):
        """Reparameterized sample from q_phi(z | tau_T)."""
        q_dist = self.skill_posterior.dist(states_seq, actions_seq)
        z = q_dist.rsample()
        return z, q_dist

    def _low_level_log_prob(self, lidar_seq: torch.Tensor, actions_seq: torch.Tensor,
                             z: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions_seq.shape
        z_rep = z.unsqueeze(1).expand(-1, T, -1)
        o_emb_seq = self.lidar_encoder(lidar_seq)
        policy_dist = self.low_level_policy.dist(o_emb_seq, z_rep)
        log_prob = policy_dist.log_prob(actions_seq)
        return log_prob.sum(dim=(1, 2))

    def _tawm_log_prob(self, s0: torch.Tensor, sT: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        tawm_dist = self.tawm.dist(s0, z)
        log_prob = tawm_dist.log_prob(sT)                     
        return log_prob.sum(dim=-1)                          

    def _prior_log_prob(self, s0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mu0, sigma0 = self.skill_prior(s0)
        prior = torch.distributions.Normal(mu0, sigma0)
        return prior.log_prob(z).sum(dim=-1)                  

    def elbo_terms(self, batch: Dict[str, torch.Tensor], detach_z: bool = False
                    ) -> Dict[str, torch.Tensor]:

        s0, states_seq, actions_seq, sT, lidar_seq = (
            batch["s0"], batch["states_seq"], batch["actions_seq"], batch["sT"], batch["lidar_seq"]
        )
        # LidarEncoder performs the single required range normalization.  The
        # live F1TENTH policy receives raw metre ranges, so training must too.
        z, q_dist = self._sample_z(s0, states_seq, actions_seq)
        z_for_networks = z.detach() if detach_z else z

        log_pi = self._low_level_log_prob(lidar_seq, actions_seq, z_for_networks)
        log_psi = self._tawm_log_prob(s0, sT, z_for_networks)

        mu0, sigma0 = self.skill_prior(s0)
        prior_dist = torch.distributions.Normal(mu0, sigma0)
        kl = kl_divergence(q_dist, prior_dist).sum(dim=-1)              

        elbo = log_pi + log_psi - kl
        return {
            "z": z, "q_dist": q_dist, "prior_dist": prior_dist,
            "log_pi": log_pi, "log_psi": log_psi, "kl": kl, "elbo": elbo,
        }

    def m_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        terms = self.elbo_terms(batch, detach_z=True)
        loss = -terms["elbo"].mean()
        return {
            "loss": loss,
            "elbo": terms["elbo"].mean().detach(),
            "log_pi": terms["log_pi"].mean().detach(),
            "log_psi": terms["log_psi"].mean().detach(),
            "kl": terms["kl"].mean().detach(),
        }

    def e_loss_causal(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        s0, states_seq, actions_seq, lidar_seq = (
            batch["s0"], batch["states_seq"], batch["actions_seq"], batch["lidar_seq"]
        )
        z, q_dist = self._sample_z(s0, states_seq, actions_seq)

        log_q = q_dist.log_prob(z).sum(dim=-1)
        log_pi = self._low_level_log_prob(lidar_seq, actions_seq, z)
        log_prior = self._prior_log_prob(s0, z)

        kl_to_true_posterior = log_q - log_pi - log_prior
        loss = kl_to_true_posterior.mean()
        return {"loss": loss, "kl_to_true_posterior": loss.detach()}

    def naive_vi_e_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        terms = self.elbo_terms(batch)
        loss = -terms["elbo"].mean()
        return {"loss": loss, "elbo": terms["elbo"].mean().detach()}


    @torch.no_grad()
    def prior_mean_std(self, s0: torch.Tensor):
        return self.skill_prior(s0)

    @torch.no_grad()
    def predict_terminal_state(self, s0: torch.Tensor, z: torch.Tensor, sample: bool = False):
        mu, sigma = self.tawm(s0, z)
        if sample:
            return mu + sigma * torch.randn_like(sigma)
        return mu

    @torch.no_grad()
    def act(self, o: torch.Tensor, z: torch.Tensor, sample: bool = True):
        o_embed = self.lidar_encoder(o)
        mu, sigma = self.low_level_policy(o_embed, z)
        if sample:
            return mu + sigma * torch.randn_like(sigma)
        return mu
