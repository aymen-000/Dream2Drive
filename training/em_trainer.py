from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from models.skill_model import OPOSMSkillModel
from utils.logger import Logger

@dataclass
class EMTrainerConfig:
    lr_e: float = 5e-5
    lr_m: float = 5e-5
    grad_clip_norm: float = 10.0
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 20
    save_every_epochs: int = 10
    mode: str = "causal_em"
    aux_loss_weight: float = 1.0
    metrics_path: Optional[str] = None  # defaults to <checkpoint_dir>/metrics.json


class EMTrainer:
    def __init__(self, model: OPOSMSkillModel, cfg: EMTrainerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        phi_params = list(self.model.skill_posterior.parameters())
        theta_psi_omega_params = (
            list(self.model.low_level_policy.parameters())
            + list(self.model.tawm.parameters())
            + list(self.model.skill_prior.parameters())
            + list(self.model.lidar_encoder.parameters())  # encoder is shared, trained only via the M-step objective
        )

        if cfg.mode == "causal_em":
            self.opt_e = torch.optim.Adam(phi_params, lr=cfg.lr_e)
            self.opt_m = torch.optim.Adam(theta_psi_omega_params, lr=cfg.lr_m)
        elif cfg.mode == "naive_vi":
            self.opt_joint = torch.optim.Adam(
                phi_params + theta_psi_omega_params, lr=cfg.lr_m
            )
        else:
            raise ValueError(f"Unknown training mode: {cfg.mode}")

        self.logger = Logger(log_every=cfg.log_every)
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.global_step = 0

        # --- metrics history, for later plotting ---
        self.metrics_path = cfg.metrics_path or os.path.join(cfg.checkpoint_dir, "metrics.json")
        self.metrics_history: list[dict] = []
        self.validation_metrics_path = os.path.join(cfg.checkpoint_dir, "validation_metrics.json")
        self.validation_history: list[dict] = []

    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_step(self, batch, epoch: Optional[int] = None) -> dict:
        batch = self._to_device(batch)

        if self.cfg.mode == "causal_em":
            self.opt_e.zero_grad(set_to_none=True)
            self.model.lidar_encoder.zero_grad(set_to_none=True)  # encoder isn't in opt_e; zero it so e-step grads don't leak into the m-step update below
            e_out = self.model.e_loss_causal(batch)
            e_out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.skill_posterior.parameters(),
                                            self.cfg.grad_clip_norm)
            self.opt_e.step()

            self.opt_m.zero_grad(set_to_none=True)
            m_out = self.model.m_loss(batch)
            m_out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.low_level_policy.parameters())
                + list(self.model.tawm.parameters())
                + list(self.model.skill_prior.parameters())
                + list(self.model.lidar_encoder.parameters()),
                self.cfg.grad_clip_norm,
            )
            self.opt_m.step()

            metrics = {
                "e_loss": e_out["loss"].item(),
                "m_loss": m_out["loss"].item(),
                "elbo": m_out["elbo"].item(),
                "log_pi": m_out["log_pi"].item(),
                "log_psi": m_out["log_psi"].item(),
                "kl": m_out["kl"].item(),
            }
        else:
            self.opt_joint.zero_grad(set_to_none=True)
            out = self.model.naive_vi_e_loss(batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.opt_joint.step()
            metrics = {"joint_loss": out["loss"].item(), "elbo": out["elbo"].item()}

        self.global_step += 1
        self.logger.log(self.global_step, **metrics)

        # --- record + periodically persist metrics for plotting ---
        record = {"step": self.global_step, "epoch": epoch, **metrics}
        self.metrics_history.append(record)
        if self.global_step % self.cfg.log_every == 0:
            self._save_metrics()

        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Aggregate held-out ELBO terms and terminal-scan reconstruction MSE."""
        was_training = self.model.training
        self.model.eval()
        totals = {key: 0.0 for key in ("loss", "elbo", "log_pi", "log_psi", "kl", "lidar_mse")}
        count = 0
        for batch in dataloader:
            batch = self._to_device(batch)
            terms = self.model.elbo_terms(batch)
            target = torch.clamp(
                batch["lidar_T"], min=0.0, max=self.model.lidar_max_range
            ) / self.model.lidar_max_range
            o0_embed = self.model.lidar_encoder(batch["lidar_seq"][:, 0])
            pred, _ = self.model.tawm(o0_embed, terms["z"])
            batch_size = batch["lidar_T"].shape[0]
            values = {
                "loss": -terms["elbo"].mean(),
                "elbo": terms["elbo"].mean(),
                "log_pi": terms["log_pi"].mean(),
                "log_psi": terms["log_psi"].mean(),
                "kl": terms["kl"].mean(),
                "lidar_mse": (pred - target).square().mean(),
            }
            for key, value in values.items():
                totals[key] += float(value) * batch_size
            count += batch_size
        if was_training:
            self.model.train()
        if count == 0:
            raise ValueError("Validation dataloader is empty")
        return {key: value / count for key, value in totals.items()}

    def fit(self, dataloader: DataLoader, num_epochs: int, steps_per_epoch: Optional[int] = None,
            val_dataloader: Optional[DataLoader] = None):
        data_iter = iter(dataloader)
        for epoch in range(1, num_epochs + 1):
            steps = steps_per_epoch or len(dataloader)
            for _ in range(steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                self.train_step(batch, epoch=epoch)

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                val_record = {
                    "step": self.global_step,
                    "epoch": epoch,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
                self.validation_history.append(val_record)
                print(
                    f"[validation epoch {epoch:>4d}] "
                    + "  ".join(f"{key}={value:.4f}" for key, value in val_metrics.items())
                )
                self._save_validation_metrics()

            if epoch % self.cfg.save_every_epochs == 0 or epoch == num_epochs:
                self.save_checkpoint(os.path.join(self.cfg.checkpoint_dir, f"oposm_epoch{epoch}.pt"))

        self.save_checkpoint(os.path.join(self.cfg.checkpoint_dir, "oposm_final.pt"))
        self._save_metrics()  # final flush, guarantees everything is on disk

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "mode": self.cfg.mode,
            "global_step": self.global_step,
            "state_dim": self.model.state_dim,
            "action_dim": self.model.action_dim,
            "z_dim": self.model.z_dim,
            "lidar_dim": self.model.lidar_dim,
            "world_model_target": "terminal_lidar",
            "obs_embed_dim": self.model.obs_embed_dim,
            "tawm_predict_delta": self.model.tawm_predict_delta,
        }, path)
        print(f"[checkpoint] saved -> {path}")

    def _save_metrics(self, path: Optional[str] = None):
        path = path or self.metrics_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def _save_validation_metrics(self):
        with open(self.validation_metrics_path, "w") as f:
            json.dump(self.validation_history, f, indent=2)

    @staticmethod
    def load_model(model: OPOSMSkillModel, path: str, device: str = "cpu") -> OPOSMSkillModel:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model
