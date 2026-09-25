from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import gym



from models.skill_model import OPOSMSkillModel
from utils.normalization import Normalizer
from f110_gym.envs.f110_env import F110Env

@dataclass
class F1TenthEnvConfig:
    map_yaml: str
    map_ext: str
    timestep: float = 0.01
    control_hz: int = 20
    max_steer_rad: float = 0.4
    max_speed_mps: float = 8.0
    render: bool = False
    # Number of beams fed to the learned policy.  ``None`` uses the width the
    # loaded model was trained with.  Do not use a fixed stride here: F1TENTH
    # installations can expose different native scan resolutions.
    lidar_dim: Optional[int] = None


class F1TenthEnvironment:


    def __init__(self, model: OPOSMSkillModel, state_normalizer: Normalizer,
                 action_normalizer: Normalizer, cfg: F1TenthEnvConfig,
                 initial_state_raw: np.ndarray, device: str = "cpu"):
        if model is not None and state_normalizer is not None and action_normalizer is not None:
            self.model = model.to(device)
            self.model.eval()
            self.state_normalizer = state_normalizer
            self.action_normalizer = action_normalizer
        self.device = device
        self.cfg = cfg
        self.lidar_dim = cfg.lidar_dim 
        if self.lidar_dim <= 0:
            raise ValueError(f"lidar_dim must be positive, got {self.lidar_dim}")
        self.env = F110Env(map=cfg.map_yaml.split("/")[-1][:-9]) 
        self._substeps = max(1, round(1.0 / (cfg.control_hz * cfg.timestep)))

        self.crashed = False
        self.crash_step: Optional[int] = None
        self._step_count = 0
        self.reset(initial_state_raw)

    def _pose_for_reset(self, state_raw: np.ndarray) -> np.ndarray:
        x, y, yaw = state_raw[0], state_raw[1], state_raw[4] 
        return np.array([[x, y, yaw]], dtype=np.float64)

    def reset(self, state_raw: np.ndarray):
        self.obs, _, self.done, _ = self.env.reset(self._pose_for_reset(state_raw))
        self.crashed = False
        self.crash_step = None
        self._step_count = 0

    def get_state(self) -> np.ndarray:
        return np.asarray(self.env.sim.agents[0].state, dtype=np.float32).copy()

    def _raw_step(self, steer: float, speed: float):
        steer = float(np.clip(steer, -self.cfg.max_steer_rad, self.cfg.max_steer_rad))
        speed = float(np.clip(speed, 0.0, self.cfg.max_speed_mps))
        action = np.array([[steer, speed]], dtype=np.float64)
        self.obs, _, self.done, _ = self.env.step(action)
        self._step_count += 1
        if not self.crashed and bool(self.obs["collisions"][0]):
            self.crashed = True
            self.crash_step = self._step_count
            print(f"[f1tenth_env] WARNING: collision at control step {self._step_count}.")
        if self.cfg.render:
            self.env.render(mode="human")

    def step_raw_action(self, action_raw: np.ndarray) -> np.ndarray:
        for _ in range(self._substeps):
            self._raw_step(action_raw[0], action_raw[1])
        return self.get_state()

    def _policy_scan(self) -> np.ndarray:
        """Return exactly the LiDAR width expected by the learned policy.

        Native F1TENTH scan lengths vary by simulator/configuration (for
        example, 108 or 1080 beams).  Selecting every Nth beam can therefore
        silently change the input width.  Evenly sampling the complete scan
        preserves its field of view and keeps the network input stable.
        """
        full_scan = np.asarray(self.obs["scans"][0], dtype=np.float32).reshape(-1)
        if full_scan.size == 0:
            raise RuntimeError("F1TENTH returned an empty LiDAR scan")
        if full_scan.size == self.lidar_dim:
            return full_scan

        indices = np.linspace(0, full_scan.size - 1, self.lidar_dim).round().astype(np.intp)
        return full_scan[indices]

    def execute_skill(self, z: np.ndarray, horizon: int) -> np.ndarray:
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        for _ in range(horizon):
            scan = self._policy_scan()
            o_t = torch.as_tensor(scan, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                o_embed = self.model.lidar_encoder(o_t)
                a_mu, _ = self.model.low_level_policy(o_embed, z_t)
            a_raw = self.action_normalizer.inverse_transform(a_mu.cpu().numpy())[0]
            for _ in range(self._substeps):
                self._raw_step(a_raw[0], a_raw[1])
        return self.get_state()

    def close(self):
        pass
