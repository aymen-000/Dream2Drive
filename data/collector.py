from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from utils.track import Centerline


@dataclass
class PurePursuitConfig:
    lookahead_m: float = 1.6
    min_lookahead_m: float = 0.8
    max_lookahead_m: float = 3.5
    lookahead_speed_gain: float = 0.15  
    wheelbase_m: float = 0.33
    max_steer_rad: float = 0.4
    target_speed_mps: float = 5.0
    corner_slowdown_gain: float = 2.5   
    min_speed_mps: float = 1.0


class PurePursuitController:
    def __init__(self, centerline: Centerline, cfg: PurePursuitConfig):
        self.centerline = centerline
        self.cfg = cfg

    def _lookahead_point(self, xy: np.ndarray, lookahead_m: float) -> np.ndarray:
        idx = self.centerline.nearest_index(xy)
        target_dist = self.centerline.cum_dist[idx] + lookahead_m
        target_dist = target_dist % self.centerline.total_length
        target_idx = int(np.searchsorted(self.centerline.cum_dist, target_dist, side="left"))
        target_idx = target_idx % self.centerline.num_points
        return self.centerline.points[target_idx]

    def act(self, x: float, y: float, yaw: float, speed: float) -> np.ndarray:
        """Returns raw [steer_cmd, speed_cmd] for the current raw pose."""
        cfg = self.cfg
        lookahead = float(np.clip(
            cfg.min_lookahead_m + cfg.lookahead_speed_gain * speed,
            cfg.min_lookahead_m, cfg.max_lookahead_m,
        ))
        target = self._lookahead_point(np.array([x, y], dtype=np.float32), lookahead)

        dx = target[0] - x
        dy = target[1] - y
        cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy

        curvature = 2.0 * local_y / max(lookahead ** 2, 1e-6)
        steer_cmd = np.arctan(cfg.wheelbase_m * curvature)
        steer_cmd = float(np.clip(steer_cmd, -cfg.max_steer_rad, cfg.max_steer_rad))

        speed_cmd = cfg.target_speed_mps - cfg.corner_slowdown_gain * abs(steer_cmd)
        speed_cmd = float(max(speed_cmd, cfg.min_speed_mps))

        return np.array([steer_cmd, speed_cmd], dtype=np.float32)


def add_exploration_noise(action: np.ndarray, rng: np.random.RandomState,
                           steer_std: float = 0.03, speed_std: float = 0.3,
                           max_steer_rad: float = 0.4) -> np.ndarray:
    noisy = action.copy()
    noisy[0] = float(np.clip(action[0] + rng.normal(0, steer_std), -max_steer_rad, max_steer_rad))
    noisy[1] = max(0.0, action[1] + rng.normal(0, speed_std))
    return noisy