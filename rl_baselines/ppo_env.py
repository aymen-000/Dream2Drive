"""PPO racing environment copied from the notebook's F1TenthPPOEnv wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from f110_gym.envs.f110_env import F110Env
from scripts.config_utils import load_config
from utils.track import Centerline, discover_map, relative_progress_m


@dataclass
class PPOEnvConfig:
    map_name: str
    maps_root: str = "maps"
    lidar_dim: int = 128
    lidar_max_range: float = 10.0
    timestep: float = 0.01
    control_hz: int = 20
    max_steer_rad: float = 0.4
    max_speed_mps: float = 8.0
    max_episode_seconds: float = 120.0
    progress_weight: float = 1.0
    cross_track_weight: float = 0.15
    speed_weight: float = 0.02
    time_penalty_per_sec: float = 0.4
    lap_bonus: float = 50.0
    collision_penalty: float = 80.0
    randomize_start: bool = False


def ppo_env_config_from_project(config_path: str, map_name: str,
                                max_episode_seconds: float = 120.0,
                                randomize_start: bool = False,
                                time_penalty_per_sec: float = 0.4) -> PPOEnvConfig:
    project = load_config(config_path)
    return PPOEnvConfig(
        map_name=map_name,
        maps_root=project.data.maps_root,
        lidar_dim=project.data.lidar_dim,
        lidar_max_range=project.data.lidar_max_range,
        timestep=project.f1tenth_env.timestep,
        control_hz=project.f1tenth_env.control_hz,
        max_steer_rad=project.f1tenth_env.max_steer_rad,
        max_speed_mps=project.f1tenth_env.max_speed_mps,
        max_episode_seconds=max_episode_seconds,
        progress_weight=1.0,
        cross_track_weight=0.15,
        speed_weight=0.02,
        time_penalty_per_sec=time_penalty_per_sec,
        lap_bonus=50.0,
        collision_penalty=80.0,
        randomize_start=randomize_start,
    )


class F1TenthPPOEnv(gym.Env):
    """Notebook's PPO environment, using its direct F110Env step loop."""

    metadata = {"render_modes": []}

    def __init__(self, map_name: str, randomize_start: bool = False,
                 max_episode_seconds: float = 120.0,
                 config_path: str = "configs/default.yaml",
                 time_penalty_per_sec: float = 0.4):
        super().__init__()
        self.cfg = ppo_env_config_from_project(
            config_path, map_name, max_episode_seconds, randomize_start,
            time_penalty_per_sec)
        self.map_name = map_name
        self.randomize_start = randomize_start
        _, map_ext, centerline_csv = discover_map(self.cfg.maps_root, map_name)
        frame = pd.read_csv(centerline_csv)
        frame.columns = [str(column).lstrip("#").strip() for column in frame.columns]
        if not {"x_m", "y_m"}.issubset(frame.columns):
            raise ValueError(
                f"Expected x_m,y_m columns in {centerline_csv}; got {list(frame.columns)}")
        self.points = frame[["x_m", "y_m"]].to_numpy(dtype=np.float32)
        self.centerline = Centerline(self.points)
        self.segments = self.centerline.segments
        self.segment_lengths = self.centerline.segment_lengths
        self.segment_lengths_sq = self.centerline.segment_lengths_sq
        self.cumulative = self.centerline.cum_dist
        self.total_length = self.centerline.total_length
        self.sim = F110Env(map=map_name, map_ext=map_ext, timestep=self.cfg.timestep)
        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.zeros(self.cfg.lidar_dim, dtype=np.float32),
                np.array([0.0, -1.0], dtype=np.float32),
            ]),
            high=np.ones(self.cfg.lidar_dim + 2, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.substeps = max(1, round(1.0 / (self.cfg.control_hz * self.cfg.timestep)))
        self.max_steps = max(1, int(max_episode_seconds * self.cfg.control_hz))
        self.step_count = 0

    def _project(self, xy):
        return self.centerline.project_m(xy)

    def _observation(self):
        scan = np.asarray(self.last_obs["scans"][0], dtype=np.float32).reshape(-1)
        indices = np.linspace(0, scan.size - 1, self.cfg.lidar_dim).round().astype(np.intp)
        lidar = np.clip(scan[indices] / self.cfg.lidar_max_range, 0.0, 1.0)
        state = np.asarray(self.sim.sim.agents[0].state, dtype=np.float32)
        speed = np.clip(state[3] / self.cfg.max_speed_mps, 0.0, 1.0)
        steering = np.clip(state[2] / self.cfg.max_steer_rad, -1.0, 1.0)
        return np.concatenate([lidar, np.array([speed, steering], dtype=np.float32)]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        idx = 0
        if options and "start_index" in options:
            idx = int(options["start_index"]) % len(self.points)
        elif self.randomize_start:
            idx = int(self.np_random.integers(len(self.points)))
        xy = self.points[idx]
        tangent = self.segments[idx]
        yaw = float(np.arctan2(tangent[1], tangent[0]))
        self.start_s, _ = self._project(xy)
        self.prev_progress = 0.0
        self.laps = 0
        self.step_count = 0
        self.lap_times = []
        self.last_obs, _, _, _ = self.sim.reset(
            np.array([[xy[0], xy[1], yaw]], dtype=np.float64))
        return self._observation(), {"map_name": self.map_name, "start_index": idx}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        steer = float(action[0] * self.cfg.max_steer_rad)
        target_speed = float((action[1] + 1.0) * 0.5 * self.cfg.max_speed_mps)
        collision = False
        for _ in range(self.substeps):
            self.last_obs, _, _, _ = self.sim.step(
                np.array([[steer, target_speed]], dtype=np.float64))
            collision = bool(self.last_obs["collisions"][0])
            if collision:
                break
        self.step_count += 1
        state = np.asarray(self.sim.sim.agents[0].state, dtype=np.float32)
        absolute_s, cross_track = self._project(state[:2])
        progress = relative_progress_m(absolute_s, self.start_s, self.total_length)
        raw_delta = progress - self.prev_progress
        lap_completed = raw_delta < -0.5 * self.total_length
        delta_s = raw_delta
        if delta_s < -0.5 * self.total_length:
            delta_s += self.total_length
        elif delta_s > 0.5 * self.total_length:
            delta_s -= self.total_length
        delta_s = float(np.clip(delta_s, -2.0, 2.0))
        self.prev_progress = progress
        reward = (
            self.cfg.progress_weight * delta_s
            - self.cfg.cross_track_weight * cross_track
            + self.cfg.speed_weight * max(float(state[3]), 0.0)
            - self.cfg.time_penalty_per_sec / self.cfg.control_hz
        )
        if lap_completed:
            self.laps += 1
            self.lap_times.append(self.step_count / self.cfg.control_hz)
            reward += self.cfg.lap_bonus
        if collision:
            reward -= self.cfg.collision_penalty
        terminated = collision or lap_completed
        truncated = self.step_count >= self.max_steps and not terminated
        info = {
            "map_name": self.map_name,
            "laps_completed": self.laps,
            "lap_times_sec": list(self.lap_times),
            "collision": collision,
            "progress_m": progress,
            "total_track_m": self.total_length,
            "speed_mps": float(state[3]),
            "target_speed_mps": target_speed,
            "elapsed_time_sec": self.step_count / self.cfg.control_hz,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        self.sim.close_rendering()
