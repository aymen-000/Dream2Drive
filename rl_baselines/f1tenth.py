from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.f1tenth_env import F1TenthEnvConfig, F1TenthEnvironment
from scripts.config_utils import load_config
from utils.track import Centerline, discover_map


@dataclass
class RLTrackEnvConfig:
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
    time_penalty_per_sec: float = 0.2
    lap_bonus: float = 50.0
    collision_penalty: float = 50.0
    randomize_start: bool = False
    render: bool = False


class F1TenthRLTrackEnv(gym.Env):
    """LiDAR-to-steer/speed Gymnasium wrapper with progress-based reward."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: RLTrackEnvConfig):
        super().__init__()
        self.cfg = cfg
        map_yaml, map_ext, centerline_csv = discover_map(cfg.maps_root, cfg.map_name)
        self.centerline = Centerline.from_csv(centerline_csv)
        self._segments = np.roll(self.centerline.points, -1, axis=0) - self.centerline.points
        self._segment_lengths = np.linalg.norm(self._segments, axis=1)
        self._segment_lengths_sq = np.maximum(self._segment_lengths ** 2, 1e-8)
        self._cumulative = np.concatenate([[0.0], np.cumsum(self._segment_lengths)])[:-1]
        self._total_length = float(np.sum(self._segment_lengths))
        self._waypoints = self.centerline.points

        initial_yaw = float(np.arctan2(
            self._segments[0, 1], self._segments[0, 0]))
        initial_state = np.array([
            self._waypoints[0, 0], self._waypoints[0, 1], 0.0, 0.0,
            initial_yaw, 0.0, 0.0,
        ], dtype=np.float32)
        env_cfg = F1TenthEnvConfig(
            map_yaml=map_yaml, map_ext=map_ext, timestep=cfg.timestep,
            control_hz=cfg.control_hz, max_steer_rad=cfg.max_steer_rad,
            max_speed_mps=cfg.max_speed_mps, render=cfg.render,
            lidar_dim=cfg.lidar_dim, action_repeat=1,
        )
        self.sim = F1TenthEnvironment(
            model=None, state_normalizer=None, action_normalizer=None,
            cfg=env_cfg, initial_state_raw=initial_state,
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(cfg.lidar_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._max_steps = max(1, int(cfg.max_episode_seconds * cfg.control_hz))
        self._step_count = 0
        self._laps = 0
        self._prev_s = 0.0
        self._start_s = 0.0
        self._lap_start_step = 0
        self._lap_times_sec: list[float] = []

    def _project(self, xy: np.ndarray) -> tuple[float, float]:
        offset = xy[None, :] - self._waypoints
        t = np.sum(offset * self._segments, axis=1) / self._segment_lengths_sq
        t = np.clip(t, 0.0, 1.0)
        projected = self._waypoints + t[:, None] * self._segments
        delta = xy[None, :] - projected
        d2 = np.sum(delta * delta, axis=1)
        idx = int(np.argmin(d2))
        progress = float(self._cumulative[idx] + t[idx] * self._segment_lengths[idx])
        return progress, float(np.sqrt(d2[idx]))

    def _observation(self) -> np.ndarray:
        scan = self.sim.get_lidar()
        return np.clip(scan / self.cfg.lidar_max_range, 0.0, 1.0).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        start_idx = 0
        if options and "start_index" in options:
            start_idx = int(options["start_index"]) % len(self._waypoints)
        elif self.cfg.randomize_start:
            start_idx = int(self.np_random.integers(len(self._waypoints)))
        yaw = float(np.arctan2(self._segments[start_idx, 1], self._segments[start_idx, 0]))
        state = np.array([
            self._waypoints[start_idx, 0], self._waypoints[start_idx, 1], 0.0, 0.0,
            yaw, 0.0, 0.0,
        ], dtype=np.float32)
        self.sim.reset(state)
        self._step_count = 0
        self._laps = 0
        self._start_s, _ = self._project(state[:2])
        self._prev_s = 0.0
        self._lap_start_step = 0
        self._lap_times_sec = []
        return self._observation(), {"map_name": self.cfg.map_name}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        steering = float(action[0] * self.cfg.max_steer_rad)
        speed = float((action[1] + 1.0) * 0.5 * self.cfg.max_speed_mps)
        state = self.sim.step_raw_action(np.array([steering, speed], dtype=np.float32))
        self._step_count += 1

        absolute_progress, cross_track = self._project(state[:2])
        progress = float((absolute_progress - self._start_s) % self._total_length)
        raw_delta_s = progress - self._prev_s
        lap_completed = raw_delta_s < -0.5 * self._total_length
        delta_s = raw_delta_s
        if delta_s < -0.5 * self._total_length:
            delta_s += self._total_length
        elif delta_s > 0.5 * self._total_length:
            delta_s -= self._total_length
        delta_s = float(np.clip(delta_s, -2.0, 2.0))
        reward = (
            self.cfg.progress_weight * delta_s
            - self.cfg.cross_track_weight * cross_track
            + self.cfg.speed_weight * max(float(state[3]), 0.0)
            - self.cfg.time_penalty_per_sec / self.cfg.control_hz
        )
        self._prev_s = progress

        if lap_completed:
            self._laps += 1
            self._lap_times_sec.append(
                (self._step_count - self._lap_start_step) / self.cfg.control_hz)
            self._lap_start_step = self._step_count
            reward += self.cfg.lap_bonus

        collision = bool(self.sim.crashed)
        if collision:
            reward -= self.cfg.collision_penalty
        terminated = collision or lap_completed
        truncated = self._step_count >= self._max_steps and not terminated
        info = {
            "map_name": self.cfg.map_name,
            "laps_completed": self._laps,
            "lap_times_sec": list(self._lap_times_sec),
            "collision": collision,
            "progress_m": progress,
            "total_track_m": self._total_length,
            "speed_mps": float(state[3]),
            "target_speed_mps": speed,
            "elapsed_time_sec": self._step_count / self.cfg.control_hz,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        self.sim.close()


def rl_env_config_from_project(config_path: str, map_name: str,
                               max_episode_seconds: float = 120.0,
                               randomize_start: bool = False,
                               render: bool = False,
                               time_penalty_per_sec: float = 0.2) -> RLTrackEnvConfig:
    cfg = load_config(config_path)
    return RLTrackEnvConfig(
        map_name=map_name, maps_root=cfg.data.maps_root,
        lidar_dim=cfg.data.lidar_dim, lidar_max_range=cfg.data.lidar_max_range,
        timestep=cfg.f1tenth_env.timestep, control_hz=cfg.f1tenth_env.control_hz,
        max_steer_rad=cfg.f1tenth_env.max_steer_rad,
        max_speed_mps=cfg.f1tenth_env.max_speed_mps,
        max_episode_seconds=max_episode_seconds,
        randomize_start=randomize_start,
        render=render,
        time_penalty_per_sec=time_penalty_per_sec,
    )
