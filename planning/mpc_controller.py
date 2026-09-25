from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from planning.reward import CenterlineProgressReward, calculate_progress






@dataclass
class KinematicBicycleParams:
    wheelbase_m: float = 0.33
    max_steer_rad: float = 0.4
    max_speed_mps: float = 8.0
    min_speed_mps: float = 0.0
    max_accel_mps2: float = 8.0


def bicycle_rollout(state0: np.ndarray, actions: np.ndarray, dt: float,
                     params: KinematicBicycleParams) -> np.ndarray:
    """Returns (H, 4) array of [x, y, yaw, v] states, one row per action step."""
    x, y, yaw, v = state0
    out = np.zeros((actions.shape[0], 4), dtype=np.float32)
    for h in range(actions.shape[0]):
        steer = float(np.clip(actions[h, 0], -params.max_steer_rad, params.max_steer_rad))
        speed_cmd = float(np.clip(actions[h, 1], params.min_speed_mps, params.max_speed_mps))

        accel = np.clip((speed_cmd - v) / dt, -params.max_accel_mps2, params.max_accel_mps2)
        v = float(np.clip(v + accel * dt, params.min_speed_mps, params.max_speed_mps))
        yaw = yaw + (v / params.wheelbase_m) * np.tan(steer) * dt
        x = x + v * np.cos(yaw) * dt
        y = y + v * np.sin(yaw) * dt

        out[h] = [x, y, yaw, v]
    return out


@dataclass
class MPCConfig:
    horizon: int = 15
    population_size: int = 512
    cem_iters: int = 5
    elite_frac: float = 0.15
    init_steer_std: float = 0.15
    init_speed_std: float = 1.5
    dt: float = 0.05

    @property
    def num_elite(self) -> int:
        return max(1, int(round(self.population_size * self.elite_frac)))


class KinematicMPCController:

    def __init__(self, reward_fn: CenterlineProgressReward, params: KinematicBicycleParams,
                 cfg: MPCConfig, seed: int = 0,
                 classic_action_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.reward_fn = reward_fn
        self.params = params
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.classic_action_fn = classic_action_fn

        if self.reward_fn.reward_mode == "tal" and classic_action_fn is None:
            raise ValueError("reward_mode='tal' requires classic_action_fn to be provided")

    def _state4(self, raw_state: np.ndarray) -> np.ndarray:
        return np.array([raw_state[0], raw_state[1], raw_state[4], raw_state[3]], dtype=np.float32)

    def act(self, raw_state: np.ndarray) -> np.ndarray:
        cfg, params = self.cfg, self.params
        state0 = self._state4(raw_state)

        mean = np.zeros((cfg.horizon, 2), dtype=np.float32)
        mean[:, 1] = min(3.0, params.max_speed_mps)
        std = np.array([cfg.init_steer_std, cfg.init_speed_std], dtype=np.float32)
        std = np.tile(std, (cfg.horizon, 1))

        best_actions, best_reward = mean.copy(), -np.inf

        for _ in range(cfg.cem_iters):
            samples = mean[None] + std[None] * self.rng.normal(
                size=(cfg.population_size, cfg.horizon, 2)
            ).astype(np.float32)

            positions = np.zeros((cfg.population_size, cfg.horizon, 2), dtype=np.float32)
            headings = np.zeros((cfg.population_size, cfg.horizon), dtype=np.float32)
            velocities = np.zeros((cfg.population_size, cfg.horizon), dtype=np.float32)

            for k in range(cfg.population_size):
                traj = bicycle_rollout(state0, samples[k], cfg.dt, params)
                positions[k] = traj[:, 0:2]
                headings[k] = traj[:, 2]
                velocities[k] = traj[:, 3]

            init_position = np.tile(state0[0:2], (cfg.population_size, 1))

            classic_actions = None
            if self.reward_fn.reward_mode == "tal":
                classic_actions = np.zeros_like(samples)
                for k in range(cfg.population_size):
                    for l in range(cfg.horizon):
                        s_l = np.array([positions[k, l, 0], positions[k, l, 1],
                                         headings[k, l], velocities[k, l]], dtype=np.float32)
                        classic_actions[k, l] = self.classic_action_fn(s_l)

            rewards = self.reward_fn(
                positions,
                headings=headings,
                velocities=velocities,
                agent_actions=samples if self.reward_fn.reward_mode == "tal" else None,
                classic_actions=classic_actions,
                collisions=None,
                lap_done=None,
                init_position=init_position,
            )

            elite = np.argsort(-rewards)[: cfg.num_elite]
            mean = samples[elite].mean(axis=0)
            std = samples[elite].std(axis=0).clip(min=1e-3)

            top1 = elite[0]
            if rewards[top1] > best_reward:
                best_reward = float(rewards[top1])
                best_actions = samples[top1].copy()

        return best_actions[0]


def classical_mpc_race(controller: KinematicMPCController, env, waypoints_raw: np.ndarray,
             control_hz: int, max_seconds: float = 60.0, max_laps: int = 1) -> dict:

    trajectory = [env.get_state().copy()]
    wpts_list = waypoints_raw.tolist()

    seg_len = np.linalg.norm(np.diff(waypoints_raw, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_s = float(cum_dist[-1]) + 1e-8

    max_ticks = int(max_seconds * control_hz)
    laps_completed = 0
    collision_occurred = False  
    prev_s = calculate_progress(env.get_state()[0:2], wpts_list, cum_dist)

    for _ in range(max_ticks):
        raw_state = env.get_state()
        action = controller.act(raw_state)
        next_state = env.step_raw_action(action)
        trajectory.append(next_state.copy())

        if getattr(env, "crashed", False):
            collision_occurred = True
            print(f"[classical_mpc_race] stopping: collision detected "
                  f"(control step {getattr(env, 'crash_step', None)}).")
            break

        s = calculate_progress(next_state[0:2], wpts_list, cum_dist)
        if s < prev_s - total_s * 0.5: 
            laps_completed += 1
        prev_s = s

        if laps_completed >= max_laps:
            break

    return {
        "trajectory_raw": np.stack(trajectory, axis=0),
        "laps_completed": laps_completed,
        "final_progress_s": prev_s,
        "total_track_s": total_s,
        "collision": collision_occurred, 
    }