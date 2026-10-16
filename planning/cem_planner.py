from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

import numpy as np
import torch

from models.skill_model import OPOSMSkillModel
from planning.reward import LidarStructuralReward, calculate_progress, get_distance


@dataclass
class CEMPlannerConfig:
    plan_length: int = 8
    population_size: int = 500
    elite_frac: float = 0.10
    cem_iters: int = 10
    init_std: float = 1.0
    device: str = "cuda"
    x_idx: int = 0
    y_idx: int = 1
    yaw_idx: Optional[int] = 4
    v_idx: Optional[int] = 3

    # --- NEW: imagined-rollout saving ---
    save_dir: Optional[str] = "outputs/track"       # if set, plan() dumps an .npz per call here
    save_population: bool = True         # save the full last-iteration population, not just the best
    forward_reward_weight: float = 1.0
    symmetry_penalty_weight: float = 0.2
    lidar_collision_penalty: float = 100.0
    safe_distance_m: float = 0.35
    forward_beam_fraction: float = 0.30

    @property
    def num_elite(self) -> int:
        return max(1, int(round(self.population_size * self.elite_frac)))


RewardFn = Callable[..., np.ndarray]


class Environment(Protocol):

    def get_state(self) -> np.ndarray: ...
    def get_lidar(self) -> np.ndarray: ...

    def execute_skill(self, z: np.ndarray, horizon: int) -> np.ndarray:
        """Executes skill z for `horizon` low-level control ticks and
        returns the resulting raw state."""
        ...


class CEMPlanner:
    def __init__(self, model: OPOSMSkillModel,
                 reward_fn: LidarStructuralReward, cfg: CEMPlannerConfig,
                 collision_checker: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.model = model
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.collision_checker = collision_checker
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if cfg.save_dir is not None:
            os.makedirs(cfg.save_dir, exist_ok=True)

    def _rollout_population(self, o0: torch.Tensor, epsilons: torch.Tensor) -> torch.Tensor:
        K, L, _ = epsilons.shape
        lidar = o0.expand(K, -1).clone()
        predicted_lidar = []
        with torch.no_grad():
            for l in range(L):
                lidar_embed = self.model.lidar_encoder(lidar)
                mu0, sigma0 = self.model.skill_prior(lidar_embed)
                z = mu0 + sigma0 * epsilons[:, l, :]
                mu_next, _ = self.model.tawm(lidar_embed, z)
                lidar = mu_next.clamp(0.0, 1.0) * self.model.lidar_max_range
                predicted_lidar.append(lidar)
        return torch.stack(predicted_lidar, dim=1)

    def plan(self, o0_raw: np.ndarray, step_idx: Optional[int] = None) -> dict:
        cfg = self.cfg
        z_dim = self.model.z_dim

        o0 = torch.as_tensor(o0_raw, dtype=torch.float32, device=self.device).reshape(1, -1)

        mean = torch.zeros(cfg.plan_length, z_dim, device=self.device)
        std = torch.full((cfg.plan_length, z_dim), cfg.init_std, device=self.device)

        best_epsilon, best_reward, best_predicted_lidar = None, -np.inf, None

        # kept from the LAST cem_iter only, for visualization of the final population
        last_population_lidar, last_rewards, last_elite_idx = None, None, None

        for _ in range(cfg.cem_iters):
            eps = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                cfg.population_size, cfg.plan_length, z_dim, device=self.device
            )

            predicted_lidar = self._rollout_population(o0, eps)
            predicted_lidar_np = predicted_lidar.detach().cpu().numpy()
            rewards = self.reward_fn(predicted_lidar_np)

            elite_idx = np.argsort(-rewards)[: cfg.num_elite]
            elite_eps = eps[elite_idx]
            mean = elite_eps.mean(dim=0)
            std = elite_eps.std(dim=0).clamp_min(1e-3)

            top1 = elite_idx[0]
            if rewards[top1] > best_reward:
                best_reward = float(rewards[top1])
                best_epsilon = eps[top1].detach().clone()
                best_predicted_lidar = predicted_lidar[top1].detach().clone()

            # NEW: remember this iteration's population -- overwritten each loop,
            # so after the loop these hold the LAST iteration's data
            last_population_lidar = predicted_lidar_np
            last_rewards = rewards
            last_elite_idx = elite_idx

        print(f"[CEMPlanner] best reward: {best_reward:.3f}")

        # NEW: dump the imagined rollouts for this planning call
        if cfg.save_dir is not None:
            self._save_planning_step(
                step_idx=step_idx,
                population_lidar=last_population_lidar,
                population_rewards=last_rewards,
                elite_idx=last_elite_idx,
                best_predicted_lidar=best_predicted_lidar.cpu().numpy(),
                best_reward=best_reward,
            )

        return {
            "epsilon_seq": best_epsilon.cpu().numpy(),
            "predicted_lidar": best_predicted_lidar.cpu().numpy(),
            "reward": best_reward,
            "predicted_collisions": (
                best_predicted_lidar[:, self.reward_fn.forward_slice].min(dim=-1).values.cpu().numpy()
                < self.reward_fn.safe_distance
            ),
        }

    def _save_planning_step(self, step_idx, population_lidar, population_rewards,
                             elite_idx, best_predicted_lidar, best_reward) -> None:
        """Save one planning call's imagined rollouts to cfg.save_dir/plan_step_XXXX.npz."""
        cfg = self.cfg
        fname = f"plan_step_{step_idx:05d}.npz" if step_idx is not None else "plan_latest.npz"
        path = os.path.join(cfg.save_dir, fname)

        save_kwargs = dict(
            elite_idx=elite_idx,
            best_predicted_lidar=best_predicted_lidar,   # (L, beams)
            best_reward=np.float32(best_reward),
        )
        if cfg.save_population:
            save_kwargs["population_lidar"] = population_lidar.astype(np.float32)  # (K, L, beams)
            save_kwargs["population_rewards"] = population_rewards.astype(np.float32)       # (K,)

        np.savez_compressed(path, **save_kwargs)

    @torch.no_grad()
    def epsilon_to_z(self, o_raw: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        o_embed = self.model.lidar_encoder(torch.as_tensor(o_raw, dtype=torch.float32, device=self.device).reshape(1, -1))
        eps_t = torch.as_tensor(epsilon, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu0, sigma0 = self.model.skill_prior(o_embed)
            z = mu0 + sigma0 * eps_t
        return z.squeeze(0).cpu().numpy()


def calculate_progress_local(point, wpts, cum_dist, prev_i=None, window=15):
    n = len(wpts)
    if prev_i is None:
        search_idx = range(n)
    else:
        search_idx = [i % n for i in range(prev_i - window, prev_i + window)]
    dists = [get_distance(point, wpts[i]) for i in search_idx]
    min_local = int(np.argmin(dists))
    min_i = search_idx[min_local]
    if min_i == n - 1:
        min_i -= 1
    p_i, p_ii = wpts[min_i], wpts[min_i + 1]
    seg_vec = np.array(p_ii) - np.array(p_i)
    seg_len = np.linalg.norm(seg_vec) + 1e-8
    t = np.clip(np.dot(np.array(point) - np.array(p_i), seg_vec) / (seg_len ** 2), 0.0, 1.0)
    s = cum_dist[min_i] + t * seg_len
    return s, min_i


def mpc_race(planner: CEMPlanner, env: Environment, waypoints_raw: np.ndarray,
             skill_horizon: int, max_skills: int = 200, max_laps: int = 1,
             replan_every_skills: int = 1, save_dir: Optional[str] = None) -> dict:
    trajectory = [env.get_state().copy()]
    executed_zs: List[np.ndarray] = []

    cfg = planner.cfg
    wpts_list = waypoints_raw.tolist()
    seg_len = np.linalg.norm(np.diff(waypoints_raw, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_s = float(cum_dist[-1]) + 1e-8

    # NEW: wire up saving for this race
    if save_dir is not None:
        planner.cfg.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(os.path.join(save_dir, "track.npz"), waypoints=waypoints_raw)

    laps_completed = 0
    lap_times_sec: List[float] = []
    stopped_for_predicted_collision = False
    collision_occurred = False  
    plan_call_idx = 0  
    start_elapsed = env.get_elapsed_time() if hasattr(env, "get_elapsed_time") else 0.0

    prev_s, prev_i = calculate_progress_local(
        env.get_state()[[cfg.x_idx, cfg.y_idx]], wpts_list, cum_dist, prev_i=None
    )

    for _ in range(max_skills):
        if laps_completed >= max_laps or collision_occurred:
            break
        plan_lidar = env.get_lidar()
        plan = planner.plan(plan_lidar, step_idx=plan_call_idx)
        plan_call_idx += 1
        execute_count = min(replan_every_skills, plan["epsilon_seq"].shape[0])
        #predicted_collisions = plan.get("predicted_collisions")
        #if predicted_collisions is not None and np.any(predicted_collisions[:execute_count]):
        #    first_collision = int(np.flatnonzero(predicted_collisions)[0])
        #    print(
        #        "[mpc_race] stopping: the best imagined plan predicts a map "
        #        f"collision at skill {first_collision + 1}."
        #    )
        #    stopped_for_predicted_collision = True
        #    break

        for k in range(execute_count):
            eps_k = plan["epsilon_seq"][k]
            prior_lidar = plan_lidar if k == 0 else plan["predicted_lidar"][k - 1]
            z_k = planner.epsilon_to_z(prior_lidar, eps_k)
            next_state = env.execute_skill(z_k, skill_horizon)
            trajectory.append(next_state.copy())
            executed_zs.append(z_k)
            if getattr(env, "crashed", False):
                collision_occurred = True
                print(f"[mpc_race] stopping: collision detected "
                      f"(control step {getattr(env, 'crash_step', None)}).")
                break

            s, cur_i = calculate_progress_local(
                next_state[[cfg.x_idx, cfg.y_idx]], wpts_list, cum_dist, prev_i=prev_i
            )
            if s < prev_s - total_s * 0.5:  # wrapped past s=0 -> lap completed
                laps_completed += 1
                if hasattr(env, "get_elapsed_time"):
                    lap_times_sec.append(env.get_elapsed_time() - start_elapsed)
            prev_s = s
            prev_i = cur_i  # carry the index forward for the next step's window search
            print(f"[mpc_race] progress s={s:.3f}/{total_s:.3f}, laps={laps_completed}")

        if laps_completed >= max_laps or collision_occurred:
            break

    return {
        "trajectory_raw": np.stack(trajectory, axis=0),
        "executed_zs": np.stack(executed_zs, axis=0) if executed_zs else np.zeros((0, planner.model.z_dim)),
        "laps_completed": laps_completed,
        "lap_times_sec": lap_times_sec,
        "final_progress_s": prev_s,
        "total_track_s": total_s,
        "stopped_for_predicted_collision": stopped_for_predicted_collision,
        "collision": collision_occurred,   
    }
