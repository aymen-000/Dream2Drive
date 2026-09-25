from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

import numpy as np
import torch

from models.skill_model import OPOSMSkillModel
from utils.normalization import Normalizer
from planning.reward import CenterlineProgressReward, calculate_progress, get_distance


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
    save_dir: Optional[str] = None       # if set, plan() dumps an .npz per call here
    save_population: bool = True         # save the full last-iteration population, not just the best

    @property
    def num_elite(self) -> int:
        return max(1, int(round(self.population_size * self.elite_frac)))


RewardFn = Callable[..., np.ndarray]


class Environment(Protocol):

    def get_state(self) -> np.ndarray: ...

    def execute_skill(self, z: np.ndarray, horizon: int) -> np.ndarray:
        """Executes skill z for `horizon` low-level control ticks and
        returns the resulting raw state."""
        ...


class CEMPlanner:
    def __init__(self, model: OPOSMSkillModel, state_normalizer: Normalizer,
                 reward_fn: CenterlineProgressReward, cfg: CEMPlannerConfig,
                 collision_checker: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.model = model
        self.normalizer = state_normalizer
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.collision_checker = collision_checker
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if self.reward_fn.reward_mode == "cth" and (cfg.yaw_idx is None):
            raise ValueError(
                "reward_mode='cth' needs cfg.yaw_idx set to the column of "
                "predicted_raw containing yaw -- this planner has no way to "
                "know that on its own; set it explicitly."
            )
        if self.reward_fn.reward_mode == "cth" and (cfg.v_idx is None):
            raise ValueError(
                "reward_mode='cth' needs cfg.v_idx set to the column of "
                "predicted_raw containing velocity -- set it explicitly."
            )
        if self.reward_fn.reward_mode == "tal":
            raise NotImplementedError(
                "reward_mode='tal' requires a per-(k,l) classical-planner "
                "action to compare against the skill model's implied action, "
                "which this skill-based planner has no direct analogue for "
                "(skills aren't [steer, speed] actions). Not implemented here "
            )

        if cfg.save_dir is not None:
            os.makedirs(cfg.save_dir, exist_ok=True)

    def _rollout_population(self, s0_norm: torch.Tensor, epsilons: torch.Tensor) -> torch.Tensor:
        K, L, _ = epsilons.shape
        s = s0_norm.unsqueeze(0).expand(K, -1).clone()
        predicted = []
        with torch.no_grad():
            for l in range(L):
                mu0, sigma0 = self.model.skill_prior(s)
                z = mu0 + sigma0 * epsilons[:, l, :]
                mu_next, _ = self.model.tawm(s, z)
                s = mu_next
                predicted.append(s)
        return torch.stack(predicted, dim=1)

    def plan(self, s0_raw: np.ndarray, step_idx: Optional[int] = None) -> dict:
        """
        step_idx: optional index for this planning call (e.g. mpc_race's replan
        counter). Used only to name the saved .npz file when cfg.save_dir is set.
        """
        cfg = self.cfg
        z_dim = self.model.z_dim

        s0_norm = torch.as_tensor(
            self.normalizer.transform(s0_raw[None, :])[0], dtype=torch.float32, device=self.device
        )

        mean = torch.zeros(cfg.plan_length, z_dim, device=self.device)
        std = torch.full((cfg.plan_length, z_dim), cfg.init_std, device=self.device)

        best_epsilon, best_reward, best_predicted_norm, best_collisions = None, -np.inf, None, None

        # kept from the LAST cem_iter only, for visualization of the final population
        last_positions, last_rewards, last_elite_idx = None, None, None

        for _ in range(cfg.cem_iters):
            eps = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                cfg.population_size, cfg.plan_length, z_dim, device=self.device
            )

            predicted_norm = self._rollout_population(s0_norm, eps)
            predicted_raw = self.normalizer.inverse_transform(
                predicted_norm.detach().cpu().numpy().reshape(-1, predicted_norm.shape[-1])
            ).reshape(predicted_norm.shape)

            positions = predicted_raw[:, :, [cfg.x_idx, cfg.y_idx]]  # (K, L, 2)

            headings = None
            velocities = None
            if self.reward_fn.reward_mode == "cth":
                headings = predicted_raw[:, :, cfg.yaw_idx]
                velocities = predicted_raw[:, :, cfg.v_idx]
            # TODO : I nedd here to add if condition to solve the problem of TAL type reward since it needs actions

            init_position = np.tile(s0_raw[[cfg.x_idx, cfg.y_idx]], (cfg.population_size, 1))
            collisions = self._imagined_collisions(positions)

            rewards = self.reward_fn(
                positions,
                headings=headings,
                velocities=velocities,
                agent_actions=None,
                classic_actions=None,
                collisions=None,
                lap_done=None,
                init_position=init_position,
            )

            elite_idx = np.argsort(-rewards)[: cfg.num_elite]
            elite_eps = eps[elite_idx]
            mean = elite_eps.mean(dim=0)
            std = elite_eps.std(dim=0).clamp_min(1e-3)

            top1 = elite_idx[0]
            if rewards[top1] > best_reward:
                best_reward = float(rewards[top1])
                best_epsilon = eps[top1].detach().clone()
                best_predicted_norm = predicted_norm[top1].detach().clone()
                best_collisions = None if collisions is None else collisions[top1].copy()

            # NEW: remember this iteration's population -- overwritten each loop,
            # so after the loop these hold the LAST iteration's data
            last_positions = positions
            last_rewards = rewards
            last_elite_idx = elite_idx

        best_predicted_raw = self.normalizer.inverse_transform(best_predicted_norm.cpu().numpy())
        print(f"[CEMPlanner] best reward: {best_reward:.3f}")

        # NEW: dump the imagined rollouts for this planning call
        if cfg.save_dir is not None:
            self._save_planning_step(
                step_idx=step_idx,
                s0_raw=s0_raw,
                population_positions=last_positions,
                population_rewards=last_rewards,
                elite_idx=last_elite_idx,
                best_predicted_positions=best_predicted_raw[:, [cfg.x_idx, cfg.y_idx]],
                best_reward=best_reward,
            )

        return {
            "epsilon_seq": best_epsilon.cpu().numpy(),
            "predicted_states_raw": best_predicted_raw,
            "reward": best_reward,
            "predicted_collisions": best_collisions,
        }

    def _save_planning_step(self, step_idx, s0_raw, population_positions, population_rewards,
                             elite_idx, best_predicted_positions, best_reward) -> None:
        """Save one planning call's imagined rollouts to cfg.save_dir/plan_step_XXXX.npz."""
        cfg = self.cfg
        fname = f"plan_step_{step_idx:05d}.npz" if step_idx is not None else "plan_latest.npz"
        path = os.path.join(cfg.save_dir, fname)

        save_kwargs = dict(
            init_position=s0_raw[[cfg.x_idx, cfg.y_idx]],
            elite_idx=elite_idx,
            best_predicted_positions=best_predicted_positions,   # (L, 2)
            best_reward=np.float32(best_reward),
        )
        if cfg.save_population:
            save_kwargs["population_positions"] = population_positions.astype(np.float32)  # (K, L, 2)
            save_kwargs["population_rewards"] = population_rewards.astype(np.float32)       # (K,)

        np.savez_compressed(path, **save_kwargs)

    def _imagined_collisions(self, positions: np.ndarray) -> Optional[np.ndarray]:
        """Check every imagined terminal skill state against the map.

        The world model predicts terminal states only.  Interpolating a
        straight line between two terminals is not a valid vehicle rollout on
        a curved track, so collision scoring is intentionally applied to the
        states the model actually imagines.
        """
        if self.collision_checker is None:
            return None
        return np.asarray(self.collision_checker(positions), dtype=bool)

    def epsilon_to_z(self, s_raw: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        s_norm = torch.as_tensor(
            self.normalizer.transform(s_raw[None, :])[0], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        eps_t = torch.as_tensor(epsilon, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu0, sigma0 = self.model.skill_prior(s_norm)
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
    stopped_for_predicted_collision = False
    collision_occurred = False  
    plan_call_idx = 0  

    prev_s, prev_i = calculate_progress_local(
        env.get_state()[[cfg.x_idx, cfg.y_idx]], wpts_list, cum_dist, prev_i=None
    )

    for _ in range(max_skills):
        if laps_completed >= max_laps or collision_occurred:
            break
        plan = planner.plan(env.get_state(), step_idx=plan_call_idx)  # NEW: step_idx passed through
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
            s_raw = env.get_state()
            z_k = planner.epsilon_to_z(s_raw, eps_k)
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
            prev_s = s
            prev_i = cur_i  # carry the index forward for the next step's window search
            print(f"[mpc_race] progress s={s:.3f}/{total_s:.3f}, laps={laps_completed}")

        if laps_completed >= max_laps or collision_occurred:
            break

    return {
        "trajectory_raw": np.stack(trajectory, axis=0),
        "executed_zs": np.stack(executed_zs, axis=0) if executed_zs else np.zeros((0, planner.model.z_dim)),
        "laps_completed": laps_completed,
        "final_progress_s": prev_s,
        "total_track_s": total_s,
        "stopped_for_predicted_collision": stopped_for_predicted_collision,
        "collision": collision_occurred,   
    }