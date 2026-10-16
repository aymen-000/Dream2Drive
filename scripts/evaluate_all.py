from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from envs.f1tenth_env import F1TenthEnvConfig, F1TenthEnvironment
from models.skill_model import OPOSMSkillModel
from planning.cem_planner import CEMPlanner, CEMPlannerConfig, direct_skill_race, mpc_race
from planning.reward import LidarStructuralReward
from rl_baselines.f1tenth import F1TenthRLTrackEnv, rl_env_config_from_project
from scripts.config_utils import load_config, resolve_device
from scripts.evaluate import discover_available_maps
from utils.normalization import Normalizer
from utils.track import Centerline, discover_map


def load_oposm(cfg: Any, checkpoint: str, device: str) -> OPOSMSkillModel:
    model = OPOSMSkillModel(
        state_dim=len(cfg.data.state_features), action_dim=len(cfg.data.action_features),
        z_dim=cfg.model.z_dim, hidden_dim=cfg.model.hidden_dim,
        gru_hidden_dim=cfg.model.gru_hidden_dim,
        gru_bidirectional=cfg.model.gru_bidirectional, min_std=cfg.model.min_std,
        lidar_dim=cfg.data.lidar_dim, obs_embed_dim=cfg.model.obs_embed_dim,
        lidar_max_range=cfg.data.lidar_max_range,
        tawm_predict_delta=cfg.model.tawm_predict_delta,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device).eval()
    return model


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_start(centerline: Centerline, seed: int) -> tuple[np.ndarray, float]:
    points = centerline.points
    idx = int(np.random.default_rng(seed).integers(len(points)))
    next_idx = (idx + 1) % len(points)
    direction = points[next_idx] - points[idx]
    yaw = float(np.arctan2(direction[1], direction[0]))
    state = np.array([points[idx, 0], points[idx, 1], 0., 0., yaw, 0., 0.], dtype=np.float32)
    return state, float(idx)


def evaluate_oposm_episode(model, cfg, state_norm, action_norm, device, map_name,
                           seed: int, mode: str) -> dict[str, Any]:
    map_yaml, map_ext, centerline_csv = discover_map(cfg.data.maps_root, map_name)
    centerline = Centerline.from_csv(centerline_csv)
    start_state, start_index = _seed_start(centerline, seed)
    waypoints = centerline.subsample_waypoints(cfg.planning.waypoint_spacing_m)
    reward = LidarStructuralReward(
        lidar_dim=model.lidar_dim,
        forward_weight=cfg.planning.forward_reward_weight,
        symmetry_weight=cfg.planning.symmetry_penalty_weight,
        collision_penalty=cfg.planning.lidar_collision_penalty,
        safe_distance=cfg.planning.safe_distance_m,
        forward_fraction=cfg.planning.forward_beam_fraction,
    )
    planner_cfg = CEMPlannerConfig(
        plan_length=cfg.planning.plan_length,
        population_size=cfg.planning.population_size,
        elite_frac=cfg.planning.elite_frac, cem_iters=cfg.planning.cem_iters,
        init_std=cfg.planning.init_std, device=device, save_dir=None,
    )
    planner = CEMPlanner(model, reward, planner_cfg)
    env_cfg = F1TenthEnvConfig(
        map_yaml=map_yaml, map_ext=map_ext, timestep=cfg.f1tenth_env.timestep,
        control_hz=cfg.f1tenth_env.control_hz,
        max_steer_rad=cfg.f1tenth_env.max_steer_rad,
        max_speed_mps=cfg.f1tenth_env.max_speed_mps, render=False,
        lidar_dim=cfg.data.lidar_dim, action_repeat=cfg.data.subsample_stride,
    )
    env = F1TenthEnvironment(model, state_norm, action_norm, env_cfg, start_state, device=device)
    try:
        race = direct_skill_race if mode == "direct" else mpc_race
        kwargs = {} if mode == "direct" else {"replan_every_skills": cfg.planning.replan_every_skills}
        result = race(planner, env, waypoints, skill_horizon=cfg.data.subtraj_len - 1,
                      max_skills=cfg.planning.max_skills_per_episode,
                      max_laps=cfg.planning.max_laps, **kwargs)
        track_len = centerline.total_length
        start_s = centerline.cum_dist[int(start_index)]
        projected = [centerline.progress_m(state[:2])
                     for state in result["trajectory_raw"]]
        progress = max(float((s - start_s) % track_len) for s in projected)
        lap_times = [float(t) for t in result.get("lap_times_sec", [])]
        return {
            "method": "OPOSM direct" if mode == "direct" else "OPOSM CEM-MPC",
            "map_name": map_name, "seed": seed, "start_index": start_index,
            "laps_completed": int(result["laps_completed"]),
            "success": bool(result["laps_completed"] > 0),
            "collision": bool(result.get("collision", False)),
            "max_progress_fraction": float(np.clip(progress / track_len, 0., 1.)),
            "max_progress_m": progress, "track_length_m": track_len,
            "lap_times_sec": lap_times,
        }
    finally:
        env.close()


def evaluate_ppo_episode(model, config_path, map_name, seed: int, max_seconds: float) -> dict[str, Any]:
    env_cfg = rl_env_config_from_project(
        config_path, map_name, max_episode_seconds=max_seconds, randomize_start=True)
    env = F1TenthRLTrackEnv(env_cfg)
    try:
        start_index = int(np.random.default_rng(seed).integers(len(env.centerline.points)))
        obs, _ = env.reset(seed=seed, options={"start_index": start_index})
        terminated = truncated = False
        info: dict[str, Any] = {}
        max_progress = 0.
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            max_progress = max(max_progress, float(info.get("progress_m", 0.)))
        track_len = float(info.get("total_track_m", 0.))
        laps = int(info.get("laps_completed", 0))
        return {
            "method": "RL baseline", "map_name": map_name, "seed": seed,
            "start_index": start_index,
            "laps_completed": laps, "success": laps > 0,
            "collision": bool(info.get("collision", False)),
            "max_progress_fraction": float(np.clip(max_progress / track_len, 0., 1.)) if track_len else 0.,
            "max_progress_m": max_progress, "track_length_m": track_len,
            "lap_times_sec": [float(t) for t in info.get("lap_times_sec", [])],
        }
    finally:
        env.close()


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    progress = [r["max_progress_fraction"] for r in runs]
    successes = [r for r in runs if r["success"]]
    lap_times = [t for r in successes for t in r["lap_times_sec"]]
    return {
        "episodes": len(runs),
        "success_rate": float(np.mean([r["success"] for r in runs])) if runs else None,
        "successes": len(successes),
        "max_progress_fraction_mean": float(np.mean(progress)) if progress else None,
        "max_progress_fraction_std": float(np.std(progress, ddof=1)) if len(progress) > 1 else 0.,
        "successful_lap_time_mean_sec": float(np.mean(lap_times)) if lap_times else None,
        "successful_lap_time_std_sec": float(np.std(lap_times, ddof=1)) if len(lap_times) > 1 else None,
        "collision_rate": float(np.mean([r["collision"] for r in runs])) if runs else None,
    }


def _mean_ci(values: list[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if not len(x):
        return float("nan"), float("nan")
    # Normal 95% interval; report one observation as a point without error bars.
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.


def make_figures(results: dict[str, list[dict[str, Any]]], out_dir: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.labelsize": 10, "legend.fontsize": 9,
                         "figure.dpi": 150, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"RL baseline": "#4C78A8", "OPOSM direct": "#F58518", "OPOSM CEM-MPC": "#54A24B"}
    methods = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True)

    # Per episode maximum progress, with individual seed outcomes and mean 95% CI.
    ax = axes[0]
    for i, method in enumerate(methods):
        vals = [r["max_progress_fraction"] for r in results[method]]
        jitter = np.linspace(-0.08, 0.08, max(1, len(vals)))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=16, alpha=.55,
                   color=colors.get(method), edgecolor="none")
        mean, ci = _mean_ci(vals)
        ax.errorbar(i, mean, yerr=ci, fmt="D", color="black", capsize=3, ms=4, zorder=5)
    ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    ax.set_ylabel("Maximum lap progress")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
    ax.grid(axis="y", alpha=.25)
    ax.set_title("(a) Max progress")

    # Lap time is conditioned on successful episodes and labeled as such.
    ax = axes[1]
    for i, method in enumerate(methods):
        vals = [t for r in results[method] if r["success"] for t in r["lap_times_sec"]]
        if vals:
            mean, ci = _mean_ci(vals)
            ax.errorbar(i, mean, yerr=ci, fmt="o", color=colors.get(method),
                        capsize=4, ms=5, lw=1.5)
            ax.scatter(np.full(len(vals), i) + np.linspace(-.08, .08, len(vals)), vals,
                       s=14, color=colors.get(method), alpha=.45, edgecolor="none")
    ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    ax.set_ylabel("Lap time (s)")
    ax.grid(axis="y", alpha=.25)
    ax.set_title("(b) Successful lap time")

    # Wilson 95% interval for a Bernoulli success rate.
    ax = axes[2]
    for i, method in enumerate(methods):
        runs = results[method]
        n = len(runs)
        if not n:
            continue
        p = sum(r["success"] for r in runs) / n
        z = 1.96
        den = 1 + z*z/n
        center = (p + z*z/(2*n))/den
        half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
        ax.errorbar(i, p, yerr=[[p-max(0., center-half)], [min(1., center+half)-p]],
                    fmt="o", color=colors.get(method), capsize=4, ms=5)
        ax.text(i, min(1.05, p + .09), f"{sum(r['success'] for r in runs)}/{n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
    ax.grid(axis="y", alpha=.25)
    ax.set_title("(c) Lap completion")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"evaluation_summary.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_checkpoint", required=True,
                        help="OPOSM checkpoint; evaluated as direct and CEM-MPC.")
    parser.add_argument("--normalizer_dir", required=True,
                        help="Directory containing this model's state/action normalizer JSONs.")
    parser.add_argument("--baseline_checkpoints", nargs="+", required=True,
                        help="One or more PPO/SAC .zip baseline checkpoints (checkpoint seeds/models).")
    parser.add_argument("--baseline_algorithm", choices=("auto", "ppo", "sac"), default="auto",
                        help="Algorithm for baseline checkpoints; auto detects PPO/SAC from filenames.")
    parser.add_argument("--eval_maps", nargs="+", default=None)
    parser.add_argument("--include_train_maps", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--max_episode_seconds", type=float, default=120.)
    parser.add_argument("--output_dir", default="outputs/evaluation")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("Provide at least one evaluation seed")

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    state_norm = Normalizer.load(os.path.join(args.normalizer_dir, "state_normalizer.json"))
    action_norm = Normalizer.load(os.path.join(args.normalizer_dir, "action_normalizer.json"))
    model = load_oposm(cfg, args.model_checkpoint, device)
    train_maps = set(cfg.collection.map_names)
    maps = list(args.eval_maps) if args.eval_maps else [
        m for m in discover_available_maps(cfg.data.maps_root)
        if args.include_train_maps or m not in train_maps]
    if not maps:
        parser.error("No maps selected; pass --eval_maps or configure valid held-out maps")
    os.makedirs(args.output_dir, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {"RL baseline": [], "OPOSM direct": [], "OPOSM CEM-MPC": []}

    baseline_models = []
    for path in args.baseline_checkpoints:
        algo = args.baseline_algorithm
        if algo == "auto":
            stem = os.path.basename(path).lower()
            algo = "sac" if "sac" in stem else "ppo" if "ppo" in stem else ""
        if not algo:
            parser.error(f"Cannot infer PPO or SAC for {path}; set --baseline_algorithm")
        baseline_models.append((PPO if algo == "ppo" else SAC).load(path, device="auto"))
    for map_name in maps:
        for seed in args.seeds:
            print(f"[evaluate_all] map={map_name} seed={seed}")
            # Multiple baseline checkpoints are pooled as model seeds and
            # over the same evaluation seeds, with checkpoint identity retained.
            for model_idx, baseline in enumerate(baseline_models):
                _seed_everything(seed)
                row = evaluate_ppo_episode(baseline, args.config, map_name, seed, args.max_episode_seconds)
                row["checkpoint"] = os.path.abspath(args.baseline_checkpoints[model_idx])
                row["checkpoint_index"] = model_idx
                results["RL baseline"].append(row)
            for mode, label in (("direct", "OPOSM direct"), ("cem", "OPOSM CEM-MPC")):
                _seed_everything(seed)
                row = evaluate_oposm_episode(model, cfg, state_norm, action_norm,
                                             device, map_name, seed, mode)
                row["checkpoint"] = os.path.abspath(args.model_checkpoint)
                results[label].append(row)

    report = {
        "metadata": {
            "config": os.path.abspath(args.config),
            "model_checkpoint": os.path.abspath(args.model_checkpoint),
            "baseline_checkpoints": [os.path.abspath(p) for p in args.baseline_checkpoints],
            "maps": maps, "seeds": args.seeds,
            "success_definition": "at least one complete lap in an episode",
            "progress_definition": "maximum projected progress from episode start divided by lap length",
            "lap_time_definition": "simulated time to complete a lap; aggregated over successful episodes only",
            "intervals": "95% normal confidence intervals for means; Wilson intervals for success rates",
        },
        "summary": {method: summarize(runs) for method, runs in results.items()},
        "per_map_summary": {
            map_name: {
                method: summarize([r for r in runs if r["map_name"] == map_name])
                for method, runs in results.items()
            }
            for map_name in maps
        },
        "runs": results,
    }
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(report, f, indent=2)
    make_figures(results, args.output_dir)
    print(f"[evaluate_all] saved JSON and figures under {args.output_dir}")


if __name__ == "__main__":
    main()
